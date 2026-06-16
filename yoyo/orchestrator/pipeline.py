"""AimBot pipeline orchestrator — coordinates capture, detection, and mouse threads.

Architecture:
    ┌──────────┐   frame_deque    ┌──────────┐   result_queue   ┌──────────┐
    │ Capture  │ ────────────────→│ Detector │ ────────────────→│  Mouse   │
    │ Thread   │  (maxlen=2)      │ Thread   │  (maxsize=1)     │ Thread   │
    └──────────┘                  └──────────┘                  └──────────┘

- frame_deque: bounded deque, auto-drops oldest frames when full (always freshest)
- result_queue: Queue(1), always overwritten with latest detection
- The main thread handles hotkey monitoring and overlay updates
"""

import logging
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from ..capture import DXGICapture
from ..detector import YOLODetector
from ..detector.postprocessing import Detection
from ..mouse import MouseController

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Single detection cycle result, passed from detector to mouse thread."""
    detections: List[Detection]
    target: Optional[Detection]
    frame_timestamp: float
    detect_time_ms: float
    frame: Optional[np.ndarray] = None  # For overlay (optional)


@dataclass
class PipelineStats:
    """Running statistics for the pipeline."""
    capture_fps: float = 0.0
    detect_fps: float = 0.0
    detect_time_ms: float = 0.0
    last_detection: Optional[Detection] = None
    active: bool = False
    running: bool = False


class AimBotPipeline:
    """Orchestrates the capture→detect→mouse pipeline with multi-threading.

    Manages thread lifecycle, queue coordination, hotkey toggling,
    and provides stats for the overlay/UI.

    Usage:
        pipeline = AimBotPipeline(app_config)
        pipeline.start()
        # ... main loop polls pipeline.stats for overlay updates ...
        pipeline.stop()
    """

    def __init__(
        self,
        capture: DXGICapture,
        detector: YOLODetector,
        mouse: MouseController,
        toggle_hotkey: str = "ctrl+shift+a",
        quit_hotkey: str = "ctrl+shift+q",
    ):
        """Initialize the pipeline.

        Args:
            capture: Configured DXGICapture instance.
            detector: Configured YOLODetector instance.
            mouse: Configured MouseController instance.
            toggle_hotkey: Hotkey string to toggle aiming on/off.
            quit_hotkey: Hotkey string to quit the application.
        """
        self._capture = capture
        self._detector = detector
        self._mouse = mouse
        self._toggle_hotkey = toggle_hotkey
        self._quit_hotkey = quit_hotkey

        # Thread-safe queues
        self._frame_deque: deque = deque(maxlen=2)  # Capture → Detector
        self._result_queue: queue.Queue = queue.Queue(maxsize=1)  # Detector → Mouse

        # State
        self._active = False  # Aiming enabled?
        self._running = False
        self._quit = False

        # Threads
        self._detect_thread: Optional[threading.Thread] = None
        self._mouse_thread: Optional[threading.Thread] = None

        # Stats (thread-safe via lock)
        self._stats_lock = threading.Lock()
        self._stats = PipelineStats()

        # Callbacks for overlay updates (called from detector thread)
        self._on_frame_callbacks: List[Callable] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def stats(self) -> PipelineStats:
        """Get a copy of current pipeline statistics (thread-safe)."""
        with self._stats_lock:
            return PipelineStats(
                capture_fps=self._stats.capture_fps,
                detect_fps=self._stats.detect_fps,
                detect_time_ms=self._stats.detect_time_ms,
                last_detection=self._stats.last_detection,
                active=self._stats.active,
                running=self._stats.running,
            )

    @property
    def is_active(self) -> bool:
        return self._active

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def should_quit(self) -> bool:
        return self._quit

    def add_frame_callback(self, callback: Callable) -> None:
        """Register a callback for frame+detection results (for overlay).

        Callback signature: callback(frame: np.ndarray, detections: List[Detection],
                                      target: Optional[Detection], stats: PipelineStats)
        Called from the detector thread — keep it fast.
        """
        self._on_frame_callbacks.append(callback)

    def start(self) -> bool:
        """Start the pipeline: capture, detector, and mouse threads.

        Returns:
            True if started successfully.
        """
        if self._running:
            return True

        # Start capture
        if not self._capture.start():
            logger.error("Failed to start screen capture.")
            return False

        # Start background threads
        self._running = True
        self._quit = False

        self._detect_thread = threading.Thread(
            target=self._detector_loop,
            daemon=True,
            name="yoyo-detector",
        )
        self._mouse_thread = threading.Thread(
            target=self._mouse_loop,
            daemon=True,
            name="yoyo-mouse",
        )

        self._detect_thread.start()
        self._mouse_thread.start()

        # Start hotkey monitoring in a separate daemon thread
        self._hotkey_thread = threading.Thread(
            target=self._hotkey_loop,
            daemon=True,
            name="yoyo-hotkeys",
        )
        self._hotkey_thread.start()

        with self._stats_lock:
            self._stats.running = True

        logger.info("Pipeline started.")
        return True

    def stop(self) -> None:
        """Stop all pipeline threads and release resources."""
        self._running = False
        self._active = False
        self._quit = True

        # Wake up blocking queues
        try:
            self._result_queue.put_nowait(None)
        except queue.Full:
            pass

        # Join threads
        for thread in [self._detect_thread, self._mouse_thread]:
            if thread and thread.is_alive():
                thread.join(timeout=2.0)

        self._capture.stop()

        with self._stats_lock:
            self._stats.running = False
            self._stats.active = False

        logger.info("Pipeline stopped.")

    def toggle_aiming(self) -> bool:
        """Toggle aiming on/off. Returns new state."""
        self._active = not self._active
        with self._stats_lock:
            self._stats.active = self._active
        state = "ON" if self._active else "OFF"
        logger.info(f"Aiming: {state}")
        print(f"\n[AimBot] Aiming: {state}")
        return self._active

    # ------------------------------------------------------------------
    # Thread Loops
    # ------------------------------------------------------------------

    def _detector_loop(self) -> None:
        """Detector thread: grab frames from capture, run inference, push results."""
        if not self._detector.is_initialized:
            logger.error("Detector not initialized — detector thread exiting.")
            return

        frame_count = 0
        last_fps_report = time.perf_counter()

        while self._running:
            # Get latest frame from capture (blocking poll)
            frame = self._capture.get_latest_frame()

            if frame is None:
                time.sleep(0.001)
                continue

            # Run detection
            t0 = time.perf_counter()
            detections = self._detector.detect(frame.image)
            dt_ms = (time.perf_counter() - t0) * 1000

            frame_count += 1

            # Select target
            target = self._detector.select_best(
                detections,
                crosshair_pos=None,  # Use center of frame = center of capture region
                frame_size=(frame.width, frame.height),
            )

            # Push result to mouse thread (non-blocking, overwrite old)
            result = PipelineResult(
                detections=detections,
                target=target,
                frame_timestamp=frame.timestamp,
                detect_time_ms=dt_ms,
                frame=frame.image,  # For overlay
            )
            try:
                self._result_queue.put_nowait(result)
            except queue.Full:
                # Discard stale result — mouse thread is behind
                pass

            # Update stats periodically
            now = time.perf_counter()
            if now - last_fps_report >= 1.0:
                with self._stats_lock:
                    self._stats.detect_fps = frame_count / (now - last_fps_report)
                    self._stats.detect_time_ms = dt_ms
                    self._stats.last_detection = target
                    self._stats.active = self._active
                frame_count = 0
                last_fps_report = now

            # Notify overlay callbacks
            for cb in self._on_frame_callbacks:
                try:
                    cb(frame.image, detections, target, self.stats)
                except Exception:
                    pass

    def _mouse_loop(self) -> None:
        """Mouse thread: consume detection results and move the cursor."""
        last_target_pos: Optional[Tuple[int, int]] = None

        while self._running:
            try:
                result: Optional[PipelineResult] = self._result_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if result is None:
                continue

            # Only move if aiming is active and we have a target
            if self._active and result.target is not None:
                aim_point = result.target.aim_point or result.target.center
                ax, ay = aim_point

                # Convert frame-local coordinates to absolute screen coordinates
                region = self._capture.get_region()
                if region is not None:
                    rx, ry, _, _ = region
                    ax += rx
                    ay += ry

                # Avoid jitter: only move if target changed significantly
                if last_target_pos is not None:
                    lx, ly = last_target_pos
                    if abs(ax - lx) < 3 and abs(ay - ly) < 3:
                        continue  # Same target, skip redundant move

                self._mouse.move_to(ax, ay)
                last_target_pos = (ax, ay)
            else:
                last_target_pos = None

    def _hotkey_loop(self) -> None:
        """Monitor global hotkeys for toggle and quit."""
        try:
            from pynput import keyboard
        except ImportError:
            logger.warning(
                "pynput not installed. Hotkeys disabled.\n"
                "  Install with: pip install pynput"
            )
            return

        # Parse hotkey strings
        toggle_combo = self._parse_hotkey(self._toggle_hotkey)
        quit_combo = self._parse_hotkey(self._quit_hotkey)

        # Track pressed keys
        pressed: set = set()

        def on_press(key):
            key_str = self._key_to_string(key)
            if key_str:
                pressed.add(key_str)

            # Check toggle combo
            if toggle_combo and toggle_combo.issubset(pressed):
                # Check that only the combo keys are pressed (or at least combo is fully active)
                if len(pressed) >= len(toggle_combo):
                    self.toggle_aiming()

            # Check quit combo
            if quit_combo and quit_combo.issubset(pressed):
                if len(pressed) >= len(quit_combo):
                    self._quit = True

        def on_release(key):
            key_str = self._key_to_string(key)
            if key_str:
                pressed.discard(key_str)

        listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        listener.daemon = True
        listener.start()

        # Keep thread alive while pipeline runs
        while self._running:
            time.sleep(0.1)

        listener.stop()

    # ------------------------------------------------------------------
    # Hotkey parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_hotkey(hotkey_str: str) -> set:
        """Parse a hotkey string like 'ctrl+shift+a' into a set of key names."""
        if not hotkey_str:
            return set()
        parts = hotkey_str.lower().strip().split("+")
        result = set()
        for p in parts:
            p = p.strip()
            # Normalize common modifiers
            if p in ("ctrl", "control"):
                result.add("ctrl")
            elif p in ("shift",):
                result.add("shift")
            elif p in ("alt",):
                result.add("alt")
            else:
                result.add(p)
        return result

    @staticmethod
    def _key_to_string(key) -> Optional[str]:
        """Convert a pynput key to a normalized string."""
        try:
            if hasattr(key, 'char') and key.char is not None:
                return key.char.lower()
        except Exception:
            pass

        try:
            name = key.name
            if name:
                # Normalize pynput modifier names
                if name in ('ctrl_l', 'ctrl_r'):
                    return 'ctrl'
                if name in ('shift_l', 'shift_r'):
                    return 'shift'
                if name in ('alt_l', 'alt_r'):
                    return 'alt'
                if name.startswith('ctrl'):
                    return 'ctrl'
                if name.startswith('shift'):
                    return 'shift'
                if name.startswith('alt'):
                    return 'alt'
                return name.lower()
        except Exception:
            pass

        return str(key).lower()
