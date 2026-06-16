"""DXGI screen capture backend using dxcam.

Provides high-performance, low-latency screen capture on Windows via
the DXGI Desktop Duplication API. dxcam handles the D3D11 texture→numpy
zero-copy conversion internally.

Supports both auto-detection of game windows by title and manual region capture.
"""

import threading
import time
from typing import Optional, Tuple

import dxcam
import numpy as np

from .frame import Frame
from .window_finder import find_window_by_title


class DXGICapture:
    """High-performance screen capture using DXGI Desktop Duplication.

    Wraps dxcam to provide threaded frame capture with configurable
    FPS and region of interest.

    Usage:
        cap = DXGICapture(region=(0, 0, 1920, 1080), target_fps=60)
        cap.start()
        while running:
            frame = cap.get_latest_frame()
            if frame:
                process(frame)
        cap.stop()
    """

    def __init__(
        self,
        region: Optional[Tuple[int, int, int, int]] = None,
        window_title: str = "",
        target_fps: int = 60,
        output_color: str = "BGR",
    ):
        """Initialize the DXGI capture backend.

        Args:
            region: Capture region as (x, y, width, height). Overrides window_title.
            window_title: Keyword to auto-detect game window. Used if region is None.
            target_fps: Desired capture frame rate.
            output_color: Color format — "BGR" or "BGRA". BGR is 3-channel for OpenCV/YOLO.
        """
        self._region = region
        self._window_title = window_title
        self._target_fps = target_fps
        self._output_color = output_color

        self._camera: Optional[dxcam.DXCamera] = None
        self._running = False
        self._latest_frame: Optional[Frame] = None
        self._lock = threading.Lock()
        self._capture_region: Optional[Tuple[int, int, int, int]] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> bool:
        """Start screen capture in a background thread.

        Returns:
            True if capture started successfully, False otherwise.
        """
        if self._running:
            return True

        # Resolve capture region
        region = self._resolve_region()
        if region is None:
            print("[DXGICapture] Error: No capture region configured and no window found.")
            return False

        x, y, w, h = region
        self._capture_region = region

        # Initialize dxcam camera
        # dxcam region is (left, top, right, bottom)
        dxcam_region = (x, y, x + w, y + h)

        try:
            self._camera = dxcam.create(
                output_idx=0,             # Primary display
                output_color=self._output_color,
                max_buffer_len=64,
            )
        except Exception as e:
            print(f"[DXGICapture] Failed to create dxcam camera: {e}")
            print("  Hint: dxcam requires a GPU with DXGI support (any modern GPU).")
            return False

        if self._camera is None:
            print("[DXGICapture] dxcam.create() returned None — no supported GPU found.")
            return False

        # Start threaded capture
        self._running = True
        self._camera.start(
            target_fps=self._target_fps,
            video_mode=True,
            region=dxcam_region,
        )

        # Start a consumer thread that reads from dxcam as fast as possible
        self._grab_thread = threading.Thread(
            target=self._grab_loop,
            daemon=True,
            name="dxcam-grab",
        )
        self._grab_thread.start()

        print(f"[DXGICapture] Started — region={region}, target_fps={self._target_fps}")
        return True

    def stop(self) -> None:
        """Stop screen capture and release resources."""
        self._running = False
        if self._grab_thread and self._grab_thread.is_alive():
            self._grab_thread.join(timeout=2.0)

        if self._camera:
            try:
                self._camera.stop()
            except Exception:
                pass

        self._camera = None
        print("[DXGICapture] Stopped.")

    def get_latest_frame(self) -> Optional[Frame]:
        """Return the most recently captured frame, or None.

        Thread-safe. Called by the detector thread.
        """
        with self._lock:
            return self._latest_frame

    def get_region(self) -> Optional[Tuple[int, int, int, int]]:
        """Return the active capture region as (x, y, w, h)."""
        return self._capture_region

    @property
    def is_running(self) -> bool:
        return self._running

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _resolve_region(self) -> Optional[Tuple[int, int, int, int]]:
        """Determine the capture region from config or window detection."""
        # Explicit region takes priority
        if self._region is not None:
            x, y, w, h = self._region
            if w > 0 and h > 0:
                return (x, y, w, h)

        # Try auto-detect by window title
        if self._window_title:
            info = find_window_by_title(self._window_title)
            if info:
                print(f"[DXGICapture] Found window: \"{info.title}\" at {info.x},{info.y} {info.width}x{info.height}")
                return (info.x, info.y, info.width, info.height)
            else:
                print(f"[DXGICapture] No window matching \"{self._window_title}\" found.")

        return None

    def _grab_loop(self) -> None:
        """Continuously grab frames from dxcam in a dedicated thread."""
        if self._camera is None:
            return

        # Track frame timing for FPS reporting
        last_report = time.perf_counter()
        frame_count = 0

        while self._running:
            try:
                # dxcam.get_latest_frame() blocks briefly waiting for a new frame
                img = self._camera.get_latest_frame()
            except Exception:
                time.sleep(0.001)
                continue

            if img is None:
                time.sleep(0.001)
                continue

            # img should already be numpy array in BGR or BGRA format
            if not isinstance(img, np.ndarray):
                img = np.array(img)

            frame = Frame(
                image=img,
                timestamp=time.perf_counter(),
                region=self._capture_region,
            )

            with self._lock:
                self._latest_frame = frame

            # Periodic FPS logging
            frame_count += 1
            now = time.perf_counter()
            if now - last_report >= 5.0:
                fps = frame_count / (now - last_report)
                print(f"[DXGICapture] {fps:.1f} FPS")
                frame_count = 0
                last_report = now
