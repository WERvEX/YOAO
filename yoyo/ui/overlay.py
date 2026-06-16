"""Detection overlay window — visualizes detections, FPS, and status.

Uses OpenCV named windows for simplicity and performance.
Can be disabled via config for minimal-overhead operation.

Thread-safe: receives frame + detection data from the pipeline's
detector thread callback and renders in the main thread via update().
"""

import logging
import threading
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np

from ..detector.postprocessing import Detection

logger = logging.getLogger(__name__)


# Visual constants
COLOR_TARGET = (0, 255, 0)        # Green — selected target
COLOR_DETECTION = (0, 165, 255)   # Orange — other detections
COLOR_AIM_POINT = (0, 0, 255)     # Red — aim point
COLOR_STATUS_ON = (0, 255, 0)     # Green — aiming active
COLOR_STATUS_OFF = (0, 0, 255)    # Red — aiming inactive
COLOR_FPS = (255, 255, 255)       # White — FPS text
COLOR_CROSSHAIR = (255, 255, 0)   # Cyan — crosshair
LINE_THICKNESS = 2
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.5


class DetectionOverlay:
    """Transparent-style overlay window showing detection results.

    Usage:
        overlay = DetectionOverlay(enabled=True, show_boxes=True, ...)
        pipeline.add_frame_callback(overlay.on_frame)
        # In main loop:
        while running:
            overlay.update()
            if overlay.should_close:
                break
        overlay.destroy()
    """

    WINDOW_NAME = "YOAO — Detection Overlay"

    def __init__(
        self,
        enabled: bool = True,
        show_boxes: bool = True,
        show_labels: bool = True,
        show_fps: bool = True,
        show_status: bool = True,
        window_alpha: float = 0.7,
    ):
        """Initialize the overlay.

        Args:
            enabled: Show the overlay window.
            show_boxes: Draw bounding boxes.
            show_labels: Show class/confidence text.
            show_fps: Display FPS counter.
            show_status: Display aiming active/inactive indicator.
            window_alpha: Window opacity (0-1). Note: OpenCV doesn't support
                         true transparency; this adjusts the overlay blending.
        """
        self._enabled = enabled
        self._show_boxes = show_boxes
        self._show_labels = show_labels
        self._show_fps = show_fps
        self._show_status = show_status
        self._window_alpha = window_alpha

        # Shared state (written by detector callback, read by main thread)
        self._lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_detections: List[Detection] = []
        self._latest_target: Optional[Detection] = None
        self._active = False
        self._fps = 0.0
        self._detect_ms = 0.0

        self._should_close = False

        if self._enabled:
            cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.WINDOW_NAME, 960, 540)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, val: bool) -> None:
        self._enabled = val
        if not val:
            cv2.destroyWindow(self.WINDOW_NAME)
        else:
            cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL)

    @property
    def should_close(self) -> bool:
        return self._should_close

    def on_frame(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        target: Optional[Detection],
        stats,
    ) -> None:
        """Callback from pipeline detector thread. Stores data for rendering.

        Args:
            frame: The captured frame (BGR).
            detections: All detections for this frame.
            target: The selected target (or None).
            stats: PipelineStats snapshot.
        """
        if not self._enabled:
            return

        with self._lock:
            # Store a copy to avoid holding the frame buffer too long
            self._latest_frame = frame.copy() if frame is not None else None
            self._latest_detections = detections
            self._latest_target = target
            self._active = stats.active
            self._fps = stats.detect_fps
            self._detect_ms = stats.detect_time_ms

    def update(self) -> None:
        """Render and display the overlay. Call from main thread in a loop."""
        if not self._enabled:
            return

        with self._lock:
            frame = self._latest_frame
            detections = list(self._latest_detections)
            target = self._latest_target
            active = self._active
            fps = self._fps
            detect_ms = self._detect_ms

        if frame is None:
            return

        display = frame.copy()

        h, w = display.shape[:2]
        crosshair = (w // 2, h // 2)

        # Draw crosshair
        cv2.drawMarker(
            display, crosshair, COLOR_CROSSHAIR,
            markerType=cv2.MARKER_CROSS, markerSize=10, thickness=1,
        )

        # Draw all detections
        if self._show_boxes:
            for det in detections:
                is_target = (target is not None and det is target)
                color = COLOR_TARGET if is_target else COLOR_DETECTION
                thickness = LINE_THICKNESS + 1 if is_target else LINE_THICKNESS

                x1, y1, x2, y2 = det.bbox
                cv2.rectangle(display, (x1, y1), (x2, y2), color, thickness)

                # Draw aim point
                if det.aim_point:
                    cv2.circle(display, det.aim_point, 4, COLOR_AIM_POINT, -1)

                # Label
                if self._show_labels:
                    label = f"cls:{det.class_id} {det.confidence:.2f}"
                    (tw, th), _ = cv2.getTextSize(label, FONT, FONT_SCALE, 1)
                    cv2.rectangle(
                        display,
                        (x1, y1 - th - 4),
                        (x1 + tw + 4, y1),
                        color,
                        -1,
                    )
                    cv2.putText(
                        display, label, (x1 + 2, y1 - 2),
                        FONT, FONT_SCALE, (0, 0, 0), 1,
                    )

                    # Mark if this is the target
                    if is_target:
                        target_label = "TARGET"
                        cv2.putText(
                            display, target_label,
                            (x1, y2 + 15), FONT, 0.6, COLOR_TARGET, 2,
                        )

        # Status bar at top
        if self._show_status:
            status_text = "AIM: ON" if active else "AIM: OFF"
            status_color = COLOR_STATUS_ON if active else COLOR_STATUS_OFF
            cv2.putText(
                display, status_text, (10, 20),
                FONT, 0.6, status_color, 2,
            )

        if self._show_fps:
            fps_text = f"Detect: {fps:.1f} FPS | {detect_ms:.1f}ms"
            cv2.putText(
                display, fps_text, (10, 45),
                FONT, FONT_SCALE, COLOR_FPS, 1,
            )

        # Show the frame
        cv2.imshow(self.WINDOW_NAME, display)

        # Check for window close or keypress
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            self._should_close = True

    def destroy(self) -> None:
        """Close the overlay window."""
        try:
            cv2.destroyWindow(self.WINDOW_NAME)
        except Exception:
            pass
        self._enabled = False
