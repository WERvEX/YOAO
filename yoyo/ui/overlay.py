"""Transparent screen overlay using PyQt6 — drawn directly on the desktop.

Creates a frameless, topmost, click-through transparent window positioned
over the capture region. Detection boxes, labels, FPS, and status are
rendered using QPainter.

Key properties:
  - Qt.WA_TranslucentBackground: per-pixel transparency
  - Qt.WindowStaysOnTopHint: always above game window
  - Qt.WA_TransparentForMouseEvents: click-through (no input interception)
  - pyqtSignal: thread-safe update from detector thread → Qt main thread
"""

import logging
import threading
from typing import List, Optional, Tuple

import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QFont, QPainter, QPen
from PyQt6.QtWidgets import QWidget

from ..detector.postprocessing import Detection

logger = logging.getLogger(__name__)


class ScreenOverlay(QWidget):
    """Transparent screen overlay rendered with PyQt6 + QPainter.

    Renders detection bounding boxes, aim points, FPS, and status on a
    frameless, click-through, topmost window.

    Usage:
        overlay = ScreenOverlay(region=(0, 0, 1920, 1080))
        overlay.show()
        # ... from detector callback thread:
        pipeline.add_frame_callback(overlay.on_frame)
        # ... main thread runs Qt event loop:
        app.exec()
    """

    # Signal emitted from detector thread to trigger repaint on Qt main thread
    _frame_ready = pyqtSignal()

    def __init__(
        self,
        region: Optional[Tuple[int, int, int, int]] = None,
        enabled: bool = True,
        show_boxes: bool = True,
        show_labels: bool = True,
        show_fps: bool = True,
        show_status: bool = True,
    ):
        """Initialize the screen overlay.

        Args:
            region: Screen region to cover as (x, y, w, h). None = full primary screen.
            enabled: Whether the overlay is visible.
            show_boxes: Draw detection bounding boxes.
            show_labels: Show class/confidence labels.
            show_fps: Display FPS counter.
            show_status: Show aiming active/inactive indicator.
        """
        super().__init__()

        self._enabled = enabled
        self._show_boxes = show_boxes
        self._show_labels = show_labels
        self._show_fps = show_fps
        self._show_status = show_status

        # Window position and size
        if region is not None:
            self._x, self._y, self._w, self._h = region
        else:
            from PyQt6.QtGui import QGuiApplication
            screen = QGuiApplication.primaryScreen()
            if screen:
                geom = screen.geometry()
                self._x, self._y = 0, 0
                self._w, self._h = geom.width(), geom.height()
            else:
                self._x, self._y, self._w, self._h = 0, 0, 1920, 1080

        # Thread-safe rendering data
        self._lock = threading.Lock()
        self._detections: List[Detection] = []
        self._target: Optional[Detection] = None
        self._active: bool = False
        self._fps: float = 0.0
        self._detect_ms: float = 0.0

        # Setup window
        self._setup_window()

        # Connect signal: emitted from detector thread → repaint on Qt thread
        self._frame_ready.connect(self._on_frame_ready)

    # ------------------------------------------------------------------
    # Public API (compatible with old Win32 ScreenOverlay)
    # ------------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, val: bool) -> None:
        self._enabled = val
        if val:
            self.show()
        else:
            self.hide()

    def show(self) -> None:
        """Show the overlay window."""
        if self._enabled:
            super().show()

    def hide(self) -> None:
        """Hide the overlay window."""
        super().hide()

    def update(
        self,
        detections: List[Detection],
        target: Optional[Detection],
        active: bool = False,
        fps: float = 0.0,
        detect_ms: float = 0.0,
    ) -> None:
        """Thread-safe: store detection data and request repaint."""
        with self._lock:
            self._detections = list(detections)
            self._target = target
            self._active = active
            self._fps = fps
            self._detect_ms = detect_ms
        # Signal Qt main thread to repaint (thread-safe)
        self._frame_ready.emit()

    def on_frame(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        target: Optional[Detection],
        stats,
    ) -> None:
        """Callback compatible with AimBotPipeline.add_frame_callback.

        Called from detector thread. `frame` is ignored — overlay renders
        on the screen, not on a frame copy.
        """
        if not self._enabled:
            return
        self.update(
            detections=detections,
            target=target,
            active=stats.active,
            fps=stats.detect_fps,
            detect_ms=stats.detect_time_ms,
        )

    # ------------------------------------------------------------------
    # Qt Window Setup
    # ------------------------------------------------------------------

    def _setup_window(self) -> None:
        """Configure the overlay as a frameless, transparent, topmost, click-through window."""
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)

        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )

        self.setGeometry(self._x, self._y, self._w, self._h)
        self.setStyleSheet("background: transparent;")

        # Use a fixed-size font for status/fps text
        self._font = QFont("Consolas", 12)
        self._font.setStyleHint(QFont.StyleHint.Monospace)

    # ------------------------------------------------------------------
    # Qt Slots
    # ------------------------------------------------------------------

    def _on_frame_ready(self) -> None:
        """Slot: called on Qt main thread when new frame data is available."""
        # Call QWidget.update() — NOT our override which would recurse infinitely.
        super().update()

    # ------------------------------------------------------------------
    # QPainter Rendering
    # ------------------------------------------------------------------

    def paintEvent(self, event) -> None:
        """Render detection data via QPainter (called by Qt on main thread)."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Read latest data under lock
        with self._lock:
            detections = list(self._detections)
            target = self._target
            active = self._active
            fps = self._fps
            detect_ms = self._detect_ms

        # --- Crosshair at screen center ---
        cx, cy = self._w // 2, self._h // 2
        painter.setPen(QPen(QColor(0, 255, 0), 1))  # Green
        painter.drawLine(cx - 12, cy, cx + 12, cy)
        painter.drawLine(cx, cy - 12, cx, cy + 12)

        # --- Detection boxes ---
        if self._show_boxes:
            pen_target = QPen(QColor(0, 255, 0), 2)    # Green for target
            pen_detect = QPen(QColor(255, 165, 0), 2)  # Orange for others
            pen_aim = QPen(QColor(255, 0, 0), 2)       # Red for aim point

            for det in detections:
                is_target = (target is not None and det is target)
                pen = pen_target if is_target else pen_detect
                painter.setPen(pen)

                x1, y1, x2, y2 = det.bbox
                painter.drawRect(x1, y1, x2 - x1, y2 - y1)

                # Aim point cross
                if det.aim_point:
                    ax, ay = det.aim_point
                    painter.setPen(pen_aim)
                    painter.drawLine(ax - 5, ay, ax + 5, ay)
                    painter.drawLine(ax, ay - 5, ax, ay + 5)

                # Label
                if self._show_labels:
                    label = f"[{det.class_id}] {det.confidence:.2f}"
                    painter.setPen(QColor(255, 255, 255))
                    # Draw shadow
                    painter.drawText(x1 + 2, y1 - 5, label)
                    if is_target:
                        painter.setPen(QPen(QColor(0, 255, 0)))
                        painter.drawText(x1 + 1, y2 + 16, "TARGET")

        # --- Status bar (top-left) ---
        if self._show_status:
            status_text = "AIM: ON" if active else "AIM: OFF"
            color = QColor(0, 255, 0) if active else QColor(255, 0, 0)
            painter.setPen(color)
            painter.setFont(self._font)
            painter.drawText(12, 20, status_text)

        if self._show_fps:
            fps_text = f"Detect: {fps:.1f} FPS | {detect_ms:.1f}ms"
            painter.setPen(QColor(255, 255, 255))
            painter.setFont(self._font)
            painter.drawText(12, 40, fps_text)

        painter.end()
