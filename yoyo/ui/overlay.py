"""Transparent screen overlay — drawn directly on the desktop using Win32 layered windows.

Creates a click-through, topmost, transparent window positioned over the capture
region. Detection boxes, labels, FPS, and status are rendered using GDI primitives
with no external window framework overhead.

Key properties:
  - WS_EX_LAYERED + WS_EX_TRANSPARENT: transparent, click-through (doesn't steal focus)
  - WS_EX_TOPMOST: stays above the game window
  - UpdateLayeredWindow: per-pixel alpha blending for smooth rendering
  - GDI drawing: minimal overhead, no OpenCV window event loop
"""

import ctypes
import ctypes.wintypes
import logging
import threading
from typing import List, Optional, Tuple

import numpy as np

from ..detector.postprocessing import Detection

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Missing wintypes (Python version-dependent availability)
# ---------------------------------------------------------------------------
if not hasattr(ctypes.wintypes, "HCURSOR"):
    ctypes.wintypes.HCURSOR = ctypes.wintypes.HANDLE


# ---------------------------------------------------------------------------
# Custom Win32 structures
# ---------------------------------------------------------------------------

WNDPROC = ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_long, ctypes.c_uint,
                              ctypes.c_long, ctypes.c_long)


class WNDCLASSW(ctypes.Structure):
    _fields_ = [
        ("style", ctypes.wintypes.UINT),
        ("lpfnWndProc", WNDPROC),
        ("cbClsExtra", ctypes.c_int),
        ("cbWndExtra", ctypes.c_int),
        ("hInstance", ctypes.wintypes.HINSTANCE),
        ("hIcon", ctypes.wintypes.HICON),
        ("hCursor", ctypes.wintypes.HCURSOR),
        ("hbrBackground", ctypes.wintypes.HBRUSH),
        ("lpszMenuName", ctypes.wintypes.LPCWSTR),
        ("lpszClassName", ctypes.wintypes.LPCWSTR),
    ]


class BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [
        ("biSize", ctypes.c_uint32),
        ("biWidth", ctypes.c_int32),
        ("biHeight", ctypes.c_int32),
        ("biPlanes", ctypes.c_uint16),
        ("biBitCount", ctypes.c_uint16),
        ("biCompression", ctypes.c_uint32),
        ("biSizeImage", ctypes.c_uint32),
        ("biXPelsPerMeter", ctypes.c_int32),
        ("biYPelsPerMeter", ctypes.c_int32),
        ("biClrUsed", ctypes.c_uint32),
        ("biClrImportant", ctypes.c_uint32),
    ]

# ---------------------------------------------------------------------------
# Win32 constants
# ---------------------------------------------------------------------------
WS_EX_LAYERED = 0x00080000
WS_EX_TRANSPARENT = 0x00000020
WS_EX_TOPMOST = 0x00000008
WS_POPUP = 0x80000000
ULW_ALPHA = 0x00000002
AC_SRC_OVER = 0x00
AC_SRC_ALPHA = 0x01

# GDI
SRCCOPY = 0x00CC0020

# Colors (BGR for GDI)
COLOR_BG = 0x00000000        # Fully transparent
COLOR_TARGET = 0x0000FF00    # Green
COLOR_DETECTION = 0x00A5FF   # Orange
COLOR_AIM = 0x000000FF       # Red
COLOR_CROSSHAIR = 0x00FFFF00 # Cyan
COLOR_STATUS_ON = 0x0000FF00 # Green
COLOR_STATUS_OFF = 0x000000FF # Red
COLOR_TEXT = 0x00FFFFFF      # White


class ScreenOverlay:
    """Transparent screen overlay drawn directly on the Windows desktop.

    Renders detection bounding boxes, aim points, FPS, and status
    using GDI on a layered window. The window is click-through and
    topmost — it appears over the game without intercepting input.

    Usage:
        overlay = ScreenOverlay(region=(0, 0, 1920, 1080))
        overlay.show()
        # ... from detector callback thread:
        overlay.update(detections, target, active=True, fps=60.0, detect_ms=15.0)
        # ... on shutdown:
        overlay.hide()
    """

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
            region: Screen region to cover as (x, y, w, h). None = full virtual screen.
            enabled: Whether the overlay is visible.
            show_boxes: Draw detection bounding boxes.
            show_labels: Show class/confidence labels.
            show_fps: Display FPS counter.
            show_status: Show aiming active/inactive indicator.
        """
        self._enabled = enabled
        self._show_boxes = show_boxes
        self._show_labels = show_labels
        self._show_fps = show_fps
        self._show_status = show_status

        # Window dimensions
        if region is not None:
            self._x, self._y, self._w, self._h = region
        else:
            self._x = ctypes.windll.user32.GetSystemMetrics(0)
            self._y = ctypes.windll.user32.GetSystemMetrics(1)
            self._w = ctypes.windll.user32.GetSystemMetrics(78)  # SM_CXVIRTUALSCREEN
            self._h = ctypes.windll.user32.GetSystemMetrics(79)  # SM_CYVIRTUALSCREEN

        # GDI handles
        self._hwnd = None
        self._hdc = None
        self._mem_dc = None
        self._bitmap = None
        self._old_bitmap = None
        self._font: Optional[Tuple] = None  # (hfont, height)

        # Thread safety for update data
        self._lock = threading.Lock()
        self._visible = False

        # Pre-register window class
        self._register_class()

    # ------------------------------------------------------------------
    # Public API
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
        """Create and show the transparent overlay window."""
        if not self._enabled or self._visible:
            return

        self._create_window()
        self._visible = True

    def hide(self) -> None:
        """Destroy the overlay window."""
        if not self._visible:
            return

        self._destroy_gdi()
        if self._hwnd:
            ctypes.windll.user32.DestroyWindow(self._hwnd)
            self._hwnd = None
        self._visible = False

    def update(
        self,
        detections: List[Detection],
        target: Optional[Detection],
        active: bool = False,
        fps: float = 0.0,
        detect_ms: float = 0.0,
    ) -> None:
        """Render and display detection data on the overlay.

        Thread-safe — can be called from any thread. The actual
        GDI drawing happens synchronously (GDI is not thread-safe,
        but we serialize through the lock and the single rendering call).

        Args:
            detections: All detections for this frame.
            target: The selected target to aim at, or None.
            active: Whether aiming is currently active.
            fps: Detection FPS.
            detect_ms: Last detection time in milliseconds.
        """
        if not self._visible or not self._hwnd:
            return

        # Copy data under lock, then draw outside lock
        with self._lock:
            # Shallow copies — the Detection objects are immutable-ish for rendering
            dets = list(detections)
            tgt = target
            is_active = active
            cur_fps = fps
            cur_ms = detect_ms

        self._draw_frame(dets, tgt, is_active, cur_fps, cur_ms)

    def on_frame(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        target: Optional[Detection],
        stats,
    ) -> None:
        """Callback compatible with AimBotPipeline.add_frame_callback.

        Note: `frame` is accepted but ignored — the overlay renders
        graphics on top of the screen, not on the captured frame.
        `stats` is a PipelineStats object.
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
    # Win32 Window
    # ------------------------------------------------------------------

    def _register_class(self) -> None:
        """Register a Win32 window class for the overlay."""
        module_handle = ctypes.windll.kernel32.GetModuleHandleW(None)

        wndproc = ctypes.WINFUNCTYPE(
            ctypes.c_long, ctypes.c_long, ctypes.c_uint,
            ctypes.c_long, ctypes.c_long,
        )

        def _wnd_proc(hwnd, msg, wparam, lparam):
            # Minimal window proc — we don't handle input
            return ctypes.windll.user32.DefWindowProcW(hwnd, msg, wparam, lparam)

        self._wndproc = wndproc(_wnd_proc)

        class_name = "YOAOScreenOverlay"

        wc = WNDCLASSW()
        wc.lpfnWndProc = self._wndproc
        wc.hInstance = module_handle
        wc.lpszClassName = class_name
        wc.hbrBackground = ctypes.windll.gdi32.GetStockObject(5)  # NULL_BRUSH
        wc.style = 0
        wc.cbClsExtra = 0
        wc.cbWndExtra = 0
        wc.hIcon = None
        wc.hCursor = None

        atom = ctypes.windll.user32.RegisterClassW(ctypes.byref(wc))
        if atom == 0:
            err = ctypes.get_last_error()
            if err != 1410:  # ERROR_CLASS_ALREADY_EXISTS
                logger.error(f"RegisterClassW failed: {err}")
        self._class_name = class_name

    def _create_window(self) -> None:
        """Create the layered, transparent, topmost, click-through window."""
        module_handle = ctypes.windll.kernel32.GetModuleHandleW(None)

        ex_style = WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOPMOST
        style = WS_POPUP

        self._hwnd = ctypes.windll.user32.CreateWindowExW(
            ex_style,
            self._class_name,
            "YOAO Overlay",
            style,
            self._x, self._y, self._w, self._h,
            None,  # parent
            None,  # menu
            module_handle,
            None,  # lparam
        )

        if not self._hwnd:
            logger.error(f"CreateWindowExW failed: {ctypes.get_last_error()}")
            return

        # Setup GDI resources
        self._init_gdi()

        # Show the window
        ctypes.windll.user32.ShowWindow(self._hwnd, 1)  # SW_SHOWNORMAL
        # Make fully transparent initially
        ctypes.windll.user32.SetLayeredWindowAttributes(
            self._hwnd, 0, 255, 0x00000002  # LWA_ALPHA
        )

    # ------------------------------------------------------------------
    # GDI Drawing
    # ------------------------------------------------------------------

    def _init_gdi(self) -> None:
        """Initialize GDI resources for double-buffered drawing."""
        screen_dc = ctypes.windll.user32.GetDC(0)
        self._hdc = ctypes.windll.gdi32.CreateCompatibleDC(screen_dc)
        self._mem_dc = ctypes.windll.gdi32.CreateCompatibleDC(screen_dc)
        ctypes.windll.user32.ReleaseDC(0, screen_dc)

        # Create a 32-bit bitmap for per-pixel alpha
        bi = BITMAPINFOHEADER()
        bi.biSize = ctypes.sizeof(BITMAPINFOHEADER)
        bi.biWidth = self._w
        bi.biHeight = -self._h  # Negative = top-down DIB
        bi.biPlanes = 1
        bi.biBitCount = 32
        bi.biCompression = 0  # BI_RGB

        ppv_bits = ctypes.c_void_p()
        self._bitmap = ctypes.windll.gdi32.CreateDIBSection(
            self._mem_dc,
            ctypes.byref(bi),
            0,  # DIB_RGB_COLORS
            ctypes.byref(ppv_bits),
            None,
            0,
        )
        self._dib_bits = ppv_bits

        self._old_bitmap = ctypes.windll.gdi32.SelectObject(self._mem_dc, self._bitmap)

        # Create a font
        font_height = 16
        self._hfont = ctypes.windll.gdi32.CreateFontW(
            font_height, 0, 0, 0,
            400,  # FW_NORMAL
            0, 0, 0,
            1,  # DEFAULT_CHARSET
            0, 0, 0, 0,
            "Consolas",
        )
        ctypes.windll.gdi32.SelectObject(self._mem_dc, self._hfont)

    def _destroy_gdi(self) -> None:
        """Release GDI resources."""
        if self._hfont:
            ctypes.windll.gdi32.DeleteObject(self._hfont)
            self._hfont = None
        if self._old_bitmap and self._mem_dc:
            ctypes.windll.gdi32.SelectObject(self._mem_dc, self._old_bitmap)
            self._old_bitmap = None
        if self._bitmap:
            ctypes.windll.gdi32.DeleteObject(self._bitmap)
            self._bitmap = None
        if self._mem_dc:
            ctypes.windll.gdi32.DeleteDC(self._mem_dc)
            self._mem_dc = None
        if self._hdc:
            ctypes.windll.gdi32.DeleteDC(self._hdc)
            self._hdc = None

    def _draw_frame(
        self,
        detections: List[Detection],
        target: Optional[Detection],
        active: bool,
        fps: float,
        detect_ms: float,
    ) -> None:
        """Draw a full frame of detection data onto the overlay."""
        if not self._mem_dc or not self._hwnd:
            return

        hdc = self._mem_dc

        # Clear with full transparency
        brush = ctypes.windll.gdi32.CreateSolidBrush(0x00000000)
        rect = ctypes.wintypes.RECT(0, 0, self._w, self._h)
        ctypes.windll.user32.FillRect(hdc, ctypes.byref(rect), brush)
        ctypes.windll.gdi32.DeleteObject(brush)

        # Create a pen for outlines (2px, green for target)
        pen_target = ctypes.windll.gdi32.CreatePen(0, 2, COLOR_TARGET)
        pen_detect = ctypes.windll.gdi32.CreatePen(0, 2, COLOR_DETECTION)
        pen_aim = ctypes.windll.gdi32.CreatePen(0, 2, COLOR_AIM)

        # Set text color and background mode
        ctypes.windll.gdi32.SetTextColor(hdc, COLOR_TEXT)
        ctypes.windll.gdi32.SetBkMode(hdc, 1)  # TRANSPARENT

        # Draw crosshair at center
        cx, cy = self._w // 2, self._h // 2
        ctypes.windll.gdi32.SelectObject(hdc, pen_target)
        ctypes.windll.gdi32.MoveToEx(hdc, cx - 10, cy, None)
        ctypes.windll.gdi32.LineTo(hdc, cx + 10, cy)
        ctypes.windll.gdi32.MoveToEx(hdc, cx, cy - 10, None)
        ctypes.windll.gdi32.LineTo(hdc, cx, cy + 10)

        # Draw detection boxes
        if self._show_boxes:
            for det in detections:
                is_target = (target is not None and det is target)
                pen = pen_target if is_target else pen_detect
                ctypes.windll.gdi32.SelectObject(hdc, pen)

                x1, y1, x2, y2 = det.bbox
                ctypes.windll.gdi32.MoveToEx(hdc, x1, y1, None)
                ctypes.windll.gdi32.LineTo(hdc, x2, y1)
                ctypes.windll.gdi32.LineTo(hdc, x2, y2)
                ctypes.windll.gdi32.LineTo(hdc, x1, y2)
                ctypes.windll.gdi32.LineTo(hdc, x1, y1)

                # Draw aim point as a small filled circle (cross)
                if det.aim_point:
                    ax, ay = det.aim_point
                    ctypes.windll.gdi32.SelectObject(hdc, pen_aim)
                    ctypes.windll.gdi32.MoveToEx(hdc, ax - 4, ay, None)
                    ctypes.windll.gdi32.LineTo(hdc, ax + 4, ay)
                    ctypes.windll.gdi32.MoveToEx(hdc, ax, ay - 4, None)
                    ctypes.windll.gdi32.LineTo(hdc, ax, ay + 4)

                # Label
                if self._show_labels:
                    label = f"[{det.class_id}] {det.confidence:.2f}"
                    # Draw text with shadow for readability
                    ctypes.windll.gdi32.SetTextColor(hdc, 0x00000000)  # Black shadow
                    ctypes.windll.gdi32.TextOutW(hdc, x1 + 2, y1 - 17, label, len(label))
                    ctypes.windll.gdi32.SetTextColor(hdc, COLOR_TEXT)
                    ctypes.windll.gdi32.TextOutW(hdc, x1 + 1, y1 - 18, label, len(label))

                    if is_target:
                        tlabel = "TARGET"
                        ctypes.windll.gdi32.TextOutW(hdc, x1 + 1, y2 + 2, tlabel, len(tlabel))

        # Status bar
        if self._show_status:
            status_text = "AIM: ON" if active else "AIM: OFF"
            st_color = COLOR_STATUS_ON if active else COLOR_STATUS_OFF
            ctypes.windll.gdi32.SetTextColor(hdc, st_color)
            ctypes.windll.gdi32.TextOutW(hdc, 10, 5, status_text, len(status_text))

        if self._show_fps:
            fps_text = f"Detect: {fps:.1f} FPS | {detect_ms:.1f}ms"
            ctypes.windll.gdi32.SetTextColor(hdc, COLOR_TEXT)
            ctypes.windll.gdi32.TextOutW(hdc, 10, 25, fps_text, len(fps_text))

        # Cleanup pens
        ctypes.windll.gdi32.DeleteObject(pen_target)
        ctypes.windll.gdi32.DeleteObject(pen_detect)
        ctypes.windll.gdi32.DeleteObject(pen_aim)

        # Present: blit memory DC to window using UpdateLayeredWindow
        blend = ctypes.wintypes.BLENDFUNCTION()
        blend.BlendOp = AC_SRC_OVER
        blend.BlendFlags = 0
        blend.SourceConstantAlpha = 255
        blend.AlphaFormat = AC_SRC_ALPHA  # Per-pixel alpha

        ppt_src = ctypes.wintypes.POINT(0, 0)
        ppt_dst = ctypes.wintypes.POINT(self._x, self._y)
        psize = ctypes.wintypes.SIZE(self._w, self._h)

        ctypes.windll.user32.UpdateLayeredWindow(
            self._hwnd,
            self._hdc,  # screen DC (not used when using blend+alpha)
            ctypes.byref(ppt_dst),
            ctypes.byref(psize),
            self._mem_dc,
            ctypes.byref(ppt_src),
            0,  # color key (not used)
            ctypes.byref(blend),
            ULW_ALPHA,
        )
