"""Win32 mouse control via SendInput.

Provides low-latency mouse movement using the Windows SendInput API
through ctypes. Supports absolute positioning (screen coordinates)
and smooth movement along generated paths.

The SendInput approach uses MOUSEEVENTF_MOVE with absolute coordinates,
which sends input at the kernel level — lower latency than pyautogui/pynput
but still detectable by kernel-level anti-cheat.
"""

import ctypes
import time
from typing import Optional, Tuple

from .path_generator import PathConfig, PathGenerator


# Windows API types and constants
class _POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]


class _MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", ctypes.c_long),
        ("dy", ctypes.c_long),
        ("mouseData", ctypes.c_ulong),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
    ]


class _INPUT(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_ulong),
        ("mi", _MOUSEINPUT),
    ]


INPUT_MOUSE = 0
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_ABSOLUTE = 0x8000
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004

# Virtual screen dimensions (for absolute coordinate normalization)
SM_CXVIRTUALSCREEN = 78
SM_CYVIRTUALSCREEN = 79


class MouseController:
    """Low-latency mouse controller using Win32 SendInput.

    Usage:
        ctrl = MouseController(smoothness=0.8, move_speed=1.0)
        ctrl.move_to(500, 300)  # Smooth movement to screen coordinates
        ctrl.teleport_to(500, 300)  # Instant cursor jump (no path)
    """

    def __init__(
        self,
        smoothness: float = 0.8,
        move_speed: float = 1.0,
        jitter_enabled: bool = True,
        jitter_amplitude: float = 2.0,
    ):
        """Initialize the mouse controller.

        Args:
            smoothness: Path curvature 0-1.
            move_speed: Movement speed multiplier.
            jitter_enabled: Add Gaussian micro-jitter to paths.
            jitter_amplitude: Jitter magnitude in pixels.
        """
        path_cfg = PathConfig(
            smoothness=smoothness,
            jitter_enabled=jitter_enabled,
            jitter_amplitude=jitter_amplitude,
            move_speed=move_speed,
        )
        self._path_gen = PathGenerator(path_cfg)

        # Cache screen dimensions for absolute coordinate mapping
        self._screen_width = ctypes.windll.user32.GetSystemMetrics(
            SM_CXVIRTUALSCREEN
        )
        self._screen_height = ctypes.windll.user32.GetSystemMetrics(
            SM_CYVIRTUALSCREEN
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def move_to(self, x: int, y: int) -> None:
        """Move the cursor to (x, y) along a human-like Bezier path.

        Args:
            x: Target absolute screen X coordinate.
            y: Target absolute screen Y coordinate.
        """
        current = self.get_position()
        if current is None:
            self.teleport_to(x, y)
            return

        # Generate waypoints with timing
        for (wx, wy), delay in self._path_gen.generate_with_timing(
            current, (x, y)
        ):
            self._send_move(wx, wy)
            if delay > 0.001:
                time.sleep(delay)

        # Ensure final position
        self._send_move(x, y)

    def teleport_to(self, x: int, y: int) -> None:
        """Instantly move the cursor to (x, y) — no path, minimum latency.

        Args:
            x: Target absolute screen X coordinate.
            y: Target absolute screen Y coordinate.
        """
        self._send_move(x, y)

    def get_position(self) -> Optional[Tuple[int, int]]:
        """Get the current cursor position in screen coordinates.

        Returns:
            (x, y) tuple, or None on failure.
        """
        pt = _POINT()
        if ctypes.windll.user32.GetCursorPos(ctypes.byref(pt)):
            return (pt.x, pt.y)
        return None

    @property
    def screen_size(self) -> Tuple[int, int]:
        """Get the virtual screen dimensions."""
        return (self._screen_width, self._screen_height)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _send_move(self, x: int, y: int) -> None:
        """Send a single absolute mouse move event via SendInput."""
        # Normalize coordinates to [0, 65535] for MOUSEEVENTF_ABSOLUTE
        nx = int((x / self._screen_width) * 65535) if self._screen_width > 0 else 0
        ny = int((y / self._screen_height) * 65535) if self._screen_height > 0 else 0

        inp = _INPUT()
        inp.type = INPUT_MOUSE
        inp.mi.dx = nx
        inp.mi.dy = ny
        inp.mi.mouseData = 0
        inp.mi.dwFlags = MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE
        inp.mi.time = 0
        inp.mi.dwExtraInfo = None

        ctypes.windll.user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(inp))
