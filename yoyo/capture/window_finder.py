"""Window auto-detection utility for finding game windows by title."""

import ctypes
import ctypes.wintypes
import sys
from typing import List, NamedTuple, Optional


class WindowInfo(NamedTuple):
    """Information about a found window."""
    hwnd: int
    title: str
    x: int
    y: int
    width: int
    height: int


def find_window_by_title(keyword: str) -> Optional[WindowInfo]:
    """Find a window whose title contains the given keyword (case-insensitive).

    Args:
        keyword: Substring to match in window titles.

    Returns:
        WindowInfo if found, None otherwise.
    """
    if not keyword:
        return None

    keyword_lower = keyword.lower()

    def _enum_callback(hwnd: int, _lparam: int) -> bool:
        nonlocal result
        if not ctypes.windll.user32.IsWindowVisible(hwnd):
            return True  # continue enumeration

        length = ctypes.windll.user32.GetWindowTextLengthW(hwnd)
        if length == 0:
            return True

        buf = ctypes.create_unicode_buffer(length + 1)
        ctypes.windll.user32.GetWindowTextW(hwnd, buf, length + 1)
        title = buf.value

        if keyword_lower in title.lower():
            rect = ctypes.wintypes.RECT()
            ctypes.windll.user32.GetWindowRect(hwnd, ctypes.byref(rect))
            result = WindowInfo(
                hwnd=hwnd,
                title=title,
                x=rect.left,
                y=rect.top,
                width=rect.right - rect.left,
                height=rect.bottom - rect.top,
            )
            return False  # stop enumeration
        return True

    WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_int, ctypes.c_int)
    result: Optional[WindowInfo] = None

    ctypes.windll.user32.EnumWindows(WNDENUMPROC(_enum_callback), 0)

    if result is not None and (result.width <= 0 or result.height <= 0):
        # Minimized or invalid window
        return None

    return result


def list_visible_windows() -> List[WindowInfo]:
    """List all visible windows with non-empty titles (debug utility)."""
    windows: List[WindowInfo] = []

    def _enum_callback(hwnd: int, _lparam: int) -> bool:
        if not ctypes.windll.user32.IsWindowVisible(hwnd):
            return True

        length = ctypes.windll.user32.GetWindowTextLengthW(hwnd)
        if length == 0:
            return True

        buf = ctypes.create_unicode_buffer(length + 1)
        ctypes.windll.user32.GetWindowTextW(hwnd, buf, length + 1)
        title = buf.value

        if title:
            rect = ctypes.wintypes.RECT()
            ctypes.windll.user32.GetWindowRect(hwnd, ctypes.byref(rect))
            windows.append(WindowInfo(
                hwnd=hwnd,
                title=title,
                x=rect.left,
                y=rect.top,
                width=rect.right - rect.left,
                height=rect.bottom - rect.top,
            ))
        return True

    WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_int, ctypes.c_int)
    ctypes.windll.user32.EnumWindows(WNDENUMPROC(_enum_callback), 0)
    return windows
