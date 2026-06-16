"""UI module — detection overlay and status display."""


def __getattr__(name):
    if name == "ScreenOverlay":
        from .overlay import ScreenOverlay
        return ScreenOverlay
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["ScreenOverlay"]
