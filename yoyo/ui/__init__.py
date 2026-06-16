"""UI module — detection overlay and status display."""


def __getattr__(name):
    if name == "DetectionOverlay":
        from .overlay import DetectionOverlay
        return DetectionOverlay
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["DetectionOverlay"]
