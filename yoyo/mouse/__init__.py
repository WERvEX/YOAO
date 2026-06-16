"""Mouse control module — Win32 SendInput + natural path generation."""


def __getattr__(name):
    if name == "MouseController":
        from .controller import MouseController
        return MouseController
    if name == "PathGenerator":
        from .path_generator import PathGenerator
        return PathGenerator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["MouseController", "PathGenerator"]
