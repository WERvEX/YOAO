"""Screen capture module — DXGI-based frame acquisition."""


def __getattr__(name):
    if name == "DXGICapture":
        from .dxgi_backend import DXGICapture
        return DXGICapture
    if name == "Frame":
        from .frame import Frame
        return Frame
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["DXGICapture", "Frame"]
