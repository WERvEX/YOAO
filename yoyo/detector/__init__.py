"""YOLO object detection module — ONNX Runtime inference pipeline."""


def __getattr__(name):
    if name == "YOLODetector":
        from .yolo_detector import YOLODetector
        return YOLODetector
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["YOLODetector"]
