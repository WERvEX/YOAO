"""Typed configuration dataclasses for all YOAO subsystems.

Every configurable value lives here, with sensible defaults.
The orchestrator loads config.yaml and maps it onto these dataclasses.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class RegionConfig:
    """Manual screen capture region (top-left origin)."""
    x: int = 0
    y: int = 0
    width: int = 1920
    height: int = 1080

    @property
    def as_tuple(self) -> Tuple[int, int, int, int]:
        return (self.x, self.y, self.width, self.height)


@dataclass
class CaptureConfig:
    """Screen capture configuration."""
    window_title: str = ""              # Keyword to match game window
    auto_detect: bool = True            # Auto-detect window by title
    region: RegionConfig = field(default_factory=RegionConfig)
    target_fps: int = 60                # Desired capture frame rate


@dataclass
class DetectorConfig:
    """YOLO detector configuration."""
    model_path: str = "models/yolov8n.onnx"
    confidence_threshold: float = 0.5
    nms_threshold: float = 0.45
    input_size: List[int] = field(default_factory=lambda: [640, 640])
    target_classes: List[int] = field(default_factory=lambda: [0])  # 0=person (COCO)
    target_mode: str = "nearest_crosshair"  # nearest_crosshair | highest_confidence | largest_bbox


@dataclass
class MouseConfig:
    """Mouse control configuration."""
    move_speed: float = 1.0             # Overall speed multiplier
    smoothness: float = 0.8             # Path curvature (0=linear, 1=max curves)
    jitter_enabled: bool = True         # Micro-jitter for human-like movement
    jitter_amplitude: float = 2.0       # Jitter amplitude in pixels
    head_offset: float = -0.15          # Vertical offset from bbox center (fraction of h)


@dataclass
class PipelineConfig:
    """Pipeline orchestration configuration."""
    toggle_hotkey: str = "ctrl+shift+a"
    quit_hotkey: str = "ctrl+shift+q"
    show_fps: bool = True


@dataclass
class OverlayConfig:
    """Detection overlay window configuration."""
    enabled: bool = True
    show_boxes: bool = True
    show_labels: bool = True
    show_fps: bool = True
    show_status: bool = True
    window_alpha: float = 0.7           # Transparent overlay


@dataclass
class OnnxConfig:
    """ONNX Runtime configuration."""
    execution_provider: str = "auto"    # auto | cuda | directml | cpu
    intra_op_threads: int = 2
    enable_profiling: bool = False


@dataclass
class AppConfig:
    """Top-level application configuration."""
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    mouse: MouseConfig = field(default_factory=MouseConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    overlay: OverlayConfig = field(default_factory=OverlayConfig)
    onnx: OnnxConfig = field(default_factory=OnnxConfig)
