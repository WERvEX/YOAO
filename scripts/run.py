"""Main entry point for the YOAO aiming system.

Usage:
    python scripts/run.py
    python scripts/run.py --config config.yaml
    python scripts/run.py --no-overlay
    python scripts/run.py --list-windows
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Tuple

import yaml
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from yoyo.capture import DXGICapture
from yoyo.capture.window_finder import list_visible_windows
from yoyo.config.schema import (
    AppConfig,
    CaptureConfig,
    DetectorConfig,
    MouseConfig,
    OnnxConfig,
    OverlayConfig,
    PipelineConfig,
    RegionConfig,
)
from yoyo.detector import YOLODetector
from yoyo.mouse import MouseController
from yoyo.orchestrator import AimBotPipeline
from yoyo.ui import ScreenOverlay

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("yoyo")


def load_config(config_path: str) -> AppConfig:
    """Load configuration from YAML file and create AppConfig.

    Args:
        config_path: Path to config.yaml.

    Returns:
        Populated AppConfig instance.
    """
    path = Path(config_path)
    if not path.exists():
        logger.warning(f"Config file not found: {config_path}. Using defaults.")
        return AppConfig()

    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    # Extract nested configs and convert to dataclass instances
    cap_raw = dict(raw.get("capture", {}))
    region_raw = cap_raw.pop("region", {})
    capture = CaptureConfig(
        region=RegionConfig(**region_raw),
        **cap_raw,
    )

    return AppConfig(
        capture=capture,
        detector=DetectorConfig(**raw.get("detector", {})),
        mouse=MouseConfig(**raw.get("mouse", {})),
        pipeline=PipelineConfig(**raw.get("pipeline", {})),
        overlay=OverlayConfig(**raw.get("overlay", {})),
        onnx=OnnxConfig(**raw.get("onnx", {})),
    )


def build_components(
    config: AppConfig,
) -> Tuple[DXGICapture, YOLODetector, MouseController]:
    """Build core pipeline components from configuration.

    Args:
        config: Application configuration.

    Returns:
        Tuple of (capture, detector, mouse) instances.
    """
    # Capture
    region = (
        config.capture.region.as_tuple
        if not config.capture.auto_detect
        else None
    )
    capture = DXGICapture(
        region=region,
        window_title=config.capture.window_title,
        target_fps=config.capture.target_fps,
    )

    # Detector
    detector = YOLODetector(
        model_path=config.detector.model_path,
        input_size=tuple(config.detector.input_size),
        confidence_threshold=config.detector.confidence_threshold,
        nms_threshold=config.detector.nms_threshold,
        target_classes=config.detector.target_classes,
        target_mode=config.detector.target_mode,
        head_offset=config.mouse.head_offset,
        execution_provider=config.onnx.execution_provider,
        intra_op_threads=config.onnx.intra_op_threads,
    )

    # Mouse
    mouse = MouseController(
        smoothness=config.mouse.smoothness,
        move_speed=config.mouse.move_speed,
        jitter_enabled=config.mouse.jitter_enabled,
        jitter_amplitude=config.mouse.jitter_amplitude,
    )

    return capture, detector, mouse


def main():
    parser = argparse.ArgumentParser(
        description="YOAO — Real-time Game Aiming Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run.py                      # Run with defaults
  python scripts/run.py --config my.yaml     # Custom config
  python scripts/run.py --no-overlay          # Run without overlay
  python scripts/run.py --list-windows        # List visible windows
        """,
    )
    parser.add_argument(
        "--config", "-c",
        default=str(PROJECT_ROOT / "config.yaml"),
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--no-overlay",
        action="store_true",
        help="Disable the detection overlay window",
    )
    parser.add_argument(
        "--list-windows",
        action="store_true",
        help="List all visible windows and exit",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override model path from config",
    )
    args = parser.parse_args()

    # List windows mode
    if args.list_windows:
        windows = list_visible_windows()
        print(f"\nVisible windows ({len(windows)}):")
        for w in sorted(windows, key=lambda x: x.title.lower()):
            print(f"  [{w.x},{w.y} {w.width}x{w.height}] {w.title}")
        return

    # Load config
    config = load_config(args.config)
    if args.model:
        config.detector.model_path = args.model
    if args.no_overlay:
        config.overlay.enabled = False

    # Build components
    print("\n" + "=" * 60)
    print("  YOAO — Real-time Game Aiming Assistant")
    print("=" * 60)
    print(f"  Config:      {args.config}")
    print(f"  Model:       {config.detector.model_path}")
    print(f"  Overlay:     {'ON' if config.overlay.enabled else 'OFF'}")
    print(f"  Toggle Key:  {config.pipeline.toggle_hotkey}")
    print(f"  Quit Key:    {config.pipeline.quit_hotkey}")
    print("=" * 60)
    print()

    capture, detector, mouse = build_components(config)

    # Initialize detector (load ONNX model)
    print("Initializing detector...")
    if not detector.initialize():
        print("ERROR: Failed to initialize detector. Check model path and ONNX Runtime installation.")
        sys.exit(1)
    print("Detector initialized.\n")

    # Qt application MUST exist before any QWidget (ScreenOverlay) is created
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    # Create pipeline
    pipeline = AimBotPipeline(
        capture=capture,
        detector=detector,
        mouse=mouse,
        toggle_hotkey=config.pipeline.toggle_hotkey,
        quit_hotkey=config.pipeline.quit_hotkey,
    )

    # Start pipeline
    print("Starting pipeline...")
    if not pipeline.start():
        print("ERROR: Failed to start pipeline.")
        sys.exit(1)

    # Create and show screen overlay positioned over the capture region
    overlay: Optional[ScreenOverlay] = None
    if config.overlay.enabled:
        region = capture.get_region()
        if region is not None:
            overlay = ScreenOverlay(
                region=region,
                enabled=True,
                show_boxes=config.overlay.show_boxes,
                show_labels=config.overlay.show_labels,
                show_fps=config.overlay.show_fps,
                show_status=config.overlay.show_status,
            )
            overlay.show()
            pipeline.add_frame_callback(overlay.on_frame)

    print(f"Pipeline running. Press {config.pipeline.toggle_hotkey} to toggle aiming.")
    print(f"Press {config.pipeline.quit_hotkey} to quit.\n")

    # Qt event loop on main thread (replaces the old while+sleep poll loop).
    # A QTimer checks pipeline liveness at 10 Hz and quits the app when done.
    def _check_pipeline():
        if not pipeline.is_running or pipeline.should_quit:
            app.quit()

    timer = QTimer()
    timer.timeout.connect(_check_pipeline)
    timer.start(100)  # 10 Hz

    try:
        app.exec()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        timer.stop()
        print("Stopping pipeline...")
        pipeline.stop()
        if overlay is not None:
            overlay.hide()
        print("Done.")


if __name__ == "__main__":
    main()
