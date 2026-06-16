"""Download and export YOLOv8n model to ONNX format.

Usage:
    python scripts/download_model.py
    python scripts/download_model.py --model yolov8n --output models/yolov8n.onnx
    python scripts/download_model.py --model yolov8s --output models/yolov8s.onnx

This script downloads a YOLOv8 model from ultralytics and exports it to ONNX
with FP16 precision for efficient GPU inference via ONNX Runtime.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path so we can import yoyo if needed
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def download_and_export(model_name: str, output_path: str, input_size: int = 640) -> None:
    """Download YOLOv8 model and export to ONNX format.

    Args:
        model_name: YOLO model variant name (e.g. 'yolov8n', 'yolov8s', 'yolov11n')
        output_path: Path to save the ONNX model
        input_size: Model input resolution (square)
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Error: ultralytics package not found. Install with: pip install ultralytics")
        print("This is only needed for model download/export, not for inference.")
        sys.exit(1)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {model_name}.pt from ultralytics...")
    model = YOLO(f"{model_name}.pt")

    print(f"Exporting to ONNX (imgsz={input_size}, half=True)...")
    # Export to ONNX with FP16 for faster GPU inference
    model.export(
        format="onnx",
        imgsz=input_size,
        half=True,          # FP16 precision
        simplify=True,      # Simplify ONNX graph
        opset=12,           # ONNX opset version
    )

    # ultralytics exports to same directory as the .pt file with .onnx extension
    # Move to our desired output path
    default_onnx = Path(f"{model_name}.onnx")
    if default_onnx.exists() and default_onnx.resolve() != output_path.resolve():
        import shutil
        shutil.move(str(default_onnx), str(output_path))
        print(f"Model moved to: {output_path}")
    else:
        print(f"Model saved at: {default_onnx}")

    print(f"✓ Done. ONNX model ready at: {output_path}")
    print(f"  Model: {model_name}")
    print(f"  Input size: {input_size}x{input_size}")
    print(f"  Precision: FP16")
    print(f"  File size: {output_path.stat().st_size / 1e6:.1f} MB")


def main():
    parser = argparse.ArgumentParser(
        description="Download YOLO model and export to ONNX format"
    )
    parser.add_argument(
        "--model", "-m",
        default="yolov8n",
        choices=["yolov5n", "yolov8n", "yolov8s", "yolov8m", "yolov11n", "yolov11s"],
        help="YOLO model variant (default: yolov8n)"
    )
    parser.add_argument(
        "--output", "-o",
        default="models/yolov8n.onnx",
        help="Output path for ONNX model (default: models/yolov8n.onnx)"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image size (default: 640)"
    )
    args = parser.parse_args()

    download_and_export(args.model, args.output, args.imgsz)


if __name__ == "__main__":
    main()
