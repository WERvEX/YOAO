"""Fine-tune YOLOv8 for game-specific head detection.

This script provides a skeleton for fine-tuning a YOLO model on custom
annotated game screenshots. The workflow is:

  1. Collect game screenshots (e.g. using the YOAO capture pipeline)
  2. Annotate heads in YOLO format (use LabelImg, CVAT, Roboflow, etc.)
  3. Organize data:
       dataset/
         images/
           train/
           val/
         labels/
           train/    # .txt files in YOLO format: class_id cx cy w h (norm 0-1)
           val/
         dataset.yaml:
           path: ./dataset
           train: images/train
           val: images/val
           names:
             0: head
  4. Run: python scripts/fine_tune.py --data dataset/dataset.yaml --epochs 100
  5. Output: runs/train/exp/weights/best.onnx → copy to models/

Zero code changes needed in the inference pipeline — just swap the ONNX file.

Usage:
    python scripts/fine_tune.py --data dataset/dataset.yaml
    python scripts/fine_tune.py --data dataset.yaml --model yolov8n --epochs 200 --batch 16
    python scripts/fine_tune.py --data dataset.yaml --resume runs/train/exp/weights/last.pt
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune YOLOv8 for game-specific head detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example workflow:
  1. python scripts/fine_tune.py --data dataset/dataset.yaml --epochs 100
  2. Copy the best ONNX model:
     cp runs/train/exp/weights/best.onnx models/yolov8n_head.onnx
  3. Update config.yaml:
     detector:
       model_path: "models/yolov8n_head.onnx"
       target_classes: [0]  # 'head' class
        """,
    )
    parser.add_argument(
        "--data", "-d",
        required=True,
        help="Path to dataset YAML configuration file",
    )
    parser.add_argument(
        "--model", "-m",
        default="yolov8n.pt",
        help="Base model to fine-tune (default: yolov8n.pt)",
    )
    parser.add_argument(
        "--epochs", "-e",
        type=int,
        default=100,
        help="Number of training epochs (default: 100)",
    )
    parser.add_argument(
        "--batch", "-b",
        type=int,
        default=16,
        help="Batch size (default: 16, reduce if OOM)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Training image size (default: 640)",
    )
    parser.add_argument(
        "--device",
        default="0",
        help="Device: '0' for GPU 0, 'cpu' for CPU, '0,1' for multi-GPU",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Resume training from checkpoint (.pt file)",
    )
    parser.add_argument(
        "--export-onnx",
        action="store_true",
        default=True,
        help="Export best model to ONNX after training (default: True)",
    )
    parser.add_argument(
        "--no-export",
        action="store_true",
        help="Skip ONNX export after training",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print("Error: ultralytics package not found.")
        print("  Install with: pip install ultralytics")
        sys.exit(1)

    # Validate dataset path
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"Error: Dataset config not found: {args.data}")
        print("\nExpected format (dataset.yaml):")
        print("  path: ./dataset")
        print("  train: images/train")
        print("  val: images/val")
        print("  names:")
        print("    0: head")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  Fine-tuning: {args.model}")
    print(f"  Dataset:     {args.data}")
    print(f"  Epochs:      {args.epochs}")
    print(f"  Batch size:  {args.batch}")
    print(f"  Image size:  {args.imgsz}")
    print(f"  Device:      {args.device}")
    print(f"{'='*60}\n")

    # Load model
    if args.resume:
        print(f"Resuming from: {args.resume}")
        model = YOLO(args.resume)
    else:
        print(f"Loading base model: {args.model}")
        model = YOLO(args.model)

    # Train
    print("\nStarting training...\n")
    results = model.train(
        data=str(data_path),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        # Additional recommended settings for head detection
        patience=20,            # Early stopping patience
        save=True,              # Save checkpoints
        save_period=10,         # Save every 10 epochs
        exist_ok=True,          # Overwrite existing output dir
        pretrained=True,        # Use pre-trained weights
        optimizer="auto",       # Automatic optimizer selection
        verbose=True,           # Detailed output
        # Data augmentation (adjust based on your game)
        hsv_h=0.015,            # Small HSV shift (game colors are consistent)
        hsv_s=0.3,
        hsv_v=0.2,
        degrees=0.0,            # No rotation (game doesn't rotate)
        translate=0.1,          # Small translation
        scale=0.3,              # Scale augmentation
        fliplr=0.5,             # Horizontal flip
        mosaic=0.5,             # Mosaic augmentation
    )

    # Export to ONNX
    if args.export_onnx and not args.no_export:
        print("\nExporting best model to ONNX...")
        # The best weights are saved at runs/train/exp/weights/best.pt
        best_pt = Path(results.save_dir) / "weights" / "best.pt"
        if best_pt.exists():
            best_model = YOLO(str(best_pt))
            best_model.export(
                format="onnx",
                imgsz=args.imgsz,
                half=True,
                simplify=True,
                opset=12,
            )
            onnx_path = best_pt.with_suffix(".onnx")
            print(f"\n✓ ONNX model exported to: {onnx_path}")
            print(f"  File size: {onnx_path.stat().st_size / 1e6:.1f} MB")
            print(f"\nTo use this model:")
            print(f"  1. Copy to models/:")
            print(f"     cp {onnx_path} models/yolov8n_head.onnx")
            print(f"  2. Update config.yaml:")
            print(f"     detector:")
            print(f"       model_path: \"models/yolov8n_head.onnx\"")
            print(f"       target_classes: [0]")
        else:
            print(f"Warning: best.pt not found at {best_pt}")
            print("  Check training output directory for weights.")

    print("\n✓ Fine-tuning complete.")
    print(f"  Results saved to: {results.save_dir}")


if __name__ == "__main__":
    main()
