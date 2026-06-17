# YOAO — Real-time Game Aiming Assistant

YOAO is a real-time aiming assistant that captures your screen, detects targets (person/head) via YOLO, and instantly moves your mouse to the target. All rendering is done via a transparent overlay — no game file modification.

## Features

- **Real-time Detection** — YOLOv8 ONNX inference, runs on GPU via DirectML/CUDA
- **Instant Mouse** — `SendInput` teleport (<1ms), no Bezier delay
- **Transparent Overlay** — PyQt6 overlay draws detection boxes, aim points, FPS on screen
- **Fine-tunable** — Train in PyTorch, export to ONNX, drop in `models/` — no code change
- **Hotkey Toggle** — Configurable activation and quit hotkeys

## How It Works

```
Screen (DXGI) → YOLO ONNX → Target Select → Mouse Move
                  ↓
           PyQt6 Overlay (boxes, FPS, status)
```

1. **Capture** — DXGI Desktop Duplication captures game frames at 60+ FPS
2. **Detect** — ONNX Runtime runs YOLO inference, returns bounding boxes
3. **Select** — Picks nearest-to-crosshair or highest-confidence target
4. **Move** — `SendInput` teleports mouse to aim point (bbox center + head offset)

## Requirements

- Windows 10/11 with a GPU (DXGI support)
- Python 3.10+
- Your own YOLO ONNX model (see [Model Setup](#model-setup))

## Quick Start

```bash
# Install
git clone https://github.com/yourname/YOAO.git
cd YOAO
pip install -r requirements.txt

# Place your model
cp /path/to/your/best.onnx models/

# Edit config.yaml (set capture region, model path, etc.)
# Then run:
python scripts/run.py
```

**Hotkeys (default):**
| Key | Action |
|-----|--------|
| `Alt+I` | Toggle aiming |
| `Alt+O` | Quit |

## Configuration

Edit `config.yaml`:

```yaml
capture:
  region:
    width: 1920    # your screen resolution
    height: 1080

detector:
  model_path: "models/best.onnx"
  confidence_threshold: 0.5
  target_classes: [0]           # 0 = person (single-class model)

mouse:
  head_offset: -0.15            # aim above bbox center

pipeline:
  toggle_hotkey: "alt+i"
  quit_hotkey: "alt+o"

overlay:
  enabled: true
  show_boxes: true
  show_fps: true
```

## Model Setup

YOAO uses ONNX models. Two paths:

### Option A — Fine-tune your own (recommended)

```bash
# Fine-tune YOLOv8n on your dataset
python scripts/fine_tune.py --data dataset.yaml --epochs 100

# Export to ONNX (or use ultralytics directly)
yolo export model=runs/train/weights/best.pt format=onnx

# Copy to models/
cp runs/train/weights/best.onnx models/
```

### Option B — Pre-trained YOLO

```bash
python scripts/download_model.py   # downloads yolov8n.onnx (COCO, 80 classes)
```

Then set `target_classes: [0]` for person detection.

## Project Structure

```
YOAO/
├── yoyo/
│   ├── capture/          # DXGI screen capture
│   ├── detector/         # YOLO ONNX inference + pre/postprocessing
│   ├── mouse/            # Win32 SendInput controller
│   ├── orchestrator/     # Pipeline (threads, queues, timing)
│   ├── config/           # Typed config dataclasses
│   └── ui/               # PyQt6 transparent overlay
├── models/               # ONNX model files
├── scripts/              # Entry points
├── config.yaml           # Runtime config
└── requirements.txt
```

## Important Notes

- **DPI Scaling** — Windows display scaling must be set to 100%. DXGI captures physical pixels but PyQt6 uses logical coordinates; scaling breaks alignment.
- **Windowed/Borderless** — Works best with windowed or borderless fullscreen games.
- **Use at your own risk** — Some games may detect screen capture or input simulation.

## License

MIT
