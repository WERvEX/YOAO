"""Frame preprocessing for YOLO ONNX models.

Converts captured BGRA/BGR frames to the normalized tensor format
expected by YOLO ONNX models: (1, 3, H, W) float32, values in [0, 1].

Uses letterbox resizing (preserve aspect ratio + pad) to match standard
YOLO training preprocessing. Returns padding info so postprocessing can
correctly map detection coordinates back to the original frame.
"""

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np


@dataclass
class LetterboxInfo:
    """Letterbox padding parameters for coordinate reverse-mapping."""
    pad_left: int = 0
    pad_top: int = 0
    scale: float = 1.0         # resize scale factor (new / orig)
    new_width: int = 640        # unpadded image width after resize
    new_height: int = 360       # unpadded image height after resize


def preprocess(
    image: np.ndarray,
    input_size: Tuple[int, int] = (640, 640),
) -> Tuple[np.ndarray, LetterboxInfo]:
    """Preprocess a captured frame for YOLO ONNX inference using letterbox.

    Pipeline: BGRA→BGR→RGB → letterbox resize → normalize → NCHW → add batch dim.

    Args:
        image: Input image as numpy array (H, W, C). Supports BGRA (4ch) or BGR (3ch).
        input_size: Model input dimensions (width, height). Default (640, 640).

    Returns:
        (tensor, lb_info) — preprocessed tensor (1, 3, H, W) float32, and
        letterbox info for reverse-mapping detection coordinates.
    """
    # Drop alpha channel if present
    if image.shape[2] == 4:
        image = image[:, :, :3]

    # BGR → RGB
    image = image[:, :, ::-1]

    orig_h, orig_w = image.shape[:2]
    target_w, target_h = input_size

    # Letterbox: resize preserving aspect ratio, then pad
    r = min(target_w / orig_w, target_h / orig_h)
    new_w = int(orig_w * r)
    new_h = int(orig_h * r)

    if (new_w, new_h) != (orig_w, orig_h):
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Pad to target size (top-bottom, left-right)
    pad_w = target_w - new_w
    pad_h = target_h - new_h
    pad_left = pad_w // 2
    pad_top = pad_h // 2

    # Pad with gray (114, 114, 114) — standard YOLO padding value
    image = cv2.copyMakeBorder(
        image, pad_top, pad_h - pad_top,
        pad_left, pad_w - pad_left,
        cv2.BORDER_CONSTANT, value=(114, 114, 114),
    )

    lb_info = LetterboxInfo(
        pad_left=pad_left,
        pad_top=pad_top,
        scale=r,
        new_width=new_w,
        new_height=new_h,
    )

    # HWC → CHW, normalize to [0, 1], convert to float32
    tensor = np.ascontiguousarray(image, dtype=np.float32) / 255.0
    tensor = np.transpose(tensor, (2, 0, 1))  # HWC → CHW

    # Add batch dimension: CHW → NCHW
    tensor = np.expand_dims(tensor, axis=0)

    return np.ascontiguousarray(tensor), lb_info


def get_scale_factors(
    original_size: Tuple[int, int],
    input_size: Tuple[int, int] = (640, 640),
) -> Tuple[float, float]:
    """Compute scale factors to map detection coordinates back to original image.

    Deprecated: use LetterboxInfo from preprocess() instead for accurate
    reverse-mapping. This function assumes simple stretch resize and does NOT
    account for letterbox padding.

    Args:
        original_size: Original image (width, height).
        input_size: Model input (width, height).

    Returns:
        (scale_x, scale_y) multipliers.
    """
    orig_w, orig_h = original_size
    input_w, input_h = input_size
    return (orig_w / input_w, orig_h / input_h)
