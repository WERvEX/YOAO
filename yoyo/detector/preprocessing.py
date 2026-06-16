"""Frame preprocessing for YOLO ONNX models.

Converts captured BGRA/BGR frames to the normalized tensor format
expected by YOLOv8 ONNX models: (1, 3, H, W) float32, values in [0, 1].
"""

from typing import Tuple

import cv2
import numpy as np


def preprocess(
    image: np.ndarray,
    input_size: Tuple[int, int] = (640, 640),
) -> np.ndarray:
    """Preprocess a captured frame for YOLO ONNX inference.

    Pipeline: BGRA→BGR→RGB → resize → normalize → NCHW → add batch dim.

    Args:
        image: Input image as numpy array (H, W, C). Supports BGRA (4ch) or BGR (3ch).
        input_size: Model input dimensions (width, height). Default (640, 640).

    Returns:
        Preprocessed tensor as numpy float32 array with shape (1, 3, H, W),
        values normalized to [0, 1]. Channel order is RGB.
    """
    # Convert color: dxcam outputs BGR or BGRA → we need RGB
    if image.shape[2] == 4:
        # BGRA → BGR (drop alpha)
        image = image[:, :, :3]

    # BGR → RGB
    image = image[:, :, ::-1]

    # Resize to model input size
    # INTER_LINEAR is fast and sufficient for downsampling
    orig_h, orig_w = image.shape[:2]
    target_w, target_h = input_size
    if (orig_w, orig_h) != (target_w, target_h):
        image = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    # HWC → CHW, normalize to [0, 1], convert to float32
    tensor = np.ascontiguousarray(image, dtype=np.float32) / 255.0
    tensor = np.transpose(tensor, (2, 0, 1))  # HWC → CHW

    # Add batch dimension: CHW → NCHW
    tensor = np.expand_dims(tensor, axis=0)

    return np.ascontiguousarray(tensor)


def get_scale_factors(
    original_size: Tuple[int, int],
    input_size: Tuple[int, int] = (640, 640),
) -> Tuple[float, float]:
    """Compute scale factors to map detection coordinates back to original image.

    Args:
        original_size: Original image (width, height).
        input_size: Model input (width, height).

    Returns:
        (scale_x, scale_y) multipliers.
    """
    orig_w, orig_h = original_size
    input_w, input_h = input_size
    return (orig_w / input_w, orig_h / input_h)
