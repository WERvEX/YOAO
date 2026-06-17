"""Postprocessing for YOLO ONNX outputs — supports two output formats.

Format A — Raw anchor-based (YOLOv8 standard):
  Shape: (1, 84, 8400) or (1, 4+C, N) for 640×640 input.
  Channels 0-3: bbox cx, cy, w, h (normalized 0-1).
  Channels 4+: class scores (80 for COCO).
  Needs: NMS, cxcywh→xyxy, class argmax.

Format B — Post-processed (exported with NMS, e.g. best.onnx):
  Shape: (1, 300, 6) = [x1, y1, x2, y2, confidence, class_id].
  Already in xyxy pixel coords (model input space), already NMS'd.
  Needs: scaling to frame space only.

Target selection strategies are applied here — the detector returns either
the best single target or all detections for overlay rendering.
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .preprocessing import LetterboxInfo


@dataclass
class Detection:
    """A single detection result in screen-space coordinates."""
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2) in screen pixels
    center: Tuple[int, int]          # (cx, cy) bbox center
    confidence: float
    class_id: int
    # Optional head aim-point (offset from bbox center toward top)
    aim_point: Optional[Tuple[int, int]] = None


def parse_detections(
    output: np.ndarray,
    confidence_threshold: float = 0.5,
    nms_threshold: float = 0.45,
    target_classes: Optional[List[int]] = None,
    input_size: Tuple[int, int] = (640, 640),
    frame_size: Tuple[int, int] = (1920, 1080),
    head_offset: float = -0.15,
    lb_info: Optional[LetterboxInfo] = None,
) -> List[Detection]:
    """Parse YOLO ONNX output into Detection objects — auto-detects output format.

    Supports two formats:
      - Raw anchor-based: (1, 84, 8400) — YOLOv8 standard output
      - Post-processed:   (1, N, 6) — model exported with NMS, e.g. best.onnx

    Args:
        output: Raw ONNX model output.
        confidence_threshold: Minimum class confidence to keep a detection.
        nms_threshold: IoU threshold for NMS (only used for raw format).
        target_classes: List of class IDs to consider. None = all classes.
        input_size: Model input (width, height).
        frame_size: Original frame (width, height).
        head_offset: Vertical aim offset as fraction of bbox height.
        lb_info: Letterbox padding info for correct coordinate reverse-mapping.

    Returns:
        List of Detection objects sorted by confidence (descending).
    """
    if output.ndim == 3 and output.shape[2] == 6:
        # Post-processed format: (1, N, 6) = [x1, y1, x2, y2, score, class]
        return _parse_postprocessed(
            output, confidence_threshold, target_classes,
            input_size, frame_size, head_offset, lb_info,
        )

    # ── Raw anchor-based format ────────────────────────────────────────

    # Squeeze batch dimension if present
    if output.ndim == 3:
        output = output[0]  # (84, 8400)

    # YOLOv8 output: rows = [x, y, w, h, class_0, ..., class_79] for each of 8400 anchors
    # Transpose to (8400, 84) for easier processing
    if output.shape[0] == 84:
        output = output.T  # (8400, 84)

    boxes_raw = output[:, :4]   # (8400, 4)  — cx, cy, w, h (normalized)
    scores_raw = output[:, 4:]  # (8400, num_classes)

    # Get best class and its score for each anchor
    class_ids = np.argmax(scores_raw, axis=1)
    confidences = np.max(scores_raw, axis=1)

    # Apply confidence threshold
    mask = confidences >= confidence_threshold
    if target_classes is not None and len(target_classes) > 0:
        class_mask = np.isin(class_ids, target_classes)
        mask = mask & class_mask

    boxes_raw = boxes_raw[mask]
    confidences = confidences[mask]
    class_ids = class_ids[mask]

    if len(boxes_raw) == 0:
        return []

    # Convert cx,cy,w,h → x1,y1,x2,y2 (normalized 0-1 in model space)
    cx, cy, w, h = (
        boxes_raw[:, 0],
        boxes_raw[:, 1],
        boxes_raw[:, 2],
        boxes_raw[:, 3],
    )
    x1 = np.maximum(0, cx - w / 2)
    y1 = np.maximum(0, cy - h / 2)
    x2 = np.minimum(1, cx + w / 2)
    y2 = np.minimum(1, cy + h / 2)
    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

    # Apply NMS
    keep_indices = nms(boxes_xyxy, confidences, nms_threshold)
    if len(keep_indices) == 0:
        return []

    boxes_xyxy = boxes_xyxy[keep_indices]
    confidences = confidences[keep_indices]
    class_ids = class_ids[keep_indices]

    # Scale coordinates to original frame size (reverse letterbox)
    input_w, input_h = input_size
    frame_w, frame_h = frame_size

    if lb_info is not None:
        # Letterbox reverse mapping: denormalize → un-pad → scale to original
        pad_x = lb_info.pad_left
        pad_y = lb_info.pad_top
        scale = lb_info.scale
    else:
        # Fallback: simple stretch resize
        pad_x, pad_y = 0, 0
        scale = 1.0

    detections: List[Detection] = []
    for i in range(len(boxes_xyxy)):
        if lb_info is not None:
            # Denormalize to letterbox pixel coords: norm * input_size
            bx1_px = boxes_xyxy[i, 0] * input_w
            by1_px = boxes_xyxy[i, 1] * input_h
            bx2_px = boxes_xyxy[i, 2] * input_w
            by2_px = boxes_xyxy[i, 3] * input_h
            # Reverse letterbox
            bx1 = int((bx1_px - pad_x) / scale)
            by1 = int((by1_px - pad_y) / scale)
            bx2 = int((bx2_px - pad_x) / scale)
            by2 = int((by2_px - pad_y) / scale)
        else:
            scale_x = frame_w / input_w
            scale_y = frame_h / input_h
            bx1 = int(boxes_xyxy[i, 0] * scale_x)
            by1 = int(boxes_xyxy[i, 1] * scale_y)
            bx2 = int(boxes_xyxy[i, 2] * scale_x)
            by2 = int(boxes_xyxy[i, 3] * scale_y)

        # Clamp to frame bounds
        bx1 = max(0, min(frame_w, bx1))
        by1 = max(0, min(frame_h, by1))
        bx2 = max(0, min(frame_w, bx2))
        by2 = max(0, min(frame_h, by2))

        bbox = (bx1, by1, bx2, by2)
        bbox_h = by2 - by1
        center = ((bx1 + bx2) // 2, (by1 + by2) // 2)

        # Compute aim point: offset from center toward top of bbox
        aim_y = int(center[1] + head_offset * bbox_h)
        aim_point = (center[0], max(0, aim_y))

        detections.append(Detection(
            bbox=bbox,
            center=center,
            confidence=float(confidences[i]),
            class_id=int(class_ids[i]),
            aim_point=aim_point,
        ))

    # Sort by confidence descending
    detections.sort(key=lambda d: d.confidence, reverse=True)
    return detections


def _parse_postprocessed(
    output: np.ndarray,
    confidence_threshold: float,
    target_classes: Optional[List[int]],
    input_size: Tuple[int, int],
    frame_size: Tuple[int, int],
    head_offset: float,
    lb_info: Optional[LetterboxInfo] = None,
) -> List[Detection]:
    """Parse post-processed ONNX output: (1, N, 6) where each row is
    [x1, y1, x2, y2, confidence, class_id].

    This format comes from models exported with built-in NMS. Coordinates are
    in model input space (e.g. 640×640 letterbox). We reverse the letterbox
    padding + scaling to map to original frame pixel coordinates.
    """
    dets = output[0]  # (N, 6)

    if dets.shape[1] != 6:
        return []

    x1_arr = dets[:, 0]
    y1_arr = dets[:, 1]
    x2_arr = dets[:, 2]
    y2_arr = dets[:, 3]
    scores = dets[:, 4]
    class_ids = dets[:, 5].astype(np.int32)

    # Filter by confidence + target classes
    mask = scores >= confidence_threshold
    if target_classes is not None and len(target_classes) > 0:
        mask = mask & np.isin(class_ids, target_classes)

    indices = np.where(mask)[0]
    if len(indices) == 0:
        return []

    frame_w, frame_h = frame_size

    # Reverse letterbox: coordinates are in the letterbox'd 640×640 space.
    # To map back to original frame coords: subtract padding, divide by scale.
    if lb_info is not None:
        pad_x = lb_info.pad_left
        pad_y = lb_info.pad_top
        scale = lb_info.scale
    else:
        # Fallback: assume simple stretch resize
        input_w, input_h = input_size
        pad_x, pad_y = 0, 0
        scale = input_w / frame_w  # same as 1/scale_x

    detections: List[Detection] = []
    for idx in indices:
        # Map from letterbox space → original frame space
        if lb_info is not None:
            bx1 = int((x1_arr[idx] - pad_x) / scale)
            by1 = int((y1_arr[idx] - pad_y) / scale)
            bx2 = int((x2_arr[idx] - pad_x) / scale)
            by2 = int((y2_arr[idx] - pad_y) / scale)
        else:
            # Old stretch-resize fallback
            input_w, input_h = input_size
            scale_x = frame_w / input_w
            scale_y = frame_h / input_h
            bx1 = int(x1_arr[idx] * scale_x)
            by1 = int(y1_arr[idx] * scale_y)
            bx2 = int(x2_arr[idx] * scale_x)
            by2 = int(y2_arr[idx] * scale_y)

        # Clamp to frame bounds
        bx1 = max(0, min(frame_w, bx1))
        by1 = max(0, min(frame_h, by1))
        bx2 = max(0, min(frame_w, bx2))
        by2 = max(0, min(frame_h, by2))

        # Skip degenerate boxes
        if bx2 <= bx1 or by2 <= by1:
            continue

        bbox = (bx1, by1, bx2, by2)
        bbox_h = by2 - by1
        center = ((bx1 + bx2) // 2, (by1 + by2) // 2)

        # Aim point: offset from bbox center toward top (head area)
        aim_y = int(center[1] + head_offset * bbox_h)
        aim_point = (center[0], max(0, aim_y))

        detections.append(Detection(
            bbox=bbox,
            center=center,
            confidence=float(scores[idx]),
            class_id=int(class_ids[idx]),
            aim_point=aim_point,
        ))

    # Sort by confidence descending
    detections.sort(key=lambda d: d.confidence, reverse=True)
    return detections


def nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float,
) -> np.ndarray:
    """Pure-numpy Non-Maximum Suppression.

    Args:
        boxes: Array of shape (N, 4) with (x1, y1, x2, y2) in any consistent coords.
        scores: Array of shape (N,) with confidence scores.
        iou_threshold: Overlap threshold — boxes with IoU above this are suppressed.

    Returns:
        Indices of kept boxes.
    """
    if len(boxes) == 0:
        return np.array([], dtype=np.intp)

    # Sort by score descending
    order = np.argsort(scores)[::-1]
    keep: List[int] = []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = (x2 - x1) * (y2 - y1)

    while len(order) > 0:
        i = order[0]
        keep.append(i)

        if len(order) == 1:
            break

        # Compute IoU of the current best box with the rest
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        inter = w * h
        iou = inter / (area[i] + area[order[1:]] - inter)

        # Keep indices with IoU below threshold
        remaining = np.where(iou <= iou_threshold)[0]
        order = order[remaining + 1]

    return np.array(keep, dtype=np.intp)


def select_target(
    detections: List[Detection],
    mode: str = "nearest_crosshair",
    crosshair_pos: Optional[Tuple[int, int]] = None,
    frame_size: Optional[Tuple[int, int]] = None,
) -> Optional[Detection]:
    """Select the best target from a list of detections.

    Args:
        detections: Parsed detection results.
        mode: Selection strategy:
            - "nearest_crosshair": Closest bbox center to crosshair position.
            - "highest_confidence": Highest confidence score.
            - "largest_bbox": Largest bounding box area.
        crosshair_pos: (x, y) of crosshair. Defaults to center of frame.
        frame_size: (width, height) of frame. Used if crosshair_pos is None.

    Returns:
        The selected Detection, or None if no detections.
    """
    if not detections:
        return None

    if mode == "nearest_crosshair":
        if crosshair_pos is None and frame_size is not None:
            crosshair_pos = (frame_size[0] // 2, frame_size[1] // 2)
        if crosshair_pos is None:
            return detections[0]

        cx, cy = crosshair_pos

        def dist(d: Detection) -> float:
            dx = d.center[0] - cx
            dy = d.center[1] - cy
            return math.sqrt(dx * dx + dy * dy)

        return min(detections, key=dist)

    elif mode == "highest_confidence":
        return detections[0]  # Already sorted by confidence

    elif mode == "largest_bbox":
        def area(d: Detection) -> int:
            return (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1])
        return max(detections, key=area)

    else:
        return detections[0]
