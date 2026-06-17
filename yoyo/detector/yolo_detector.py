"""YOLO object detector using ONNX Runtime.

Loads an ONNX-exported YOLOv8 model, runs inference on captured frames,
and returns parsed detections. The detector is model-agnostic — any YOLOv8-family
ONNX model with compatible output format works.

Key design: Training stays in PyTorch/ultralytics; inference uses ONNX Runtime.
Swap models by replacing the .onnx file and updating config.
"""

import logging
import time
from typing import List, Optional, Tuple

import numpy as np

from .preprocessing import preprocess
from .postprocessing import Detection, parse_detections, select_target

logger = logging.getLogger(__name__)


class YOLODetector:
    """YOLO object detector via ONNX Runtime.

    Usage:
        detector = YOLODetector(
            model_path="models/yolov8n.onnx",
            input_size=(640, 640),
            confidence_threshold=0.5,
            nms_threshold=0.45,
            target_classes=[0],        # 0=person for COCO
            target_mode="nearest_crosshair",
            head_offset=-0.15,
            execution_provider="auto",
        )
        detector.initialize()

        detections = detector.detect(frame_image)
        target = detector.select_best(detections, crosshair=(960, 540))
    """

    def __init__(
        self,
        model_path: str = "models/yolov8n.onnx",
        input_size: Tuple[int, int] = (640, 640),
        confidence_threshold: float = 0.5,
        nms_threshold: float = 0.45,
        target_classes: Optional[List[int]] = None,
        target_mode: str = "nearest_crosshair",
        head_offset: float = -0.15,
        execution_provider: str = "auto",
        intra_op_threads: int = 2,
    ):
        """Initialize the YOLO detector.

        Args:
            model_path: Path to ONNX model file.
            input_size: Model input (width, height).
            confidence_threshold: Minimum confidence for detections.
            nms_threshold: IoU threshold for NMS.
            target_classes: Class IDs to target (None = all classes).
            target_mode: "nearest_crosshair" | "highest_confidence" | "largest_bbox".
            head_offset: Fraction of bbox height to offset aim point (negative = up).
            execution_provider: "auto" | "cuda" | "directml" | "cpu".
            intra_op_threads: ONNX Runtime intra-op parallelism threads.
        """
        self._model_path = model_path
        self._input_size = input_size
        self._confidence_threshold = confidence_threshold
        self._nms_threshold = nms_threshold
        self._target_classes = target_classes
        self._target_mode = target_mode
        self._head_offset = head_offset
        self._execution_provider = execution_provider
        self._intra_op_threads = intra_op_threads

        self._session = None
        self._input_name: str = ""
        self._initialized = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def initialize(self) -> bool:
        """Load the ONNX model and create inference session.

        Returns:
            True if initialization succeeded.
        """
        try:
            import onnxruntime as ort
        except ImportError:
            logger.error("onnxruntime not installed. Run: pip install onnxruntime")
            return False

        # Resolve execution provider
        providers = self._resolve_providers(ort)

        try:
            sess_options = ort.SessionOptions()
            sess_options.intra_op_num_threads = self._intra_op_threads
            sess_options.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            )

            self._session = ort.InferenceSession(
                self._model_path,
                sess_options=sess_options,
                providers=providers,
            )
        except Exception as e:
            logger.error(f"Failed to create ONNX session: {e}")
            return False

        # Get model metadata
        inputs = self._session.get_inputs()
        if not inputs:
            logger.error("ONNX model has no inputs.")
            return False

        self._input_name = inputs[0].name
        input_shape = inputs[0].shape
        logger.info(
            f"ONNX model loaded: {self._model_path}\n"
            f"  Input: {self._input_name} shape={input_shape}\n"
            f"  Providers: {self._session.get_providers()}\n"
            f"  Active provider: {self._session.get_provider_options()}"
        )

        self._initialized = True
        return True

    def detect(
        self,
        image: np.ndarray,
    ) -> List[Detection]:
        """Run detection on a single frame.

        Args:
            image: Input image as numpy array (H, W, C), BGR or BGRA.

        Returns:
            List of Detection objects in frame-pixel coordinates.
        """
        if not self._initialized:
            logger.warning("Detector not initialized. Call initialize() first.")
            return []

        frame_h, frame_w = image.shape[:2]

        # Preprocess (letterbox — returns tensor + padding info for reverse mapping)
        tensor, lb_info = preprocess(image, self._input_size)

        # Inference
        start = time.perf_counter()
        outputs = self._session.run(None, {self._input_name: tensor})
        elapsed_ms = (time.perf_counter() - start) * 1000

        if not outputs or outputs[0] is None:
            return []

        raw_output = outputs[0]

        # Parse detections
        detections = parse_detections(
            output=raw_output,
            confidence_threshold=self._confidence_threshold,
            nms_threshold=self._nms_threshold,
            target_classes=self._target_classes,
            input_size=self._input_size,
            frame_size=(frame_w, frame_h),
            head_offset=self._head_offset,
            lb_info=lb_info,
        )

        return detections

    def select_best(
        self,
        detections: List[Detection],
        crosshair_pos: Optional[Tuple[int, int]] = None,
        frame_size: Optional[Tuple[int, int]] = None,
    ) -> Optional[Detection]:
        """Select the best target from detections using the configured strategy.

        Args:
            detections: Parsed detection results.
            crosshair_pos: Screen coordinates of the crosshair.
            frame_size: Frame (width, height) — used if crosshair_pos is None.

        Returns:
            Best Detection or None.
        """
        return select_target(
            detections,
            mode=self._target_mode,
            crosshair_pos=crosshair_pos,
            frame_size=frame_size,
        )

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    @property
    def input_size(self) -> Tuple[int, int]:
        return self._input_size

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _resolve_providers(self, ort) -> List[str]:
        """Determine ONNX Runtime execution providers based on config.

        Order of preference for "auto":
          1. CUDA (NVIDIA)
          2. DirectML (AMD/Intel GPU on Windows)
          3. CPU (fallback)
        """
        if self._execution_provider == "cpu":
            return ["CPUExecutionProvider"]

        if self._execution_provider == "cuda":
            if "CUDAExecutionProvider" in ort.get_available_providers():
                return ["CUDAExecutionProvider", "CPUExecutionProvider"]
            logger.warning("CUDAExecutionProvider not available, falling back to auto.")
            return self._auto_providers(ort)

        if self._execution_provider == "directml":
            if "DmlExecutionProvider" in ort.get_available_providers():
                return ["DmlExecutionProvider", "CPUExecutionProvider"]
            logger.warning("DmlExecutionProvider not available, falling back to auto.")
            return self._auto_providers(ort)

        # "auto" — try in priority order
        return self._auto_providers(ort)

    @staticmethod
    def _auto_providers(ort) -> List[str]:
        available = ort.get_available_providers()
        preferred = ["CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider"]
        result = [p for p in preferred if p in available]
        if not result:
            result = ["CPUExecutionProvider"]
        logger.info(f"Available ONNX providers: {available}")
        logger.info(f"Using providers: {result}")
        return result
