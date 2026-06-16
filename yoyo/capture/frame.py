"""Frame dataclass — carries captured screen data through the pipeline."""

import time
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np


@dataclass
class Frame:
    """A single captured screen frame.

    Attributes:
        image: BGRA or BGR image as numpy array (H, W, C).
        timestamp: Monotonic capture timestamp (seconds).
        region: Screen region this frame covers as (x, y, width, height).
    """
    image: np.ndarray
    timestamp: float = field(default_factory=time.perf_counter)
    region: Optional[Tuple[int, int, int, int]] = None  # (x, y, w, h)

    @property
    def width(self) -> int:
        return self.image.shape[1]

    @property
    def height(self) -> int:
        return self.image.shape[0]

    @property
    def center(self) -> Tuple[int, int]:
        """Screen-space center of this frame (crosshair position)."""
        if self.region is None:
            return (self.width // 2, self.height // 2)
        rx, ry, _, _ = self.region
        return (rx + self.width // 2, ry + self.height // 2)
