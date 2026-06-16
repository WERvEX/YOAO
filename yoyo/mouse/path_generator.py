"""Natural mouse path generation using Bezier curves.

Generates human-like mouse movement paths with:
  - Cubic Bezier curves with randomized control points
  - Gaussian micro-jitter along the path
  - Fitts's Law-based timing (longer distance + smaller target = more time)
  - Configurable smoothness and speed
"""

import math
import random
import time
from dataclasses import dataclass
from typing import Iterator, List, Tuple


@dataclass
class PathConfig:
    """Configuration for path generation."""
    smoothness: float = 0.8         # 0=linear, 1=max curves
    jitter_enabled: bool = True
    jitter_amplitude: float = 2.0   # Pixels
    move_speed: float = 1.0         # Multiplier (higher = faster)
    min_duration: float = 0.01      # Minimum move duration (seconds)
    max_duration: float = 0.25      # Maximum move duration


class PathGenerator:
    """Generate natural mouse movement paths.

    Produces a sequence of waypoints from start to target position using
    cubic Bezier curves with configurable curvature and micro-jitter.

    Usage:
        gen = PathGenerator(config)
        waypoints = gen.generate(start=(100, 200), target=(500, 300))
        for x, y in waypoints:
            mouse.move_to(x, y)
    """

    def __init__(self, config: PathConfig):
        self._config = config
        self._rng = random.Random()

    def generate(
        self,
        start: Tuple[int, int],
        target: Tuple[int, int],
    ) -> List[Tuple[int, int]]:
        """Generate a list of waypoints from start to target.

        Args:
            start: (x, y) current cursor position.
            target: (x, y) desired cursor position.

        Returns:
            List of (x, y) waypoints including the final target.
        """
        sx, sy = start
        tx, ty = target
        dx = tx - sx
        dy = ty - sy
        distance = math.sqrt(dx * dx + dy * dy)

        if distance < 2:
            return [(tx, ty)]

        # Determine number of steps based on Fitts's Law:
        # More steps for longer distances, scaled by speed
        steps = max(
            5,
            int(distance / (10.0 * self._config.move_speed)),
        )

        # Generate control points for cubic Bezier
        cp1, cp2 = self._create_control_points(
            start, target, self._config.smoothness
        )

        # Sample the Bezier curve
        waypoints: List[Tuple[int, int]] = []
        for i in range(steps + 1):
            t = i / steps
            x, y = self._bezier_point(sx, sy, cp1, cp2, tx, ty, t)

            # Add micro-jitter
            if self._config.jitter_enabled and 0.1 < t < 0.9:
                jx = self._rng.gauss(0, self._config.jitter_amplitude * 0.3)
                jy = self._rng.gauss(0, self._config.jitter_amplitude * 0.3)
                x += int(round(jx))
                y += int(round(jy))

            waypoints.append((int(round(x)), int(round(y))))

        # Ensure we end exactly at target
        waypoints[-1] = target

        return waypoints

    def generate_with_timing(
        self,
        start: Tuple[int, int],
        target: Tuple[int, int],
    ) -> Iterator[Tuple[Tuple[int, int], float]]:
        """Generate waypoints with per-point delay for timed playback.

        Yields:
            ((x, y), delay_seconds) tuples. The delay is the time to wait
            after the previous move before sending this point.
        """
        waypoints = self.generate(start, target)
        if not waypoints:
            return

        sx, sy = start
        tx, ty = target
        total_distance = math.sqrt((tx - sx) ** 2 + (ty - sy) ** 2)

        # Fitts's Law: MT = a + b * log2(D/W + 1)
        # Approximate: duration grows with log of distance
        base_duration = max(
            self._config.min_duration,
            min(
                self._config.max_duration,
                (0.05 + 0.02 * math.log2(total_distance + 1)) / self._config.move_speed,
            ),
        )

        n = len(waypoints)
        for i, wp in enumerate(waypoints):
            # Vary timing slightly per point — start slow, speed up in middle, slow at end
            t = i / n if n > 1 else 0
            speed_factor = self._speed_profile(t)
            delay = base_duration / n * speed_factor
            yield (wp, delay)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _create_control_points(
        self,
        start: Tuple[int, int],
        target: Tuple[int, int],
        smoothness: float,
    ) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Create randomized control points for a cubic Bezier curve.

        The control points are offset perpendicular to the line from start→target.
        Higher smoothness = larger perpendicular offset = more curved path.
        """
        sx, sy = start
        tx, ty = target
        dx = tx - sx
        dy = ty - sy
        distance = math.sqrt(dx * dx + dy * dy)
        if distance < 1:
            distance = 1

        # Unit vector perpendicular to the direction
        px = -dy / distance
        py = dx / distance

        # Maximum perpendicular offset scales with distance and smoothness
        max_offset = distance * 0.3 * smoothness

        # Random offsets for two control points
        offset1 = self._rng.uniform(-max_offset, max_offset)
        offset2 = self._rng.uniform(-max_offset, max_offset)

        # Control point 1: ~1/3 along the path
        cp1_x = sx + dx * 0.3 + px * offset1
        cp1_y = sy + dy * 0.3 + py * offset1

        # Control point 2: ~2/3 along the path
        cp2_x = sx + dx * 0.7 + px * offset2
        cp2_y = sy + dy * 0.7 + py * offset2

        return (cp1_x, cp1_y), (cp2_x, cp2_y)

    @staticmethod
    def _bezier_point(
        x0: float, y0: float,
        cp1: Tuple[float, float],
        cp2: Tuple[float, float],
        x3: float, y3: float,
        t: float,
    ) -> Tuple[float, float]:
        """Evaluate a cubic Bezier curve at parameter t in [0, 1]."""
        mt = 1 - t
        mt2 = mt * mt
        mt3 = mt2 * mt
        t2 = t * t
        t3 = t2 * t

        x = mt3 * x0 + 3 * mt2 * t * cp1[0] + 3 * mt * t2 * cp2[0] + t3 * x3
        y = mt3 * y0 + 3 * mt2 * t * cp1[1] + 3 * mt * t2 * cp2[1] + t3 * y3
        return (x, y)

    @staticmethod
    def _speed_profile(t: float) -> float:
        """Return a speed multiplier for a point at normalized time t.

        Models human movement: accelerate at start, decelerate near target.
        Uses a sine-based profile peaking in the middle.
        """
        # sin(pi * t) peaks at t=0.5 → invert to get slower-at-peak
        # We want: fast in middle, slow at ends
        # speed = 1 + 0.5 * sin(pi * t)
        # Actually we want delay factor: high at ends (slow), low in middle (fast)
        return 0.5 + 1.5 * abs(math.sin(math.pi * t))
