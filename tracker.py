"""
tracker.py — CentroidTracker with exponential-weighted velocity smoothing.
Pure Python + NumPy. No external dependencies.

NOTE: The file uploaded as tracker.py was the old Phase 2 pipeline script.
This is the correct CentroidTracker class that pipeline_hw.py imports.
"""

import numpy as np
from collections import deque


class CentroidTracker:
    def __init__(self, history: int = 8):
        self._history = deque(maxlen=history)   # ring buffer of (vx, vy)
        self._prev_cx = None
        self._prev_cy = None
        self.vx       = 0.0
        self.vy       = 0.0

    def update(self, cx: float, cy: float) -> None:
        """Call once per frame with the target centroid."""
        if self._prev_cx is not None:
            self.vx = cx - self._prev_cx
            self.vy = cy - self._prev_cy
            self._history.append((self.vx, self.vy))
        self._prev_cx = cx
        self._prev_cy = cy

    def smooth_velocity(self):
        """Exponential-weighted mean: recent frames have higher weight."""
        if not self._history:
            return 0.0, 0.0
        n       = len(self._history)
        weights = np.exp(np.linspace(-1.0, 0.0, n))
        weights /= weights.sum()
        vxs = np.array([h[0] for h in self._history])
        vys = np.array([h[1] for h in self._history])
        return float(np.dot(weights, vxs)), float(np.dot(weights, vys))

    def predict_next(self):
        """Returns (vx, vy) — expected pixel displacement next frame."""
        return self.smooth_velocity()

    def reset(self) -> None:
        """Call when the target is lost."""
        self._prev_cx = None
        self._prev_cy = None
        self._history.clear()
        self.vx = 0.0
        self.vy = 0.0
