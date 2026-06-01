[↑ Modules](../README.md) | [Back to Repo Root](../../README.md) | [Zone Masking Deep Dive →](../../docs/04_zone_masking_algorithm.md#centroid-tracker)

---

# `tracker.py`: Centroid Tracker with Velocity Smoothing

Tracks the centroid of a detected person across frames and provides an exponential-weighted velocity estimate. This velocity drives the motion-predictive padding in `adaptive_roi.py`.

## Why Velocity Smoothing?

Raw frame-to-frame centroid displacement is **noisy**: the YOLO bounding box jitters slightly even for a stationary person. A simple instantaneous velocity estimate would cause the adaptive padding to flicker.

Exponential-weighted averaging over the last 8 frames gives:
- **High weight** to the most recent 1–2 frames (responsive to actual direction changes)- **Low weight** to older frames (dampens jitter and outliers)
## API

### `CentroidTracker(history=8)`

Creates a tracker for a single target. One instance per tracked person slot.

| Method | Description |
|--------|-------------|
| `update(cx, cy)` | Call once per frame with the target centroid. Computes instantaneous velocity and appends to the ring buffer. |
| `predict_next()` | Returns `(vx, vy)`: expected pixel displacement next frame, exponentially smoothed. |
| `smooth_velocity()` | Internal: computes the exponential-weighted mean over the history buffer. |
| `reset()` | Clear all history. Call when a target is lost to prevent stale velocity affecting the next target. |

## Exponential Weighting

```python
def smooth_velocity(self):
    n = len(self._history)
    weights = np.exp(np.linspace(-1.0, 0.0, n))  # e^-1 to e^0
    weights /= weights.sum()                        # normalize to sum=1
    vxs = np.array([h[0] for h in self._history])
    vys = np.array([h[1] for h in self._history])
    return float(np.dot(weights, vxs)), float(np.dot(weights, vys))
```

For `history=8`, the weight distribution is:

| Frame age | Weight (approx) |
|-----------|----------------|
| Current (age 0) | **37.1%** |
| 1 frame ago | 27.1% |
| 2 frames ago | 19.9% |
| 3 frames ago | 14.5% |
| 4+ frames ago | < 1.4% each |

A person who was walking but just stopped will see their velocity estimate drop to ~0 within 2–3 frames.

## Usage Example

```python
from tracker import CentroidTracker

# One tracker per target slot (pipeline uses MAX_TARGETS=5 slots)
trackers = {i: CentroidTracker(history=8) for i in range(5)}

# Per frame, per target:
for idx, (x, y, w, h) in enumerate(detected_boxes):
    cx, cy = x + w // 2, y + h // 2
    trackers[idx].update(cx, cy)
    vx, vy = trackers[idx].predict_next()
    # vx, vy now fed to adaptive_pad()

# When a target disappears:
trackers[lost_idx].reset()
```

## Design Notes

- **Pure Python + NumPy**: no external tracking library (SORT, DeepSORT etc.) required- **One tracker per slot**, not per identity: targets are matched to slots by detection order, not by appearance. This is sufficient for our use case because we only need velocity, not identity- **Ring buffer** (`deque(maxlen=8)`): automatically discards old frames, constant memory- The tracker does **not** do Kalman filtering or data association: it's a deliberate simplicity choice for an embedded system with limited CPU budget
## See Also

- [`adaptive_roi/`](../adaptive_roi/): consumes the velocity output from this module- [Zone Masking: Centroid Tracker](../../docs/04_zone_masking_algorithm.md#centroid-tracker)