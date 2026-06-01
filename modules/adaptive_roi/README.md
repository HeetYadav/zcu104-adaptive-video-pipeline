[↑ Modules](../README.md) | [Back to Repo Root](../../README.md) | [Zone Masking Deep Dive →](../../docs/04_zone_masking_algorithm.md#motion-predictive-roi)

---

# `adaptive_roi.py`: Motion-Predictive ROI Padding

Expands a raw YOLO bounding box asymmetrically in the direction of motion, so a moving person never exits Zone 1 between detection frames.

## The Problem It Solves

YOLOv4 inference runs at ~10–15 FPS. The compositor runs at 30 FPS. Between detection updates (every 2–3 frames), a walking person moves ~5–20 pixels. A tight bounding box would let the person's edge clip into Zone 3 (black) between frames: visually jarring and analytically incorrect.

**Adaptive ROI padding** solves this by:
1. Getting the person's current velocity `(vx, vy)` from the `CentroidTracker`
2. Adding extra padding **in the direction of travel**: the box leads the person
3. Adding minimal padding in the opposite direction: no wasted bits

## API

### `adaptive_pad(x, y, w, h, vx, vy, frame_w, frame_h, base_pad=20, vel_scale=3.0, max_expand=80)`

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `x, y, w, h` | `int` |: | Raw YOLO bounding box (pixels) |
| `vx, vy` | `float` |: | Velocity from `CentroidTracker.predict_next()` |
| `frame_w, frame_h` | `int` |: | Frame dimensions for boundary clamping |
| `base_pad` | `int` | `20` | Minimum padding on every side (px) |
| `vel_scale` | `float` | `3.0` | Extra px of padding per px/frame of velocity |
| `max_expand` | `int` | `80` | Hard cap on velocity-driven expansion (px) |

**Returns:** `(ax, ay, aw, ah)`: expanded box, clamped to frame boundaries.

## How It Works

```python
# Directional padding: expand MORE in the direction of travel
pad_left   = base_pad + clamp(max(0, -vx) * vel_scale, 0, max_expand)
pad_right  = base_pad + clamp(max(0, +vx) * vel_scale, 0, max_expand)
pad_top    = base_pad + clamp(max(0, -vy) * vel_scale, 0, max_expand)
pad_bottom = base_pad + clamp(max(0, +vy) * vel_scale, 0, max_expand)
```

### Worked Example

Person walking **right** at 15 px/frame (`vx=+15, vy=0`):

| Side | Calculation | Result |
|------|-------------|--------|
| Left | `20 + max(0, -15) × 3.0` | `20 px` (minimal: behind the person) |
| Right | `20 + max(0, +15) × 3.0 = 20 + 45` | `65 px` (leading the person) |
| Top | `20 + 0` | `20 px` |
| Bottom | `20 + 0` | `20 px` |

The Zone 1 box extends 65 px in front of the person's current position, giving the detection algorithm plenty of time to update before the person reaches the edge.

## Parameter Tuning Guide

| Scenario | Recommendation |
|----------|---------------|
| Slow surveillance camera, mostly static subjects | Reduce `vel_scale` to `1.5`, increase `base_pad` to `30` |
| Fast-moving subjects (sports, robotics) | Increase `vel_scale` to `5.0`, increase `max_expand` to `120` |
| High detection FPS (small gap between updates) | Reduce `base_pad` to `10`: less prediction needed |
| Low detection FPS (large gap between updates) | Increase `base_pad` to `40`: more buffer needed |

## Usage Example

```python
from adaptive_roi import adaptive_pad
from tracker import CentroidTracker

tracker = CentroidTracker(history=8)

# After YOLOv4 detection:
x, y, w, h = detection_box
cx, cy = x + w // 2, y + h // 2

tracker.update(cx, cy)
vx, vy = tracker.predict_next()

# Get motion-padded Zone 1 box
ax, ay, aw, ah = adaptive_pad(x, y, w, h, vx, vy, frame_w, frame_h)
```

## See Also

- [`tracker/`](../tracker/): provides `(vx, vy)` velocity estimates- [Zone Masking Algorithm](../../docs/04_zone_masking_algorithm.md#motion-predictive-roi)