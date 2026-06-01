[↑ Modules](../README.md) | [Back to Repo Root](../../README.md) | [Zone Masking Deep Dive →](../../docs/04_zone_masking_algorithm.md)

---

# `zone_mask.py`: 3-Zone Spatial Masking Engine

The core algorithm module. Builds the composited output frame by applying three spatial zones to a camera frame, making background pixels pure black to maximize H.264 compression efficiency.

## The 3-Zone Concept

```
┌──────────────────────────────────────────┐
│  Zone 3: Pure Black → ~0 bits in H.264   │
│   ┌──────────────────────────────────┐   │
│   │  Zone 2: 50% Downsampled Ring    │   │
│   │   ┌─────────────────────────┐    │   │
│   │   │  Zone 1: Full Res ROI   │    │   │
│   │   │  (Person detected here) │    │   │
│   │   └─────────────────────────┘    │   │
│   └──────────────────────────────────┘   │
└──────────────────────────────────────────┘
```

| Zone | Coverage | Quality | H.264 Bitrate |
|------|----------|---------|---------------|
| Zone 1 | Tight bounding box around person | **Full resolution** | High: all texture preserved |
| Zone 2 | 1.6× ring around Zone 1 | **50% downsampled** then upsampled | ~Half: reduced detail |
| Zone 3 | Everything else | **Pure black (0x000000)** | **~Zero**: encoder skips black macroblocks |

## API

### `build_zone_mask_multi(frame, adapted_boxes, ring_scale=1.6, zone2_downsample=0.5)`

The primary function. Takes a camera frame and a list of expanded bounding boxes, returns the composited output frame.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `frame` | `numpy.ndarray` (H, W, 3) BGR | Raw camera frame from IP Webcam |
| `adapted_boxes` | `list[(ax, ay, aw, ah)]` | Zone 1 bounding boxes from `adaptive_pad()` |
| `ring_scale` | `float` | How much larger Zone 2 is vs Zone 1 (default 1.6×) |
| `zone2_downsample` | `float` | Downsample factor for Zone 2 (default 0.5 = 50%) |

**Returns:** `(composited_frame, ring_boxes)`
- `composited_frame`: numpy array, same shape as input. Zone 3 is all zeros.- `ring_boxes`: list of `(rx, ry, rw, rh)` for Zone 2 rects, parallel to `adapted_boxes`
### `draw_zone_overlay_multi(frame, adapted_boxes, ring_boxes)`

Draws visual zone boundaries on the composited frame in-place.
- Zone 1 → solid **green** rectangle labeled `Z1`- Zone 2 → **dashed amber** rectangle labeled `Z2`
**Returns:** The same frame (mutated in-place).

---

## Implementation: Two-Pass Painter's Algorithm

```python
# Start: everything black (Zone 3 baseline)
out = np.zeros_like(frame)

# Pass 1: paint all Zone 2 rings (lower priority)
for (ax, ay, aw, ah) in adapted_boxes:
    rx, ry, rw, rh = _ring_box(ax, ay, aw, ah, fw, fh)
    crop = frame[ry:ry+rh, rx:rx+rw]
    small = cv2.resize(crop, (sw, sh), cv2.INTER_AREA)    # downsample
    z2_up = cv2.resize(small, (rw, rh), cv2.INTER_LINEAR) # upsample
    out[ry:ry+rh, rx:rx+rw] = z2_up

# Pass 2: paint all Zone 1 ROIs on top (highest priority: always wins)
for (ax, ay, aw, ah) in adapted_boxes:
    out[ay:ay+ah, ax:ax+aw] = frame[ay:ay+ah, ax:ax+aw]
```

**Why two passes?** If you interleave Zone 1 and Zone 2 per-target, a Zone 2 ring from target B could overwrite Zone 1 of target A when they overlap. Two passes guarantee Zone 1 always wins for any pixel.

## Why `np.zeros_like` for Zone 3?

`np.zeros_like(frame)` creates an array of the same shape filled with zeros: pure black in BGR. This means any pixel NOT painted by Zone 1 or Zone 2 stays exactly `(0, 0, 0)`.

An H.264 encoder in VBR mode, presented with solid black 16×16 macroblocks:
- DCT of all-zeros = all-zero coefficients- After quantization: still all zero- After entropy coding: essentially a **skip flag**: ~0 bits
This is the mechanism behind the bandwidth savings.

## Usage Example

```python
from zone_mask import build_zone_mask_multi, draw_zone_overlay_multi

# composited: Zone 3=black, Zone 2=blurred, Zone 1=full-res
# ring_boxes: Zone 2 rects (one per target)
composited, ring_boxes = build_zone_mask_multi(frame, adapted_boxes)

# Add visual overlay for monitoring
out = draw_zone_overlay_multi(composited, adapted_boxes, ring_boxes)
```

## See Also

- [Zone Masking Algorithm: Full Documentation](../../docs/04_zone_masking_algorithm.md)- [`adaptive_roi/`](../adaptive_roi/): provides the `adapted_boxes` input