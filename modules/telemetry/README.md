[↑ Modules](../README.md) | [Back to Repo Root](../../README.md)

---

# `telemetry.py` — Per-Zone JPEG Byte Measurement

A diagnostic module used in the Phase 2 software pipeline (`pipeline_sw`) to independently measure the JPEG-compressed byte size of each zone. Quantifies the compression benefit of Zone 2 downsampling and Zone 3 blacking.

> [!NOTE]
> This module is **not used** in the Phase 3 pipelines (`pipeline_hw.py`, `pipeline_hw_1.py`). Phase 3 uses VCU H.264 VBR telemetry instead. `telemetry.py` is kept as a diagnostic and educational tool.

## What It Measures

Instead of measuring total frame bandwidth, `telemetry.py` encodes each zone **independently** as JPEG and counts the compressed bytes. This directly quantifies how much each zone contributes to the output bitstream.

| Measurement | What it tells you |
|-------------|-----------------|
| `Z1_bytes` | Bytes used by full-resolution ROI crop |
| `Z2_bytes` | Bytes used by the 50%-downsampled proximity ring crop |
| `Z3_bytes_per_200px` | Bytes for a 200×200 black background sample — should be very small |
| `Z1/Z2 ratio` | How much more efficient Zone 1 is vs Zone 2 (higher = more benefit from downsampling) |

## API

### `measure_zone_bytes(frame, ax, ay, aw, ah, rx, ry, rw, rh, jpeg_quality=85)`

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `frame` | `numpy.ndarray` | The composited output frame (after zone masking) |
| `ax, ay, aw, ah` | `int` | Zone 1 bounding box coordinates |
| `rx, ry, rw, rh` | `int` | Zone 2 ring bounding box coordinates |
| `jpeg_quality` | `int` | JPEG quality level for measurement (default 85) |

**Returns:** `(z1_bytes, z2_bytes, z3_bytes_per_200px)`

**Prints:**
```
[ZONES] Z1=18432B (full-res ROI) | Z2=24576B (50% ring) | Z3~512B/200px (black bg) | ratio Z1/Z2=0.75
```

## How it Works

```python
def jpeg_size(crop):
    ok, buf = cv2.imencode('.jpg', crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return len(buf) if ok else 0

z1_bytes = jpeg_size(frame[ay:ay+ah, ax:ax+aw])    # Zone 1 crop
z2_bytes = jpeg_size(frame[ry:ry+rh, rx:rx+rw])    # Zone 2 crop (blurred)

# Zone 3: sample a 200x200 block from the corner (should be black = tiny)
bg_sample = frame[0:200, 0:200]
z3_bytes = jpeg_size(bg_sample)
```

## Interpreting the Output

**Good result (ROI masking working correctly):**
```
Z3~512B/200px (black bg)       ← very small: Zone 3 is black, compresses to near-zero
ratio Z1/Z2=0.75               ← Zone 2 slightly larger because ring area > ROI area
```

**Bad result (mask not being applied):**
```
Z3~45000B/200px (black bg)     ← huge: Zone 3 contains real pixel data, mask not working
```

## Usage Example

```python
from telemetry import measure_zone_bytes

# After compositing:
composited, ring_boxes = build_zone_mask_multi(frame, adapted_boxes)

if adapted_boxes and ring_boxes:
    ax, ay, aw, ah = adapted_boxes[0]
    rx, ry, rw, rh = ring_boxes[0]
    z1, z2, z3 = measure_zone_bytes(composited, ax, ay, aw, ah, rx, ry, rw, rh)
```

## See Also

- [`tools/benchmark/`](../../tools/benchmark/) — the Phase 3 automated benchmark (supersedes this module for final measurements)
- [Pipeline SW](../../pipelines/pipeline_sw/) — the pipeline that uses this module
