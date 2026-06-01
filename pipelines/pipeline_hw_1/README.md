[↑ Pipelines](../README.md) | [Back to Repo Root](../../README.md)

---

# `pipeline_hw_1.py` — Phase 3 Baseline: DPU + MJPEG (No VCU)

> **This is the benchmark baseline.** It uses the DPU for YOLOv4 inference (same as `pipeline_hw.py`) but serves the output as MJPEG over HTTP instead of using the VCU H.264 encoder. Used by `realBenchmark.py` to establish the pre-VCU bandwidth figure.

## What It Does

1. **Grabs** JPEG frames from IP Webcam via HTTP polling
2. **Detects** persons using YOLOv4 INT8 inference on the DPU (same as `pipeline_hw.py`)
3. **Composites** each frame with the 3-zone ROI mask
4. **Encodes** as MJPEG (CPU-based `cv2.imencode`) — **no VCU involved**
5. **Serves** MJPEG stream on port 5000 and emits bandwidth telemetry

## How to Run

```bash
# From repo root on the board
python3 pipelines/pipeline_hw_1/pipeline_hw_1.py

# View in VLC:
# http://<board-ip>:5000/stream
```

## How It Differs from `pipeline_hw.py`

| Feature | `pipeline_hw_1.py` | `pipeline_hw.py` |
|---------|-------------------|-----------------|
| Inference | DPU (hardware) | DPU (hardware) |
| Encoding | MJPEG — CPU | H.264 VCU — hardware silicon |
| Bandwidth | ~4,000–8,000 kbps | ~120–700 kbps |
| CPU encoding load | High (JPEG per frame) | **Zero** |
| Stream to VLC | MJPEG HTTP — no buffering | MJPEG HTTP — no buffering |
| VCU usage | ❌ None | ✅ omxh264enc |

## Architecture — 4 Threads + HTTP Server

```
Thread 1 (Grabber)    → polls phone /shot.jpg
Thread 2 (Detector)   → DPU YOLOv4 → boxes (identical to pipeline_hw)
Thread 3 (Compositor) → zone_mask → cv2.imencode (MJPEG on CPU)
Thread 4 (HTTP Server)→ serves MJPEG multipart stream
```

## Telemetry Output

```
[Telemetry] frame=    30 | targets=2 | vx=+5.2 vy=-1.1 | 121.4 KB/frame | BW: 7521.8 kbps (7.7 FPS)
```

| Field | Meaning |
|-------|---------|
| `vx=+5.2 vy=-1.1` | Target 0 velocity (pixels/frame) from `CentroidTracker` |
| `121.4 KB/frame` | Average compressed JPEG size |
| `BW: 7521.8 kbps` | Bandwidth = KB/frame × 8 × FPS |

This is the number that `realBenchmark.py` collects and compares against the VCU H.264 figure from `pipeline_hw.py`.

## Why This Exists

The bandwidth savings of the VCU H.264 pipeline are only meaningful when compared to a fair baseline. This pipeline provides that baseline:
- **Same camera source** (IP Webcam)
- **Same DPU inference** (identical YOLOv4 model and preprocessing)
- **Same 3-zone mask** (identical `zone_mask.py` logic)
- **Different encoder** — MJPEG instead of H.264

The difference in output bandwidth is entirely due to the H.264 VBR encoder exploiting the black pixels in Zone 3.

## Dependencies

- **Python:** `cv2`, `numpy`, `threading`, `http.server`
- **Vitis AI:** `vart`, `xir`
- **Modules:** [`zone_mask`](../../modules/zone_mask/), [`adaptive_roi`](../../modules/adaptive_roi/), [`tracker`](../../modules/tracker/)

## See Also

- [Benchmark Results](../../docs/08_benchmark_results.md) — how this pipeline's output is compared to `pipeline_hw.py`
- [`pipeline_hw/`](../pipeline_hw/) — the VCU hardware pipeline
- [`tools/benchmark/`](../../tools/benchmark/) — the automated benchmark runner
