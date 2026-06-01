[↑ Pipelines](../README.md) | [Back to Repo Root](../../README.md)

---

# `pipeline.py` — Phase 2: Software-Only Baseline

> **Historical reference pipeline.** This was the Phase 2 implementation before the DPU and VCU were added. It uses OpenCV's built-in DNN module (CPU inference) with YOLOv4-tiny Darknet weights. Kept for reproducibility and comparison.

> [!WARNING]
> This pipeline requires `yolov4-tiny.cfg` and `yolov4-tiny.weights` in the repo root. These files are excluded from the repository by `.gitignore` (23 MB). Download them separately or use the Phase 3 pipelines which use the DPU `.xmodel` instead.

## What It Does

1. **Grabs** JPEG frames from IP Webcam via HTTP polling
2. **Detects** persons using **YOLOv4-tiny via OpenCV DNN** — runs on the ARM CPU (no DPU)
3. **Composites** frames with the 3-zone ROI mask (same `zone_mask.py` as Phase 3)
4. **Encodes** as MJPEG (CPU)
5. **Serves** MJPEG stream + emits per-zone byte telemetry using `telemetry.py`

## How to Run

```bash
# Requires yolov4-tiny.cfg + yolov4-tiny.weights in repo root
python3 pipelines/pipeline_sw/pipeline.py
```

## Key Difference from Phase 3

| Feature | Phase 2 (`pipeline.py`) | Phase 3 (`pipeline_hw.py`) |
|---------|------------------------|---------------------------|
| Inference | OpenCV DNN — ARM CPU | DPU B4096 — FPGA fabric |
| Model | YOLOv4-tiny (Darknet `.weights`) | YOLOv4 full (`.xmodel` INT8) |
| Inference speed | ~80–150 ms/frame on CPU | ~15–25 ms on DPU |
| CPU load | Very high | Low (DPU + VCU offload) |
| TILED detection | Yes (4-patch for long range) | No (single pass 416×416) |

## Tiled Detection

Phase 2 includes an experimental **tiled detection mode** that divides the frame into 4 overlapping patches before inference. This improves detection range for small/distant faces at the cost of 4× the inference time.

```python
TILED_DETECTION = True   # 4 overlapping patches → 4× range
TILED_DETECTION = False  # Single pass → 4× faster
```

At 20 ft, a face is ~15 px in 640×480. A single-pass detector at half resolution cannot see it. Tiling gives 2× pixel density per patch, effectively doubling detection range.

This approach was superseded in Phase 3 by the DPU's higher accuracy and direct 416×416 input at higher speed.

## Telemetry Output

Phase 2 uses `telemetry.py` (`measure_zone_bytes`) to print per-zone byte counts:

```
[ZONES] Z1=18432B (full-res ROI) | Z2=24576B (50% ring) | Z3~512B/200px (black bg) | ratio Z1/Z2=0.75
```

This was the original measurement tool before `realBenchmark.py` automated the comparison.

## Dependencies

- **Python:** `cv2`, `numpy`, `threading`, `http.server`
- **Model files:** `yolov4-tiny.cfg`, `yolov4-tiny.weights` (not in repo — too large)
- **Modules:** [`zone_mask`](../../modules/zone_mask/), [`adaptive_roi`](../../modules/adaptive_roi/), [`tracker`](../../modules/tracker/), [`telemetry`](../../modules/telemetry/)

## See Also

- [Project Overview — Phase Evolution](../../docs/01_project_overview.md#project-evolution)
- [`pipeline_hw/`](../pipeline_hw/) — the production Phase 3 pipeline
