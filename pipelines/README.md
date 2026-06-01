[↑ Back to README](../../README.md) | [Docs →](../../docs/)

---

# Pipelines

This folder contains the two runnable pipeline entry points. Run them directly on the ZCU104 board.

## Which Pipeline Should I Run?

| Pipeline | Folder | Hardware Used | Purpose |
|----------|--------|--------------|---------|
| **`pipeline_hw.py`** | [`pipeline_hw/`](pipeline_hw/) | DPU + VCU (Full HW) | ✅ **Primary**: production pipeline with hardware H.264 telemetry |
| **`pipeline_hw_1.py`** | [`pipeline_hw_1/`](pipeline_hw_1/) | DPU only | 📊 **Benchmark baseline**: DPU inference with MJPEG output |

## Quick Commands

```bash
# Full hardware pipeline (recommended)
python3 pipelines/pipeline_hw/pipeline_hw.py

# MJPEG baseline (for benchmark comparison)
python3 pipelines/pipeline_hw_1/pipeline_hw_1.py
```

## Dependencies

All pipelines import from [`modules/`](../modules/). The import path is resolved automatically using `__file__`, so you can run from any directory.

The `yolov4_leaky_spp_m/` model directory must exist in the **repo root** (or wherever you run from on the board).
