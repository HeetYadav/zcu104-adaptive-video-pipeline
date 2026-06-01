[↑ Back to README](../../README.md) | [Docs →](../../docs/)

---

# Pipelines

This folder contains the three runnable pipeline entry points — each represents a distinct phase in the project's evolution. Run them directly on the ZCU104 board.

## Which Pipeline Should I Run?

| Pipeline | Folder | Hardware Used | Purpose |
|----------|--------|--------------|---------|
| **`pipeline_hw.py`** | [`pipeline_hw/`](pipeline_hw/) | DPU + VCU (Full HW) | ✅ **Primary** — production pipeline with hardware H.264 telemetry |
| **`pipeline_hw_1.py`** | [`pipeline_hw_1/`](pipeline_hw_1/) | DPU only | 📊 **Benchmark baseline** — DPU inference with MJPEG output |
| **`pipeline.py`** | [`pipeline_sw/`](pipeline_sw/) | CPU only | 📚 **Historical** — Phase 2 software-only baseline |

## Quick Commands

```bash
# Full hardware pipeline (recommended)
python3 pipelines/pipeline_hw/pipeline_hw.py

# MJPEG baseline (for benchmark comparison)
python3 pipelines/pipeline_hw_1/pipeline_hw_1.py

# Software-only baseline (historical reference)
python3 pipelines/pipeline_sw/pipeline.py
```

## Dependencies

All three pipelines import from [`modules/`](../modules/). The import path is resolved automatically using `__file__`, so you can run from any directory.

The `yolov4_leaky_spp_m/` model directory must exist in the **repo root** (or wherever you run from on the board).
