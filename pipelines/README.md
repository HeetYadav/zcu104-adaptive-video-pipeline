[↑ Back to README](../../README.md) | [Docs →](../../docs/)

---

# Pipelines

This folder contains the three runnable pipeline entry points. Two run directly on the ZCU104 board, and one runs on any standard computer (laptop/desktop).

## Which Pipeline Should I Run?

| Pipeline | Folder | Hardware Used | Purpose |
|----------|--------|--------------|---------|
| **`pipeline_hw.py`** | [`pipeline_hw/`](pipeline_hw/) | DPU + VCU (Full HW) | ✅ **Primary**: production pipeline with hardware H.264 telemetry |
| **`pipeline_hw_1.py`** | [`pipeline_hw_1/`](pipeline_hw_1/) | DPU only | 📊 **Benchmark baseline**: DPU inference with MJPEG output |
| **`pipeline_sim.py`** | [`pipeline_sim/`](pipeline_sim/) | CPU only (Laptop/Desktop) | 💻 **Simulation**: test the algorithm locally without an FPGA |

## Quick Commands

```bash
# CPU Simulation (run on your laptop)
python3 pipelines/pipeline_sim/pipeline_sim.py

# Full hardware pipeline (recommended on board)
python3 pipelines/pipeline_hw/pipeline_hw.py

# MJPEG baseline (for benchmark comparison on board)
python3 pipelines/pipeline_hw_1/pipeline_hw_1.py
```

## Dependencies

All pipelines import from [`modules/`](../modules/). The import path is resolved automatically using `__file__`, so you can run from any directory.

The `yolov4_leaky_spp_m/` model directory must exist in the **repo root** (or wherever you run from on the board).
