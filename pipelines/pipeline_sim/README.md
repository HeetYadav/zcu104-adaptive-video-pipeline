[↑ Pipelines](../README.md) | [Back to Repo Root](../../README.md) | [Simulation Guide →](../../docs/00_prerequisites.md)

---

# `pipeline_sim/` — CPU-Only Simulation Pipeline

Run the complete ZCU104 ROI bandwidth management algorithm **on any laptop** — no FPGA, no ZCU104, no PetaLinux required.

This pipeline mirrors `pipeline_hw.py`'s 4-thread architecture exactly, replacing:

| `pipeline_hw.py` (board) | `pipeline_sim.py` (laptop) |
|--------------------------|---------------------------|
| DPU (FPGA inference) | OpenCV DNN (CPU YOLOv4) |
| VCU (H.264 hardware) | Bandwidth model from active pixel ratio |
| IP Webcam app (phone) | Local video file, webcam, or RTSP URL |
| PetaLinux board | Any Windows / Linux / macOS machine |

---

## Requirements

```bash
pip install opencv-python numpy
```

**Python 3.7+ is required.** No other dependencies.

---

## Model Files

You need YOLOv4 `.cfg` and `.weights` files. Place them in the **repo root** (same directory as `README.md`).

### Option A: YOLOv4-Tiny (Recommended for CPU — faster)

```bash
# .cfg (already included in the repo root)
# yolov4-tiny.cfg is present ✓

# Download weights (~23 MB):
wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights
```

### Option B: Full YOLOv4 (Higher accuracy, slower on CPU)

```bash
# Download .cfg:
wget https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg

# Download weights (~244 MB):
wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights
```

> [!TIP]
> For learning the algorithm, start with YOLOv4-Tiny (`--tiny` flag). It runs at 5–15 FPS on a laptop CPU vs. 0.5–2 FPS for full YOLOv4.

---

## Usage

```bash
# Run from the repo root:

# Video file (recommended for first run)
python3 pipelines/pipeline_sim/pipeline_sim.py --input path/to/video.mp4 --tiny

# Webcam (device 0)
python3 pipelines/pipeline_sim/pipeline_sim.py --input 0 --tiny

# RTSP stream
python3 pipelines/pipeline_sim/pipeline_sim.py --input rtsp://192.168.1.100/stream --tiny

# Full YOLOv4 (slower but more accurate)
python3 pipelines/pipeline_sim/pipeline_sim.py --input video.mp4

# Save output to a file
python3 pipelines/pipeline_sim/pipeline_sim.py --input video.mp4 --tiny --output result.mp4

# Headless (no display window — for servers)
python3 pipelines/pipeline_sim/pipeline_sim.py --input video.mp4 --tiny --no-display
```

### All Options

```
--input         Video source: file path, webcam index (0), or RTSP URL  [required]
--output        Save composited output to this video file
--conf          Detection confidence threshold (default: 0.30)
--nms           NMS IoU threshold (default: 0.40)
--tiny          Use YOLOv4-Tiny weights (faster on CPU)
--cfg           Path to custom .cfg file
--weights       Path to custom .weights file
--max-targets   Maximum simultaneous persons to track (default: 5)
--no-display    Disable OpenCV window (useful on headless servers)
```

---

## What You Will See

An OpenCV window showing the 3-zone composited output:

- **Black background** — Zone 3: pure zero pixels (near-zero bits in H.264 VBR)
- **Green rectangle** labeled `Z1` — Zone 1: full-resolution ROI around each person
- **Dashed amber rectangle** labeled `Z2` — Zone 2: 50%-downsampled proximity ring
- **"CPU SIM — No FPGA" watermark** — visual reminder this is simulation mode

Terminal telemetry every 30 frames:

```
[Detector] 2 person(s) detected
    [Telemetry] frame=    30 | targets=2 | BW:  258.3 kbps ( 7.2 FPS)  [CPU sim]
    [Telemetry] frame=    60 | targets=2 | BW:  241.7 kbps ( 7.5 FPS)  [CPU sim]
```

The bandwidth figure uses the **same active-pixel-ratio model** as `pipeline_hw.py` — it represents what the VCU hardware H.264 encoder would produce for that scene.

---

## How It Differs from the Real Pipeline

| Aspect | `pipeline_hw.py` (board) | `pipeline_sim.py` (laptop) |
|--------|--------------------------|---------------------------|
| **Inference speed** | ~15–25 ms (DPU INT8) | ~200–2000 ms (CPU FP32) |
| **Inference accuracy** | Slightly lower (INT8 quantization) | Higher (FP32 weights) |
| **Bandwidth figure** | Measured from VCU hardware | Modelled from pixel area ratio |
| **Frame rate** | 8–10 FPS (camera-limited) | 1–15 FPS (CPU inference limited) |
| **Power** | ~10 W total | Varies by laptop |

The **zone masking algorithm, motion prediction, and bandwidth model are identical** in both pipelines. Simulation mode teaches you the algorithm; the board proves it on real silicon.

---

## Stopping

Press **Q** in the video window, or **Ctrl+C** in the terminal.

---

## See Also

- [Project Overview →](../../docs/01_project_overview.md)
- [Zone Masking Algorithm →](../../docs/04_zone_masking_algorithm.md)
- [DPU Inference (how it differs from CPU) →](../../docs/05_dpu_inference.md)
- [Hardware Setup →](../../docs/02_hardware_setup.md) ← for when you get a ZCU104
