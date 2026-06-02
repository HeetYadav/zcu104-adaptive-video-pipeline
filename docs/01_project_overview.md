[← README](../README.md) | [↑ Docs Index](../README.md#documentation) | [Next: Hardware Setup →](02_hardware_setup.md)

---

# 01: Project Overview

## Table of Contents
- [The Problem](#the-problem)
- [The Hypothesis](#the-hypothesis)
- [Why FPGA?](#why-fpga)
- [Project Evolution](#project-evolution)
- [Key Results](#key-results)
---

## The Problem

Modern surveillance, robotics, and edge-vision systems face a fundamental tension: **they need to stream high-quality video for analysis, but the networks and storage they run on are bandwidth-constrained.**

A 1080p camera streaming uncompressed video generates approximately **1.5 Gbps**. Even compressed MJPEG at moderate quality outputs **4,000–8,000 kbps** depending on scene complexity. On embedded systems where multiple cameras share a single network link, this is unsustainable.

The naive solution: simply compress harder: destroys image quality in the regions where it matters most: around the people and objects being monitored.

**The real question is:** *Can we compress the background without touching the foreground?*

---

## The Hypothesis

> If we detect people using an AI model, and then apply a 3-zone spatial masking strategy before H.264 encoding, the encoder's Variable Bitrate algorithm will naturally allocate near-zero bits to the black background while preserving full quality in the region of interest.

This hypothesis has three sub-claims:

1. **Detection accuracy:** A YOLOv4 model running on dedicated AI hardware (the DPU) can detect persons reliably enough in real-time to drive the masking decision
2. **Compression physics:** An H.264 encoder in VBR mode, presented with large areas of pure black (`0x000000`), will reduce its bitrate proportionally: black pixels compress to near-zero because they produce zero DCT coefficients
3. **Hardware feasibility:** The DPU (AI inference) and VCU (H.264 encoding) on the ZCU104 can both operate simultaneously, with the ARM CPU handling only the orchestration logic

All three claims are validated experimentally in this project.

---

## Why FPGA?

A natural question is: *why not just run this on a GPU-accelerated server?*

The answer is **latency, power, and deployment scenario**.

| Concern | GPU Server | ZCU104 FPGA |
|---------|-----------|-------------|
| Power draw | 150–300 W | ~10 W total |
| Network dependency | Requires sending raw video to server first | Processing happens at the camera, only compressed ROI stream sent |
| Latency | Round-trip to server adds 50–200 ms | Sub-frame latency, all local |
| Cost | High ($$$) | Moderate |
| Deployment | Data center | Edge device, weather-hardened |

The ZCU104 is an **edge device**: it sits next to the camera, processes the video locally, and only sends the bandwidth-reduced stream downstream.

Furthermore, the ZCU104's **DPU** (Deep Learning Processing Unit) is a dedicated INT8 matrix-multiply engine burned into the FPGA fabric. It runs YOLOv4 inference at a fraction of the power cost of even a small GPU.

---

## Project Evolution

The project is organized around two key pipelines, representing the baseline and the final hardware-accelerated solution:

### Baseline: `pipeline_hw_1.py` (DPU Inference)

- **Inference:** YOLOv4 on the DPU (FPGA fabric) via Vitis AI Runtime (`vart`)- **Encoding:** MJPEG (CPU-based)- **Output:** MJPEG HTTP stream on port 5000- **Bandwidth:** ~6,000–8,000 kbps (MJPEG is uncompressed)- **CPU load:** Moderate: CPU handles the compositor and HTTP server- **Key achievement:** Proves the DPU can run YOLOv4 reliably and feed bounding boxes to the compositor in real-time
### Final: `pipeline_hw.py` (Full Hardware Acceleration)

- **Inference:** YOLOv4 on DPU (unchanged from Baseline)- **Encoding:** VCU hardware H.264 encoder (`omxh264enc`) via GStreamer: runs on dedicated silicon in the FPGA PL, zero CPU cycles- **Visualization:** MJPEG stream (smooth, immediate, parallel to VCU encoding)- **Telemetry:** VCU H.264 bandwidth calculated from active pixel area ratio- **Output:** MJPEG HTTP stream on port 5000 (view in VLC), H.264 telemetry in terminal- **Bandwidth (H.264 equivalent):** ~120–700 kbps depending on how many persons are in frame- **CPU load:** Minimal: CPU handles only Python thread coordination
---

## Key Results

| Metric | Value |
|--------|-------|
| MJPEG Baseline Bandwidth | **8,821.1 kbps** |
| VCU H.264 + ROI Bandwidth | **841.4 kbps** |
| Average Bandwidth Reduction | **90.5% (10.5x less)** |
| MJPEG Baseline Framerate | 8.6 FPS |
| VCU H.264 + ROI Framerate | **9.5 FPS (+10% faster)** |
| Inference hardware | DPU B4096 on FPGA fabric |
| Encoding hardware | VCU H.264 encoder on FPGA PL |
| CPU utilization during Final HW | Low (orchestration only) |
| Detection model | YOLOv4 (leaky SPP, quantized to INT8) |
| Target class | Person |
| Inference input resolution | 416×416 px |

---

[← README](../README.md) | [↑ Docs Index](../README.md#documentation) | [Next: Hardware Setup →](02_hardware_setup.md)
