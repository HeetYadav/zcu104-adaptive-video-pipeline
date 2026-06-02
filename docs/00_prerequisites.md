[← README](../README.md) | [↑ Docs Index](../README.md#documentation) | [Next: Project Overview →](01_project_overview.md)

---

# 00: Prerequisites

Before diving into this project, make sure you are comfortable with the following concepts. Each section is written to be read in **5 minutes or less**, and links to deeper resources if you want to go further.

## Table of Contents

- [What is an FPGA?](#what-is-an-fpga)
- [What is the ZCU104 SoC?](#what-is-the-zcu104-soc)
- [What is the DPU?](#what-is-the-dpu)
- [What is INT8 Quantization?](#what-is-int8-quantization)
- [What is GStreamer?](#what-is-gstreamer)
- [Linux and Networking Basics](#linux-and-networking-basics)
- [Python Knowledge Needed](#python-knowledge-needed)
- [Learning Resources](#learning-resources)

---

## What is an FPGA?

An **FPGA (Field-Programmable Gate Array)** is an integrated circuit whose logic can be reconfigured after manufacturing — unlike a CPU or GPU whose silicon is fixed forever.

You program an FPGA by writing hardware description code (VHDL or Verilog), which a synthesis tool converts into a bitstream that configures the FPGA's internal lookup tables and routing.

**What makes FPGAs useful for this project:**

| Feature | Why It Matters Here |
|---------|-------------------|
| Custom silicon | The DPU and VCU are specialized circuits burned into the FPGA fabric — not general-purpose CPU cores |
| Parallel execution | DPU runs inference while VCU encodes video simultaneously — physically parallel, not software-scheduled |
| Low power | The ZCU104 runs the entire pipeline at ~10 W vs ~150 W for a GPU server |

> [!NOTE]
> You do **NOT** need to write any VHDL or Verilog to use this project. The FPGA is already programmed by the Vitis AI TRD image. You interact with it entirely through Python.

---

## What is the ZCU104 SoC?

The ZCU104 uses a **Zynq UltraScale+ MPSoC (ZU7EV)** — a System-on-Chip that combines:

```
┌──────────────────────────────────────────────────────────┐
│                   ZCU104 ZU7EV SoC                       │
│                                                          │
│  ┌─────────────────────────┐  ┌───────────────────────┐ │
│  │   Processing System     │  │  Programmable Logic   │ │
│  │   (PS) — fixed silicon  │  │  (PL) — FPGA fabric   │ │
│  │                         │  │                       │ │
│  │  • Quad ARM Cortex-A53  │  │  • DPU (AI engine)    │ │
│  │  • 4 GB LPDDR4 RAM      │  │  • VCU (H.264 codec)  │ │
│  │  • Ethernet, USB, eMMC  │  │  • 504K LUTs          │ │
│  │  • Runs PetaLinux OS    │  │  • 1728 DSP slices    │ │
│  └─────────────────────────┘  └───────────────────────┘ │
│                    ↕ AXI Interconnect ↕                  │
└──────────────────────────────────────────────────────────┘
```

**PS (Processing System):** The ARM cores. This is where Python runs. Think of it as a small Linux computer.

**PL (Programmable Logic):** The FPGA fabric. This is where the DPU and VCU live. They are configured by loading a bitstream (the Vitis AI TRD image does this at boot).

**Key insight:** The DPU and VCU run on PL while Python runs on PS. They share the same LPDDR4 memory via DMA — Python passes a pointer to an image, and the DPU/VCU processes it directly without the ARM CPU being involved in the computation.

---

## What is the DPU?

The **DPU (Deep Learning Processing Unit)** is a hardware accelerator IP core from AMD/Xilinx that implements a matrix-multiply engine optimized for neural network inference.

| Property | Value in This Project |
|----------|----------------------|
| Variant | B4096 (4096 parallel multiply-add ops/cycle) |
| Precision | INT8 (8-bit signed integers) |
| Model format | `.xmodel` (compiled by Vitis AI quantizer) |
| Python interface | `vart` (Vitis AI Runtime) |

**Analogy:** If a GPU is a general-purpose parallel processor, the DPU is a narrow specialist — it only does matrix multiply in INT8, but it does it extremely efficiently at a fraction of the power.

**What INT8 means:** Instead of 32-bit floating point (FP32), weights and activations are quantized to 8-bit integers. This is 4× smaller and 4× faster to multiply — with minimal accuracy loss after proper calibration.

---

## What is INT8 Quantization?

Neural networks are typically trained in **FP32** (32-bit floating point). Running them on embedded hardware requires **quantization** — converting weights and activations to lower precision.

**INT8 Symmetric Quantization:**

```
FP32 value range:  [-1.5, +1.5]   (for example)
INT8 range:        [-128, +127]

Scale factor = max(abs(range)) / 127
            = 1.5 / 127 ≈ 0.0118

To quantize:   int8_val = round(fp32_val / scale)
To dequantize: fp32_approx = int8_val * scale
```

**Why it matters for this project:** The YOLOv4 model's input tensor expects INT8 values in the range [-128, +127]. If you pass uint8 (range [0, 255]), the Xilinx IR library crashes with `xir::DataType::UNKNOWN`. The critical fix (subtract 128) re-centers the uint8 range into the signed INT8 space.

See [DPU Inference — Critical Bug Fix →](05_dpu_inference.md#critical-bug-fix--int8-type-mismatch)

---

## What is GStreamer?

**GStreamer** is an open-source multimedia pipeline framework for Linux. It lets you chain processing elements together to build audio/video pipelines declaratively.

**Pipeline syntax (the `!` operator):**

```
element1 ! element2 ! element3
```

This reads as: "pipe output of element1 into element2, pipe output of element2 into element3."

**This project's VCU pipeline:**

```
appsrc ! videoconvert ! video/x-raw,format=NV12 ! omxh264enc control-rate=variable target-bitrate=1500 ! fakesink sync=false
```

| Element | Role |
|---------|------|
| `appsrc` | Receives raw frames pushed from Python |
| `videoconvert` | Converts BGR → NV12 (VCU input format) |
| `omxh264enc` | Hardware H.264 encoder (the VCU) |
| `fakesink` | Discards output (telemetry mode) |

You don't need deep GStreamer knowledge to use this project. The pipeline string is already written. See [VCU Encoding →](06_vcu_encoding.md) for a full element-by-element breakdown.

---

## Linux and Networking Basics

The ZCU104 runs **PetaLinux** — a minimal embedded Linux. You'll need to be comfortable with:

| Task | Command |
|------|---------|
| SSH into the board | `ssh root@<board-ip>` |
| Copy files to board | `scp -O -o HostKeyAlgorithms=+ssh-rsa file root@<ip>:/home/root/` |
| Find the board's IP | `hostname -I` (run on the board) |
| Kill a process | `pkill python3` |
| Check running processes | `ps aux \| grep python3` |
| Run a script | `python3 pipeline_hw.py` |
| Stop a script | `Ctrl+C` |
| Check kernel logs | `dmesg \| tail -20` |

**Network setup:** All three devices (board, phone, laptop) must be on the **same subnet** — typically connected to the same Wi-Fi router.

> [!TIP]
> The full hardware setup walkthrough (flashing the SD card, connecting via SSH, configuring the phone camera) is in [docs/02_hardware_setup.md](02_hardware_setup.md).

---

## Python Knowledge Needed

This project uses:

| Concept | Where Used |
|---------|-----------|
| `threading.Thread` + `threading.Lock` | 4-thread pipeline architecture |
| `numpy` arrays | Frame buffers, DPU tensor buffers, zone masking |
| `cv2` (OpenCV) | Frame decode, resize, color convert, JPEG encode, GStreamer writer |
| `http.server` | MJPEG HTTP streaming server |
| `subprocess` | Benchmark runner launches pipelines as subprocesses |

You do **not** need to know CUDA, TensorFlow, PyTorch, or any other ML framework. The DPU runtime (`vart`) abstracts all hardware details.

---

## Learning Resources

### FPGA / ZCU104
- 📖 [ZCU104 Evaluation Board User Guide (UG1267)](https://docs.xilinx.com/r/en-US/ug1267-zcu104-eval-bd) — board pinout, boot mode switches, peripherals
- 📖 [Zynq UltraScale+ MPSoC Technical Reference Manual (UG1085)](https://docs.xilinx.com/r/en-US/ug1085-zynq-ultrascale-trm) — deep SoC architecture

### Vitis AI / DPU
- 📖 [Vitis AI User Guide (UG1414)](https://docs.xilinx.com/r/en-US/ug1414-vitis-ai) — model quantization, compilation, runtime
- 🎬 [Xilinx Vitis AI YouTube Playlist](https://www.youtube.com/playlist?list=PLr6QJqmr2rVYGPi7CJDG7Y_dFUXxVcAY) — hands-on tutorials
- 📦 [Xilinx Model Zoo (GitHub)](https://github.com/Xilinx/Vitis-AI/tree/master/model_zoo) — pre-quantized models for ZCU104

### GStreamer
- 📖 [GStreamer Application Development Manual](https://gstreamer.freedesktop.org/documentation/application-development/) — core concepts
- 🛠️ `gst-inspect-1.0 omxh264enc` — run on the board to see all VCU encoder parameters

### Object Detection (YOLO)
- 📖 [YOLOv4 Paper](https://arxiv.org/abs/2004.10934) — the model used in this project
- 🎬 [Understanding YOLO](https://www.youtube.com/watch?v=9s_FpMpdYW8) — visual explanation of anchor boxes and detection heads

---

[← README](../README.md) | [↑ Docs Index](../README.md#documentation) | [Next: Project Overview →](01_project_overview.md)
