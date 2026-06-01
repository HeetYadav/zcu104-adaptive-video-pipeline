[← Zone Masking](04_zone_masking_algorithm.md) | [↑ Back to README](../README.md) | [Next: VCU Encoding →](06_vcu_encoding.md)

---

# 05: DPU Inference

## Table of Contents
- [What is the DPU?](#what-is-the-dpu)- [The YOLOv4 Model](#the-yolov4-model)- [Preprocessing Pipeline](#preprocessing-pipeline)- [Critical Bug Fix: INT8 Type Mismatch](#critical-bug-fix--int8-type-mismatch)- [RTLD_GLOBAL: The Hidden C++ ABI Fix](#rtld_global--the-hidden-c-abi-fix)- [DPU Runner Initialization](#dpu-runner-initialization)- [Output Decoding: Boxes, Scores, NMS](#output-decoding)- [Inference Timing](#inference-timing)
---

## What is the DPU?

The **DPU (Deep Learning Processing Unit)** is a parameterizable IP core from Xilinx that implements a dedicated matrix-multiply accelerator optimized for deep learning inference. It is instantiated in the **Programmable Logic (FPGA fabric)** of the ZCU104, not on the ARM CPU.

| Property | Value |
|----------|-------|
| **DPU Variant** | B4096 (B-series, 4096 parallel operations/cycle) |
| **Precision** | INT8 (8-bit integer arithmetic) |
| **Interface** | AXI4 Master + AXI4-Lite Slave |
| **Memory** | Direct DMA to LPDDR4 (shared with ARM) |
| **Power** | ~2–4 W (vs ~25 W for a comparable GPU inference) |
| **Runtime** | Vitis AI Runtime (`vart`) |

The DPU is controlled by the **Vitis AI Runtime (vart)**: a C++ library with Python bindings (`import vart`). The runtime handles:
- Loading the compiled `.xmodel` into DPU fabric registers- Allocating DMA buffers for input/output tensors- Submitting inference jobs and waiting for completion
---

## The YOLOv4 Model

| Property | Value |
|----------|-------|
| **Model** | YOLOv4 Leaky SPP Medium (`yolov4_leaky_spp_m`) |
| **Source** | Xilinx Model Zoo (pre-quantized for DPU) |
| **Task** | Object detection (80 COCO classes; we use only `person`) |
| **Input tensor** | `[1, 416, 416, 3]`: Batch=1, H=416, W=416, C=3 (RGB) |
| **Input dtype** | **INT8 (signed)** ← critical, see below |
| **Output** | 3 scale detection heads (13×13, 26×26, 52×52) |
| **Quantization** | Post-training quantization, INT8, symmetric |
| **Architecture file** | `yolov4_leaky_spp_m.xmodel` |

---

## Preprocessing Pipeline

Before sending a frame to the DPU, the frame must be preprocessed to match the model's input tensor specification:

```python
# 1. Resize to model input size
resized = cv2.resize(bgr_frame, (416, 416))

# 2. Convert BGR → RGB (YOLO models expect RGB)
rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

# 3. INT8 conversion: CRITICAL (see bug section below)
img_data = rgb.astype(np.uint8)       # still uint8 (0–255)
img_int8 = (img_data.astype(np.int16): 128).astype(np.int8)
# Maps: 0 → -128, 128 → 0, 255 → +127

# 4. Reshape to match tensor layout [1, 416, 416, 3]
input_tensor_data[0] = img_int8.reshape(1, 416, 416, 3)

# 5. Submit to DPU runner
job_id = runner.execute_async(input_tensor_data, output_tensor_data)
runner.wait(job_id)
```

---

## Critical Bug Fix: INT8 Type Mismatch

> [!CAUTION]
> This is the most important bug in the entire project. Getting this wrong causes an **immediate hard crash**: not a graceful error.

### The Crash

```
F0529 12:06:22.871002  5348 xrt_device_handle_imp.cpp:250]
  xir::DataType::UNKNOWN abort
Aborted (core dumped)
```

### Root Cause

The `.xmodel` declares its input tensor type as `INT8` (signed, range -128 to +127). When the Python code passes a `uint8` array (range 0 to 255), the Xilinx IR library (`xir`) cannot match the Python type to the expected tensor type, reports it as `UNKNOWN`, and aborts.

### The Wrong Code

```diff
- img_int8 = rgb.astype(np.uint8)           # WRONG: dtype is uint8- input_tensor_data[0] = img_int8            # xir sees uint8, expects int8 → CRASH```

### The Correct Code

```diff
+ img_int8 = (rgb.astype(np.int16): 128).astype(np.int8)
+ # Cast to int16 FIRST before subtracting 128 to avoid uint8 underflow
+ # Then cast to int8: maps [0,255] → [-128,+127]
+ input_tensor_data[0] = img_int8            # dtype matches → OK
```

> [!WARNING]
> You **must** cast to `int16` before subtracting 128. If you do `rgb.astype(np.int8): 128`, the subtraction overflows because `int8` cannot hold values below -128. Always go through `int16` as the intermediate type.

### Why the Mapping Works

The model was quantized with a zero-point of 128 (standard for unsigned-to-signed quantization). Subtracting 128 re-centers the distribution around 0, which is what the INT8 model weights expect.

---

## RTLD_GLOBAL: The Hidden C++ ABI Fix

### The Problem

The Vitis AI C++ libraries (`vart.so`, `xir.so`) share C++ RTTI (Run-Time Type Information) types, including `std::any`. When Python loads these libraries with the default `dlopen()` flags, each `.so` gets its own copy of the RTTI type information in separate symbol namespaces.

When `vart.so` creates a `std::any` and `xir.so` tries to `std::any_cast` it, it compares the two `typeinfo` objects from different namespaces: they don't match: and the cast throws `std::bad_any_cast`, crashing the process.

### The Fix

```python
# At the VERY TOP of the file, before ANY vart/xir import:
import sys as _sys, os as _os
_sys.setdlopenflags(_os.RTLD_GLOBAL | _os.RTLD_LAZY)

# NOW import the Vitis AI libraries
import vart
import xir
```

`RTLD_GLOBAL` tells the dynamic linker to export all symbols from each loaded library into the global symbol table. When `vart.so` and `xir.so` both use `RTLD_GLOBAL`, they share a **single** copy of each `typeinfo` object: the same address: so `std::any_cast` succeeds.

> [!CAUTION]
> `setdlopenflags` must be called **before** `import vart` or `import xir`. If you import them first and then set the flag, it has no effect: the libraries are already loaded with the wrong flags.

---

## DPU Runner Initialization

```python
# 1. Deserialize the .xmodel into a graph
graph = xir.Graph.deserialize("yolov4_leaky_spp_m/yolov4_leaky_spp_m.xmodel")

# 2. Find the DPU subgraph (there is exactly one for this model)
root = graph.get_root_subgraph()
subgraphs = [s for s in root.toposort_child_subgraph()
             if s.has_attr("device") and s.get_attr("device").upper() == "DPU"]
# subgraphs[0] is the DPU acceleratable portion of the model

# 3. Create the runner (allocates DMA buffers, locks DPU resources)
runner = vart.Runner.create_runner(subgraphs[0], "run")

# 4. Get tensor descriptors
input_tensors  = runner.get_input_tensors()   # [1, 416, 416, 3], INT8
output_tensors = runner.get_output_tensors()  # 3 detection heads
```

> [!NOTE]
> `create_runner()` acquires an exclusive lock on the DPU hardware. Only **one runner** can hold the DPU at a time. This is why `realBenchmark.py` kills one pipeline completely before starting the next: otherwise the second pipeline would block indefinitely waiting for the DPU lock to be released.

---

## Output Decoding

The DPU returns raw output tensors from three detection heads. Decoding them involves:

1. **Reshaping** each tensor from flat bytes to `[1, H, W, anchors*(5+classes)]`
2. **Applying sigmoid** to objectness and class scores
3. **Decoding box coordinates** from the grid-relative offsets using YOLO anchor math
4. **Filtering** by confidence threshold (`CONF_THRESH = 0.30`)
5. **Filtering** by class index 0 (person in COCO)
6. **NMS (Non-Maximum Suppression)**: removes overlapping boxes for the same person

---

## Inference Timing

| Step | Typical Time |
|------|-------------|
| Frame capture + HTTP GET | ~30–50 ms (Wi-Fi dependent) |
| Preprocess (resize + int8 convert) | ~2–5 ms |
| DPU inference (YOLOv4) | ~15–25 ms |
| Output decode + NMS | ~2–5 ms |
| **Total per inference** | **~50–85 ms (~12–20 FPS)** |

> [!TIP]
> The Detector and Compositor threads run concurrently. The Compositor paces itself to 30 FPS using the most recent `_faces` result. If detection takes 80 ms, the Compositor simply reuses the previous detection result for 2–3 frames: the person doesn't visibly jump because the motion-predictive padding keeps them comfortably inside Zone 1.

---

[← Zone Masking](04_zone_masking_algorithm.md) | [↑ Back to README](../README.md) | [Next: VCU Encoding →](06_vcu_encoding.md)
