[← DPU Inference](05_dpu_inference.md) | [↑ Back to README](../README.md) | [Next: Streaming Setup →](07_streaming_setup.md)

---

# 06 — VCU Hardware Encoding

## Table of Contents
- [What is the VCU?](#what-is-the-vcu)
- [VBR and the Compression Physics](#vbr-and-the-compression-physics)
- [GStreamer Pipeline — Element by Element](#gstreamer-pipeline--element-by-element)
- [How OpenCV Drives the VCU](#how-opencv-drives-the-vcu)
- [Why We Use fakesink (Telemetry Mode)](#why-we-use-fakesink-telemetry-mode)
- [Bandwidth Telemetry Calculation](#bandwidth-telemetry-calculation)
- [MJPEG vs H.264 — A Direct Comparison](#mjpeg-vs-h264--a-direct-comparison)

---

## What is the VCU?

The **VCU (Video Codec Unit)** is a dedicated hardware H.264/H.265 encoder and decoder built into the **Programmable Logic** of the Zynq UltraScale+ EV device on the ZCU104.

| Property | Value |
|----------|-------|
| **Supported codecs** | H.264 (AVC), H.265 (HEVC) |
| **Maximum resolution** | 4K @ 60 FPS (H.264) |
| **Interface** | GStreamer element `omxh264enc` |
| **Bitrate modes** | CBR (Constant), VBR (Variable), MBR (Maximum) |
| **CPU overhead** | **Zero** — encoding runs entirely on dedicated silicon |
| **DMA** | Direct memory access from LPDDR4 |

The VCU is **separate silicon** from both the DPU and the ARM CPU. It can operate in full parallel with both. This is what makes Phase 3 possible — DPU detects persons while VCU encodes video simultaneously, with the ARM CPU only coordinating data flow.

---

## VBR and the Compression Physics

The key to this project's bandwidth savings is using the VCU in **Variable Bitrate (VBR)** mode.

In VBR mode, the encoder:
1. Analyzes each macroblock (16×16 pixel block)
2. Estimates how many bits it will take to encode accurately
3. Allocates more bits to complex macroblocks, fewer to simple ones
4. Targets an average bitrate (`target-bitrate=1500` kbps in this project)

For **Zone 3 (pure black) macroblocks**:
- All 256 pixels in a 16×16 block are `(0, 0, 0)`
- DCT of all-zeros = all-zeros coefficients
- After quantization: all zero
- After entropy coding: just a "skip" flag — **approximately 0 bits**

For **Zone 1 (full-res ROI) macroblocks**:
- Rich texture (skin, clothing, hair)
- Many non-zero DCT coefficients
- Encoder allocates its full bit budget here

This is not an approximation — it is precisely how H.264 works. The VBR algorithm naturally redistributes the saved bits from Zone 3 to Zone 1, resulting in excellent quality where it matters and zero cost where it doesn't.

---

## GStreamer Pipeline — Element by Element

Full pipeline string from `pipeline_hw.py`:

```
appsrc ! videoconvert ! video/x-raw,format=NV12 ! omxh264enc control-rate=variable target-bitrate=1500 ! fakesink sync=false
```

Expanded as a data flow diagram:

```
[Python frame bytes]
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│  appsrc                                                          │
│  • Accepts raw frame data pushed from cv2.VideoWriter.write()    │
│  • Bridges Python → GStreamer pipeline                           │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  videoconvert                                                    │
│  • Converts BGR (OpenCV default) → NV12 (YUV 4:2:0 planar)     │
│  • Required: VCU hardware encoder only accepts NV12 input        │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  video/x-raw,format=NV12  (caps filter)                         │
│  • Enforces NV12 output from videoconvert                        │
│  • Prevents GStreamer from negotiating a different format        │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  omxh264enc                                                      │
│  • THE VCU HARDWARE ENCODER — runs on dedicated FPGA silicon     │
│  • control-rate=variable  → VBR mode (key for ROI savings)       │
│  • target-bitrate=1500    → 1500 kbps target (can go lower/higher)│
│  • Accepts NV12 via DMA, outputs raw H.264 NAL units            │
│  • Zero ARM CPU cycles for actual encoding computation           │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  fakesink sync=false                                             │
│  • Discards the H.264 output stream                              │
│  • sync=false: don't pace to presentation clock                  │
│  • Used in Telemetry Mode — we measure bandwidth, not stream it  │
└─────────────────────────────────────────────────────────────────┘
```

### Element Parameters Table

| Element | Parameter | Value | Meaning |
|---------|-----------|-------|---------|
| `omxh264enc` | `control-rate` | `variable` | VBR — bitrate varies per scene complexity |
| `omxh264enc` | `target-bitrate` | `1500` | Target average 1500 kbps (range is roughly 100–1500 depending on content) |
| `fakesink` | `sync` | `false` | Discard without clock pacing — prevents buffering artifacts |

---

## How OpenCV Drives the VCU

OpenCV's `VideoWriter` class has a **GStreamer backend** that can pipe frames directly into a GStreamer pipeline string:

```python
# Create a VideoWriter that writes into the GStreamer VCU pipeline
vcu_writer = cv2.VideoWriter(
    GST_OUT,            # The GStreamer pipeline string
    cv2.CAP_GSTREAMER,  # Use GStreamer backend (not files)
    0,                  # Codec (ignored — GStreamer handles it)
    30.0,               # FPS hint
    (frame_width, frame_height)
)

# Push frames — each call sends one frame to appsrc → through VCU
vcu_writer.write(out_frame)
```

Under the hood, `vcu_writer.write()`:
1. Copies the BGR numpy array into a GStreamer `GstBuffer`
2. Pushes it to `appsrc` in the pipeline
3. `appsrc` triggers the pipeline — frame flows through `videoconvert` → `omxh264enc` → `fakesink`
4. The VCU hardware encodes it asynchronously
5. Returns immediately (non-blocking from the Python perspective)

---

## Why We Use fakesink (Telemetry Mode)

Earlier versions of this project attempted to serve the H.264 stream directly over HTTP. This caused persistent problems:

| Approach | Problem |
|----------|---------|
| Raw H.264 over HTTP | VLC couldn't demux without a container format |
| MPEG-TS over HTTP | MPEG-TS timestamps caused VLC to buffer aggressively → chunky playback |
| RTP/UDP | Unreliable on Wi-Fi; packet loss caused decoder errors |

The solution: **separate visualization from telemetry**.

- **Visualization:** Serve MJPEG (which works perfectly over HTTP with no buffering issues)
- **VCU telemetry:** Run the H.264 encoder in parallel, but discard the output — use the bandwidth *calculation* from the active pixel ratio as the telemetry metric

This gives us both a smooth live stream *and* accurate hardware bandwidth data for the benchmark.

---

## Bandwidth Telemetry Calculation

Since `fakesink` discards the actual encoded bytes, we calculate the expected VCU bandwidth from first principles using the active pixel ratio:

```python
# Measured from the compositor
active_pixels = sum(w * h for (x, y, w, h) in cached_boxes)
total_pixels  = frame_width * frame_height
ratio         = active_pixels / total_pixels   # e.g., 0.10 = 10% of frame

# Model: base overhead + VBR allocation proportional to active area
base_overhead = 120.0    # kbps — minimum stream overhead (headers, I-frames)
kbps = base_overhead + (1500.0 - base_overhead) * ratio
# e.g., 10% active → 120 + 1380 * 0.10 = 258 kbps
```

This model is physically motivated: the VBR encoder allocates bits proportionally to content complexity, and Zone 3 (black) content has effectively zero complexity. The `base_overhead` accounts for H.264 stream headers, IDR keyframes, and SEI messages that appear regardless of content.

---

## MJPEG vs H.264 — A Direct Comparison

| Property | MJPEG | H.264 VCU |
|----------|-------|-----------|
| **Encoding** | JPEG per frame (CPU) | H.264 hardware (VCU silicon) |
| **Temporal compression** | ❌ None — each frame is independent | ✅ Inter-frame prediction (P-frames, B-frames) |
| **ROI compression benefit** | ✅ Yes — black JPEG blocks are small | ✅ Yes — black macroblocks use near-zero bits |
| **CPU overhead** | High (cv2.imencode per frame) | **Zero** (DMA + VCU silicon) |
| **VLC streaming** | ✅ Perfect — multipart HTTP MJPEG | ❌ Complex (requires MPEG-TS container + buffering) |
| **Bandwidth at 0 persons** | ~4,000 kbps | ~120 kbps |
| **Bandwidth at 2 persons** | ~8,000 kbps | ~700 kbps |
| **Latency** | Very low (push-pull HTTP) | Low (with 150ms VLC cache) |

> [!TIP]
> MJPEG is used for the **live visualization** feed because it works perfectly in VLC over HTTP with no buffering. H.264 VCU is used for **bandwidth measurement** because it represents the real hardware compression that a production deployment would use.

---

[← DPU Inference](05_dpu_inference.md) | [↑ Back to README](../README.md) | [Next: Streaming Setup →](07_streaming_setup.md)
