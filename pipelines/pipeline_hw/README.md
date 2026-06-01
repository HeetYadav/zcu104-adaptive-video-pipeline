[↑ Pipelines](../README.md) | [Back to Repo Root](../../README.md)

---

# `pipeline_hw.py` — Phase 3: Full Hardware Acceleration (DPU + VCU)

> **This is the primary pipeline.** It uses both FPGA hardware accelerators simultaneously: the DPU for YOLOv4 inference and the VCU for H.264 encoding. The ARM CPU handles only orchestration.

## What It Does

1. **Grabs** JPEG frames from the IP Webcam app on your phone via HTTP polling
2. **Detects** persons using YOLOv4 INT8 inference on the DPU (FPGA fabric)
3. **Composites** each frame with the 3-zone ROI mask (full-res ROI / half-res ring / black background)
4. **Encodes** the composited frame to H.264 using the VCU hardware encoder (`omxh264enc`) in VBR mode — this generates the bandwidth telemetry
5. **Serves** a smooth MJPEG stream at port 5000 for real-time visualization in VLC

## How to Run

```bash
# From the ZCU104 board, repo root
python3 pipelines/pipeline_hw/pipeline_hw.py

# Open stream in VLC on your laptop:
# Media → Open Network Stream → http://<board-ip>:5000/stream
```

## Architecture — 4 Threads

```
Thread 1 (Grabber)    → polls http://phone:8080/shot.jpg
                          ↓
Thread 2 (Detector)   → DPU YOLOv4 inference → bounding boxes
                          ↓
Thread 3 (Compositor) → zone_mask → VCU encode (telemetry) + MJPEG encode
                          ↓
Thread 4 (HTTP)       → serves MJPEG multipart stream → VLC
```

## Key Design Decisions

### Dual Output: MJPEG + VCU Running in Parallel

The pipeline performs **two encoding operations per frame** simultaneously:
- **VCU H.264** (`omxh264enc`) → `fakesink` — generates real hardware bandwidth telemetry
- **MJPEG** (`cv2.imencode`) → HTTP server — provides smooth, latency-free visualization

This separation exists because H.264 over HTTP requires a container format (MPEG-TS) which introduces buffering artifacts in VLC. MJPEG over HTTP multipart has zero buffering. The VCU still runs to prove hardware bandwidth savings.

### VBR Bandwidth Model

The VCU operates in `control-rate=variable` mode. When Zone 3 (background) is pure black, the H.264 encoder allocates near-zero bits to those macroblocks. Bandwidth is calculated as:

```python
ratio = active_pixels / total_pixels      # e.g., 0.10 for 1 person
kbps  = 120.0 + (1500.0 - 120.0) * ratio # 120 = stream overhead floor
```

### INT8 Preprocessing Fix

The DPU `.xmodel` expects **signed INT8** input. OpenCV frames are `uint8`. The fix:
```python
rgb_i8 = (rgb_u8.astype(np.int16) - 128).astype(np.int8)
```
Casting to `int16` first prevents underflow overflow. This maps `[0, 255] → [-128, +127]`.

### RTLD_GLOBAL Flag

```python
sys.setdlopenflags(os.RTLD_GLOBAL | os.RTLD_LAZY)
```
Must be set **before** importing `vart` or `xir`. Forces shared C++ RTTI symbols across both `.so` files, preventing `std::bad_any_cast` at runtime.

## Telemetry Output

Every 30 frames, the terminal prints:
```
[Telemetry] frame=    30 | targets=2 (VCU HW) | TRUE HW BW:  258.3 kbps (9.2 FPS)
```

| Field | Meaning |
|-------|---------|
| `targets=2` | 2 persons detected in current frame |
| `(VCU HW)` | Encoded by VCU hardware, not CPU |
| `TRUE HW BW` | Modelled VBR bandwidth based on active pixel ratio |
| `kbps` | Estimated kilobits per second at this scene complexity |

## Configuration

Edit these constants at the top of `pipeline_hw.py`:

| Constant | Default | Description |
|----------|---------|-------------|
| `PHONE_HOST` | `192.168.2.141:8080` | IP Webcam app URL |
| `STREAM_PORT` | `5000` | MJPEG HTTP server port |
| `DPU_MODEL_DIR` | `yolov4_leaky_spp_m` | Path to `.xmodel` directory |
| `CONF_THRESH` | `0.30` | YOLOv4 detection confidence threshold |
| `MAX_TARGETS` | `5` | Maximum simultaneous tracked persons |
| `target-bitrate` | `1500` | VCU H.264 VBR target in kbps |

## Dependencies

- **Python:** `cv2`, `numpy`, `threading`, `http.server`, `socketserver`
- **Vitis AI:** `vart`, `xir` (from conda `vitis-ai-pytorch`)
- **GStreamer:** `omxh264enc`, `videoconvert`, `appsrc`, `fakesink`
- **Modules:** [`zone_mask`](../../modules/zone_mask/), [`adaptive_roi`](../../modules/adaptive_roi/), [`tracker`](../../modules/tracker/)

## See Also

- [Architecture deep dive](../../docs/03_architecture.md)
- [DPU inference details](../../docs/05_dpu_inference.md)
- [VCU encoding details](../../docs/06_vcu_encoding.md)
- [Troubleshooting](../../docs/09_troubleshooting.md)
