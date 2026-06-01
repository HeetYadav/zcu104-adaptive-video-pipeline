[← VCU Encoding](06_vcu_encoding.md) | [↑ Back to README](../README.md) | [Next: Benchmark Results →](08_benchmark_results.md)

---

# 07 — Streaming Setup

## Table of Contents
- [Starting the Pipeline](#starting-the-pipeline)
- [Opening in VLC](#opening-in-vlc)
- [Opening in a Web Browser](#opening-in-a-web-browser)
- [VLC Settings for Low Latency](#vlc-settings-for-low-latency)
- [URL Reference](#url-reference)
- [What You Should See](#what-you-should-see)
- [Stopping the Pipeline](#stopping-the-pipeline)

---

## Starting the Pipeline

### Step 1 — Verify everything is ready

```bash
# On the board
python3 preflight.py
```

All checks must pass before proceeding.

### Step 2 — Start the pipeline

```bash
# Phase 3 — Full Hardware (DPU + VCU): recommended
python3 pipeline_hw.py

# Phase 3 Baseline — DPU + MJPEG (no VCU):
python3 pipeline_hw_1.py
```

**Expected startup output:**

```
[DPU] Loading model from yolov4_leaky_spp_m ...
[OK]  DPU runner created.
[Grabber] Connecting to http://192.168.2.141:8080/shot.jpg
[Detector] DPU ready.  Input: 1×416×416×3 INT8
[Compositor] VCU hardware encoder ready (Telemetry Mode).
[Compositor] Starting MJPEG stream for visualization on port 5000...
============================================================
  ROI Pipeline  [DPU + VCU Hardware — Full Acceleration]
  Input  ← http://192.168.2.141:8080/shot.jpg
  Output → http://<board-ip>:5000/stream
============================================================
```

The pipeline is ready when you see the `============` footer.

> [!NOTE]
> The DRM log lines like `[drm] Pid 5331 opened device` are **normal** — they come from the Linux DRM subsystem as the DPU firmware loads the XCLBIN bitstream. They print to console but do not affect operation.

---

## Opening in VLC

> [!IMPORTANT]
> **📸 Add screenshot here:** `assets/photos/vlc_stream_output.png`
> Screenshot of VLC playing the stream with detection boxes visible — green Zone 1, amber Zone 2 dashed ring, black background (Zone 3).

1. Open **VLC Media Player** on your laptop
2. Go to **Media → Open Network Stream** (or press `Ctrl+N`)
3. Enter the URL:
   ```
   http://192.168.137.176:5000/stream
   ```
   *(Replace `192.168.137.176` with your board's actual IP address)*
4. Click **Play**

> [!TIP]
> Set VLC's network cache to **150 ms** for the smoothest playback — see [VLC Settings for Low Latency](#vlc-settings-for-low-latency) below.

---

## Opening in a Web Browser

The MJPEG stream is also viewable directly in any modern browser (Chrome, Firefox, Edge):

```
http://<board-ip>:5000/stream
```

The browser renders the multipart MJPEG stream natively. No plugins required.

> [!NOTE]
> Safari on iOS does **not** support multipart MJPEG in the browser. Use VLC on iOS instead.

---

## VLC Settings for Low Latency

By default, VLC buffers 1000 ms of network video before playing. For a live stream, this means 1 second of delay and periodic "catch-up" playback. Reduce it:

**VLC → Tools → Preferences → Input / Codecs**

| Setting | Default | Recommended |
|---------|---------|-------------|
| Network caching (ms) | 1000 | **150** |

> [!WARNING]
> Setting the cache below 100 ms may cause VLC to drop frames on slower Wi-Fi connections. 150 ms is a good balance between latency and stability.

After changing the setting, you must **restart VLC** for it to take effect. The setting is saved permanently.

---

## URL Reference

| URL | Description |
|-----|-------------|
| `http://<board-ip>:5000/stream` | **Main stream** — MJPEG with zone overlay |
| `http://<board-ip>:5000/` | Alias for `/stream` |
| `http://<board-ip>:8080/shot.jpg` | Phone camera snapshot (not the pipeline output) |

---

## What You Should See

> [!IMPORTANT]
> **📸 Add screenshot here:** `assets/photos/vlc_detection_zones.png`
> Close-up view showing the three zones clearly labelled: "Z1" in the green box around the person, "Z2" in the amber dashed ring, and the rest of the frame in pure black.

When working correctly:

- **Background** — pure black (Zone 3)
- **Around each detected person** — a solid **green rectangle** labeled `Z1` (Zone 1 — full resolution)
- **Surrounding the green rectangle** — a dashed **amber rectangle** labeled `Z2` (Zone 2 — half resolution)
- **Terminal** — `[Detector]` lines showing detected person count, and `[Telemetry]` lines every 30 frames showing bandwidth and FPS

Example telemetry output:
```
[Detector] 2 person(s) detected
[Detector] 2 person(s) detected
[Telemetry] frame=    30 | targets=2 (VCU HW) | TRUE HW BW:  258.3 kbps (9.2 FPS)
[Detector] 2 person(s) detected
[Telemetry] frame=    60 | targets=2 (VCU HW) | TRUE HW BW:  241.7 kbps (9.5 FPS)
```

---

## Stopping the Pipeline

Press **Ctrl+C** in the terminal running the pipeline. It will print `Pipeline stopped.` and exit cleanly.

If the pipeline crashes or hangs (common after a `SIGKILL` from `realBenchmark.py`), kill any remaining Python processes:

```bash
pkill -f pipeline_hw.py
# or
pkill python3
```

> [!WARNING]
> If `pipeline_hw.py` or `pipeline_hw_1.py` is still running when you try to start it again, you'll get `OSError: [Errno 98] Address already in use` because port 5000 is still occupied. Both pipeline scripts now set `allow_reuse_address = True`, which minimizes the TIME_WAIT window, but you may still need to wait 2–3 seconds between runs.

---

[← VCU Encoding](06_vcu_encoding.md) | [↑ Back to README](../README.md) | [Next: Benchmark Results →](08_benchmark_results.md)
