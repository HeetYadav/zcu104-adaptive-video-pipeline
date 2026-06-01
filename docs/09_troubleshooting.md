[← Benchmark Results](08_benchmark_results.md) | [↑ Back to README](../README.md)

---

# 09 — Troubleshooting

Every hard-won fix from development. Each issue follows a consistent structure: **Symptom → Root Cause → Fix → Prevention**.

## Table of Contents

- [xir::DataType::UNKNOWN Crash](#xirdatatypeunknown-crash)
- [OSError: Address already in use](#oserror-address-already-in-use)
- [DPU Resource Lock — "Waiting for process to release"](#dpu-resource-lock)
- [DRM Log Flooding the Terminal](#drm-log-flooding)
- [VLC Shows Black Screen](#vlc-shows-black-screen)
- [Pipeline Output is Choppy (Pause → Fast-Forward → Pause)](#choppy-output)
- [Benchmark Reports 0.0 kbps / 0.0 FPS](#benchmark-reports-00)
- [SCP Transfer Fails — Host Key Rejected](#scp-transfer-fails)
- [Phone Camera Feed Not Connecting](#phone-camera-feed-not-connecting)
- [VCU GStreamer Pipeline Failed to Open](#vcu-gstreamer-pipeline-failed)

---

<details>
<summary><strong>xir::DataType::UNKNOWN Crash</strong></summary>

**Symptom**

The pipeline crashes immediately after loading the model, printing:
```
F0529 12:06:22.871002  5348 xrt_device_handle_imp.cpp:250]
  xir::DataType::UNKNOWN abort
Aborted (core dumped)
```

**Root Cause**

The `.xmodel` declares its input tensor type as `INT8` (signed, -128 to +127). When the pipeline passes a `uint8` (unsigned, 0–255) numpy array, the Xilinx IR library (`xir`) cannot match the Python dtype to the expected tensor type, reports `UNKNOWN`, and calls `abort()`.

**Fix**

Change the preprocessing to subtract 128 and cast to `int8`:

```diff
- img_data = rgb.astype(np.uint8)
- input_tensor_data[0] = img_data

+ img_int8 = (rgb.astype(np.int16) - 128).astype(np.int8)
+ input_tensor_data[0] = img_int8
```

Cast to `int16` *before* subtracting 128 to prevent uint8 underflow (0 - 128 would overflow if you subtract in uint8 space).

**Prevention**

Always check the input tensor dtype before submitting:
```python
in_t = runner.get_input_tensors()
print(f"Expected dtype: {in_t[0].dtype}")   # Should be xir.DataType.INT8
```

</details>

---

<details>
<summary><strong>OSError: Address already in use</strong></summary>

**Symptom**

```
OSError: [Errno 98] Address already in use
File "pipeline_hw.py", line 468, in <module>
    httpd = socketserver.ThreadingTCPServer(('0.0.0.0', STREAM_PORT), MjpegServer)
```

**Root Cause**

After a pipeline is killed (e.g., by `realBenchmark.py`'s `SIGKILL`), Linux keeps the TCP port in `TIME_WAIT` state for up to 60 seconds to handle any delayed packets still in transit. When the next pipeline starts immediately and tries to bind to the same port (5000), Linux refuses with `EADDRINUSE`.

**Fix**

Set `allow_reuse_address = True` before creating the server. This is already implemented in both `pipeline_hw.py` and `pipeline_hw_1.py`:

```python
socketserver.TCPServer.allow_reuse_address = True
httpd = socketserver.ThreadingTCPServer(('0.0.0.0', STREAM_PORT), MjpegServer)
```

Or for `HTTPServer`:
```python
HTTPServer.allow_reuse_address = True
```

If you still get this error, wait 5 seconds and try again — the kernel's `TIME_WAIT` window will have expired.

**Prevention**

Always use `allow_reuse_address = True` for any server that might be restarted frequently (development or benchmark scenarios).

</details>

---

<details>
<summary><strong>DPU Resource Lock — "Waiting for process to release"</strong></summary>

**Symptom**

```
I0529 12:06:22.871002  5348 xrt_device_handle_imp.cpp:250]
  waiting for process [5331] to release the resource:/tmp/vart_device_0
```

The pipeline hangs at this line indefinitely.

**Root Cause**

Only one process can hold the DPU resource at a time. The previous pipeline (PID 5331) was killed but the DPU resource lock file (`/tmp/vart_device_0`) was not cleaned up before the new process (PID 5348) started.

This happens when:
- The previous `python3 pipeline_hw.py` was killed with `SIGKILL` (instant kill, no cleanup)
- The previous run crashed without releasing the DPU
- `realBenchmark.py` killed the first pipeline and the second started before the OS fully released the resource

**Fix**

Kill all remaining Python processes and delete the lock file:

```bash
pkill -9 python3
rm -f /tmp/vart_device_0
```

Wait 2 seconds, then start the pipeline again.

**Prevention**

`realBenchmark.py` already waits for the process group to terminate before starting the next pipeline. The issue only occurs if you interrupt `realBenchmark.py` itself mid-run with Ctrl+C.

</details>

---

<details>
<summary><strong>DRM Log Flooding the Terminal</strong></summary>

**Symptom**

The terminal is flooded with lines like:
```
[ 9147.625721] [drm] Pid 5331 opened device
[ 9147.629663] [drm] Pid 5331 closed device
[12813.000007] [drm] zocl_xclbin_read_axlf The XCLBIN already loaded
[12813.009622] [drm] bitstream 9f1d8d6f-... locked, ref=1
```

**Root Cause**

These are **kernel log messages** from the `zocl` DRM driver — the Linux kernel module that manages access to the FPGA fabric (XCLBIN bitstream loading, DPU resource arbitration). They print to the kernel ring buffer and appear in the console because the ZCU104 PetaLinux image has `printk` console logging enabled.

**These messages are harmless.** They confirm the DPU bitstream is loading and the resource lock is being managed correctly.

**Fix**

To suppress them during a session:
```bash
# Suppress kernel messages below WARNING level from appearing on console
echo 4 > /proc/sys/kernel/printk
```

To restore:
```bash
echo 7 > /proc/sys/kernel/printk
```

**Prevention**

For long benchmark runs, redirect kernel messages before starting:
```bash
echo 4 > /proc/sys/kernel/printk && python3 realBenchmark.py
```

</details>

---

<details>
<summary><strong>VLC Shows Black Screen</strong></summary>

**Symptom**

VLC opens the URL and shows a black screen with no video, or shows "Your input can't be opened."

**Root Cause (most common: wrong URL path)**

The MJPEG stream is served at `/stream`, not `/`. VLC will get an empty response at the root path.

**Fix**

Use the exact URL:
```
http://<board-ip>:5000/stream
```

Not:
```
http://<board-ip>:5000          ← might work, might not (depends on HTTP handler)
http://<board-ip>:5000/video    ← wrong path
```

**Root Cause (second most common: pipeline not running)**

Check the board terminal — the pipeline may have crashed silently.

**Fix**

```bash
# On the board
ps aux | grep python3         # Check if the pipeline is running
python3 pipeline_hw.py        # Restart if not
```

**Root Cause (VLC cache too high)**

VLC is waiting to fill its buffer before playing.

**Fix**

Reduce VLC cache: **Tools → Preferences → Input/Codecs → Network caching → 150 ms**

**Prevention**

Always confirm the pipeline printed the `Output → http://<board-ip>:5000/stream` banner before opening VLC.

</details>

---

<details>
<summary><strong>Pipeline Output is Choppy (Pause → Fast-Forward → Pause)</strong></summary>

**Symptom**

The VLC stream plays for 2–3 seconds, freezes, then plays 5 seconds of footage in fast-forward, then freezes again in a repeating loop.

**Root Cause**

This was caused by the MPEG-TS container format during earlier development attempts. MPEG-TS embeds PTS (Presentation Timestamps) in the container. VLC uses those timestamps to determine playback pacing. When the encoder produces bursty output (variable-latency DPU + compositor timing), the PTS gaps cause VLC to buffer aggressively, resulting in the freeze → fast-forward cycle.

**Fix**

The current pipeline does **not** use MPEG-TS. It serves raw MJPEG over multipart HTTP, which has no embedded timestamps. VLC renders MJPEG frames as they arrive — no buffering logic.

If you've accidentally switched back to MPEG-TS (e.g., re-added `mpegtsmux` to `GST_OUT`), revert to the current `fakesink` approach.

**Prevention**

Do not add `mpegtsmux`, `mp4mux`, or any container muxer to `GST_OUT`. The VCU pipeline uses `fakesink` — the H.264 output is for telemetry calculation, not for streaming.

</details>

---

<details>
<summary><strong>Benchmark Reports 0.0 kbps / 0.0 FPS</strong></summary>

**Symptom**

`realBenchmark.py` completes but reports:
```
Average Bandwidth:      0.0 kbps
Average Framerate:      0.0 FPS
```

**Root Cause 1: Pipeline crashed before telemetry**

The pipeline may have thrown an exception in the first 4 seconds (during initialization), before any `[Telemetry]` lines were emitted.

**Fix**

Run `realBenchmark.py` and watch the indented output lines carefully — they now show ALL pipeline output, including tracebacks. Look for error messages.

**Root Cause 2: Phone camera not reachable**

If the Grabber thread can't reach the IP Webcam app, it logs connection errors silently and `_grab_frame` stays `None`. The Compositor has nothing to process, so no telemetry is emitted.

**Fix**

```bash
# Verify phone is reachable from the board
ping 192.168.2.141

# Verify IP Webcam is serving
curl http://192.168.2.141:8080/shot.jpg -o /tmp/test.jpg && echo OK
```

**Root Cause 3: `pipeline_hw_1.py` crashed due to missing `socketserver` import**

Older versions of `pipeline_hw_1.py` had `socketserver.TCPServer.allow_reuse_address = True` but didn't import `socketserver`. The fix is to use `HTTPServer.allow_reuse_address = True` instead, which is already fixed in the current version.

**Prevention**

Always run `python3 preflight.py` before the benchmark. Preflight verifies the phone connection, DPU model, and GStreamer plugins.

</details>

---

<details>
<summary><strong>SCP Transfer Fails — Host Key Rejected</strong></summary>

**Symptom**

```
Unable to negotiate with 192.168.137.176 port 22:
no matching host key type found. Their offer: ssh-rsa
```

**Root Cause**

Modern OpenSSH clients (version 8.8+) disable `ssh-rsa` host key verification by default because SHA-1 is deprecated. The ZCU104's PetaLinux SSH server only offers `ssh-rsa` host keys, so the connection is rejected.

**Fix**

Always use these flags with `scp`:
```cmd
scp -O -o HostKeyAlgorithms=+ssh-rsa <file> root@<board-ip>:/home/root/
```

| Flag | Purpose |
|------|---------|
| `-O` | Use legacy SCP protocol (not SFTP) |
| `-o HostKeyAlgorithms=+ssh-rsa` | Re-enable ssh-rsa host key support for this connection |

**Prevention**

Add a `~/.ssh/config` entry on your Windows laptop:
```
Host 192.168.137.176
    HostKeyAlgorithms +ssh-rsa
    PubkeyAcceptedAlgorithms +ssh-rsa
```
After this, plain `scp` without flags will work for this host.

</details>

---

<details>
<summary><strong>Phone Camera Feed Not Connecting</strong></summary>

**Symptom**

The pipeline starts but prints repeated connection errors, or the stream shows a static frame that never updates.

**Root Cause**

One of:
- IP Webcam app is not running on the phone
- Phone screen is off (some Android versions suspend background apps when screen is off)
- Phone IP address changed (DHCP reassigned)
- Phone and board are on different subnets

**Fix**

1. On your phone: open IP Webcam, ensure the server is running (you'll see the URL and FPS counter)
2. Check the phone's current IP in the IP Webcam app screen
3. Update `PHONE_HOST` in `pipeline_hw.py` if the IP changed:
   ```python
   PHONE_HOST = "192.168.x.xxx:8080"   # ← Update here
   ```
4. Verify from the board:
   ```bash
   curl http://<phone-ip>:8080/shot.jpg -o /tmp/test.jpg
   ```

**Prevention**

Assign the phone a static IP in your router's DHCP settings using its MAC address. This prevents IP changes between sessions.

</details>

---

<details>
<summary><strong>VCU GStreamer Pipeline Failed to Open</strong></summary>

**Symptom**

```
[ERROR] VCU GStreamer pipeline failed to open!
```

Or `vcu_writer.isOpened()` returns `False`.

**Root Cause**

The GStreamer pipeline string in `GST_OUT` failed to initialize. Common causes:
- `omxh264enc` plugin is not installed
- `videoconvert` plugin is missing
- The bitstream loaded on the FPGA does not include the VCU IP core

**Fix**

Run the preflight check:
```bash
python3 preflight.py
```

Check the `[4] GStreamer plugins` section. If `omxh264enc` fails:
```bash
gst-inspect-1.0 omxh264enc
```

If the plugin is missing, you may be running a bitstream that doesn't include the VCU. The VCU requires the `vcu` bitstream variant of the Vitis AI TRD image.

**Prevention**

Preflight always checks for `omxh264enc`. If it's not present, preflight will warn you before the pipeline starts.

</details>

---

[← Benchmark Results](08_benchmark_results.md) | [↑ Back to README](../README.md)
