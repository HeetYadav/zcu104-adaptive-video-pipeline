[↑ Tools](../README.md) | [Back to Repo Root](../../README.md) | [Benchmark Results Doc →](../../docs/08_benchmark_results.md)

---

# `benchmark/` — Benchmark Tools

Contains two complementary measurement scripts:

| Script | Purpose |
|--------|---------|
| `realBenchmark.py` | **Automated end-to-end** — starts each pipeline, collects telemetry, produces a final comparison report |
| `benchmark.py` | **Deep hardware benchmark** — measures DPU vs CPU inference speed, VCU encoding throughput, and per-zone ROI bandwidth contribution |

---

## `realBenchmark.py` — Automated Pipeline Comparison

### What It Does

1. Starts `pipeline_hw_1.py` (MJPEG baseline) as a subprocess
2. Waits 4 seconds for the DPU model to load
3. Simulates a VLC client connecting (`urllib.request.urlopen(http://127.0.0.1:5000/stream)`)
4. Collects `[Telemetry]` lines for 15 seconds
5. Kills the pipeline cleanly
6. Repeats for `pipeline_hw.py` (VCU H.264 pipeline)
7. Prints a final comparison report

### How to Run

```bash
# On the ZCU104 board — both pipelines must NOT be running
pkill python3
python3 tools/benchmark/realBenchmark.py
```

### Expected Output

```
=======================================================
  AUTOMATED REAL-WORLD PIPELINE EVALUATION SCRIPT
=======================================================
...

=======================================================
                 FINAL PIPELINE REPORT
=======================================================

[1] Hardware DPU + MJPEG Pipeline (pipeline_hw_1.py)
    Average Bandwidth:   6210.4 kbps
    Average Framerate:      7.8 FPS

[2] Hardware H.264 VCU Pipeline (pipeline_hw.py)
    Average Bandwidth:    258.3 kbps
    Average Framerate:      9.2 FPS

[3] Comparison
    Bandwidth Savings: The True Hardware Pipeline uses
                       24.0x LESS bandwidth than MJPEG!
=======================================================
```

### How It Collects Telemetry

`realBenchmark.py` reads `[Telemetry]` lines from each pipeline's stdout using regex:

```python
# For all pipelines:
pattern = r"BW:\s*([\d.]+)\s*kbps"
```

Readings below 10 kbps (pipeline initialization artifacts) are discarded. The final average is taken over all valid readings within the 15-second window.

### Client Simulation: Why It's Needed

The MJPEG HTTP server only starts pushing frames when **a client is connected**. Without a client, the compositor thread would have nothing to serve, and `_out_frame` would never be populated — resulting in 0 FPS in the telemetry.

`simulate_client()` opens a persistent HTTP connection to `http://127.0.0.1:5000/stream` from within `realBenchmark.py`, which triggers the server to start streaming.

---

## `benchmark.py` — Deep Hardware Benchmark

### What It Measures

**Claim 1 — DPU vs CPU inference:**
Runs YOLOv4 inference on the DPU and on the CPU (via OpenCV DNN) alternately, measures wall-clock time per inference over 50 runs each.

**Claim 2 — VCU vs CPU encoding:**
Pushes 100 frames through `omxh264enc` (VCU) and through `cv2.imencode` (CPU JPEG), measures frames per second.

**Claim 3 — ROI bandwidth reduction:**
Generates a synthetic test frame with a person in the center, applies the full zone masking pipeline, measures JPEG byte count with and without masking.

### How to Run

```bash
# IMPORTANT: Kill all pipelines first — DPU can only be held by one process
pkill -f pipeline_hw
python3 tools/benchmark/benchmark.py
```

### Example Output

```
============================================================
  CLAIM 1 — DPU vs CPU Inference
============================================================
  DPU (hardware):  22.4 ms/frame  →  44.6 FPS
  CPU (OpenCV DNN): 187.3 ms/frame →   5.3 FPS
  Speedup:  8.4×  faster on DPU

============================================================
  CLAIM 2 — VCU vs CPU Encoding  
============================================================
  VCU omxh264enc:     29.8 FPS (hardware)
  CPU MJPEG (JPEG):   14.2 FPS
  VCU advantage:  2.1×  faster encoding

============================================================
  CLAIM 3 — ROI Bandwidth Reduction
============================================================
  Full frame JPEG:     156.3 KB  (6123 kbps @ 5 FPS)
  ROI-masked JPEG:      14.1 KB  (  55 kbps @ 5 FPS)  
  Reduction:  91.0%
============================================================
```

### Key Design: DPU Lock Check

`benchmark.py` checks `/tmp/vart_device_0` before starting to detect if another process holds the DPU:

```python
def _check_dpu_free():
    if os.path.exists("/tmp/vart_device_0"):
        # another process holds the DPU
        print("[ERROR] DPU is held by another process.")
        print("  → Run: pkill python3")
        sys.exit(1)
```

This prevents the 60-second silent hang that occurs when `vart.Runner.create_runner()` waits for the lock.

## See Also

- [Benchmark Results Documentation](../../docs/08_benchmark_results.md)
- [Streaming Setup](../../docs/07_streaming_setup.md)
