[↑ Tools](../README.md) | [Back to Repo Root](../../README.md) | [Benchmark Results Doc →](../../docs/08_benchmark_results.md)

---

# `benchmark/`: Benchmark Tools

Contains two complementary measurement scripts:

| Script | Purpose |
|--------|---------|
| `realBenchmark.py` | **Automated end-to-end**: starts each pipeline, collects telemetry, produces a final comparison report |

---

## `realBenchmark.py`: Automated Pipeline Comparison

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
# On the ZCU104 board: both pipelines must NOT be running
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
    Average Bandwidth:   8821.1 kbps
    Average Framerate:      8.6 FPS

[2] Hardware DPU + VCU H.264 Pipeline (pipeline_hw.py)
    Average Bandwidth:    841.4 kbps
    Average Framerate:      9.5 FPS

[3] Comparison
    Bandwidth Savings (DPU+MJPEG vs DPU+VCU):
                       10.5x LESS bandwidth than MJPEG!
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

The MJPEG HTTP server only starts pushing frames when **a client is connected**. Without a client, the compositor thread would have nothing to serve, and `_out_frame` would never be populated: resulting in 0 FPS in the telemetry.

`simulate_client()` opens a persistent HTTP connection to `http://127.0.0.1:5000/stream` from within `realBenchmark.py`, which triggers the server to start streaming.

## See Also

- [Benchmark Results Documentation](../../docs/08_benchmark_results.md)- [Streaming Setup](../../docs/07_streaming_setup.md)