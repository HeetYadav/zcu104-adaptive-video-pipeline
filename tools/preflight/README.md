[↑ Tools](../README.md) | [Back to Repo Root](../../README.md)

---

# `preflight.py` — Hardware Preflight Checker

> [!TIP]
> **Always run this before starting any pipeline.** It verifies every dependency in under 10 seconds and exits with a clear PASS/FAIL for each check.

Checks 6 categories of requirements in sequence. If all pass, you're ready to run `pipeline_hw.py`. If any fail, it prints the exact fix command.

## How to Run

```bash
# From repo root on the ZCU104 board
python3 tools/preflight/preflight.py
```

## What It Checks

| # | Category | What is verified |
|---|----------|-----------------|
| 1 | **Python packages** | `vart`, `xir`, `opencv-python` (with GStreamer backend), `numpy` |
| 2 | **Local modules** | `tracker.py`, `adaptive_roi.py`, `zone_mask.py` exist in expected locations |
| 3 | **DPU model files** | `yolov4_leaky_spp_m/` directory, `.xmodel` file, `meta.json`. Attempts to create a DPU runner as live hardware test |
| 4 | **GStreamer plugins** | `appsrc`, `videoconvert`, `rtph264pay`, `udpsink`, `omxh264enc` (VCU) |
| 5 | **Network** | Pings phone at `PHONE_HOST`, pings laptop at `LAPTOP_IP` |
| 6 | **DPU hardware driver** | Runs `xdputil query` to verify the DPU kernel driver is alive |

## Output Format

```
============================================================
  ZCU104 ROI Pipeline — Preflight Check
============================================================

[1] Python packages
[PASS] vart (Vitis AI Runtime)
[PASS] xir (Xilinx IR)
[PASS] opencv-python (4.5.x)
[PASS] OpenCV built with GStreamer backend
[PASS] numpy (1.x.x)

[2] Local pipeline modules
[PASS] tracker.py exists
[FAIL] adaptive_roi.py exists
       FIX: Copy adaptive_roi.py from modules/adaptive_roi/ to this directory.

...

[6] DPU hardware driver
[PASS] xdputil query (DPU driver alive)

============================================================
  1 check(s) failed. Fix the issues above first.
============================================================
```

- `[PASS]` — check succeeded (green)
- `[FAIL]` — check failed, with a `FIX:` line showing exactly what to do (red)
- `[WARN]` — informational warning, not a blocking failure (yellow)

## Interpreting Key Checks

### `omxh264enc (VCU) found` vs not found

- **Found:** You have the full VCU bitstream loaded. `pipeline_hw.py` will work with hardware H.264 encoding.
- **Not found:** You are on a non-VCU bitstream. `pipeline_hw.py` will fail at the VCU step. Use `pipeline_hw_1.py` (MJPEG only) instead.

### DPU runner creation

If the DPU runner fails to create even though the `.xmodel` exists, it usually means:
- Wrong architecture: the `.xmodel` was compiled for a different DPU (e.g., ZCU102 instead of ZCU104)
- DPU lock held: a previous Python process is still holding the DPU resource. Run `pkill python3`.

## Configuration

Edit `PHONE_HOST` and `LAPTOP_IP` at the top of `preflight.py` to match your network setup:

```python
PHONE_HOST = "192.168.2.141"
LAPTOP_IP  = "192.168.137.197"
```

## See Also

- [Hardware Setup Guide](../../docs/02_hardware_setup.md)
- [Troubleshooting](../../docs/09_troubleshooting.md)
