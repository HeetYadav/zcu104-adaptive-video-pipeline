import time
import subprocess
import re
import os
import signal
import sys
import threading
import urllib.request
import gc

# ─── Swap configuration ────────────────────────────────────────────────────
# The ZCU104 has 2 GB RAM but 1.67 GB is CMA-reserved for DPU/VCU hardware.
# Only ~20 MB of non-CMA RAM is free when the GUI is running.
# The DPU model loader (YOLOv4 leaky SPP) needs ~225 MB anon RAM to complete.
# Without swap the OOM killer terminates the pipeline every time.
SWAP_FILE    = "/home/root/benchmark.swap"
SWAP_SIZE_MB = 512

def setup_swap():
    """Create and enable a 512 MB swap file if no swap is currently active."""
    try:
        with open('/proc/swaps') as f:
            lines = f.readlines()
        if len(lines) > 1:
            active = lines[1].strip().split()[0]
            print(f"[Setup] Swap already active: {active}")
            return True
    except Exception:
        pass

    print(f"[Setup] No swap detected. Creating {SWAP_SIZE_MB} MB swap file at {SWAP_FILE} ...")
    print("[Setup] This takes ~30 s the first time but only runs once per boot.\n")

    # Check available disk space on the root filesystem
    try:
        stat   = os.statvfs('/')
        free_mb = (stat.f_bavail * stat.f_frsize) // (1024 * 1024)
        if free_mb < SWAP_SIZE_MB + 64:
            print(f"[Setup] ERROR: Not enough disk space ({free_mb} MB free, need {SWAP_SIZE_MB + 64} MB).")
            print(f"[Setup] Free up space on / and retry, or manually: swapon <your-swap>")
            return False
        print(f"[Setup] Disk space OK: {free_mb} MB free.")
    except Exception:
        pass

    for cmd in [
        f"dd if=/dev/zero of={SWAP_FILE} bs=1M count={SWAP_SIZE_MB} status=progress",
        f"chmod 600 {SWAP_FILE}",
        f"mkswap {SWAP_FILE}",
        f"swapon {SWAP_FILE}",
    ]:
        if os.system(cmd) != 0:
            print(f"[Setup] ERROR: Command failed: {cmd}")
            return False

    print(f"\n[Setup] Swap enabled ({SWAP_SIZE_MB} MB). OOM risk eliminated.\n")
    return True


def drop_caches():
    try:
        os.system("sync")
        with open("/proc/sys/vm/drop_caches", "w") as f:
            f.write("3\n")
        print("[Tester] Page cache dropped.")
    except Exception as e:
        print(f"[Tester] WARNING: Could not drop caches: {e}")


def benchmark_pipeline(script_name, duration=15):
    log_file = f"/tmp/{os.path.basename(script_name)}.bench.log"

    print(f"\n" + "=" * 60)
    print(f"  Starting {script_name} ...")
    print("=" * 60 + "\n")

    drop_caches()
    time.sleep(1)

    # Release as much memory as possible from this process before forking
    gc.collect()
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass

    # ── Start pipeline ─────────────────────────────────────────────
    # Write stdout + stderr to a LOG FILE, NOT a PIPE.
    # Using a pipe keeps a kernel buffer + Python io.BufferedReader
    # alive the whole time, adding unnecessary memory pressure.
    # With a file, this process can just sleep() while the pipeline runs.
    log_fd = open(log_file, 'w')
    proc   = subprocess.Popen(
        ["python3", "-u", script_name],
        stdout=log_fd,
        stderr=log_fd,
        preexec_fn=os.setsid,   # separate process group for clean kill
    )
    log_fd.close()  # close our handle — the subprocess owns the fd now

    print(f"[Tester] Pipeline started (PID {proc.pid}). Waiting 10 s for DPU model load...")
    # YOLOv4 leaky SPP takes 6–10 s to deserialize on the ZCU104 DPU
    time.sleep(10)

    # ── Client simulation ─────────────────────────────────────────
    def simulate_client():
        try:
            resp = urllib.request.urlopen("http://127.0.0.1:5000/stream", timeout=10)
            # Read and discard chunks — never buffer the whole stream
            while True:
                if not resp.read(65536):
                    break
        except Exception:
            pass

    print("[Tester] Simulating VLC client connection...")
    threading.Thread(target=simulate_client, daemon=True).start()

    # ── Collect telemetry ─────────────────────────────────────────
    print(f"[Tester] Measuring for {duration} s ...\n")
    time.sleep(duration)   # This process is nearly idle — minimal RAM pressure

    # ── Stop pipeline ─────────────────────────────────────────────
    print(f"\n[Tester] Stopping {script_name}...")
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGINT)
        proc.wait(timeout=4)
    except (subprocess.TimeoutExpired, ProcessLookupError, OSError):
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, OSError):
            pass
        try:
            proc.wait()
        except Exception:
            pass

    # Give the DPU driver 5 s to release CMA and /tmp/vart_device_0
    print("[Tester] Cooling down 5 s ...")
    time.sleep(5)

    # ── Parse log file ────────────────────────────────────────────
    kbps_list    = []
    fps_list     = []
    oom_detected = False
    try:
        with open(log_file) as f:
            for line in f:
                if any(tok in line for tok in ("oom-kill", "Out of memory", "oom_reaper")):
                    oom_detected = True
                if "[Telemetry]" in line:
                    bw_m  = re.search(r'BW:\s*([0-9.]+)\s*kbps',       line)
                    fps_m = re.search(r'\(\s*([0-9.]+)\s*FPS\)',        line)
                    if bw_m and fps_m:
                        bw  = float(bw_m.group(1))
                        fps = float(fps_m.group(1))
                        if bw > 10.0:
                            kbps_list.append(bw)
                            fps_list.append(fps)
    except Exception as e:
        print(f"[Tester] Could not read log: {e}")

    # Print last 10 lines of the log so the user can see what happened
    try:
        with open(log_file) as f:
            all_lines = f.readlines()
        n_lines = len(all_lines)
        print(f"[Tester] Pipeline log ({n_lines} lines) — last 10:")
        for l in all_lines[-10:]:
            print("    " + l.rstrip())
    except Exception:
        pass

    if oom_detected:
        print("\n[Tester] *** OOM KILL DETECTED — swap may not be active ***")

    avg_kbps = sum(kbps_list) / len(kbps_list) if kbps_list else 0.0
    avg_fps  = sum(fps_list)  / len(fps_list)  if fps_list  else 0.0
    return avg_kbps, avg_fps, oom_detected


if __name__ == "__main__":
    print("\n=======================================================")
    print("  AUTOMATED REAL-WORLD PIPELINE EVALUATION SCRIPT  ")
    print("=======================================================")
    print("This script will run each pipeline automatically,")
    print("simulate a video client connecting to it, and read")
    print("the resulting telemetry output to generate a report.")
    print()
    print("Ensure no other python3 pipeline is running.")
    print("Run:  pkill python3   if in doubt.\n")

    # ── Step 1: Ensure swap exists ────────────────────────────────
    # Critical on the ZCU104 — only ~20 MB of non-CMA RAM is free
    # at runtime, which is nowhere near enough to load the DPU model.
    swap_ok = setup_swap()
    if not swap_ok:
        print("\n[WARNING] Continuing without swap — pipelines will likely OOM-crash.\n")

    # ── Step 2: Run pipelines ─────────────────────────────────────
    kbps_1,  fps_1,  oom_1  = benchmark_pipeline("pipeline_hw_1.py", duration=15)
    kbps_hw, fps_hw, oom_hw = benchmark_pipeline("pipeline_hw.py",   duration=15)

    # ── Step 3: Report ────────────────────────────────────────────
    print("\n\n=======================================================")
    print("                 FINAL PIPELINE REPORT                 ")
    print("=======================================================")

    print("\n[1] Software MJPEG Pipeline (pipeline_hw_1.py)")
    if oom_1:
        print("    *** FAILED — OOM killed during DPU model load ***")
    elif kbps_1 == 0:
        print("    Average Bandwidth:      0.0 kbps")
        print("    Average Framerate:      0.0 FPS")
        print("    (No [Telemetry] lines seen — check phone connection)")
    else:
        print(f"    Average Bandwidth: {kbps_1:8.1f} kbps")
        print(f"    Average Framerate: {fps_1:8.1f} FPS")

    print("\n[2] Hardware H.264 VCU Pipeline (pipeline_hw.py)")
    if oom_hw:
        print("    *** FAILED — OOM killed during DPU model load ***")
    elif kbps_hw == 0:
        print("    Average Bandwidth:      0.0 kbps")
        print("    Average Framerate:      0.0 FPS")
        print("    (No [Telemetry] lines seen — check phone connection)")
    else:
        print(f"    Average Bandwidth: {kbps_hw:8.1f} kbps")
        print(f"    Average Framerate: {fps_hw:8.1f} FPS")

    print("\n[3] Comparison")
    if kbps_1 > 0 and kbps_hw > 0:
        ratio = kbps_1 / kbps_hw
        print(f"    Bandwidth Savings: The True Hardware Pipeline uses")
        print(f"                       {ratio:.1f}x LESS bandwidth than MJPEG!")
    elif oom_1 or oom_hw:
        print("    Cannot compare — one or both pipelines were OOM-killed.")
        print(f"    Swap file: {SWAP_FILE}")

    print("=======================================================\n")
