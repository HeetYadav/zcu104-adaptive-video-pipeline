import time
import subprocess
import re
import os
import signal
import sys
import threading
import urllib.request
import gc

def kill_desktop_gui():
    """
    The ZCU104 has 2 GB RAM but 1.67 GB is CMA-reserved for hardware.
    The desktop GUI (Xorg, matchbox, etc.) wastes ~100 MB of the precious
    remaining non-CMA RAM. Killing it gives the DPU enough memory to load.
    """
    print("[Setup] Freeing RAM by shutting down Desktop GUI (Xorg)...")
    os.system("killall -9 Xorg matchbox-desktop matchbox-panel matchbox-window dbus-daemon 2>/dev/null")
    time.sleep(2)

def drop_caches():
    try:
        os.system("sync")
        with open("/proc/sys/vm/drop_caches", "w") as f:
            f.write("3\n")
        print("[Tester] Page cache dropped.")
    except Exception as e:
        pass

def benchmark_pipeline(script_name, duration=15):
    log_file = f"/tmp/{os.path.basename(script_name)}.bench.log"

    print(f"\n" + "=" * 60)
    print(f"  Starting {script_name} ...")
    print("=" * 60 + "\n")

    drop_caches()
    time.sleep(1)

    gc.collect()
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass

    log_fd = open(log_file, 'w')
    proc   = subprocess.Popen(
        ["python3", "-u", script_name],
        stdout=log_fd,
        stderr=log_fd,
        preexec_fn=os.setsid,
    )
    log_fd.close()

    print(f"[Tester] Pipeline started (PID {proc.pid}). Waiting 10 s for DPU model load...")
    time.sleep(10)

    def simulate_client():
        try:
            resp = urllib.request.urlopen("http://127.0.0.1:5000/stream", timeout=10)
            while True:
                if not resp.read(65536):
                    break
        except Exception:
            pass

    print("[Tester] Simulating VLC client connection...")
    threading.Thread(target=simulate_client, daemon=True).start()

    print(f"[Tester] Measuring for {duration} s ...\n")
    time.sleep(duration)

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

    print("[Tester] Cooling down 5 s ...")
    time.sleep(5)

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
        print("\n[Tester] *** OOM KILL DETECTED — Not enough RAM ***")

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

    # Clean up the failed swap file to give you your disk space back!
    if os.path.exists("/home/root/benchmark.swap"):
        os.remove("/home/root/benchmark.swap")

    # Step 1: Kill the Desktop GUI to free up 100+ MB of RAM
    kill_desktop_gui()

    # Step 2: Run pipelines
    # 0. Pure Software Pipeline (OpenCV + ResNet)
    kbps_sw, fps_sw, oom_sw = benchmark_pipeline("pipeline.py", duration=15)
    
    # 1. DPU + MJPEG Pipeline
    kbps_1,  fps_1,  oom_1  = benchmark_pipeline("pipeline_hw_1.py", duration=15)
    
    # 2. DPU + VCU Hardware Pipeline
    kbps_hw, fps_hw, oom_hw = benchmark_pipeline("pipeline_hw.py",   duration=15)

    # Step 3: Report
    print("\n\n=======================================================")
    print("                 FINAL PIPELINE REPORT                 ")
    print("=======================================================")

    print("\n[0] Pure Software Pipeline (pipeline_sw/pipeline.py)")
    if oom_sw:
        print("    *** FAILED — OOM killed ***")
    elif kbps_sw == 0:
        print("    Average Bandwidth:      0.0 kbps")
        print("    Average Framerate:      0.0 FPS")
        print("    (No [Telemetry] lines seen — check phone connection)")
    else:
        print(f"    Average Bandwidth: {kbps_sw:8.1f} kbps")
        print(f"    Average Framerate: {fps_sw:8.1f} FPS")

    print("\n[1] Hardware DPU + MJPEG Pipeline (pipeline_hw_1.py)")
    if oom_1:
        print("    *** FAILED — OOM killed during DPU model load ***")
    elif kbps_1 == 0:
        print("    Average Bandwidth:      0.0 kbps")
        print("    Average Framerate:      0.0 FPS")
        print("    (No [Telemetry] lines seen — check phone connection)")
    else:
        print(f"    Average Bandwidth: {kbps_1:8.1f} kbps")
        print(f"    Average Framerate: {fps_1:8.1f} FPS")

    print("\n[2] Hardware DPU + VCU H.264 Pipeline (pipeline_hw.py)")
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
        print(f"    Bandwidth Savings (DPU+MJPEG vs DPU+VCU):")
        print(f"                       {ratio:.1f}x LESS bandwidth than MJPEG!")
        
    if fps_sw > 0 and fps_hw > 0:
        fps_ratio = fps_hw / fps_sw
        print(f"    Framerate Boost (Pure SW vs DPU+VCU):")
        print(f"                       {fps_ratio:.1f}x HIGHER framerate!")
        
    if oom_sw or oom_1 or oom_hw:
        print("\n    Note: One or more pipelines were OOM-killed.")

    print("=======================================================\n")
