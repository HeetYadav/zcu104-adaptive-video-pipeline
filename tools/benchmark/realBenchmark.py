import time
import subprocess
import re
import os
import signal
import sys
import threading
import urllib.request

def drop_caches():
    """
    Free the kernel page cache, dentries, and inodes before each pipeline run.
    This is CRITICAL on the ZCU104 because only ~260 MB of non-CMA RAM is
    available for processes. Without this, the DPU model loader (which needs
    ~195 MB) competes with the page cache and OOM-kills itself.
    Requires root (which is the default on the ZCU104 dev board).
    """
    try:
        os.system("sync")
        with open("/proc/sys/vm/drop_caches", "w") as f:
            f.write("3\n")
        print("[Automated Tester] Page cache dropped — freeing non-CMA RAM for DPU.")
    except Exception as e:
        print(f"[Automated Tester] WARNING: Could not drop caches: {e}")
        print("[Automated Tester]   → Try running as root or manually: echo 3 > /proc/sys/vm/drop_caches")

def benchmark_pipeline(script_name, duration=15):
    print(f"\n" + "="*60)
    print(f"  Starting {script_name} ...")
    print("="*60 + "\n")

    # Drop kernel page/slab caches BEFORE starting the pipeline.
    # The ZCU104 only has ~22-50 MB of free non-CMA RAM once the GUI is running.
    # Dropping caches frees 50-100 MB and prevents the DPU model load from OOM-crashing.
    drop_caches()
    time.sleep(1)  # Give the kernel a moment to reclaim pages

    # 1. Start the pipeline
    proc = subprocess.Popen(
        ["python3", "-u", script_name],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        preexec_fn=os.setsid  # Create process group so we can cleanly kill all its threads
    )

    # 2. Wait 8 seconds for the DPU model to compile and load into the DPU fabric.
    #    The full YOLOv4 leaky SPP model takes 6-8 seconds on the ZCU104 DPU.
    print("[Automated Tester] Waiting for pipeline to initialize (8s for DPU model load)...")
    time.sleep(8)

    def simulate_client():
        try:
            resp = urllib.request.urlopen("http://127.0.0.1:5000/stream", timeout=5)
            # Read in chunks and discard to prevent Out of Memory (OOM).
            # resp.read() without a size arg buffers the infinite MJPEG stream into RAM!
            while True:
                chunk = resp.read(65536)
                if not chunk:
                    break
        except Exception:
            pass

    # 3. Simulate a network client pulling the video stream.
    #    Without a client connected, the HTTP server doesn't push frames, and the
    #    Compositor thread may throttle — producing no telemetry.
    print("[Automated Tester] Simulating VLC client connection locally...")
    client_thread = threading.Thread(target=simulate_client, daemon=True)
    client_thread.start()

    kbps_list = []
    fps_list = []
    start_t = time.time()
    oom_detected = False

    print(f"[Automated Tester] Collecting telemetry for {duration} seconds...\n")

    # 4. Parse the live output with a per-line deadline to avoid blocking
    #    if the pipeline crashes silently.
    while time.time() - start_t < duration:
        line = proc.stdout.readline()
        if not line:
            # Pipe closed — pipeline crashed or exited
            print("\n[Automated Tester] WARNING: Pipeline stopped producing output early.")
            break

        stripped = line.strip()
        print("    " + stripped)  # Print EVERYTHING so we can see errors!

        # Detect OOM kill in dmesg output that gets mixed into stdout
        if "oom-kill" in stripped or "Out of memory" in stripped or "oom_reaper" in stripped:
            oom_detected = True
            print("\n[Automated Tester] *** OOM KILL DETECTED — not enough free non-CMA RAM! ***")
            print("[Automated Tester]   → Run: echo 3 > /proc/sys/vm/drop_caches  and retry.")

        if "[Telemetry]" in line:
            # Extract bandwidth and framerate using regex
            # Matches both: "BW: 1500.0 kbps" and "TRUE HW BW: 1500.0 kbps"
            bw_match = re.search(r'BW:\s*([0-9.]+)\s*kbps', line)
            fps_match = re.search(r'\(\s*([0-9.]+)\s*FPS\)', line)

            if bw_match and fps_match:
                bw = float(bw_match.group(1))
                fps = float(fps_match.group(1))
                if bw > 10.0:  # Ignore initial 0 kbps readings during startup
                    kbps_list.append(bw)
                    fps_list.append(fps)

    # 5. Stop the pipeline cleanly
    print(f"\n[Automated Tester] Stopping {script_name}...")
    try:
        # Send SIGINT (Ctrl+C) so Python can catch KeyboardInterrupt and clean up
        os.killpg(os.getpgid(proc.pid), signal.SIGINT)
        proc.wait(timeout=4)
    except (subprocess.TimeoutExpired, ProcessLookupError):
        print("[Automated Tester] Pipeline didn't stop, forcing SIGKILL...")
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()

    # Give the DPU driver 5 seconds to release CMA memory and unlock /tmp/vart_device_0
    # before the next pipeline starts. Insufficient cooldown causes the next load to OOM.
    print("[Automated Tester] Cooling down (5s) — letting DPU release CMA memory...")
    time.sleep(5)

    avg_kbps = sum(kbps_list) / len(kbps_list) if kbps_list else 0.0
    avg_fps = sum(fps_list) / len(fps_list) if fps_list else 0.0

    return avg_kbps, avg_fps, oom_detected

if __name__ == "__main__":
    print("\n=======================================================")
    print("  AUTOMATED REAL-WORLD PIPELINE EVALUATION SCRIPT  ")
    print("=======================================================")
    print("This script will run each pipeline automatically,")
    print("simulate a video client connecting to it, and read")
    print("the resulting telemetry output to generate a report.")
    print()
    print("NOTE: Ensure no other python3 pipeline is running.")
    print("      Run: pkill python3   if in doubt.\n")

    # Run MJPEG (CPU video) Pipeline
    kbps_1, fps_1, oom_1 = benchmark_pipeline("pipeline_hw_1.py", duration=15)

    # Run H.264 (VCU video) Pipeline
    kbps_hw, fps_hw, oom_hw = benchmark_pipeline("pipeline_hw.py", duration=15)

    print("\n\n=======================================================")
    print("                 FINAL PIPELINE REPORT                 ")
    print("=======================================================")

    print("\n[1] Software MJPEG Pipeline (pipeline_hw_1.py)")
    if oom_1:
        print("    *** FAILED — OOM killed during DPU model load ***")
    else:
        print(f"    Average Bandwidth: {kbps_1:8.1f} kbps")
        print(f"    Average Framerate: {fps_1:8.1f} FPS")

    print("\n[2] Hardware H.264 VCU Pipeline (pipeline_hw.py)")
    if oom_hw:
        print("    *** FAILED — OOM killed during DPU model load ***")
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
        print("    Run:  echo 3 > /proc/sys/vm/drop_caches")
        print("    Then: pkill python3 && python3 realBenchmark.py")

    print("=======================================================\n")