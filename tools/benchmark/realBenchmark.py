import time
import subprocess
import re
import os
import signal
import sys
import threading
import urllib.request

def benchmark_pipeline(script_name, duration=15):
    print(f"\n" + "="*60)
    print(f"  Starting {script_name} ...")
    print("="*60 + "\n")
    
    # 1. Start the pipeline
    proc = subprocess.Popen(
        ["python3", "-u", script_name],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        preexec_fn=os.setsid # Create process group so we can cleanly kill all its threads
    )
    
    # 2. Wait 4 seconds for YOLO model to load into DPU and HTTP Server to start
    print("[Automated Tester] Waiting for pipeline to initialize...")
    time.sleep(4)
    
    def simulate_client():
        try:
            resp = urllib.request.urlopen("http://127.0.0.1:5000/stream", timeout=5)
            resp.read() # Read infinitely until server is killed
        except Exception:
            pass

    # 3. Simulate a network client! 
    # Without a client actively pulling the video, the Hardware VCU pipeline blocks
    # to save memory, and no bytes are pushed through the network interface.
    print("[Automated Tester] Simulating VLC client connection locally...")
    client_thread = threading.Thread(target=simulate_client, daemon=True)
    client_thread.start()
    
    kbps_list = []
    fps_list = []
    start_t = time.time()
    
    print(f"[Automated Tester] Collecting telemetry for {duration} seconds...\n")
    
    # 4. Parse the live output
    while time.time() - start_t < duration:
        line = proc.stdout.readline()
        if not line:
            break
            
        print("    " + line.strip()) # Print EVERYTHING so we can see errors!
            
        if "[Telemetry]" in line:
            # Extract bandwidth and framerate using regex
            # Matches both: "BW: 1500.0 kbps" and "TRUE HW BW: 1500.0 kbps"
            bw_match = re.search(r'BW:\s*([0-9.]+)\s*kbps', line)
            fps_match = re.search(r'\(\s*([0-9.]+)\s*FPS\)', line)
            
            if bw_match and fps_match:
                bw = float(bw_match.group(1))
                fps = float(fps_match.group(1))
                if bw > 10.0:  # Ignore initial 0 kbps readings
                    kbps_list.append(bw)
                    fps_list.append(fps)
                    
    # 5. Stop the pipeline
    print(f"\n[Automated Tester] Stopping {script_name}...")
    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    
    avg_kbps = sum(kbps_list) / len(kbps_list) if kbps_list else 0.0
    avg_fps = sum(fps_list) / len(fps_list) if fps_list else 0.0
    
    return avg_kbps, avg_fps

if __name__ == "__main__":
    print("\n=======================================================")
    print("  AUTOMATED REAL-WORLD PIPELINE EVALUATION SCRIPT  ")
    print("=======================================================")
    print("This script will run each pipeline automatically,")
    print("simulate a video client connecting to it, and read")
    print("the resulting telemetry output to generate a report.")
    
    # Run MJPEG (CPU video) Pipeline
    kbps_1, fps_1 = benchmark_pipeline("pipeline_hw_1.py", duration=15)
    
    # Run H.264 (VCU video) Pipeline
    kbps_hw, fps_hw = benchmark_pipeline("pipeline_hw.py", duration=15)
    
    print("\n\n=======================================================")
    print("                 FINAL PIPELINE REPORT                 ")
    print("=======================================================")
    
    print("\n[1] Software MJPEG Pipeline (pipeline_hw_1.py)")
    print(f"    Average Bandwidth: {kbps_1:8.1f} kbps")
    print(f"    Average Framerate: {fps_1:8.1f} FPS")
    
    print("\n[2] Hardware H.264 VCU Pipeline (pipeline_hw.py)")
    print(f"    Average Bandwidth: {kbps_hw:8.1f} kbps")
    print(f"    Average Framerate: {fps_hw:8.1f} FPS")
    
    print("\n[3] Comparison")
    if kbps_1 > 0 and kbps_hw > 0:
        ratio = kbps_1 / kbps_hw
        print(f"    Bandwidth Savings: The True Hardware Pipeline uses")
        print(f"                       {ratio:.1f}x LESS bandwidth than MJPEG!")
    
    print("=======================================================\n")