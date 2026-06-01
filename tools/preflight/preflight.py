#!/usr/bin/env python3
"""
preflight.py: Run this FIRST on the ZCU104 before starting pipeline_hw.py.
Checks every dependency and hardware driver so failures are diagnosed
before the pipeline starts, not mid-run.

Usage:  python3 preflight.py
"""

import subprocess
import sys
import os

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
WARN = "\033[93m[WARN]\033[0m"

errors = 0

def check(label, ok, fix=""):
    global errors
    if ok:
        print(f"{PASS} {label}")
    else:
        print(f"{FAIL} {label}")
        if fix:
            print(f"       FIX: {fix}")
        errors += 1

def warn(label, msg=""):
    print(f"{WARN} {label}")
    if msg:
        print(f"       NOTE: {msg}")


print("=" * 60)
print("  ZCU104 ROI Pipeline: Preflight Check")
print("=" * 60)

# ── Python packages ───────────────────────────────────────────────
print("\n[1] Python packages")

try:
    import vart
    check("vart (Vitis AI Runtime)", True)
except ImportError:
    check("vart (Vitis AI Runtime)", False,
          "conda activate vitis-ai-pytorch  OR  pip install vart")

try:
    import xir
    check("xir (Xilinx IR)", True)
except ImportError:
    check("xir (Xilinx IR)", False,
          "conda activate vitis-ai-pytorch  OR  pip install xir")

try:
    import cv2
    check(f"opencv-python ({cv2.__version__})", True)
    # Check GStreamer backend
    gst_ok = "GStreamer" in cv2.getBuildInformation()
    check("OpenCV built with GStreamer backend", gst_ok,
          "The pre-built cv2 on the Vitis AI image should have GStreamer. "
          "If not: pip install opencv-python-headless and rebuild.")
except ImportError:
    check("opencv-python", False, "pip install opencv-python-headless")

try:
    import numpy as np
    check(f"numpy ({np.__version__})", True)
except ImportError:
    check("numpy", False, "pip install numpy")

# ── Local modules ─────────────────────────────────────────────────
print("\n[2] Local pipeline modules")
for mod in ["tracker", "adaptive_roi", "zone_mask"]:
    path = f"{mod}.py"
    check(f"{path} exists", os.path.isfile(path),
          f"Copy {path} from the deliverables package to this directory.")

# ── DPU model files ───────────────────────────────────────────────
print("\n[3] DPU model files")
MODEL_DIR   = "yolov4_leaky_spp_m"
XMODEL_PATH = f"{MODEL_DIR}/yolov4_leaky_spp_m.xmodel"
META_PATH   = f"{MODEL_DIR}/meta.json"

check(f"Model directory {MODEL_DIR}/", os.path.isdir(MODEL_DIR),
      f"Download: wget https://www.xilinx.com/bin/public/openDownload?filename=yolov4_leaky_spp_m-zcu104-r3.0.0.tar.gz "
      f"then tar -xvf *.tar.gz")
check(f"{XMODEL_PATH}", os.path.isfile(XMODEL_PATH),
      "Extract the model zoo archive in this directory.")
check(f"{META_PATH}", os.path.isfile(META_PATH),
      "meta.json must be in the same directory as the .xmodel file.")

# Try loading the model
if os.path.isfile(XMODEL_PATH):
    try:
        import xir
        import vart
        graph     = xir.Graph.deserialize(XMODEL_PATH)
        subgraphs = [s for s in graph.get_root_subgraph().toposort_child_subgraph()
                     if s.has_attr("device") and s.get_attr("device").upper() == "DPU"]
        check(f"DPU subgraph found in .xmodel ({len(subgraphs)} subgraph(s))",
              len(subgraphs) > 0,
              "The .xmodel may be compiled for a different DPU arch. "
              "Download the ZCU104-specific model zoo archive.")
        if subgraphs:
            runner = vart.Runner.create_runner(subgraphs[0], "run")
            in_t   = runner.get_input_tensors()
            out_t  = runner.get_output_tensors()
            check(f"DPU runner created  (input: {in_t[0].dims})", True)
            print(f"       Output tensors: {[t.dims for t in out_t]}")
    except Exception as exc:
        check("DPU runner creation", False, str(exc))

# ── GStreamer plugins ─────────────────────────────────────────────
print("\n[4] GStreamer plugins")

def gst_inspect(plugin):
    r = subprocess.run(
        ["gst-inspect-1.0", plugin],
        capture_output=True, text=True,
        timeout=5
    )
    return r.returncode == 0

# These are mandatory
for plugin in ["rtph264pay", "udpsink", "videoconvert", "appsrc"]:
    check(f"gst-inspect {plugin}", gst_inspect(plugin),
          "sudo apt-get install gstreamer1.0-plugins-good gstreamer1.0-plugins-base")

# Encoder: prefer omxh264enc (VCU hardware), fall back to x264enc (software)
omx_ok = gst_inspect("omxh264enc")
x264_ok = gst_inspect("x264enc")

if omx_ok:
    check("omxh264enc (VCU hardware encoder): PRIMARY", True)
    if x264_ok:
        warn("x264enc also available (software fallback)")
    else:
        warn("x264enc not installed",
             "Not needed: omxh264enc is the active encoder. "
             "Install if you ever need a software fallback: "
             "sudo apt-get install gstreamer1.0-plugins-ugly")
elif x264_ok:
    check("omxh264enc (VCU hardware encoder)", False,
          "VCU not available on this bitstream.")
    warn("x264enc (software fallback) available",
         "Pipeline will work but encoder is software. "
         "Change omxh264enc → x264enc tune=zerolatency bitrate=1500 speed-preset=ultrafast "
         "in _gst_pipeline() in pipeline_hw.py")
else:
    check("omxh264enc", False, "No H.264 encoder found at all!")
    check("x264enc", False,
          "Install one: sudo apt-get install gstreamer1.0-plugins-ugly")

# ── Network connectivity ──────────────────────────────────────────
print("\n[5] Network")
PHONE_HOST = "192.168.2.141"
LAPTOP_IP  = "192.168.137.197"

def ping(host):
    r = subprocess.run(["ping", "-c", "1", "-W", "1", host],
                       capture_output=True)
    return r.returncode == 0

check(f"Ping phone ({PHONE_HOST})", ping(PHONE_HOST),
      "Check phone IP in pipeline_hw.py PHONE_HOST. "
      "Phone must run IP Webcam app and be on the same network.")
check(f"Ping laptop ({LAPTOP_IP})", ping(LAPTOP_IP),
      "Run `ipconfig` on the laptop. Find the IP on the same adapter "
      "as the ZCU104 (usually 'Ethernet' or 'Wi-Fi'). "
      "Update LAPTOP_IP in pipeline_hw.py. "
      "Also check: Windows Firewall → allow UDP port 5000 inbound.")

# ── xdputil ───────────────────────────────────────────────────────
print("\n[6] DPU hardware driver")
try:
    r = subprocess.run(
        ["xdputil", "query"],
        capture_output=True, text=True,
        timeout=5          # xdputil can hang if DPU driver is unresponsive
    )
    if r.returncode == 0:
        check("xdputil query (DPU driver alive)", True)
        for line in r.stdout.splitlines()[:10]:
            print(f"       {line}")
    else:
        check("xdputil query", False,
              "DPU driver not responding. Re-flash the bitstream or reboot.")
except subprocess.TimeoutExpired:
    warn("xdputil query timed out (5 s)",
         "DPU hardware may still work: timeout is common on some bitstreams. "
         "Try: xdputil query  manually to check.")
except FileNotFoundError:
    warn("xdputil not found",
         "Not fatal: xdputil is a diagnostic tool, not required to run the pipeline.")

# ── Summary ───────────────────────────────────────────────────────
print("\n" + "=" * 60)
if errors == 0:
    print(f"\033[92m  All checks passed. Run: python3 pipeline_hw.py\033[0m")
else:
    print(f"\033[91m  {errors} check(s) failed. Fix the issues above first.\033[0m")
print("=" * 60)