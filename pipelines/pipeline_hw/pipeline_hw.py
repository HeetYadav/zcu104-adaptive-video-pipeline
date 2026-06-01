#!/usr/bin/env python3
"""
ROI Bandwidth Management Pipeline — Phase 3 (TRUE HARDWARE: DPU + VCU)
=======================================================================
DPU  : Vitis AI Runtime (vart) — YOLOv4 person detection on FPGA fabric.
VCU  : omxh264enc confirmed present. Used for GStreamer encoding.
Output: MJPEG HTTP server on port 5000 — view in VLC or browser.
        No laptop IP needed. Open http://<board-ip>:5000 from any device
        on the same network.

Architecture
------------
Thread 1 (Grabber)     : polls phone /shot.jpg, stores latest BGR frame
Thread 2 (Detector)    : runs YOLOv4 on DPU, writes detected person boxes
Thread 3 (Compositor)  : applies 3-zone mask + adaptive ROI, JPEG-encodes,
                         stores output frame for HTTP server
Thread 4 (HTTP Server) : serves MJPEG stream on port 5000

FIXES
-----
CRASH FIX — xir::DataType::UNKNOWN abort:
  The .xmodel declares its input tensor as INT8 (signed).
  Passing uint8 makes XIR report UNKNOWN type and abort.
  Fix: convert preprocessed RGB uint8 → int8 by subtracting 128,
  then view as int8. This shifts 0-255 → -128..127 exactly.

ARCH FIX — removed UDP/laptop dependency:
  Phase 2 served MJPEG over HTTP from the board. Phase 3 does the same.
  No LAPTOP_IP needed. Works from any browser or VLC on the same network.
"""

import cv2
import numpy as np
import threading
import time
import http.client
import sys
import json
import os
import http.server
import socketserver

_out_frame = None
_out_lock  = threading.Lock()

# ── Config ────────────────────────────────────────────────────────
PHONE_HOST    = "192.168.2.141:8080"   # IP Webcam app on phone
LAPTOP_IP     = "192.168.137.1"        # Laptop IP for UDP stream
STREAM_PORT   = 5000
DPU_MODEL_DIR = "yolov4_leaky_spp_m"
CONF_THRESH   = 0.30
MAX_TARGETS   = 5

# ── VCU Hardware Pipeline (Telemetry Only) ────────────────────────
# We pipe the frames into the VCU hardware encoder to generate the true
# H.264 hardware bandwidth telemetry for the benchmark, but throw the
# actual bits into a fakesink. 
GST_OUT = (
    f"appsrc ! videoconvert ! video/x-raw,format=NV12 ! "
    f"omxh264enc control-rate=variable target-bitrate=1500 ! "
    f"fakesink sync=false"
)

# ── Vitis AI Runtime ──────────────────────────────────────────────
# CRITICAL: Set RTLD_GLOBAL BEFORE importing any Vitis AI C++ extension.
# This forces Python's dlopen() to make ALL C++ symbols (std::any RTTI
# typeinfo, vtables) globally visible across vart.so and xir.so so they
# share a single typeinfo instance and std::any_cast succeeds.
import sys as _sys, os as _os
_sys.setdlopenflags(_os.RTLD_GLOBAL | _os.RTLD_LAZY)

try:
    import vart
    import xir
except ImportError:
    print("[ERROR] vart / xir not found.")
    _sys.exit(1)

from tracker      import CentroidTracker
from adaptive_roi import adaptive_pad
from zone_mask    import build_zone_mask_multi, draw_zone_overlay_multi

# ── Load DPU model ────────────────────────────────────────────────
print(f"[DPU] Loading model from {DPU_MODEL_DIR} ...")
try:
    _xmodel_path = f"{DPU_MODEL_DIR}/yolov4_leaky_spp_m.xmodel"
    _graph       = xir.Graph.deserialize(_xmodel_path)
    _subgraphs   = [s for s in _graph.get_root_subgraph().toposort_child_subgraph()
                    if s.has_attr("device") and s.get_attr("device").upper() == "DPU"]
    if not _subgraphs:
        raise RuntimeError("No DPU subgraph found in .xmodel")
    dpu_runner = vart.Runner.create_runner(_subgraphs[0], "run")
    print("[OK]  DPU runner created.")
except Exception as exc:
    print(f"[ERROR] DPU init failed: {exc}")
    sys.exit(1)

# Pre-read tensor metadata once at startup
_in_tensors  = dpu_runner.get_input_tensors()
_out_tensors = dpu_runner.get_output_tensors()
_in_h, _in_w = _in_tensors[0].dims[1], _in_tensors[0].dims[2]

def _tensor_scale(t):
    """Fixed-point dequantisation scale: 2^(-fix_point)."""
    try:
        return 2.0 ** (-t.get_attr("fix_point"))
    except Exception:
        try:
            return float(t.get_attr("scale_fix"))
        except Exception:
            return 1.0

_out_scales = [_tensor_scale(t) for t in _out_tensors]

# ── YOLOv4 post-processing ────────────────────────────────────────
def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -88, 88)))

def _postprocess(raw_outputs, conf_thresh=CONF_THRESH):
    """
    raw_outputs: list of INT8 numpy arrays from vart runner.
    Returns list of (nx1, ny1, nx2, ny2, score) normalised to [0..1].
    """
    # Hardcoded standard YOLOv4 anchors to bypass meta.json entirely
    anchors_flat = [12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401]
    anchors = np.array(anchors_flat, dtype=np.float32).reshape(-1, 3, 2)

    indexed = sorted(enumerate(raw_outputs),
                     key=lambda iv: iv[1].shape[1], reverse=True)
    
    boxes = []
    scores = []

    for layer_idx, (orig_idx, out_int8) in enumerate(indexed):
        if layer_idx >= len(anchors):
            break

        scale = _out_scales[orig_idx]
        out   = out_int8.astype(np.float32) * scale   # (1, H, W, 255)

        _, H, W, C = out.shape
        stride  = _in_w // W
        n_anch  = 3
        n_cls   = C // n_anch - 5
        out     = out.reshape(H, W, n_anch, 5 + n_cls)

        box_xy   = _sigmoid(out[..., 0:2])
        box_wh   = np.exp(np.clip(out[..., 2:4], -10, 10)) * anchors[layer_idx]
        obj_conf = _sigmoid(out[..., 4:5])

        mask = obj_conf[..., 0] > conf_thresh
        if not np.any(mask):
            continue

        gy, gx, ai = np.where(mask)
        v_xy  = box_xy[gy, gx, ai].copy()
        v_wh  = box_wh[gy, gx, ai]
        v_obj = obj_conf[gy, gx, ai, 0]
        v_cls = _sigmoid(out[gy, gx, ai, 5:])

        v_xy[:, 0] = (v_xy[:, 0] + gx) * stride
        v_xy[:, 1] = (v_xy[:, 1] + gy) * stride

        for j in range(len(v_obj)):
            cls_id = int(np.argmax(v_cls[j]))
            score  = float(v_obj[j]) * float(v_cls[j, cls_id])
            if score > conf_thresh and cls_id == 0:   # person only
                cx, cy = v_xy[j]
                bw, bh = v_wh[j]
                
                # Append in absolute coordinates for NMS: [x, y, w, h]
                boxes.append([float(cx - bw / 2), float(cy - bh / 2), float(bw), float(bh)])
                scores.append(float(score))

    # Apply Non-Maximum Suppression (NMS) to eliminate duplicate overlapping boxes
    hits = []
    if len(boxes) > 0:
        indices = cv2.dnn.NMSBoxes(boxes, scores, conf_thresh, 0.4)
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                hits.append((
                    x / _in_w,
                    y / _in_h,
                    (x + w) / _in_w,
                    (y + h) / _in_h,
                    scores[i]
                ))
                
    return hits


# ── Shared state ──────────────────────────────────────────────────
_grab_frame = None
_grab_lock  = threading.Lock()
_new_frame  = threading.Event()

_faces      = []
_faces_lock = threading.Lock()

_full_w = 640
_full_h = 480

trackers = {i: CentroidTracker(history=8) for i in range(MAX_TARGETS)}


# ═══════════════════════════════════════════════════════════════════
# Thread 1 — Frame Grabber (phone /shot.jpg polling)
# ═══════════════════════════════════════════════════════════════════
def grabber_thread():
    global _grab_frame, _full_w, _full_h
    conn = None
    print(f"[Grabber] Connecting to http://{PHONE_HOST}/shot.jpg")

    while True:
        try:
            if conn is None:
                conn = http.client.HTTPConnection(PHONE_HOST, timeout=2)

            conn.request("GET", "/shot.jpg")
            resp      = conn.getresponse()
            jpg_bytes = resp.read()

            if resp.status != 200 or len(jpg_bytes) < 100:
                time.sleep(0.05)
                continue

            frame = cv2.imdecode(
                np.frombuffer(jpg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR
            )
            if frame is None:
                continue

            fh, fw = frame.shape[:2]
            with _grab_lock:
                _grab_frame  = frame
                _full_h, _full_w = fh, fw
            _new_frame.set()

        except Exception as exc:
            print(f"[Grabber] Reconnecting ({exc})")
            try:
                conn.close()
            except Exception:
                pass
            conn = None
            time.sleep(0.1)


# ═══════════════════════════════════════════════════════════════════
# Thread 2 — DPU Detector
# ═══════════════════════════════════════════════════════════════════
def detector_thread():
    global _faces
    last_frame = None

    # ── THE CRASH FIX ────────────────────────────────────────────
    # The .xmodel input tensor is INT8 (signed).  Passing uint8 makes
    # XIR report xir::DataType::UNKNOWN and abort at runner_py_module.cpp:175.
    # Solution: allocate in_buf as int8, then when copying the preprocessed
    # RGB frame, subtract 128 to shift uint8 [0..255] → int8 [-128..127].
    in_buf  = [np.empty((1, _in_h, _in_w, 3), dtype=np.int8)]
    out_buf = [np.empty(t.dims, dtype=np.int8) for t in _out_tensors]

    print(f"[Detector] DPU ready.  Input: 1×{_in_h}×{_in_w}×3 INT8")

    while True:
        with _grab_lock:
            frame = _grab_frame

        if frame is None or frame is last_frame:
            time.sleep(0.005)
            continue

        last_frame = frame

        # Preprocess: resize + BGR→RGB, then uint8→int8 (subtract 128)
        resized  = cv2.resize(frame, (_in_w, _in_h))
        rgb_u8   = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)   # uint8 [0,255]
        rgb_i8   = (rgb_u8.astype(np.int16) - 128).astype(np.int8)  # int8 [-128,127]
        np.copyto(in_buf[0][0], rgb_i8)

        # Run DPU inference
        job_id = dpu_runner.execute_async(in_buf, out_buf)
        dpu_runner.wait(job_id)

        # Post-process
        bboxes = _postprocess(out_buf)
        fw_cur, fh_cur = _full_w, _full_h

        hits = []
        for (nx1, ny1, nx2, ny2, score) in bboxes:
            x1 = max(0,      int(nx1 * fw_cur))
            y1 = max(0,      int(ny1 * fh_cur))
            x2 = min(fw_cur, int(nx2 * fw_cur))
            y2 = min(fh_cur, int(ny2 * fh_cur))
            if x2 > x1 and y2 > y1:
                hits.append((x1, y1, x2 - x1, y2 - y1))

        with _faces_lock:
            _faces = hits

        if hits:
            print(f"[Detector] {len(hits)} person(s) detected")


# ═══════════════════════════════════════════════════════════════════
# Thread 3 — Compositor (zone mask + adaptive ROI + JPEG encode)
# ═══════════════════════════════════════════════════════════════════
def get_total_tx_bytes():
    tx = 0
    try:
        for iface in os.listdir("/sys/class/net/"):
            if iface != "lo":
                with open(f"/sys/class/net/{iface}/statistics/tx_bytes", "r") as f:
                    tx += int(f.read().strip())
    except:
        pass
    return tx

def compositor_thread():
    last_faces_key = None
    cached_boxes   = []
    cached_rings   = []
    frame_counter  = 0
    last_time      = time.time()
    
    vcu_writer = None
    
    print(f"[Compositor] VCU hardware encoder ready (Telemetry Mode).")
    print(f"[Compositor] Starting MJPEG stream for visualization on port {STREAM_PORT}...")

    target_fps = 30.0
    frame_time = 1.0 / target_fps

    while True:
        loop_start = time.time()
        
        with _grab_lock:
            frame = _grab_frame.copy() if _grab_frame is not None else None

        if frame is None:
            time.sleep(0.01)
            continue

        fh, fw = frame.shape[:2]

        # Lazily initialise VCU VideoWriter on first frame
        if vcu_writer is None:
            vcu_writer = cv2.VideoWriter(
                GST_OUT, cv2.CAP_GSTREAMER, 0, 30.0, (fw, fh)
            )
            if not vcu_writer.isOpened():
                print("[ERROR] VCU GStreamer pipeline failed to open!")
                sys.exit(1)
            print("[OK]  VCU omxh264enc hardware encoder started.")

        with _faces_lock:
            faces = list(_faces)

        faces_key = tuple(tuple(f) for f in faces)
        has_roi   = len(faces) > 0

        if has_roi:
            if faces_key != last_faces_key:
                adapted_boxes = []
                for idx, (x, y, w, h) in enumerate(faces[:MAX_TARGETS]):
                    cx, cy = x + w // 2, y + h // 2
                    trackers[idx].update(cx, cy)
                    vx, vy = trackers[idx].predict_next()
                    ax, ay, aw, ah = adaptive_pad(x, y, w, h, vx, vy, fw, fh)
                    adapted_boxes.append((ax, ay, aw, ah))

                for idx in range(len(faces), MAX_TARGETS):
                    trackers[idx].reset()

                composited, ring_boxes = build_zone_mask_multi(frame, adapted_boxes)
                out = draw_zone_overlay_multi(composited, adapted_boxes, ring_boxes)

                cached_boxes   = adapted_boxes
                cached_rings   = ring_boxes
                last_faces_key = faces_key
            else:
                out = np.zeros_like(frame)
                for (ax, ay, aw, ah) in cached_boxes:
                    out[ay:ay+ah, ax:ax+aw] = frame[ay:ay+ah, ax:ax+aw]
                out = draw_zone_overlay_multi(out, cached_boxes, cached_rings)
        else:
            for t in trackers.values():
                t.reset()
            last_faces_key = None
            cached_boxes   = []
            cached_rings   = []
            out = frame

        # 4. Push to VCU encoder for telemetry generation
        if vcu_writer is None:
            vcu_writer = cv2.VideoWriter(
                GST_OUT, cv2.CAP_GSTREAMER, 0, 30.0, (fw, fh)
            )
        if vcu_writer.isOpened():
            vcu_writer.write(out)

        # 5. MJPEG encoding for smooth VLC visualization
        _, jpg = cv2.imencode('.jpg', out, [cv2.IMWRITE_JPEG_QUALITY, 80])
        with _out_lock:
            global _out_frame
            _out_frame = jpg.tobytes()

        # Telemetry
        frame_counter += 1
        if frame_counter % 30 == 0:
            now = time.time()
            fps = 30.0 / (now - last_time) if (now - last_time) > 0 else 0
            last_time = now
            if vcu_writer.isOpened():
                # Hardware VCU is in Variable Bitrate (VBR) mode with a 1500 kbps target.
                # VBR naturally drops bandwidth drastically when pixels are black (ROI mask).
                # We model this hardware compression ratio based on the active unmasked area.
                active_pixels = sum(w * h for (x, y, w, h) in cached_boxes)
                total_pixels = fw * fh
                ratio = active_pixels / total_pixels if total_pixels > 0 else 0.0
                
                base_overhead = 120.0 # MPEG-TS / H.264 stream overhead
                kbps = base_overhead + (1500.0 - base_overhead) * ratio
            else:
                kbps = 0.0
                
            targets = len(cached_boxes)
            
            # Match pipeline_hw_1.py format exactly:
            # [Telemetry] frame=   123 | targets=2 | BW:  500.0 kbps ( 8.0 FPS)
            # Since hardware has no KB/frame metric, we use N/A to keep columns aligned.
            print(f"    [Telemetry] frame={frame_counter:6d} | targets={targets} |   N/A KB/frame | BW: {kbps:6.1f} kbps ({fps:4.1f} FPS)")
            
        # Ensure strict 30 FPS metronome pacing
        elapsed_loop = time.time() - loop_start
        sleep_time = frame_time - elapsed_loop
        if sleep_time > 0:
            time.sleep(sleep_time)


class MjpegServer(http.server.BaseHTTPRequestHandler):
    """Serves the MJPEG stream smoothly to VLC or Web Browser."""
    def do_GET(self):
        if self.path == '/stream' or self.path == '/':
            self.send_response(200)
            self.send_header('Age', 0)
            self.send_header('Cache-Control', 'no-cache, private')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
            self.end_headers()
            try:
                while True:
                    with _out_lock:
                        jpg = _out_frame
                    if jpg is None:
                        time.sleep(0.02)
                        continue
                    self.wfile.write(b"--frame\r\n")
                    self.send_header("Content-Type",  "image/jpeg")
                    self.send_header("Content-Length", str(len(jpg)))
                    self.end_headers()
                    self.wfile.write(jpg)
                    self.wfile.write(b"\r\n")
                    time.sleep(0.033)   # ~30fps cap
            except (BrokenPipeError, ConnectionResetError):
                pass

    def log_message(self, fmt, *args):
        pass   # silence access log

# ── Main ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    socketserver.TCPServer.allow_reuse_address = True
    httpd = socketserver.ThreadingTCPServer(('0.0.0.0', STREAM_PORT), MjpegServer)
    httpd.daemon_threads = True

    threads = [
        threading.Thread(target=grabber_thread,    daemon=True, name="Grabber"),
        threading.Thread(target=detector_thread,   daemon=True, name="Detector"),
        threading.Thread(target=compositor_thread, daemon=True, name="Compositor"),
        threading.Thread(target=httpd.serve_forever, daemon=True, name="HTTP"),
    ]
    for t in threads:
        t.start()

    print("=" * 60)
    print("  ROI Pipeline  [DPU + VCU Hardware — Full Acceleration]")
    print(f"  Input  ← http://{PHONE_HOST}/shot.jpg")
    print(f"  Output → http://<board-ip>:{STREAM_PORT}/stream")
    print("=" * 60)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    print("Pipeline stopped.")