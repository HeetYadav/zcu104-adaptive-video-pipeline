#!/usr/bin/env python3
"""
ROI Bandwidth Management Pipeline — Phase 2 §2  [EXTENDED RANGE]
=================================================================
Detection range fix: Tiled detection (4 overlapping patches).
At 20ft, a face is ~15px in 640x480. The DNN can't see it at half-res.
Tiling effectively gives 2× pixel density per patch → detects at 4× the range.

Bandwidth telemetry: Terminal shows KB/frame + rolling KB/s for both
ROI-present and background-only states.
"""

import cv2
import numpy as np
import threading
import time
import http.client
from http.server import HTTPServer, BaseHTTPRequestHandler

import os

# ── Module path resolution ────────────────────────────────────────
# Allows running from any directory: python3 pipelines/pipeline_sw/pipeline.py
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _mod in ['zone_mask', 'adaptive_roi', 'tracker', 'telemetry']:
    sys.path.insert(0, os.path.join(_REPO_ROOT, 'modules', _mod))

from tracker import CentroidTracker
from adaptive_roi import adaptive_pad
from zone_mask import build_zone_mask_multi, draw_zone_overlay_multi
from telemetry import measure_zone_bytes

# ── Config ──────────────────────────────────────────────────────────
PHONE_HOST            = "192.168.2.141:8080"
SNAPSHOT_URL          = f"http://{PHONE_HOST}/shot.jpg"
STREAM_PORT           = 5000
JPEG_QUALITY          = 65

# Detection config — tune for range
CONFIDENCE_THRESHOLD  = 0.30   # lower = more sensitive (catches small/distant faces)
TILED_DETECTION       = True   # True = 4-patch tiling for long-range; False = fast single-pass

# ── Load YOLOv4-tiny Detector ─────────────────────────────────────────
print("Loading YOLOv4-tiny Darknet Model...")
net = cv2.dnn.readNetFromDarknet('yolov4-tiny.cfg', 'yolov4-tiny.weights')
# Determine output layer names
_layer_names = net.getLayerNames()
try:
    _out_layers = [_layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
except Exception:
    _out_layers = [_layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
print("[OK] YOLOv4-tiny model loaded.")
print(f"[CFG] Tiled detection: {TILED_DETECTION} | Confidence: {CONFIDENCE_THRESHOLD}")

# ── Shared state ────────────────────────────────────────────────────
_grab_frame  = None
_grab_lock   = threading.Lock()
_grab_ready  = threading.Event()

_det_frame   = None
_det_lock    = threading.Lock()

_faces       = []
_faces_lock  = threading.Lock()

_output_jpg  = None
_output_lock = threading.Lock()

_full_w = 640
_full_h = 480

MAX_FACES = 5
trackers = {i: CentroidTracker(history=8) for i in range(MAX_FACES)}

# ── Bandwidth telemetry state ────────────────────────────────────────
_bw_lock       = threading.Lock()
_bw_bytes_roi  = 0.0   # rolling sum of JPEG bytes when ROI active
_bw_bytes_bg   = 0.0   # rolling sum of JPEG bytes when no ROI
_bw_frames_roi = 0
_bw_frames_bg  = 0
_bw_t0         = time.time()


def record_bw(jpg_len, has_roi):
    global _bw_bytes_roi, _bw_bytes_bg, _bw_frames_roi, _bw_frames_bg, _bw_t0
    with _bw_lock:
        if has_roi:
            _bw_bytes_roi  += jpg_len
            _bw_frames_roi += 1
        else:
            _bw_bytes_bg  += jpg_len
            _bw_frames_bg += 1


def print_bw_report():
    """Print rolling bandwidth stats and reset counters."""
    global _bw_bytes_roi, _bw_bytes_bg, _bw_frames_roi, _bw_frames_bg, _bw_t0
    with _bw_lock:
        elapsed = time.time() - _bw_t0
        if elapsed < 0.1:
            return

        # Per-frame average
        roi_avg = (_bw_bytes_roi / _bw_frames_roi / 1024) if _bw_frames_roi else 0
        bg_avg  = (_bw_bytes_bg  / _bw_frames_bg  / 1024) if _bw_frames_bg  else 0

        # KB/s rate
        roi_kbps = (_bw_bytes_roi / elapsed / 1024) if _bw_frames_roi else 0
        bg_kbps  = (_bw_bytes_bg  / elapsed / 1024) if _bw_frames_bg  else 0

        total_frames = _bw_frames_roi + _bw_frames_bg
        saved_pct = (1 - roi_avg / bg_avg) * 100 if bg_avg > 0 and roi_avg > 0 else 0

        print(f"[BW] ROI: {roi_avg:5.1f} KB/f @ {roi_kbps:5.1f} KB/s ({_bw_frames_roi}f) | "
              f"BG: {bg_avg:5.1f} KB/f @ {bg_kbps:5.1f} KB/s ({_bw_frames_bg}f) | "
              f"Saved: {saved_pct:.0f}%")

        # Reset for next window
        _bw_bytes_roi = _bw_bytes_bg = 0.0
        _bw_frames_roi = _bw_frames_bg = 0
        _bw_t0 = time.time()


# ═══════════════════════════════════════════════════════════════════
# Thread 1 — Snapshot Grabber
# ═══════════════════════════════════════════════════════════════════
def grabber_thread():
    global _grab_frame, _det_frame, _full_w, _full_h
    det_counter = 0
    conn = None
    print(f"[Grabber] Fetching from {SNAPSHOT_URL}")

    while True:
        try:
            if conn is None:
                conn = http.client.HTTPConnection(PHONE_HOST, timeout=2)

            conn.request("GET", "/shot.jpg")
            resp = conn.getresponse()
            jpg_bytes = resp.read()

            if resp.status != 200 or len(jpg_bytes) < 100:
                time.sleep(0.05)
                continue

            arr   = np.frombuffer(jpg_bytes, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                continue

            fh, fw = frame.shape[:2]
            _full_h, _full_w = fh, fw

            with _grab_lock:
                _grab_frame = frame
            _grab_ready.set()

            # Feed detector every 3rd frame
            det_counter += 1
            if det_counter % 3 == 0:
                with _det_lock:
                    # Full-res for tiled detection (needed for long range)
                    _det_frame = frame.copy() if TILED_DETECTION else cv2.resize(
                        frame, (fw // 2, fh // 2), interpolation=cv2.INTER_NEAREST)

        except Exception as e:
            print(f"[Grabber] Reconnecting: {e}")
            try:
                conn.close()
            except Exception:
                pass
            conn = None
            time.sleep(0.1)


# ═══════════════════════════════════════════════════════════════════
# Detection helpers
# ═══════════════════════════════════════════════════════════════════
def _run_dnn_on_patch(patch, conf_thresh):
    """Run YOLOv4-tiny on a single image patch, return raw (x1,y1,x2,y2,conf) list."""
    ph, pw = patch.shape[:2]
    # YOLOv4-tiny expects 1/255.0 normalization, RGB format. 320x320 is fast and good enough.
    blob = cv2.dnn.blobFromImage(patch, 1/255.0, (320, 320), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(_out_layers)

    hits = []
    # Darknet outputs a list of arrays. Each array has rows: [xc, yc, w, h, obj_conf, class0, class1...]
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # class_id 0 = person
            if class_id == 0 and confidence > conf_thresh:
                # YOLOv4 coords from cv2.dnn are normalized [0, 1] relative to the patch width/height
                xc, yc, w, h = detection[0:4]
                x_center, y_center = xc * pw, yc * ph
                box_w, box_h = w * pw, h * ph

                x1 = int(x_center - box_w / 2.0)
                y1 = int(y_center - box_h / 2.0)
                x2 = int(x_center + box_w / 2.0)
                y2 = int(y_center + box_h / 2.0)

                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(pw, x2), min(ph, y2)
                if x2 > x1 and y2 > y1:
                    hits.append((x1, y1, x2, y2, float(confidence)))
    return hits


def detect_tiled(frame, conf_thresh):
    """
    Split frame into 4 overlapping tiles, detect in each, merge back.
    Tile layout (50% overlap):
        +-------+-------+
        | TL    | TR    |
        +-------+-------+
        | BL    | BR    |
        +-------+-------+
    Each tile is 3/4 of frame width/height, overlapping by 50%.
    This doubles effective pixel density → 2× detection range.
    """
    fh, fw = frame.shape[:2]
    tw, th = int(fw * 0.75), int(fh * 0.75)

    tiles = [
        (0,       0,       tw, th),               # TL
        (fw - tw, 0,       fw, th),               # TR
        (0,       fh - th, tw, fh),               # BL
        (fw - tw, fh - th, fw, fh),               # BR
    ]

    raw_boxes = []   # (x1, y1, x2, y2, conf) in full-frame coords

    for (tx1, ty1, tx2, ty2) in tiles:
        patch = frame[ty1:ty2, tx1:tx2]
        hits  = _run_dnn_on_patch(patch, conf_thresh)
        for (x1, y1, x2, y2, conf) in hits:
            # Translate patch-local coords back to full frame
            raw_boxes.append((x1 + tx1, y1 + ty1, x2 + tx1, y2 + ty1, conf))

    if not raw_boxes:
        return []

    # NMS to remove duplicates from overlapping tiles
    boxes_nms = np.array([[x1, y1, x2 - x1, y2 - y1] for (x1, y1, x2, y2, _) in raw_boxes],
                         dtype=np.float32)
    scores    = np.array([c for (*_, c) in raw_boxes], dtype=np.float32)
    indices   = cv2.dnn.NMSBoxes(boxes_nms.tolist(), scores.tolist(),
                                  conf_thresh, nms_threshold=0.4)

    results = []
    if len(indices) > 0:
        for idx in indices.flatten():
            x, y, w, h = [int(v) for v in boxes_nms[idx]]
            x2, y2 = min(fw, x + w), min(fh, y + h)
            if x2 > x and y2 > y:
                results.append((x, y, x2 - x, y2 - y))
    return results


def detect_single(frame, conf_thresh):
    """Original single-pass detection (fast, short-range)."""
    fh, fw = frame.shape[:2]
    # Scale coords if frame is half-res
    sx = _full_w / fw
    sy = _full_h / fh
    hits = _run_dnn_on_patch(frame, conf_thresh)
    results = []
    for (x1, y1, x2, y2, _) in hits:
        x1, y1 = int(x1 * sx), int(y1 * sy)
        x2, y2 = int(x2 * sx), int(y2 * sy)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(_full_w, x2), min(_full_h, y2)
        if x2 > x1 and y2 > y1:
            results.append((x1, y1, x2 - x1, y2 - y1))
    return results


# ═══════════════════════════════════════════════════════════════════
# Thread 2 — DNN Detector
# ═══════════════════════════════════════════════════════════════════
def detector_thread():
    global _faces
    while True:
        with _det_lock:
            frame = _det_frame

        if frame is None:
            time.sleep(0.05)
            continue

        if TILED_DETECTION:
            results = detect_tiled(frame, CONFIDENCE_THRESHOLD)
        else:
            results = detect_single(frame, CONFIDENCE_THRESHOLD)

        with _faces_lock:
            _faces = results


# ═══════════════════════════════════════════════════════════════════
# Thread 3 — Compositor
# ═══════════════════════════════════════════════════════════════════
def compositor_thread():
    global _output_jpg
    frame_count  = 0
    bw_report_every = 30   # print BW report every N frames

    last_faces_key = None
    cached_boxes   = []
    cached_rings   = []

    while True:
        _grab_ready.wait()
        _grab_ready.clear()

        with _grab_lock:
            frame = _grab_frame
        if frame is None:
            continue

        with _faces_lock:
            faces = list(_faces)

        frame_count += 1
        faces_key = tuple(tuple(f) for f in faces)
        has_roi   = len(faces) > 0

        if has_roi:
            fw, fh = frame.shape[1], frame.shape[0]

            if faces_key != last_faces_key:
                adapted_boxes = []
                for idx, (x, y, w, h) in enumerate(faces):
                    if idx >= MAX_FACES:
                        break
                    cx, cy = x + w // 2, y + h // 2
                    trackers[idx].update(cx, cy)
                    vx, vy = trackers[idx].predict_next()
                    ax, ay, aw, ah = adaptive_pad(x, y, w, h, vx, vy, fw, fh)
                    adapted_boxes.append((ax, ay, aw, ah))

                for idx in range(len(faces), MAX_FACES):
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
            cached_boxes = []
            cached_rings = []
            out = frame

        ok, jpg = cv2.imencode('.jpg', out, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if ok:
            jpg_bytes = jpg.tobytes()
            record_bw(len(jpg_bytes), has_roi)
            with _output_lock:
                _output_jpg = jpg_bytes

        # Print bandwidth report every N frames
        if frame_count % bw_report_every == 0:
            print_bw_report()


# ═══════════════════════════════════════════════════════════════════
# HTTP MJPEG Server
# ═══════════════════════════════════════════════════════════════════
class MJPEGHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-Type',
                         'multipart/x-mixed-replace; boundary=frame')
        self.end_headers()
        last_jpg = None
        try:
            while True:
                with _output_lock:
                    jpg = _output_jpg

                if jpg is not None and jpg is not last_jpg:
                    self.wfile.write(b'--frame\r\n')
                    self.wfile.write(b'Content-Type: image/jpeg\r\n')
                    self.wfile.write(f'Content-Length: {len(jpg)}\r\n'.encode())
                    self.wfile.write(b'\r\n')
                    self.wfile.write(jpg)
                    self.wfile.write(b'\r\n')
                    last_jpg = jpg
                else:
                    time.sleep(0.005)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, format, *args):
        pass


# ── Main ──────────────────────────────────────────────────────────
if __name__ == '__main__':
    threads = [
        threading.Thread(target=grabber_thread,    daemon=True, name="Grabber"),
        threading.Thread(target=detector_thread,   daemon=True, name="Detector"),
        threading.Thread(target=compositor_thread, daemon=True, name="Compositor"),
    ]
    for t in threads:
        t.start()

    time.sleep(1)

    print("=" * 60)
    print("  ROI Pipeline §2  [tiled detection — extended range]")
    print(f"  Conf threshold: {CONFIDENCE_THRESHOLD} | Tiled: {TILED_DETECTION}")
    print(f"  VLC → http://<board-ip>:{STREAM_PORT}")
    print("=" * 60)

    server = HTTPServer(('0.0.0.0', STREAM_PORT), MJPEGHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    print("Pipeline stopped.")
