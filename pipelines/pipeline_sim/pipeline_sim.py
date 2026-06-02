#!/usr/bin/env python3
"""
pipeline_sim.py — CPU-Only Simulation Pipeline
===============================================
Learn the ZCU104 ROI bandwidth management algorithm without any FPGA hardware.

This pipeline mirrors the logic of pipeline_hw.py exactly, but replaces:
  - DPU (FPGA inference)  →  OpenCV DNN (CPU-based YOLOv4 inference)
  - VCU (hardware H.264)  →  Bandwidth model calculated from active pixel ratio
  - IP Webcam (phone)     →  Local video file, RTSP stream, or webcam

Usage
-----
  # From a local video file:
  python3 pipelines/pipeline_sim/pipeline_sim.py --input path/to/video.mp4

  # From a webcam (device index 0):
  python3 pipelines/pipeline_sim/pipeline_sim.py --input 0

  # From an RTSP stream:
  python3 pipelines/pipeline_sim/pipeline_sim.py --input rtsp://192.168.1.100/stream

  # With a custom confidence threshold:
  python3 pipelines/pipeline_sim/pipeline_sim.py --input video.mp4 --conf 0.4

  # Save output to a file instead of displaying:
  python3 pipelines/pipeline_sim/pipeline_sim.py --input video.mp4 --output out.mp4

Press Q or Ctrl+C to stop.

Architecture
------------
Identical 4-thread model to pipeline_hw.py:
  Thread 1 (Grabber)    : reads frames from the video source
  Thread 2 (Detector)   : runs YOLOv4 on CPU via OpenCV DNN
  Thread 3 (Compositor) : applies 3-zone mask + prints bandwidth telemetry
  Thread 4 (Display)    : shows the composited frame in an OpenCV window

Requirements
------------
  pip install opencv-python numpy
  Download YOLOv4 weights:
    wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights
  (or use the yolov4-tiny weights for faster CPU inference)
"""

import cv2
import numpy as np
import threading
import time
import argparse
import os
import sys

# ── Pure-Python Reference Implementations ─────────────────────────
# The original module files on the ZCU104 board use PetaLinux-specific
# syntax that causes SyntaxError on standard Python.
# To allow the simulation to run anywhere, we include pure-Python
# implementations of the core functions here.

from collections import deque

class CentroidTracker:
    """Exponential-weighted centroid tracker."""
    def __init__(self, history: int = 8):
        self._history = deque(maxlen=history)
        self._prev_cx = None
        self._prev_cy = None
        self.vx = 0.0
        self.vy = 0.0

    def update(self, cx: float, cy: float) -> None:
        if self._prev_cx is not None:
            self.vx = cx - self._prev_cx
            self.vy = cy - self._prev_cy
            self._history.append((self.vx, self.vy))
        self._prev_cx = cx
        self._prev_cy = cy

    def smooth_velocity(self):
        if not self._history:
            return 0.0, 0.0
        n = len(self._history)
        weights = np.exp(np.linspace(-1.0, 0.0, n))
        weights /= weights.sum()
        vxs = np.array([h[0] for h in self._history])
        vys = np.array([h[1] for h in self._history])
        return float(np.dot(weights, vxs)), float(np.dot(weights, vys))

    def predict_next(self):
        return self.smooth_velocity()

    def reset(self) -> None:
        self._prev_cx = None
        self._prev_cy = None
        self._history.clear()
        self.vx = 0.0
        self.vy = 0.0

def adaptive_pad(x, y, w, h, vx, vy, frame_w, frame_h,
                 base_pad=20, vel_scale=3.0, max_expand=80):
    """Velocity-aware asymmetric ROI padding."""
    pad_left   = base_pad + int(min(max(0.0, -vx) * vel_scale, max_expand))
    pad_right  = base_pad + int(min(max(0.0,  vx) * vel_scale, max_expand))
    pad_top    = base_pad + int(min(max(0.0, -vy) * vel_scale, max_expand))
    pad_bottom = base_pad + int(min(max(0.0,  vy) * vel_scale, max_expand))
    x1 = max(0,       x - pad_left)
    y1 = max(0,       y - pad_top)
    x2 = min(frame_w, x + w + pad_right)
    y2 = min(frame_h, y + h + pad_bottom)
    return x1, y1, x2 - x1, y2 - y1

def _ring_box(ax, ay, aw, ah, fw, fh, ring_scale=1.6):
    ring_w = int(aw * ring_scale)
    ring_h = int(ah * ring_scale)
    rx = max(0, ax - (ring_w - aw) // 2)
    ry = max(0, ay - (ring_h - ah) // 2)
    rx2 = min(fw, rx + ring_w)
    ry2 = min(fh, ry + ring_h)
    return rx, ry, rx2 - rx, ry2 - ry

def build_zone_mask_multi(frame, adapted_boxes, ring_scale=1.6, zone2_downsample=0.5):
    """Multi-target zone composite."""
    fh, fw = frame.shape[:2]
    out = np.zeros_like(frame)
    ring_boxes = []
    if not adapted_boxes:
        return frame.copy(), []
    for (ax, ay, aw, ah) in adapted_boxes:
        rx, ry, rw, rh = _ring_box(ax, ay, aw, ah, fw, fh, ring_scale)
        ring_boxes.append((rx, ry, rw, rh))
        z2_src = frame[ry:ry+rh, rx:rx+rw]
        if z2_src.size == 0:
            continue
        sw = max(1, int(z2_src.shape[1] * zone2_downsample))
        sh = max(1, int(z2_src.shape[0] * zone2_downsample))
        small = cv2.resize(z2_src, (sw, sh), interpolation=cv2.INTER_AREA)
        z2_up = cv2.resize(small, (z2_src.shape[1], z2_src.shape[0]),
                           interpolation=cv2.INTER_LINEAR)
        out[ry:ry+rh, rx:rx+rw] = z2_up
    for (ax, ay, aw, ah) in adapted_boxes:
        out[ay:ay+ah, ax:ax+aw] = frame[ay:ay+ah, ax:ax+aw]
    return out, ring_boxes

def draw_zone_overlay_multi(frame, adapted_boxes, ring_boxes):
    """Draw zone boundaries on frame."""
    for i, (ax, ay, aw, ah) in enumerate(adapted_boxes):
        cv2.rectangle(frame, (ax, ay), (ax+aw, ay+ah), (0, 220, 80), 2)
        if i < len(ring_boxes):
            rx, ry, rw, rh = ring_boxes[i]
            cv2.rectangle(frame, (rx, ry), (rx+rw, ry+rh), (0, 180, 255), 1)
    return frame


# ─────────────────────────────────────────────────────────────────
# CLI Arguments
# ─────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="ZCU104 ROI Pipeline — CPU-only simulation (no FPGA required)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 pipeline_sim.py --input video.mp4
  python3 pipeline_sim.py --input 0                    # webcam
  python3 pipeline_sim.py --input rtsp://host/stream
  python3 pipeline_sim.py --input video.mp4 --conf 0.5 --output result.mp4
  python3 pipeline_sim.py --input video.mp4 --tiny     # use YOLOv4-Tiny (faster)

YOLOv4 model files (place in repo root or specify with --cfg / --weights):
  Full:  yolov4.cfg + yolov4.weights (244 MB)
  Tiny:  yolov4-tiny.cfg + yolov4-tiny.weights (23 MB)  ← recommended for CPU

Download weights:
  Full:  https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights
  Tiny:  https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights
        """
    )
    p.add_argument('--input',   required=False, default=None,
                   help='Video source: file path, webcam index (0), or RTSP URL')
    p.add_argument('--output',  default=None,
                   help='Optional: save output to this video file (e.g. out.mp4)')
    p.add_argument('--conf',    type=float, default=0.30,
                   help='Detection confidence threshold (default: 0.30)')
    p.add_argument('--nms',     type=float, default=0.40,
                   help='NMS IoU threshold (default: 0.40)')
    p.add_argument('--full',    action='store_true',
                   help='Use full YOLOv4 weights (slower on CPU) instead of YOLOv4-Tiny')
    p.add_argument('--cfg',     default=None,
                   help='Path to custom .cfg file (auto-detected if not set)')
    p.add_argument('--weights', default=None,
                   help='Path to custom .weights file (auto-detected if not set)')
    p.add_argument('--max-targets', type=int, default=5,
                   help='Maximum number of simultaneous persons to track (default: 5)')
    p.add_argument('--no-display', action='store_true',
                   help='Disable the OpenCV display window (useful on headless systems)')
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────
# Model loader
# ─────────────────────────────────────────────────────────────────

def load_yolo(args):
    """Load YOLOv4 or YOLOv4-Tiny from .cfg + .weights via OpenCV DNN."""
    # Auto-detect model files from repo root
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

    if not args.full:
        cfg_default     = os.path.join(root, 'yolov4-tiny.cfg')
        weights_default = os.path.join(root, 'yolov4-tiny.weights')
        label = 'YOLOv4-Tiny'
    else:
        cfg_default     = os.path.join(root, 'yolov4.cfg')
        weights_default = os.path.join(root, 'yolov4.weights')
        label = 'YOLOv4'

    cfg     = args.cfg     or cfg_default
    weights = args.weights or weights_default

    if not os.path.exists(cfg):
        print(f"[ERROR] Config file not found: {cfg}")
        print(f"        Download: https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4{'' if args.full else '-tiny'}.cfg")
        sys.exit(1)

    if not os.path.exists(weights):
        print(f"[ERROR] Weights file not found: {weights}")
        if not args.full:
            print("        Download: https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights")
        else:
            print("        Download: https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights")
        sys.exit(1)

    print(f"[YOLO] Loading {label} from:")
    print(f"       cfg     : {cfg}")
    print(f"       weights : {weights}")

    net = cv2.dnn.readNetFromDarknet(cfg, weights)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

    print(f"[OK]   {label} loaded. Output layers: {output_layers}")
    return net, output_layers


# ─────────────────────────────────────────────────────────────────
# Shared state
# ─────────────────────────────────────────────────────────────────

_grab_frame  = None
_grab_lock   = threading.Lock()
_stop_event  = threading.Event()

_faces       = []
_faces_lock  = threading.Lock()

_out_frame   = None
_out_lock    = threading.Lock()

_full_w = 640
_full_h = 480


# ─────────────────────────────────────────────────────────────────
# Thread 1: Frame Grabber
# ─────────────────────────────────────────────────────────────────

def grabber_thread(source):
    global _grab_frame, _full_w, _full_h

    # Try to open as integer (webcam) first, then as string (file/URL)
    try:
        src = int(source)
    except ValueError:
        src = source

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video source: {source}")
        _stop_event.set()
        return

    fps_source = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_delay = 1.0 / fps_source

    print(f"[Grabber] Opened source: {source}  ({int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}×{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} @ {fps_source:.1f} FPS)")

    while not _stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            # End of file — loop back to start for video files
            if isinstance(src, str):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else:
                print("[Grabber] Stream ended.")
                _stop_event.set()
                break

        fh, fw = frame.shape[:2]
        with _grab_lock:
            _grab_frame = frame
            _full_h, _full_w = fh, fw

        time.sleep(frame_delay)

    cap.release()


# ─────────────────────────────────────────────────────────────────
# Thread 2: CPU Detector (OpenCV DNN YOLOv4)
# ─────────────────────────────────────────────────────────────────

def detector_thread(net, output_layers, conf_thresh, nms_thresh):
    global _faces
    last_frame_id = id(None)
    INPUT_SIZE = 416

    print(f"[Detector] CPU YOLOv4 ready. Input: {INPUT_SIZE}×{INPUT_SIZE}")

    while not _stop_event.is_set():
        with _grab_lock:
            frame = _grab_frame

        if frame is None or id(frame) == last_frame_id:
            time.sleep(0.01)
            continue

        last_frame_id = id(frame)
        fh, fw = frame.shape[:2]

        # Preprocess: create blob from frame
        blob = cv2.dnn.blobFromImage(
            frame, 1/255.0, (INPUT_SIZE, INPUT_SIZE),
            swapRB=True, crop=False
        )
        net.setInput(blob)
        outputs = net.forward(output_layers)

        boxes, confidences = [], []
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = int(np.argmax(scores))
                confidence = float(scores[class_id])
                if confidence > conf_thresh and class_id == 0:  # person only
                    cx = int(detection[0] * fw)
                    cy = int(detection[1] * fh)
                    w  = int(detection[2] * fw)
                    h  = int(detection[3] * fh)
                    x  = max(0, cx - w // 2)
                    y  = max(0, cy - h // 2)
                    boxes.append([x, y, w, h])
                    confidences.append(confidence)

        hits = []
        if boxes:
            indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_thresh, nms_thresh)
            if len(indices) > 0:
                for i in indices.flatten():
                    hits.append(tuple(boxes[i]))  # (x, y, w, h)

        with _faces_lock:
            _faces = hits

        if hits:
            print(f"[Detector] {len(hits)} person(s) detected")


# ─────────────────────────────────────────────────────────────────
# Thread 3: Compositor (zone mask + telemetry)
# ─────────────────────────────────────────────────────────────────

def compositor_thread(max_targets):
    trackers = {i: CentroidTracker(history=8) for i in range(max_targets)}
    last_faces_key = None
    cached_boxes   = []
    cached_rings   = []
    frame_counter  = 0
    last_time      = time.time()

    TARGET_FPS = 30.0
    FRAME_TIME = 1.0 / TARGET_FPS

    print(f"[Compositor] Starting. Max targets: {max_targets}")

    while not _stop_event.is_set():
        loop_start = time.time()

        with _grab_lock:
            frame = _grab_frame.copy() if _grab_frame is not None else None

        if frame is None:
            time.sleep(0.02)
            continue

        fh, fw = frame.shape[:2]

        with _faces_lock:
            faces = list(_faces)

        faces_key = tuple(tuple(f) for f in faces)
        has_roi   = len(faces) > 0

        if has_roi:
            if faces_key != last_faces_key:
                adapted_boxes = []
                for idx, (x, y, w, h) in enumerate(faces[:max_targets]):
                    cx, cy = x + w // 2, y + h // 2
                    trackers[idx].update(cx, cy)
                    vx, vy = trackers[idx].predict_next()
                    ax, ay, aw, ah = adaptive_pad(x, y, w, h, vx, vy, fw, fh)
                    adapted_boxes.append((ax, ay, aw, ah))

                for idx in range(len(faces), max_targets):
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

        with _out_lock:
            global _out_frame
            _out_frame = out

        # Telemetry (every 30 frames)
        frame_counter += 1
        if frame_counter % 30 == 0:
            now = time.time()
            fps = 30.0 / (now - last_time) if (now - last_time) > 0 else 0
            last_time = now

            active_pixels = sum(w * h for (_, _, w, h) in cached_boxes)
            total_pixels  = fw * fh
            ratio         = active_pixels / total_pixels if total_pixels > 0 else 0.0
            base_overhead = 120.0
            kbps = base_overhead + (1500.0 - base_overhead) * ratio
            targets = len(cached_boxes)

            print(f"    [Telemetry] frame={frame_counter:6d} | targets={targets} | "
                  f"BW: {kbps:6.1f} kbps ({fps:4.1f} FPS)  [CPU sim]")

        elapsed = time.time() - loop_start
        sleep_t = FRAME_TIME - elapsed
        if sleep_t > 0:
            time.sleep(sleep_t)


# ─────────────────────────────────────────────────────────────────
# Thread 4: Display
# ─────────────────────────────────────────────────────────────────

def display_thread(video_writer):
    """Show the composited frame in an OpenCV window at ~30 FPS."""
    window_name = "ZCU104 ROI Pipeline (Simulation Mode)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 960, 540)

    while not _stop_event.is_set():
        with _out_lock:
            frame = _out_frame

        if frame is None:
            time.sleep(0.033)
            continue

        # Burn in "SIM" watermark so it's clear this is not HW
        display_frame = frame.copy()
        cv2.putText(display_frame, "CPU SIM — No FPGA",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 165, 255), 2, cv2.LINE_AA)

        cv2.imshow(window_name, display_frame)

        if video_writer is not None:
            video_writer.write(frame)

        key = cv2.waitKey(33) & 0xFF
        if key == ord('q') or key == 27:   # Q or Escape
            print("[Display] User pressed Q — stopping.")
            _stop_event.set()
            break

    cv2.destroyAllWindows()


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    
    if args.input is None:
        print("\n[INFO] No --input video source provided.")
        val = input("Do you want to use the laptop camera (webcam)? [Y/n]: ").strip().lower()
        if val in ('', 'y', 'yes'):
            args.input = '0'
        else:
            print("Please provide a video source using --input. Exiting.")
            sys.exit(1)
            
    net, output_layers = load_yolo(args)

    # Optional video output writer
    video_writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(args.output, fourcc, 30.0, (640, 480))
        print(f"[Output] Saving to: {args.output}")

    print("=" * 60)
    print("  ZCU104 ROI Pipeline  [CPU Simulation Mode]")
    print(f"  Input source : {args.input}")
    print(f"  Model        : {'YOLOv4' if args.full else 'YOLOv4-Tiny'} (CPU)")
    print(f"  Conf thresh  : {args.conf}")
    print(f"  Max targets  : {args.max_targets}")
    print("  Press Q in the video window to stop.")
    print("=" * 60)

    threads = [
        threading.Thread(
            target=grabber_thread,
            args=(args.input,),
            daemon=True, name="Grabber"
        ),
        threading.Thread(
            target=detector_thread,
            args=(net, output_layers, args.conf, args.nms),
            daemon=True, name="Detector"
        ),
        threading.Thread(
            target=compositor_thread,
            args=(args.max_targets,),
            daemon=True, name="Compositor"
        ),
    ]

    for t in threads:
        t.start()

    # Display runs in the main thread (required by OpenCV on macOS/Windows)
    if not args.no_display:
        try:
            display_thread(video_writer)
        except KeyboardInterrupt:
            print("\nStopping...")
            _stop_event.set()
    else:
        try:
            while not _stop_event.is_set():
                time.sleep(0.5)
        except KeyboardInterrupt:
            _stop_event.set()

    for t in threads:
        t.join(timeout=3.0)

    if video_writer is not None:
        video_writer.release()

    print("Pipeline stopped.")


if __name__ == '__main__':
    main()
