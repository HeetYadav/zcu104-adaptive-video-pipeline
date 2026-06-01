#!/usr/bin/env python3
"""
Laptop Webcam Server
====================
Serves laptop's built-in webcam as /shot.jpg over HTTP.
Run this on the Windows laptop.
The ZCU104 pipeline points to this IP:8080 instead of the phone.

Usage:
    python laptop_cam_server.py
"""

import cv2
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
import time

PORT = 8080

def open_camera():
    """Try different backends until one works."""
    for backend, name in [(cv2.CAP_DSHOW, "DSHOW"), (cv2.CAP_MSMF, "MSMF"), (cv2.CAP_ANY, "ANY")]:
        c = cv2.VideoCapture(0, backend)
        if c.isOpened():
            ret, frame = c.read()   # test actual grab
            if ret and frame is not None:
                print(f"[OK] Webcam opened with backend: {name}")
                c.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
                c.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                c.set(cv2.CAP_PROP_FPS, 30)
                c.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                return c
            c.release()
    return None

cap = open_camera()
if cap is None:
    print("[ERROR] Cannot open webcam with any backend!")
    print("  → Close Teams, Zoom, Discord, browser camera tabs, then retry.")
    exit(1)


_frame_lock = threading.Lock()
_latest_jpg = None


def capture_loop():
    global _latest_jpg
    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue
        ok, jpg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ok:
            with _frame_lock:
                _latest_jpg = jpg.tobytes()


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/shot.jpg':
            with _frame_lock:
                jpg = _latest_jpg
            if jpg is None:
                self.send_response(503)
                self.end_headers()
                return
            self.send_response(200)
            self.send_header('Content-Type', 'image/jpeg')
            self.send_header('Content-Length', str(len(jpg)))
            self.end_headers()
            self.wfile.write(jpg)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass   # suppress request logs


if __name__ == '__main__':
    t = threading.Thread(target=capture_loop, daemon=True)
    t.start()
    time.sleep(1)

    print("=" * 45)
    print("  Laptop Webcam Server running on port 8080")
    print("  Endpoint: http://<this-laptop-ip>:8080/shot.jpg")
    print("  Find your IP: run  ipconfig  in cmd")
    print("  Press Ctrl+C to stop")
    print("=" * 45)

    server = HTTPServer(('0.0.0.0', PORT), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    cap.release()
    print("Server stopped.")
