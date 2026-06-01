[↑ Tools](../README.md) | [Back to Repo Root](../../README.md) | [Hardware Setup →](../../docs/02_hardware_setup.md#ip-webcam-app-setup)

---

# `laptop_cam_server.py`: Laptop Webcam Camera Server

> **Alternative camera source.** If you don't have an Android phone or the IP Webcam app is not available, run this script on your Windows/Linux laptop to serve its built-in webcam over the same HTTP endpoint (`/shot.jpg`) that the pipelines use.

The pipelines poll `http://<host>:8080/shot.jpg` for camera frames. This script serves exactly that endpoint using the laptop's webcam, making it a **drop-in replacement** for the IP Webcam phone app.

## How to Run

**On your Windows laptop:**
```cmd
# Install OpenCV if needed
pip install opencv-python

# Run the server
python tools/laptop_cam_server/laptop_cam_server.py
```

**Then update the pipeline** to point to your laptop's IP:
```python
# In pipeline_hw.py or pipeline_hw_1.py, line ~47:
PHONE_HOST = "<your-laptop-ip>:8080"   # e.g., "192.168.2.100:8080"
```

Find your laptop's IP with:
```cmd
ipconfig
```
Look for the IPv4 address of your Wi-Fi adapter.

## What It Does

1. Opens the laptop webcam using OpenCV (`cv2.VideoCapture`)
2. Runs a background thread that continuously captures frames and encodes them as JPEG
3. Serves a lightweight HTTP server on port 8080 with a single endpoint: `GET /shot.jpg`
4. Each `GET /shot.jpg` request returns the **latest captured JPEG frame**

The HTTP server behavior is **identical** to IP Webcam's `/shot.jpg` endpoint: the ZCU104 pipelines cannot tell the difference.

## Backend Selection

The script tries three OpenCV camera backends in order to find one that works on your system:

```python
for backend, name in [(cv2.CAP_DSHOW, "DSHOW"),
                      (cv2.CAP_MSMF, "MSMF"),
                      (cv2.CAP_ANY,  "ANY")]:
    cap = cv2.VideoCapture(0, backend)
    if cap.isOpened():
        ...
```

| Backend | Platform | Notes |
|---------|----------|-------|
| `CAP_DSHOW` | Windows | DirectShow: fastest on Windows |
| `CAP_MSMF` | Windows | Media Foundation: fallback |
| `CAP_ANY` | Any | Auto-detect: last resort |

## Troubleshooting

**Error: "Cannot open webcam with any backend!"**

The webcam is in use by another application. Close:
- Teams, Zoom, Google Meet, Discord- Any browser tab with active camera access- OBS, Streamlabs, or any screen-capture tool
Then retry.

**Low FPS or high latency:**

The script sets `CAP_PROP_BUFFERSIZE = 1` to minimize the internal queue. Each `/shot.jpg` request gets the **most recent** frame, not a buffered one. If FPS is still low, check your USB webcam's native FPS cap (many laptop cameras are 30 FPS max).

## Configuration

Edit these variables at the top of `laptop_cam_server.py`:

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8080` | HTTP server port (must match `PHONE_HOST` in the pipeline) |
| Resolution | `640×480` | Set via `CAP_PROP_FRAME_WIDTH/HEIGHT` |
| FPS | `30` | Set via `CAP_PROP_FPS` |
| JPEG quality | `85` | Set in `cv2.imencode` parameters |

## See Also

- [Hardware Setup: IP Webcam](../../docs/02_hardware_setup.md#ip-webcam-app-setup)- [Pipeline Config: PHONE_HOST](../../pipelines/pipeline_hw/README.md#configuration)