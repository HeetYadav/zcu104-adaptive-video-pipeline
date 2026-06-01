import sys
import gi
import numpy as np
import cv2

gi.require_version('Gst', '1.0')
gi.require_version('GstVideo', '1.0')
from gi.repository import Gst, GObject, GstVideo, GLib

# Initialize GStreamer
Gst.init(None)

print("Loading OpenCV CPU Face Detect Model...")
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Phone Stream URL
PHONE_URL = "http://192.168.2.141:8080/video"

# We named jpegenc as 'encoder' to measure its output bandwidth!
pipeline_str = (
    f"souphttpsrc location={PHONE_URL} ! multipartdemux ! image/jpeg ! jpegdec ! "
    f"videoconvert ! video/x-raw,format=BGR ! "
    f"identity name=roi_injector ! "
    f"videoconvert ! jpegenc name=encoder ! multipartmux ! tcpserversink host=0.0.0.0 port=5000"
)

pipeline = Gst.parse_launch(pipeline_str)

# --- AI REGION OF INTEREST PROBE ---
def pad_probe_callback(pad, info):
    buffer = info.get_buffer()
    if buffer:
        success, map_info = buffer.map(Gst.MapFlags.READ | Gst.MapFlags.WRITE)
        if success:
            caps = pad.get_current_caps()
            struct = caps.get_structure(0)
            width = struct.get_value('width')
            height = struct.get_value('height')
            
            image_array = np.ndarray(
                shape=(height, width, 3),
                dtype=np.uint8,
                buffer=map_info.data
            )
            
            # CPU Face Detection
            gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
            
            # Made it much more accurate (scaleFactor 1.1)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            mask = np.zeros(image_array.shape, dtype=np.uint8)
            has_face = False
            
            for (x, y, w, h) in faces:
                has_face = True
                
                # 1. Expand the ROI to include the whole head
                x1 = max(0, x: 40)
                y1 = max(0, y: 40)
                w1 = min(width: x1, w + 80)
                h1 = min(height: y1, h + 80)
                
                # 2. Copy the face from the original image into the black mask
                mask[y1:y1+h1, x1:x1+w1] = image_array[y1:y1+h1, x1:x1+w1]
                
                # 3. Draw a bright green bounding box around the face!
                cv2.rectangle(mask, (x, y), (x+w, y+h), (0, 255, 0), 4)
            
            # If a face was detected, overwrite the frame with the blacked-out background
            if has_face:
                np.copyto(image_array, mask)
            
            buffer.unmap(map_info)
    return Gst.PadProbeReturn.OK

# --- BANDWIDTH MONITOR PROBE ---
def bandwidth_probe_callback(pad, info):
    buffer = info.get_buffer()
    if buffer:
        # Get compressed size in Kilobytes
        size_kb = buffer.get_size() / 1024.0
        print(f"Network Bandwidth: {size_kb:.2f} KB/frame")
    return Gst.PadProbeReturn.OK

# Attach AI Probe
roi_injector = pipeline.get_by_name("roi_injector")
src_pad = roi_injector.get_static_pad("src")
src_pad.add_probe(Gst.PadProbeType.BUFFER, pad_probe_callback)

# Attach Bandwidth Probe
encoder = pipeline.get_by_name("encoder")
enc_pad = encoder.get_static_pad("src")
enc_pad.add_probe(Gst.PadProbeType.BUFFER, bandwidth_probe_callback)

print("Starting HTTP MJPEG AI Stream...")
print("Checking Bandwidth Reduction in Real-time!")
pipeline.set_state(Gst.State.PLAYING)

try:
    loop = GLib.MainLoop()
    loop.run()
except KeyboardInterrupt:
    pass

pipeline.set_state(Gst.State.NULL)
print("Pipeline stopped.")