import os
from ultralytics import YOLO

def main():
    print("Loading YOLOv8-Nano model...")
    model = YOLO("yolov8n.pt")  # downloads if not present
    
    print("Exporting to ONNX format (opset 12)...")
    # Export for OpenCV DNN: opset 12 is widely supported, imgsz=320 keeps it fast on edge devices
    # dynamic=False creates fixed-size inputs which are generally safer for cv2.dnn
    export_path = model.export(format="onnx", opset=12, imgsz=320, dynamic=False)
    
    print(f"Export complete: {export_path}")

if __name__ == "__main__":
    main()
