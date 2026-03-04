from ultralytics import YOLO

# Load a lightweight YOLOv8 segmentation model
model = YOLO('yolov8n-seg.pt')

# Export to ONNX
success = model.export(format='onnx', opset=17, simplify=True)
print(f"Export successful: {success}")
