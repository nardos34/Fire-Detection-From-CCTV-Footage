from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="yolo_model_config.yaml",
    epochs=50,
    imgsz=640,
    batch=16
)