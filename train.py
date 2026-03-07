from ultralytics import YOLO

model = YOLO("yolov8n.pt")


model.train(
    data="landingpad.yaml", 
    epochs=50,
    imgsz=640,
    batch=16,
    name="landingpad_model"
)

print("Training finished.")