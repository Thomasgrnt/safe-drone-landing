from ultralytics import YOLO

# Charger modèle
model = YOLO("runs/detect/landingpad_model/weights/best.pt")

# Evaluation sur test
results = model.val(
    data="landingpad.yaml",
    split="test",
    imgsz=640
)

print(results)