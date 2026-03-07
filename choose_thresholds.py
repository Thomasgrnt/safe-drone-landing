import json
from ultralytics import YOLO


model = YOLO("runs/detect/landingpad_model/weights/best.pt")

tau_conf = 0.5
tau_area = 0.05

thresholds = {
    "tau_conf": tau_conf,
    "tau_area": tau_area
}

# Sauvegarde
with open("thresholds.json", "w") as f:
    json.dump(thresholds, f)

print("Thresholds saved:", thresholds)