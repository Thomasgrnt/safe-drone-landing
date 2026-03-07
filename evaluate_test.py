from ultralytics import YOLO
import cv2
import json
import pandas as pd
from pathlib import Path

# Charger modèle
model = YOLO("runs/detect/landingpad_model/weights/best.pt")

# Charger seuils
with open("thresholds.json") as f:
    thresholds = json.load(f)

tau_conf = thresholds["tau_conf"]
tau_area = thresholds["tau_area"]

test_folder = Path("dataset/images/test")

results = []

for img_path in test_folder.glob("*"):

    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]
    image_area = h * w

    detections = model.predict(source=img, conf=0.01, verbose=False)

    boxes = detections[0].boxes

    if boxes is None or len(boxes) == 0:
        decision = "NOT SAFE"
        max_conf = 0
        area_ratio = 0

    else:
        confs = boxes.conf.cpu().numpy()
        xyxy = boxes.xyxy.cpu().numpy()

        best_idx = confs.argmax()
        max_conf = confs[best_idx]

        x1, y1, x2, y2 = xyxy[best_idx]
        box_area = (x2-x1)*(y2-y1)
        area_ratio = box_area / image_area

        if max_conf >= tau_conf and area_ratio >= tau_area:
            decision = "SAFE"
        else:
            decision = "NOT SAFE"

    results.append({
        "image": img_path.name,
        "confidence": max_conf,
        "area_ratio": area_ratio,
        "decision": decision
    })


df = pd.DataFrame(results)

print(df)

df.to_csv("decision_results.csv", index=False)
