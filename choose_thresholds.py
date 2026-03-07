import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

MODEL_PATH = "runs/detect/landingpad_model/weights/best.pt"
VAL_IMAGES = Path("dataset/images/val")
VAL_LABELS = Path("dataset/labels/val")

OUTPUT_JSON = "thresholds.json"
OUTPUT_CSV = "threshold_search.csv"

CONF_GRID = np.arange(0.30, 0.71, 0.05)
AREA_GRID = np.arange(0.01, 0.101, 0.01)

FALSE_SAFE_COST = 5.0
FALSE_NOT_SAFE_COST = 1.0

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def has_ground_truth_pad(label_path: Path) -> bool:
    if not label_path.exists():
        return False
    lines = [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return len(lines) > 0


def get_best_detection(model: YOLO, image_path: Path):
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")

    h, w = img.shape[:2]
    image_area = float(h * w)

    results = model.predict(source=str(image_path), conf=0.001, verbose=False)
    result = results[0]

    if result.boxes is None or len(result.boxes) == 0:
        return 0.0, 0.0

    confs = result.boxes.conf.cpu().numpy()
    xyxy = result.boxes.xyxy.cpu().numpy()

    best_idx = int(np.argmax(confs))
    best_conf = float(confs[best_idx])

    x1, y1, x2, y2 = xyxy[best_idx]
    box_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_ratio = float(box_area / image_area)

    return best_conf, area_ratio


def decision(max_conf: float, area_ratio: float, tau_conf: float, tau_area: float) -> bool:
    return (max_conf >= tau_conf) and (area_ratio >= tau_area)


def main():
    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(f"Missing model: {MODEL_PATH}")

    model = YOLO(MODEL_PATH)

    image_paths = sorted([p for p in VAL_IMAGES.iterdir() if p.suffix.lower() in IMAGE_EXTS])
    if not image_paths:
        raise RuntimeError(f"No validation images found in {VAL_IMAGES}")

    samples = []
    for img_path in image_paths:
        gt_safe = has_ground_truth_pad(VAL_LABELS / f"{img_path.stem}.txt")
        max_conf, area_ratio = get_best_detection(model, img_path)
        samples.append({
            "image": img_path.name,
            "gt_safe": gt_safe,
            "max_conf": max_conf,
            "area_ratio": area_ratio,
        })

    rows = []
    best = None
    best_cost = float("inf")

    for tau_conf in CONF_GRID:
        for tau_area in AREA_GRID:
            tp = fp = tn = fn = 0

            for s in samples:
                pred_safe = decision(s["max_conf"], s["area_ratio"], tau_conf, tau_area)
                gt_safe = s["gt_safe"]

                if pred_safe and gt_safe:
                    tp += 1
                elif pred_safe and not gt_safe:
                    fp += 1
                elif not pred_safe and not gt_safe:
                    tn += 1
                else:
                    fn += 1

            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            accuracy = (tp + tn) / max(1, tp + tn + fp + fn)
            cost = FALSE_SAFE_COST * fp + FALSE_NOT_SAFE_COST * fn

            row = {
                "tau_conf": round(float(tau_conf), 3),
                "tau_area": round(float(tau_area), 3),
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
                "precision": round(float(precision), 4),
                "recall": round(float(recall), 4),
                "accuracy": round(float(accuracy), 4),
                "cost": round(float(cost), 4),
            }
            rows.append(row)

            if cost < best_cost:
                best_cost = cost
                best = row

    pd.DataFrame(rows).to_csv(OUTPUT_CSV, index=False)

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2)

    print("Best thresholds saved to thresholds.json")
    print(json.dumps(best, indent=2))
    print(f"Full grid search saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
