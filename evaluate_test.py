import json
from pathlib import Path

import cv2
import pandas as pd
from ultralytics import YOLO

THRESHOLDS_PATH = "thresholds.json"
DATA_YAML = "landingpad.yaml"
TEST_IMAGES = Path("dataset/images/test")
TEST_LABELS = Path("dataset/labels/test")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_best_model() -> str:
    candidates = sorted(Path(".").glob("**/landingpad_model/weights/best.pt"))
    if not candidates:
        raise FileNotFoundError("Could not find any best.pt for landingpad_model")
    model_path = str(candidates[-1])
    print(f"Using model: {model_path}")
    return model_path


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

    best_idx = int(confs.argmax())
    best_conf = float(confs[best_idx])

    x1, y1, x2, y2 = xyxy[best_idx]
    box_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_ratio = float(box_area / image_area)

    return best_conf, area_ratio


def main():
    if not Path(THRESHOLDS_PATH).exists():
        raise FileNotFoundError(f"Missing thresholds: {THRESHOLDS_PATH}")

    with open(THRESHOLDS_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    tau_conf = float(cfg["tau_conf"])
    tau_area = float(cfg["tau_area"])

    model_path = find_best_model()
    model = YOLO(model_path)

    print("=== 1) Standard YOLO detection metrics on TEST ===")
    metrics = model.val(data=DATA_YAML, split="test", imgsz=640, verbose=False)

    metric_table = pd.DataFrame({
        "Metric": ["Precision", "Recall", "mAP@50", "mAP@50-95"],
        "Value": [
            float(metrics.box.mp),
            float(metrics.box.mr),
            float(metrics.box.map50),
            float(metrics.box.map),
        ],
    })
    print(metric_table)
    metric_table.to_csv("test_detection_metrics.csv", index=False)

    print("\n=== 2) SAFE / NOT_SAFE decision evaluation on TEST ===")
    image_paths = sorted([p for p in TEST_IMAGES.iterdir() if p.suffix.lower() in IMAGE_EXTS])

    rows = []
    tp = fp = tn = fn = 0

    for img_path in image_paths:
        gt_safe = has_ground_truth_pad(TEST_LABELS / f"{img_path.stem}.txt")
        max_conf, area_ratio = get_best_detection(model, img_path)
        pred_safe = (max_conf >= tau_conf) and (area_ratio >= tau_area)

        if pred_safe and gt_safe:
            tp += 1
        elif pred_safe and not gt_safe:
            fp += 1
        elif not pred_safe and not gt_safe:
            tn += 1
        else:
            fn += 1

        rows.append({
            "image": img_path.name,
            "gt": "SAFE" if gt_safe else "NOT_SAFE",
            "pred": "SAFE" if pred_safe else "NOT_SAFE",
            "max_conf": round(max_conf, 4),
            "area_ratio": round(area_ratio, 4),
            "tau_conf": tau_conf,
            "tau_area": tau_area,
        })

    df = pd.DataFrame(rows)
    print(df)
    df.to_csv("test_decision_results.csv", index=False)

    summary = pd.DataFrame({
        "Metric": ["TP", "FP", "TN", "FN", "Decision Precision", "Decision Recall", "Decision Accuracy"],
        "Value": [
            tp,
            fp,
            tn,
            fn,
            round(tp / (tp + fp), 4) if (tp + fp) else 0.0,
            round(tp / (tp + fn), 4) if (tp + fn) else 0.0,
            round((tp + tn) / max(1, tp + tn + fp + fn), 4),
        ]
    })

    print("\nDecision summary:")
    print(summary)
    summary.to_csv("test_decision_summary.csv", index=False)

if __name__ == "__main__":
    main()
