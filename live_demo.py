import csv
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

MODEL_PATH = "best.pt"
THRESHOLDS_PATH = "thresholds.json"
LOG_PATH = Path("live_demo_log.csv")

CAMERA_INDEX = 0

GOOD_COLOR = (0, 180, 0)
BAD_COLOR = (0, 0, 255)
BOX_COLOR = (0, 255, 0)
TEXT_COLOR = (255, 255, 255)


def ensure_log_exists():
    if not LOG_PATH.exists():
        with open(LOG_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp",
                "scenario_id",
                "max_conf",
                "area_ratio",
                "tau_conf",
                "tau_area",
                "decision",
                "notes",
            ])


def load_thresholds():
    with open(THRESHOLDS_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return float(cfg["tau_conf"]), float(cfg["tau_area"])


def get_best_detection(result, frame_shape):
    h, w = frame_shape[:2]
    image_area = float(h * w)

    if result.boxes is None or len(result.boxes) == 0:
        return None, 0.0, 0.0

    confs = result.boxes.conf.cpu().numpy()
    xyxy = result.boxes.xyxy.cpu().numpy()

    best_idx = int(np.argmax(confs))
    best_conf = float(confs[best_idx])

    x1, y1, x2, y2 = map(int, xyxy[best_idx])
    area_ratio = max(0, x2 - x1) * max(0, y2 - y1) / image_area

    return (x1, y1, x2, y2), best_conf, float(area_ratio)


def append_log(scenario_id, max_conf, area_ratio, tau_conf, tau_area, decision, notes):
    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().isoformat(timespec="seconds"),
            scenario_id,
            f"{max_conf:.4f}",
            f"{area_ratio:.4f}",
            f"{tau_conf:.4f}",
            f"{tau_area:.4f}",
            decision,
            notes,
        ])


def main():
    ensure_log_exists()
    tau_conf, tau_area = load_thresholds()

    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(f"Missing model: {MODEL_PATH}")

    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    scenario_id = "live_scene_001"
    notes = ""

    print("Controls:")
    print(" q = quit")
    print(" s = save current frame result to CSV")
    print(" r = change scenario id / notes")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = model.predict(source=frame, conf=0.001, verbose=False)
        result = results[0]

        best_box, max_conf, area_ratio = get_best_detection(result, frame.shape)
        safe = (max_conf >= tau_conf) and (area_ratio >= tau_area)
        decision_text = "SAFE" if safe else "NOT_SAFE"

        if best_box is not None:
            x1, y1, x2, y2 = best_box
            cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, 2)
            cv2.putText(
                frame,
                f"landing_pad {max_conf:.2f} area={area_ratio:.3f}",
                (x1, max(25, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                BOX_COLOR,
                2
            )

        color = GOOD_COLOR if safe else BAD_COLOR
        cv2.putText(frame, f"Decision: {decision_text}", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
        cv2.putText(frame, f"max_conf={max_conf:.3f}", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, TEXT_COLOR, 2)
        cv2.putText(frame, f"area_ratio={area_ratio:.3f}", (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, TEXT_COLOR, 2)
        cv2.putText(frame, f"tau_conf={tau_conf:.3f} tau_area={tau_area:.3f}", (20, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, TEXT_COLOR, 2)
        cv2.putText(frame, f"scenario_id={scenario_id}", (20, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, TEXT_COLOR, 2)

        cv2.imshow("Safe Drone Landing - Live Demo", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("s"):
            append_log(
                scenario_id=scenario_id,
                max_conf=max_conf,
                area_ratio=area_ratio,
                tau_conf=tau_conf,
                tau_area=tau_area,
                decision=decision_text,
                notes=notes,
            )
            print("Saved one line in live_demo_log.csv")
        elif key == ord("r"):
            new_id = input("New scenario_id: ").strip()
            if new_id:
                scenario_id = new_id
            notes = input("Notes (optional): ").strip()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
