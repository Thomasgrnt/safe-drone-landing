from pathlib import Path
from ultralytics import YOLO

DATA_YAML = "landingpad.yaml"
MODEL_NAME = "yolo11n.pt"

IMG_SIZE = 640
EPOCHS = 50
BATCH = 16
PATIENCE = 15

def main():
    if not Path(DATA_YAML).exists():
        raise FileNotFoundError(f"Missing dataset config: {DATA_YAML}")

    model = YOLO(MODEL_NAME)

    model.train(
        data=DATA_YAML,
        imgsz=IMG_SIZE,
        epochs=EPOCHS,
        batch=BATCH,
        patience=PATIENCE,
        pretrained=True,
        name="landingpad_model",
        exist_ok=True,
        verbose=True,
    )

    print("\nTraining finished.")
    print("Search for best.pt with:")
    print("!find . -path '*landingpad_model/weights/best.pt'")

if __name__ == "__main__":
    main()
