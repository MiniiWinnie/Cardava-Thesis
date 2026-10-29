from pathlib import Path
import json
import pandas as pd
from ultralytics import YOLO

PROJECT_ROOT = Path(r"C:\Users\saiar\Pictures\cardava_thesis")
DATASET_DIR = PROJECT_ROOT / "dataset" / "stage2_det"
DATA_YAML = DATASET_DIR / "data.yaml"
FINAL_RUN_DIR = PROJECT_ROOT / "runs" / "stage2_final"
CHECKPOINT = FINAL_RUN_DIR / "weights" / "best.pt"
OUT_DIR = FINAL_RUN_DIR / "test_evaluation"
METRICS_CSV = OUT_DIR / "stage2_test_metrics.csv"
RAW_JSON = OUT_DIR / "stage2_test_results_raw.json"
DEVICE = "cuda:0"
IMAGE_SIZE = 640
BATCH_SIZE = 4


def verify_test_set():
    images = DATASET_DIR / "images" / "test"
    labels = DATASET_DIR / "labels" / "test"
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_files = [p for p in images.iterdir() if p.is_file() and p.suffix.lower() in exts]
    label_files = [p for p in labels.iterdir() if p.is_file() and p.suffix.lower() == ".txt"]
    if len(image_files) != 150 or len(label_files) != 150:
        raise ValueError(f"Expected 150 test images/labels, found {len(image_files)}/{len(label_files)}")
    bad = []
    for img in image_files:
        lab = labels / f"{img.stem}.txt"
        if not lab.exists():
            bad.append((img.name, "missing label"))
            continue
        lines = [x.strip() for x in lab.read_text(encoding="utf-8").splitlines() if x.strip()]
        if len(lines) != 1:
            bad.append((img.name, f"expected 1 box, found {len(lines)}"))
            continue
        parts = lines[0].split()
        if len(parts) != 5:
            bad.append((img.name, "expected 5 YOLO values"))
            continue
        try:
            cls = int(parts[0]); vals = [float(v) for v in parts[1:]]
        except ValueError:
            bad.append((img.name, "non-numeric annotation")); continue
        if cls != 0:
            bad.append((img.name, f"unexpected class {cls}")); continue
        if not all(0.0 <= v <= 1.0 for v in vals) or vals[2] <= 0 or vals[3] <= 0:
            bad.append((img.name, "invalid normalized bbox"))
    if bad:
        raise ValueError("Invalid test annotations:\n" + "\n".join(f"{n}: {r}" for n, r in bad[:20]))
    print("[PASS] Test set: 150 images / 150 labels")
    print("[PASS] Exactly one valid whole-bunch bounding box per image")


def main():
    print("=" * 70)
    print("FINAL STAGE 2 HELD-OUT TEST EVALUATION")
    print("=" * 70)
    print("Training        : NO")
    print("Hyperparameters : LOCKED (Trial 02)")
    print("Test set use    : FINAL EVALUATION")
    print(f"PyTorch/Device  : reported by Ultralytics ({DEVICE})")
    print()

    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"Final checkpoint not found:\n{CHECKPOINT}")
    if not DATA_YAML.exists():
        raise FileNotFoundError(f"data.yaml not found:\n{DATA_YAML}")

    verify_test_set()
    print(f"[PASS] Final checkpoint: {CHECKPOINT}")
    print("[PASS] Test preprocessing: LetterBox 640x640, aspect ratio preserved, no augmentation")
    print()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(CHECKPOINT))
    print(f"[PASS] Model classes: {model.names}")
    print()

    print("=" * 70)
    print("RUNNING FINAL STAGE 2 TEST SET")
    print("=" * 70)

    metrics = model.val(
        data=str(DATA_YAML),
        split="test",
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        workers=0,
        augment=False,
        plots=True,
        verbose=True,
        project=str(OUT_DIR),
        name="detector_metrics",
        exist_ok=False,
    )

    box = metrics.box
    map50_95 = float(box.map)
    map50 = float(box.map50)
    precision = float(box.mp)
    recall = float(box.mr)

    result = {
        "test_images": 150,
        "mAP50-95": map50_95,
        "mAP50": map50,
        "precision": precision,
        "recall": recall,
        "checkpoint": str(CHECKPOINT),
        "split": "test",
        "imgsz": IMAGE_SIZE,
        "batch": BATCH_SIZE,
        "augment": False,
    }

    pd.DataFrame([
        {"metric": "mAP50-95", "value": map50_95},
        {"metric": "mAP50", "value": map50},
        {"metric": "precision", "value": precision},
        {"metric": "recall", "value": recall},
    ]).to_csv(METRICS_CSV, index=False)

    RAW_JSON.write_text(json.dumps(result, indent=4), encoding="utf-8")

    print()
    print("=" * 70)
    print("FINAL STAGE 2 TEST RESULTS")
    print("=" * 70)
    print(f"Test Images       : 150")
    print(f"mAP@0.50:0.95     : {map50_95:.5f}")
    print(f"mAP@0.50          : {map50:.5f}")
    print(f"Precision          : {precision:.5f}")
    print(f"Recall             : {recall:.5f}")
    print()
    print(f"[SAVED] {METRICS_CSV}")
    print(f"[SAVED] {RAW_JSON}")
    print()
    print("[NEXT] Use the final test detections for whole-bunch pixel-length measurement and A/B/Rejected grading evaluation.")


if __name__ == "__main__":
    main()
