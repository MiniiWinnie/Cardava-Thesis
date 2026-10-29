from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from ultralytics import YOLO

PROJECT_ROOT = Path(r"C:\Users\saiar\Pictures\cardava_thesis")
DATASET_DIR = PROJECT_ROOT / "dataset" / "stage1_cls"
FINAL_RUN_DIR = PROJECT_ROOT / "runs" / "stage1_final"
CHECKPOINT = FINAL_RUN_DIR / "weights" / "best.pt"
RESULTS_DIR = FINAL_RUN_DIR / "test_evaluation"
IMAGE_SIZE = 224
BATCH_SIZE = 16
DEVICE = "cuda:0"

TEST_TRANSFORM = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.ToTensor(),
])


def extract_logits(output):
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)):
        for item in output:
            if isinstance(item, torch.Tensor):
                return item
    if isinstance(output, dict):
        for key in ("logits", "output", "pred", "preds"):
            value = output.get(key)
            if isinstance(value, torch.Tensor):
                return value
    raise RuntimeError(f"Unable to extract logits from {type(output)}")


def main():
    print("=" * 70)
    print("FINAL STAGE 1 HELD-OUT TEST EVALUATION")
    print("=" * 70)
    print(f"PyTorch Version : {torch.__version__}")
    print(f"CUDA Available  : {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")
    print(f"GPU             : {torch.cuda.get_device_name(0)}")
    print(f"Device          : {DEVICE}")
    print("Training        : NO")
    print("Hyperparameters : LOCKED (Trial 04)")
    print("Test set use    : FINAL EVALUATION")
    print()

    test_dir = DATASET_DIR / "test"
    if not test_dir.exists():
        raise FileNotFoundError(f"Test directory not found:\n{test_dir}")
    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"Final best.pt not found:\n{CHECKPOINT}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    dataset = ImageFolder(str(test_dir), transform=TEST_TRANSFORM)
    expected_classes = ["Healthy", "Unhealthy"]
    if dataset.classes != expected_classes:
        raise RuntimeError(f"Expected classes {expected_classes}, found {dataset.classes}")
    if len(dataset) != 225:
        raise RuntimeError(f"Expected 225 test images, found {len(dataset)}")

    healthy_idx = dataset.class_to_idx["Healthy"]
    unhealthy_idx = dataset.class_to_idx["Unhealthy"]
    healthy_count = sum(y == healthy_idx for _, y in dataset.samples)
    unhealthy_count = sum(y == unhealthy_idx for _, y in dataset.samples)
    if (healthy_count, unhealthy_count) != (150, 75):
        raise RuntimeError(
            f"Expected 150 Healthy and 75 Unhealthy; found {healthy_count} and {unhealthy_count}"
        )

    print("[PASS] Test set: 225 images (150 Healthy, 75 Unhealthy)")
    print("[PASS] Transform: Resize(224,224) -> ToTensor()")
    print("[PASS] No test augmentation")
    print(f"[PASS] Checkpoint: {CHECKPOINT}")
    print()

    model = YOLO(str(CHECKPOINT))
    names = model.names
    class_names = [names[i] for i in sorted(names)] if isinstance(names, dict) else list(names)
    if class_names != expected_classes:
        raise RuntimeError(f"Unexpected checkpoint classes: {class_names}")

    net = model.model.to(DEVICE)
    net.eval()

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    y_true = []
    y_pred = []
    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(DEVICE, non_blocking=True)
            output = extract_logits(net(images))
            if output.ndim != 2 or output.shape[1] != 2:
                raise RuntimeError(f"Unexpected output shape: {tuple(output.shape)}")
            preds = output.argmax(dim=1).cpu().numpy().tolist()
            y_pred.extend(preds)
            y_true.extend(labels.numpy().tolist())

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    accuracy = accuracy_score(y_true, y_pred)
    macro_precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    weighted_recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    healthy_precision, unhealthy_precision = precision_score(y_true, y_pred, labels=[0,1], average=None, zero_division=0)
    healthy_recall, unhealthy_recall = recall_score(y_true, y_pred, labels=[0,1], average=None, zero_division=0)
    healthy_f1, unhealthy_f1 = f1_score(y_true, y_pred, labels=[0,1], average=None, zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = map(int, cm.ravel())

    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )

    print("=" * 70)
    print("FINAL STAGE 1 TEST RESULTS")
    print("=" * 70)
    print(f"Accuracy          : {accuracy:.4f}")
    print(f"Macro Precision   : {macro_precision:.4f}")
    print(f"Macro Recall      : {macro_recall:.4f}")
    print(f"Macro F1          : {macro_f1:.4f}")
    print()
    print("Healthy:")
    print(f"  Precision       : {healthy_precision:.4f}")
    print(f"  Recall          : {healthy_recall:.4f}")
    print(f"  F1              : {healthy_f1:.4f}")
    print()
    print("Unhealthy:")
    print(f"  Precision       : {unhealthy_precision:.4f}")
    print(f"  Recall          : {unhealthy_recall:.4f}")
    print(f"  F1              : {unhealthy_f1:.4f}")
    print()
    print("Confusion Matrix:")
    print(f"  TN = {tn}")
    print(f"  FP = {fp}")
    print(f"  FN = {fn}")
    print(f"  TP = {tp}")
    print()
    print(report)

    metrics_df = pd.DataFrame([
        ["accuracy", accuracy],
        ["macro_precision", macro_precision],
        ["macro_recall", macro_recall],
        ["macro_f1", macro_f1],
        ["weighted_precision", weighted_precision],
        ["weighted_recall", weighted_recall],
        ["weighted_f1", weighted_f1],
        ["healthy_precision", healthy_precision],
        ["healthy_recall", healthy_recall],
        ["healthy_f1", healthy_f1],
        ["unhealthy_precision", unhealthy_precision],
        ["unhealthy_recall", unhealthy_recall],
        ["unhealthy_f1", unhealthy_f1],
    ], columns=["metric", "value"])
    metrics_df.to_csv(RESULTS_DIR / "stage1_test_metrics.csv", index=False)

    cm_df = pd.DataFrame(
        [[tn, fp], [fn, tp]],
        index=["Actual Healthy", "Actual Unhealthy"],
        columns=["Predicted Healthy", "Predicted Unhealthy"],
    )
    cm_df.to_csv(RESULTS_DIR / "stage1_test_confusion_matrix.csv")

    with open(RESULTS_DIR / "stage1_test_classification_report.txt", "w", encoding="utf-8") as f:
        f.write("FINAL STAGE 1 HELD-OUT TEST CLASSIFICATION REPORT\n")
        f.write("==============================================\n\n")
        f.write(f"Checkpoint: {CHECKPOINT}\n")
        f.write("Test set: 225 images (150 Healthy, 75 Unhealthy)\n")
        f.write("Preprocessing: Resize(224,224) -> ToTensor()\n")
        f.write("Augmentation: None\n\n")
        f.write(report)

    print(f"[SAVED] {RESULTS_DIR / 'stage1_test_metrics.csv'}")
    print(f"[SAVED] {RESULTS_DIR / 'stage1_test_confusion_matrix.csv'}")
    print(f"[SAVED] {RESULTS_DIR / 'stage1_test_classification_report.txt'}")
    print()
    print("FINAL STAGE 1 HELD-OUT TEST EVALUATION COMPLETE")


if __name__ == "__main__":
    main()
