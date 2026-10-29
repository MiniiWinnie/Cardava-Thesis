from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from ultralytics import YOLO

PROJECT_ROOT = Path(r"C:\Users\saiar\Pictures\cardava_thesis")
DATASET_DIR = PROJECT_ROOT / "dataset" / "stage1_cls"
SEARCH_DIR = PROJECT_ROOT / "runs" / "stage1_hparam_search_final"
SEARCH_RESULTS = SEARCH_DIR / "stage1_hparam_results.csv"
CORRECTED_RESULTS = SEARCH_DIR / "stage1_hparam_corrected_validation.csv"
CORRECTED_RANKING = SEARCH_DIR / "stage1_hparam_corrected_ranking.csv"
CONFUSION_DIR = SEARCH_DIR / "corrected_confusion_matrices"
IMAGE_SIZE = 224
BATCH_SIZE = 16
DEVICE = "cuda:0"

# EXACT validation preprocessing from train_stage1.py:
# Resize((224,224)) -> ToTensor(); no augmentation.
VAL_TRANSFORM = T.Compose([T.Resize((IMAGE_SIZE, IMAGE_SIZE)), T.ToTensor()])


def find_checkpoint(trial_number):
    candidates = []
    for d in SEARCH_DIR.glob(f"trial_{trial_number:02d}*"):
        if d.is_dir() and (d / "results.csv").exists() and (d / "weights" / "best.pt").exists():
            candidates.append(d)
    if not candidates:
        raise FileNotFoundError(f"No completed checkpoint found for Trial {trial_number:02d}.")
    candidates.sort(key=lambda d: (d / "weights" / "best.pt").stat().st_mtime, reverse=True)
    chosen = candidates[0]
    return chosen, chosen / "weights" / "best.pt"


def extract_logits(output):
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)):
        for x in output:
            if isinstance(x, torch.Tensor):
                return x
    if isinstance(output, dict):
        for k in ("logits", "output", "pred", "preds"):
            if isinstance(output.get(k), torch.Tensor):
                return output[k]
    raise RuntimeError(f"Could not extract logits from output type: {type(output)}")


def check_environment_and_inputs():
    print("=" * 70)
    print("STAGE 1 CORRECTED VALIDATION (EVALUATION ONLY)")
    print("=" * 70)
    print(f"PyTorch Version : {torch.__version__}")
    print(f"CUDA Available  : {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")
    print(f"GPU             : {torch.cuda.get_device_name(0)}")
    print(f"Device          : {DEVICE}")
    print("Training        : NO")
    print("Test set        : NOT USED")
    print()

    val_dir = DATASET_DIR / "val"
    if not val_dir.exists():
        raise FileNotFoundError(f"Validation directory not found: {val_dir}")
    if not SEARCH_RESULTS.exists():
        raise FileNotFoundError(f"Search results not found: {SEARCH_RESULTS}")
    CONFUSION_DIR.mkdir(parents=True, exist_ok=True)


def load_validation_dataset():
    dataset = ImageFolder(str(DATASET_DIR / "val"), transform=VAL_TRANSFORM)
    if dataset.classes != ["Healthy", "Unhealthy"]:
        raise RuntimeError(f"Unexpected class order: {dataset.classes}")
    if len(dataset) != 225:
        raise RuntimeError(f"Expected 225 validation images, found {len(dataset)}")
    counts = np.bincount([label for _, label in dataset.samples], minlength=2)
    if counts.tolist() != [150, 75]:
        raise RuntimeError(f"Expected [150,75] class counts, found {counts.tolist()}")
    print("[PASS] Validation set: 225 images (150 Healthy, 75 Unhealthy)")
    print("[PASS] Transform: Resize(224,224) -> ToTensor()")
    print("[PASS] No validation augmentation")
    print()
    return dataset


def evaluate_checkpoint(checkpoint_path, dataset):
    model = YOLO(str(checkpoint_path))
    names = model.names
    class_names = [names[i] for i in sorted(names)] if isinstance(names, dict) else list(names)
    if class_names != ["Healthy", "Unhealthy"]:
        raise RuntimeError(f"Unexpected checkpoint classes: {class_names}")

    net = model.model.to(DEVICE)
    net.eval()
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    y_true, y_pred = [], []
    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(DEVICE, non_blocking=True)
            out = extract_logits(net(images))
            if out.ndim != 2 or out.shape[1] != 2:
                raise RuntimeError(f"Unexpected logits shape: {tuple(out.shape)}")
            preds = torch.argmax(out, dim=1).cpu().numpy()
            y_true.extend(labels.numpy().tolist())
            y_pred.extend(preds.tolist())

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    return {
        "corrected_val_accuracy": float(accuracy_score(y_true, y_pred)),
        "healthy_precision": float(precision_score(y_true, y_pred, labels=[0,1], average=None, zero_division=0)[0]),
        "healthy_recall": float(recall_score(y_true, y_pred, labels=[0,1], average=None, zero_division=0)[0]),
        "healthy_f1": float(f1_score(y_true, y_pred, labels=[0,1], average=None, zero_division=0)[0]),
        "unhealthy_precision": float(precision_score(y_true, y_pred, labels=[0,1], average=None, zero_division=0)[1]),
        "unhealthy_recall": float(recall_score(y_true, y_pred, labels=[0,1], average=None, zero_division=0)[1]),
        "unhealthy_f1": float(f1_score(y_true, y_pred, labels=[0,1], average=None, zero_division=0)[1]),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "tn": int(cm[0,0]), "fp": int(cm[0,1]), "fn": int(cm[1,0]), "tp": int(cm[1,1]),
    }


def main():
    check_environment_and_inputs()
    search_df = pd.read_csv(SEARCH_RESULTS).sort_values("trial").reset_index(drop=True)
    if len(search_df) != 4:
        raise RuntimeError(f"Expected 4 trials, found {len(search_df)}")

    dataset = load_validation_dataset()
    records = []

    for _, row in search_df.iterrows():
        trial = int(row["trial"])
        out_dir, checkpoint = find_checkpoint(trial)
        print("=" * 70)
        print(f"TRIAL {trial:02d} - EXACT VALIDATION PREPROCESSING")
        print("=" * 70)
        print(f"Checkpoint: {checkpoint}")
        metrics = evaluate_checkpoint(checkpoint, dataset)
        record = row.to_dict()
        record.update({"actual_output_folder": str(out_dir), **metrics})
        records.append(record)
        print(f"Corrected accuracy      : {metrics['corrected_val_accuracy']:.4f}")
        print(f"Healthy  P/R/F1         : {metrics['healthy_precision']:.4f} / {metrics['healthy_recall']:.4f} / {metrics['healthy_f1']:.4f}")
        print(f"Unhealthy P/R/F1        : {metrics['unhealthy_precision']:.4f} / {metrics['unhealthy_recall']:.4f} / {metrics['unhealthy_f1']:.4f}")
        print(f"Macro F1                : {metrics['macro_f1']:.4f}")
        print(f"Confusion matrix        : TN={metrics['tn']} FP={metrics['fp']} FN={metrics['fn']} TP={metrics['tp']}")
        pd.DataFrame([[metrics['tn'], metrics['fp']], [metrics['fn'], metrics['tp']]],
                     index=["Actual Healthy", "Actual Unhealthy"],
                     columns=["Predicted Healthy", "Predicted Unhealthy"]).to_csv(
                         CONFUSION_DIR / f"trial_{trial:02d}_confusion_matrix.csv")
        torch.cuda.empty_cache()
        print()

    df = pd.DataFrame(records)
    df.to_csv(CORRECTED_RESULTS, index=False)

    # Primary metric: best validation accuracy recorded by the official custom validator.
    # Tie-breaker: lowest validation loss at its best epoch.
    # Further tie-breakers: corrected Unhealthy F1, then corrected Unhealthy recall.
    ranking = df.sort_values(
        by=["best_val_accuracy", "lowest_val_loss", "unhealthy_f1", "unhealthy_recall"],
        ascending=[False, True, False, False],
    ).reset_index(drop=True)
    ranking["selection_rank"] = ranking.index + 1
    ranking.to_csv(CORRECTED_RANKING, index=False)

    print("=" * 70)
    print("CORRECTED VALIDATION SUMMARY")
    print("=" * 70)
    print(ranking[["selection_rank","trial","learning_rate","batch_size","weight_decay","dropout",
                   "best_val_accuracy","best_val_accuracy_epoch","lowest_val_loss",
                   "corrected_val_accuracy","unhealthy_f1","unhealthy_recall","tn","fp","fn","tp"]].to_string(index=False))

    best = ranking.iloc[0]
    print()
    print("=" * 70)
    print("CURRENT BEST VALIDATION CONFIGURATION")
    print("=" * 70)
    print(f"Trial         : {int(best['trial'])}")
    print(f"Learning Rate : {best['learning_rate']}")
    print(f"Batch Size    : {int(best['batch_size'])}")
    print(f"Weight Decay  : {best['weight_decay']}")
    print(f"Dropout       : {best['dropout']}")
    print(f"Best Val Acc  : {best['best_val_accuracy']:.4f}")
    print(f"Best Val Epoch: {int(best['best_val_accuracy_epoch'])}")
    print(f"Lowest Val Loss: {best['lowest_val_loss']:.6f}")
    print(f"Exact Val Acc : {best['corrected_val_accuracy']:.4f}")
    print()
    print(f"[SAVED] {CORRECTED_RESULTS}")
    print(f"[SAVED] {CORRECTED_RANKING}")
    print(f"[SAVED] {CONFUSION_DIR}")
    print("[IMPORTANT] Test set was not loaded.")


if __name__ == "__main__":
    main()
