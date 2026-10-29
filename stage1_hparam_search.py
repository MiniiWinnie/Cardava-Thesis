from pathlib import Path
import random
import time

import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

# Reuse the EXACT custom Stage 1 trainer from train_stage1.py
from train_stage1 import CustomClassificationTrainer


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(r"C:\Users\saiar\Pictures\cardava_thesis")
DATASET_DIR = PROJECT_ROOT / "dataset" / "stage1_cls"
MODEL_PATH = PROJECT_ROOT / "yolo11n-cls.pt"
RUNS_DIR = PROJECT_ROOT / "runs"
SEARCH_DIR = RUNS_DIR / "stage1_hparam_search_final"


# ============================================================
# FIXED EXPERIMENT SETTINGS
# ============================================================

NUMBER_OF_TRIALS = 4
MAX_EPOCHS = 50
IMAGE_SIZE = 224
DEVICE = "cuda:0"
RANDOM_SEED = 42


# ============================================================
# HYPERPARAMETER SEARCH RANGES
# ============================================================

# From Chapter 3:
# Learning rate : 0.0005 - 0.002
# Batch size    : 8 or 16
# Weight decay  : 0.0001 - 0.001
# Dropout       : 0.0 - 0.2

LR_MIN = 0.0005
LR_MAX = 0.002
BATCH_OPTIONS = [8, 16]
WD_MIN = 0.0001
WD_MAX = 0.001
DROPOUT_MIN = 0.0
DROPOUT_MAX = 0.2

search_rng = random.Random(RANDOM_SEED)


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_random_seeds():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(RANDOM_SEED)
        torch.cuda.manual_seed_all(RANDOM_SEED)


# ============================================================
# CREATE SEARCH FOLDER
# ============================================================

def create_search_folder():
    SEARCH_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# VERIFY DATASET
# ============================================================

def verify_dataset():
    train_dir = DATASET_DIR / "train"
    val_dir = DATASET_DIR / "val"
    test_dir = DATASET_DIR / "test"

    print("=" * 70)
    print("DATASET VERIFICATION")
    print("=" * 70)

    for directory in [train_dir, val_dir, test_dir]:
        if not directory.exists():
            raise FileNotFoundError(f"Missing dataset directory:\n{directory}")
        print(f"[PASS] {directory}")

    print()
    print("[PASS] Test set located but will NOT be used during hyperparameter search.")
    print()


# ============================================================
# GENERATE RANDOM TRIAL
# ============================================================

def generate_trial():
    config = {
        "learning_rate": round(search_rng.uniform(LR_MIN, LR_MAX), 6),
        "batch_size": search_rng.choice(BATCH_OPTIONS),
        "weight_decay": round(search_rng.uniform(WD_MIN, WD_MAX), 6),
        "dropout": round(search_rng.uniform(DROPOUT_MIN, DROPOUT_MAX), 6),
    }

    assert LR_MIN <= config["learning_rate"] <= LR_MAX
    assert config["batch_size"] in BATCH_OPTIONS
    assert WD_MIN <= config["weight_decay"] <= WD_MAX
    assert DROPOUT_MIN <= config["dropout"] <= DROPOUT_MAX

    return config


# ============================================================
# LOAD OR CREATE THE OFFICIAL TRIAL PLAN
# ============================================================

def load_or_create_trial_plan():
    """
    Reuse planned_trials.csv once it exists.
    This keeps the official four configurations unchanged on reruns.
    """
    planned_path = SEARCH_DIR / "planned_trials.csv"

    if planned_path.exists():
        planned_df = pd.read_csv(planned_path)
        required = {"trial", "learning_rate", "batch_size", "weight_decay", "dropout"}
        missing = required - set(planned_df.columns)
        if missing:
            raise ValueError(
                "Existing planned_trials.csv is missing columns: "
                f"{sorted(missing)}"
            )
        if len(planned_df) != NUMBER_OF_TRIALS:
            raise ValueError(
                f"Expected {NUMBER_OF_TRIALS} planned trials, found {len(planned_df)}."
            )

        planned_df = planned_df.sort_values("trial").reset_index(drop=True)
        trial_configs = []

        for _, row in planned_df.iterrows():
            config = {
                "learning_rate": float(row["learning_rate"]),
                "batch_size": int(row["batch_size"]),
                "weight_decay": float(row["weight_decay"]),
                "dropout": float(row["dropout"]),
            }
            assert LR_MIN <= config["learning_rate"] <= LR_MAX
            assert config["batch_size"] in BATCH_OPTIONS
            assert WD_MIN <= config["weight_decay"] <= WD_MAX
            assert DROPOUT_MIN <= config["dropout"] <= DROPOUT_MAX
            trial_configs.append((int(row["trial"]), config))

        print(f"[PASS] Existing official trial plan reused:\n{planned_path}\n")
        return trial_configs

    trial_configs = []
    for trial_number in range(1, NUMBER_OF_TRIALS + 1):
        trial_configs.append((trial_number, generate_trial()))

    planned_df = pd.DataFrame([
        {
            "trial": trial_number,
            "learning_rate": config["learning_rate"],
            "batch_size": config["batch_size"],
            "weight_decay": config["weight_decay"],
            "dropout": config["dropout"],
        }
        for trial_number, config in trial_configs
    ])
    planned_df.to_csv(planned_path, index=False)
    print(f"[SAVED] {planned_path}\n")
    return trial_configs


# ============================================================
# FIND A COMPLETED TRIAL FOLDER
# ============================================================

def find_completed_trial_dir(trial_number):
    """
    Detects a completed output folder for the logical trial number.

    This recovers the current Trial 01 in trial_01-2 without retraining it.
    The old trial_01 folder contains only the configuration file and is ignored
    because it has no results.csv/best.pt pair.
    """
    pattern = f"trial_{trial_number:02d}*"
    candidates = []

    for directory in SEARCH_DIR.glob(pattern):
        if not directory.is_dir():
            continue
        results_csv = directory / "results.csv"
        best_checkpoint = directory / "weights" / "best.pt"
        if results_csv.exists() and best_checkpoint.exists():
            candidates.append(directory)

    if not candidates:
        return None

    candidates.sort(
        key=lambda p: (p / "results.csv").stat().st_mtime,
        reverse=True,
    )
    return candidates[0]


# ============================================================
# SAVE TRIAL CONFIGURATION WITHOUT CREATING trial_XX FOLDERS
# ============================================================

def save_trial_configuration(trial_number, config):
    """
    Save configuration under trial_configs/ rather than trial_XX/.
    This is the key fix for the old trial_01 -> trial_01-2 collision.
    """
    path = (
        SEARCH_DIR
        / "trial_configs"
        / f"trial_{trial_number:02d}_configuration.txt"
    )
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        f.write("STAGE 1 HYPERPARAMETER TRIAL\n")
        f.write("=============================\n\n")
        f.write(f"Trial Number : {trial_number}\n")
        f.write(f"Random Seed  : {RANDOM_SEED}\n")
        f.write(f"Learning Rate: {config['learning_rate']}\n")
        f.write(f"Batch Size   : {config['batch_size']}\n")
        f.write(f"Weight Decay : {config['weight_decay']}\n")
        f.write(f"Dropout      : {config['dropout']}\n")

        f.write("\nFIXED MODEL/TRAINING SETTINGS\n")
        f.write(f"Model        : {MODEL_PATH.name}\n")
        f.write(f"Epochs       : {MAX_EPOCHS}\n")
        f.write(f"Image Size   : {IMAGE_SIZE}\n")
        f.write("Optimizer    : AdamW\n")
        f.write(f"Device       : {DEVICE}\n")
        f.write("Workers      : 0\n")
        f.write("AMP          : False\n")
        f.write("Deterministic: True\n")

        f.write("\nFIXED AUGMENTATIONS\n")
        f.write("Horizontal Flip : p = 0.5\n")
        f.write("Vertical Flip   : p = 0.5\n")
        f.write("Brightness      : ±10%\n")
        f.write("Contrast        : ±10%\n")
        f.write("Gaussian Noise  : sigma = 5/255, p = 0.5\n")

        f.write("\nEXCLUDED AUGMENTATIONS\n")
        f.write("Rotation        : None\n")
        f.write("Translation     : None\n")
        f.write("Random Crop     : None\n")
        f.write("Random Scaling  : None\n")
        f.write("Blur            : None\n")
        f.write("Mosaic          : None\n")
        f.write("MixUp           : None\n")
        f.write("CutMix          : None\n")
        f.write("RandAugment     : None\n")
        f.write("Random Erasing  : None\n")

    print(f"[SAVED] Trial configuration: {path}")


# ============================================================
# ANALYZE RESULTS CSV
# ============================================================

def analyze_results_csv(results_csv):
    if not results_csv.exists():
        raise FileNotFoundError(f"results.csv not found:\n{results_csv}")

    df = pd.read_csv(results_csv)
    df.columns = df.columns.str.strip()
    accuracy_col = "metrics/accuracy_top1"

    if accuracy_col not in df.columns:
        raise KeyError(
            f"Could not find validation accuracy column '{accuracy_col}'.\n"
            f"Available columns:\n{list(df.columns)}"
        )

    best_accuracy_index = df[accuracy_col].idxmax()
    best_accuracy = float(df.loc[best_accuracy_index, accuracy_col])
    best_accuracy_epoch = int(df.loc[best_accuracy_index, "epoch"])

    lowest_val_loss = None
    lowest_val_loss_epoch = None
    if "val/loss" in df.columns:
        val_loss_index = df["val/loss"].idxmin()
        lowest_val_loss = float(df.loc[val_loss_index, "val/loss"])
        lowest_val_loss_epoch = int(df.loc[val_loss_index, "epoch"])

    final_accuracy = float(df.iloc[-1][accuracy_col])

    return {
        "best_val_accuracy": best_accuracy,
        "best_val_accuracy_epoch": best_accuracy_epoch,
        "lowest_val_loss": lowest_val_loss,
        "lowest_val_loss_epoch": lowest_val_loss_epoch,
        "final_val_accuracy": final_accuracy,
    }


# ============================================================
# VALIDATION METRICS FROM BEST CHECKPOINT
# ============================================================

def evaluate_best_checkpoint(checkpoint_path):
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        confusion_matrix,
    )

    validation_dir = DATASET_DIR / "val"
    model = YOLO(str(checkpoint_path))
    names = model.names

    if isinstance(names, dict):
        class_names = [names[i] for i in sorted(names)]
    else:
        class_names = list(names)

    class_to_index = {name: index for index, name in enumerate(class_names)}

    if "Healthy" not in class_to_index or "Unhealthy" not in class_to_index:
        raise RuntimeError(
            "Expected Healthy and Unhealthy classes, but model names are:\n"
            f"{class_names}"
        )

    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_records = []

    for class_name in ["Healthy", "Unhealthy"]:
        class_dir = validation_dir / class_name
        for image_path in class_dir.rglob("*"):
            if image_path.is_file() and image_path.suffix.lower() in valid_extensions:
                image_records.append((image_path, class_name))

    image_records.sort(key=lambda item: str(item[0]))

    true_labels = []
    predicted_labels = []

    # Validation only. Test images are never loaded.
    for image_path, true_class in image_records:
        predictions = model.predict(
            source=str(image_path),
            imgsz=IMAGE_SIZE,
            device=DEVICE,
            verbose=False,
        )

        if not predictions:
            raise RuntimeError(f"No prediction returned for:\n{image_path}")

        result = predictions[0]
        if result.probs is None:
            raise RuntimeError(
                f"Classification probabilities missing for:\n{image_path}"
            )

        predicted_index = int(result.probs.top1)
        true_index = class_to_index[true_class]
        true_labels.append(true_index)
        predicted_labels.append(predicted_index)

    unhealthy_index = class_to_index["Unhealthy"]

    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(
        true_labels,
        predicted_labels,
        pos_label=unhealthy_index,
        zero_division=0,
    )
    recall = recall_score(
        true_labels,
        predicted_labels,
        pos_label=unhealthy_index,
        zero_division=0,
    )
    f1 = f1_score(
        true_labels,
        predicted_labels,
        pos_label=unhealthy_index,
        zero_division=0,
    )
    matrix = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=[
            class_to_index["Healthy"],
            class_to_index["Unhealthy"],
        ],
    )

    return {
        "val_accuracy_manual": float(accuracy),
        "val_precision_unhealthy": float(precision),
        "val_recall_unhealthy": float(recall),
        "val_f1_unhealthy": float(f1),
        "tn": int(matrix[0, 0]),
        "fp": int(matrix[0, 1]),
        "fn": int(matrix[1, 0]),
        "tp": int(matrix[1, 1]),
    }


# ============================================================
# TRAIN ONE NEW TRIAL
# ============================================================

def run_new_trial(trial_number, config):
    trial_name = f"trial_{trial_number:02d}"

    print("\n" + "=" * 70)
    print(
        f"STAGE 1 HYPERPARAMETER TRIAL "
        f"{trial_number}/{NUMBER_OF_TRIALS}"
    )
    print("=" * 70)
    print(f"Learning Rate : {config['learning_rate']}")
    print(f"Batch Size    : {config['batch_size']}")
    print(f"Weight Decay  : {config['weight_decay']}")
    print(f"Dropout       : {config['dropout']}")
    print()

    save_trial_configuration(trial_number, config)
    set_random_seeds()
    start_time = time.time()

    # Every NEW trial starts from the same pretrained checkpoint.
    model = YOLO(str(MODEL_PATH))

    print("Starting training...\n")

    model.train(
        data=str(DATASET_DIR),
        epochs=MAX_EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=config["batch_size"],
        optimizer="AdamW",
        lr0=config["learning_rate"],
        weight_decay=config["weight_decay"],
        dropout=config["dropout"],
        device=DEVICE,
        seed=RANDOM_SEED,
        deterministic=True,
        amp=False,
        workers=0,

        # Disable native augmentations; the custom trainer supplies the
        # exact official five augmentations.
        auto_augment=None,
        erasing=0.0,
        degrees=0.0,
        translate=0.0,
        scale=0.0,
        shear=0.0,
        perspective=0.0,
        mosaic=0.0,
        mixup=0.0,
        cutmix=0.0,
        copy_paste=0.0,
        fliplr=0.0,
        flipud=0.0,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        bgr=0.0,
        multi_scale=0.0,

        trainer=CustomClassificationTrainer,

        # IMPORTANT: trial_XX directories are NOT pre-created.
        # Ultralytics can therefore create trial_02, trial_03, etc.
        # without turning them into -2 folders.
        project=str(SEARCH_DIR),
        name=trial_name,
        exist_ok=False,
        plots=True,
        verbose=True,
    )

    elapsed_time = time.time() - start_time
    actual_trial_dir = find_completed_trial_dir(trial_number)

    if actual_trial_dir is None:
        raise FileNotFoundError(
            f"Training completed but no completed output folder was found "
            f"for Trial {trial_number:02d} under:\n{SEARCH_DIR}"
        )

    return model, actual_trial_dir, elapsed_time


# ============================================================
# BUILD RESULT RECORD
# ============================================================

def build_trial_record(
    trial_number,
    actual_trial_dir,
    config,
    training_metrics,
    validation_metrics,
    elapsed_time,
    resumed,
):
    return {
        "trial": trial_number,
        "trial_name": f"trial_{trial_number:02d}",
        "actual_output_folder": str(actual_trial_dir),
        "resumed_existing_trial": resumed,
        "learning_rate": config["learning_rate"],
        "batch_size": config["batch_size"],
        "weight_decay": config["weight_decay"],
        "dropout": config["dropout"],
        "best_val_accuracy": training_metrics["best_val_accuracy"],
        "best_val_accuracy_epoch": training_metrics["best_val_accuracy_epoch"],
        "lowest_val_loss": training_metrics["lowest_val_loss"],
        "lowest_val_loss_epoch": training_metrics["lowest_val_loss_epoch"],
        "final_val_accuracy": training_metrics["final_val_accuracy"],
        "manual_val_accuracy": validation_metrics["val_accuracy_manual"],
        "val_precision_unhealthy": validation_metrics["val_precision_unhealthy"],
        "val_recall_unhealthy": validation_metrics["val_recall_unhealthy"],
        "val_f1_unhealthy": validation_metrics["val_f1_unhealthy"],
        "tn": validation_metrics["tn"],
        "fp": validation_metrics["fp"],
        "fn": validation_metrics["fn"],
        "tp": validation_metrics["tp"],
        # Do not invent timing for Trial 01, which was completed before
        # this fixed script was run.
        "training_time_seconds": elapsed_time if not resumed else np.nan,
    }


# ============================================================
# SAVE INTERMEDIATE RESULTS
# ============================================================

def save_results(all_results):
    results_df = pd.DataFrame(all_results)
    results_path = SEARCH_DIR / "stage1_hparam_results.csv"
    results_df.to_csv(results_path, index=False)
    return results_path


# ============================================================
# MAIN
# ============================================================

def main():
    set_random_seeds()
    create_search_folder()

    print("=" * 70)
    print("STAGE 1 HYPERPARAMETER OPTIMIZATION")
    print("=" * 70)
    print(f"PyTorch Version : {torch.__version__}")
    print(f"CUDA Available  : {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Search aborted.")

    print(f"GPU             : {torch.cuda.get_device_name(0)}")
    print(f"Device          : {DEVICE}")
    print(f"Number of Trials: {NUMBER_OF_TRIALS}")
    print(f"Random Seed     : {RANDOM_SEED}")
    print(f"Output Folder   : {SEARCH_DIR}")
    print()

    verify_dataset()
    trial_configs = load_or_create_trial_plan()

    print("=" * 70)
    print("PLANNED RANDOM SEARCH TRIALS")
    print("=" * 70)
    for trial_number, config in trial_configs:
        print(
            f"Trial {trial_number:02d} | "
            f"LR={config['learning_rate']} | "
            f"Batch={config['batch_size']} | "
            f"WD={config['weight_decay']} | "
            f"Dropout={config['dropout']}"
        )
    print()

    input(
        "\n[PAUSED] Search plan verified.\n"
        "Press ENTER to begin/resume the search, or Ctrl+C to abort..."
    )

    all_results = []

    for trial_number, config in trial_configs:
        # ----------------------------------------------------
        # RESUME SAFETY:
        # If a complete result already exists, recover it and
        # NEVER retrain that trial.
        # ----------------------------------------------------
        existing_trial_dir = find_completed_trial_dir(trial_number)

        if existing_trial_dir is not None:
            print("\n" + "=" * 70)
            print(f"[RESUME] TRIAL {trial_number:02d} ALREADY COMPLETED")
            print("=" * 70)
            print("No retraining will be performed.")
            print(f"Using existing run:\n{existing_trial_dir}")

            results_csv = existing_trial_dir / "results.csv"
            best_checkpoint = existing_trial_dir / "weights" / "best.pt"

            training_metrics = analyze_results_csv(results_csv)
            validation_metrics = evaluate_best_checkpoint(best_checkpoint)

            record = build_trial_record(
                trial_number=trial_number,
                actual_trial_dir=existing_trial_dir,
                config=config,
                training_metrics=training_metrics,
                validation_metrics=validation_metrics,
                elapsed_time=np.nan,
                resumed=True,
            )

            all_results.append(record)
            results_path = save_results(all_results)

            print()
            print(
                f"Recovered Best Validation Accuracy : "
                f"{record['best_val_accuracy']:.4f}"
            )
            print(
                f"Recovered Best Accuracy Epoch      : "
                f"{record['best_val_accuracy_epoch']}"
            )
            print(
                f"Recovered Unhealthy Precision      : "
                f"{record['val_precision_unhealthy']:.4f}"
            )
            print(
                f"Recovered Unhealthy Recall         : "
                f"{record['val_recall_unhealthy']:.4f}"
            )
            print(
                f"Recovered Unhealthy F1             : "
                f"{record['val_f1_unhealthy']:.4f}"
            )
            print(f"[SAVED] {results_path}")
            continue

        # ----------------------------------------------------
        # No completed run exists -> train this trial.
        # ----------------------------------------------------
        model, actual_trial_dir, elapsed_time = run_new_trial(
            trial_number,
            config,
        )

        results_csv = actual_trial_dir / "results.csv"
        best_checkpoint = actual_trial_dir / "weights" / "best.pt"

        if not best_checkpoint.exists():
            raise FileNotFoundError(
                f"Best checkpoint not found:\n{best_checkpoint}"
            )

        training_metrics = analyze_results_csv(results_csv)
        validation_metrics = evaluate_best_checkpoint(best_checkpoint)

        record = build_trial_record(
            trial_number=trial_number,
            actual_trial_dir=actual_trial_dir,
            config=config,
            training_metrics=training_metrics,
            validation_metrics=validation_metrics,
            elapsed_time=elapsed_time,
            resumed=False,
        )

        all_results.append(record)
        results_path = save_results(all_results)

        print()
        print("=" * 70)
        print(f"TRIAL {trial_number} COMPLETE")
        print("=" * 70)
        print(f"Actual Output Folder     : {actual_trial_dir}")
        print(f"Best Validation Accuracy : {record['best_val_accuracy']:.4f}")
        print(f"Best Accuracy Epoch      : {record['best_val_accuracy_epoch']}")
        if record["lowest_val_loss"] is not None:
            print(f"Lowest Validation Loss   : {record['lowest_val_loss']:.6f}")
        print(f"Unhealthy Precision      : {record['val_precision_unhealthy']:.4f}")
        print(f"Unhealthy Recall         : {record['val_recall_unhealthy']:.4f}")
        print(f"Unhealthy F1             : {record['val_f1_unhealthy']:.4f}")
        print(f"Training Time            : {elapsed_time / 60:.2f} minutes")
        print()
        print(f"[SAVED] {results_path}")

        del model
        torch.cuda.empty_cache()

    # ========================================================
    # FINAL RANKING
    # ========================================================

    results_df = pd.DataFrame(all_results)
    results_df = (
        results_df
        .sort_values(
            by=[
                "best_val_accuracy",
                "val_f1_unhealthy",
                "val_recall_unhealthy",
            ],
            ascending=False,
        )
        .reset_index(drop=True)
    )
    results_df["selection_rank"] = results_df.index + 1

    ranking_path = SEARCH_DIR / "stage1_hparam_final_ranking.csv"
    results_df.to_csv(ranking_path, index=False)

    print()
    print("=" * 70)
    print("STAGE 1 HYPERPARAMETER SEARCH COMPLETE")
    print("=" * 70)

    display_columns = [
        "selection_rank",
        "trial",
        "learning_rate",
        "batch_size",
        "weight_decay",
        "dropout",
        "best_val_accuracy",
        "best_val_accuracy_epoch",
        "val_precision_unhealthy",
        "val_recall_unhealthy",
        "val_f1_unhealthy",
        "training_time_seconds",
    ]
    print(results_df[display_columns].to_string(index=False))
    print()
    print(f"[SAVED] Final ranking:\n{ranking_path}")

    best = results_df.iloc[0]
    print()
    print("=" * 70)
    print("BEST VALIDATION CONFIGURATION")
    print("=" * 70)
    print(f"Trial         : {int(best['trial'])}")
    print(f"Learning Rate : {best['learning_rate']}")
    print(f"Batch Size    : {int(best['batch_size'])}")
    print(f"Weight Decay  : {best['weight_decay']}")
    print(f"Dropout       : {best['dropout']}")
    print(f"Best Val Acc  : {best['best_val_accuracy']:.4f}")
    print(f"Best Val Epoch: {int(best['best_val_accuracy_epoch'])}")
    print()
    print("[IMPORTANT] The test set was NOT used for hyperparameter selection.")


if __name__ == "__main__":
    main()
