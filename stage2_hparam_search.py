from pathlib import Path
import random
import time

import numpy as np
import pandas as pd
import torch

from ultralytics import YOLO

# Reuse the EXACT custom Stage 2 trainer
# from your verified train_stage2.py
from train_stage2 import CustomDetectionTrainer


# ============================================================
# STAGE 2 HYPERPARAMETER OPTIMIZATION
# ============================================================
# Four-trial random search.
#
# Search parameters:
#   Learning rate : 0.0005 - 0.002
#   Batch size    : 4 or 8
#   Weight decay  : 0.0001 - 0.001
#
# No dropout is used for Stage 2 detection.
#
# All other Stage 2 settings remain fixed.
#
# The test set is NEVER used for hyperparameter selection.
# ============================================================


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(
    r"C:\Users\saiar\Pictures\cardava_thesis"
)

DATASET_DIR = (
    PROJECT_ROOT /
    "dataset" /
    "stage2_det"
)

DATA_YAML = (
    DATASET_DIR /
    "data.yaml"
)

MODEL_PATH = (
    PROJECT_ROOT /
    "yolo11n.pt"
)

RUNS_DIR = (
    PROJECT_ROOT /
    "runs"
)

# New official Stage 2 search folder.
# Older Stage 2 search folders are NOT touched.
SEARCH_DIR = (
    RUNS_DIR /
    "stage2_hparam_search_final"
)


# ============================================================
# FIXED EXPERIMENT SETTINGS
# ============================================================

NUMBER_OF_TRIALS = 4

MAX_EPOCHS = 50

IMAGE_SIZE = 640

RANDOM_SEED = 42

DEVICE = "cuda:0"


# ============================================================
# HYPERPARAMETER SEARCH RANGES
# ============================================================

# Chapter 3 Stage 2 search ranges:
#
# Learning rate : approximately 0.0005 - 0.002
# Batch size    : 4 or 8
# Weight decay  : 0.0001 - 0.001
#
# No dropout is searched for the detector.

LR_MIN = 0.0005
LR_MAX = 0.002

BATCH_OPTIONS = [
    4,
    8,
]

WD_MIN = 0.0001
WD_MAX = 0.001


# ============================================================
# SEARCH RANDOM NUMBER GENERATOR
# ============================================================

search_rng = random.Random(
    RANDOM_SEED
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_random_seeds():
    random.seed(
        RANDOM_SEED
    )

    np.random.seed(
        RANDOM_SEED
    )

    torch.manual_seed(
        RANDOM_SEED
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed(
            RANDOM_SEED
        )

        torch.cuda.manual_seed_all(
            RANDOM_SEED
        )


# ============================================================
# CREATE SEARCH FOLDER
# ============================================================

def create_search_folder():
    SEARCH_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# VERIFY DATASET STRUCTURE
# ============================================================

def verify_dataset():
    required_paths = [
        DATASET_DIR / "images" / "train",
        DATASET_DIR / "images" / "val",
        DATASET_DIR / "images" / "test",
        DATASET_DIR / "labels" / "train",
        DATASET_DIR / "labels" / "val",
        DATASET_DIR / "labels" / "test",
        DATA_YAML,
    ]

    print("=" * 70)
    print("STAGE 2 DATASET VERIFICATION")
    print("=" * 70)

    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(
                f"Required dataset path not found:\n{path}"
            )

        print(
            f"[PASS] {path}"
        )

    print()


# ============================================================
# COUNT DATASET SPLITS
# ============================================================

def count_split_files():
    expected = {
        "train": 700,
        "val": 150,
        "test": 150,
    }

    extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp",
    }

    print("=" * 70)
    print("STAGE 2 SPLIT CHECK")
    print("=" * 70)

    for split, expected_count in expected.items():

        image_dir = (
            DATASET_DIR /
            "images" /
            split
        )

        label_dir = (
            DATASET_DIR /
            "labels" /
            split
        )

        image_count = sum(
            1
            for path in image_dir.iterdir()
            if (
                path.is_file()
                and path.suffix.lower() in extensions
            )
        )

        label_count = sum(
            1
            for path in label_dir.iterdir()
            if (
                path.is_file()
                and path.suffix.lower() == ".txt"
            )
        )

        print(
            f"{split.upper():5s} | "
            f"Images: {image_count:3d} | "
            f"Labels: {label_count:3d}"
        )

        if image_count != expected_count:
            raise ValueError(
                f"{split} image count is "
                f"{image_count}, expected "
                f"{expected_count}."
            )

        if label_count != expected_count:
            raise ValueError(
                f"{split} label count is "
                f"{label_count}, expected "
                f"{expected_count}."
            )

    print()
    print(
        "[PASS] Stage 2 split counts verified."
    )
    print()


# ============================================================
# VERIFY ONE WHOLE-BUNCH BOX PER IMAGE
# ============================================================

def verify_annotations():
    total_images = 0
    total_labels = 0
    invalid = 0
    wrong_box_count = 0

    print("=" * 70)
    print("STAGE 2 BOUNDING-BOX VERIFICATION")
    print("=" * 70)

    for split in [
        "train",
        "val",
        "test",
    ]:

        image_dir = (
            DATASET_DIR /
            "images" /
            split
        )

        label_dir = (
            DATASET_DIR /
            "labels" /
            split
        )

        for image_path in image_dir.iterdir():

            if (
                not image_path.is_file()
                or
                image_path.suffix.lower()
                not in {
                    ".jpg",
                    ".jpeg",
                    ".png",
                    ".bmp",
                    ".webp",
                }
            ):
                continue

            total_images += 1

            label_path = (
                label_dir /
                f"{image_path.stem}.txt"
            )

            if not label_path.exists():
                invalid += 1
                continue

            with open(
                label_path,
                "r",
                encoding="utf-8"
            ) as f:

                lines = [
                    line.strip()
                    for line in f
                    if line.strip()
                ]

            total_labels += len(lines)

            if len(lines) != 1:
                wrong_box_count += 1
                continue

            parts = lines[0].split()

            if len(parts) != 5:
                invalid += 1
                continue

            try:
                cls = int(parts[0])

                values = [
                    float(value)
                    for value in parts[1:]
                ]

            except ValueError:
                invalid += 1
                continue

            # Stage 2 detector has one class:
            # 0 = banana_bunch
            if cls != 0:
                invalid += 1
                continue

            if not all(
                0.0 <= value <= 1.0
                for value in values
            ):
                invalid += 1
                continue

            if (
                values[2] <= 0.0
                or
                values[3] <= 0.0
            ):
                invalid += 1

    print(
        f"Images checked       : {total_images}"
    )

    print(
        f"Bounding boxes found : {total_labels}"
    )

    print(
        f"Invalid annotations  : {invalid}"
    )

    print(
        f"Wrong box count      : {wrong_box_count}"
    )

    print()

    if invalid > 0:
        raise ValueError(
            "Invalid Stage 2 annotations detected."
        )

    if wrong_box_count > 0:
        raise ValueError(
            "At least one image does not contain "
            "exactly one whole-bunch bounding box."
        )

    if total_labels != total_images:
        raise ValueError(
            "Total labels do not match total images."
        )

    print(
        "[PASS] Exactly one valid whole-bunch "
        "bounding box exists for every image."
    )

    print()


# ============================================================
# GENERATE RANDOM TRIAL
# ============================================================

def generate_trial():

    config = {
        "learning_rate": round(
            search_rng.uniform(
                LR_MIN,
                LR_MAX
            ),
            6
        ),

        "batch_size": search_rng.choice(
            BATCH_OPTIONS
        ),

        "weight_decay": round(
            search_rng.uniform(
                WD_MIN,
                WD_MAX
            ),
            6
        ),
    }

    assert (
        LR_MIN
        <= config["learning_rate"]
        <= LR_MAX
    )

    assert (
        config["batch_size"]
        in BATCH_OPTIONS
    )

    assert (
        WD_MIN
        <= config["weight_decay"]
        <= WD_MAX
    )

    return config


# ============================================================
# LOAD OR CREATE OFFICIAL SEARCH PLAN
# ============================================================

def load_or_create_trial_plan():

    planned_path = (
        SEARCH_DIR /
        "planned_trials.csv"
    )

    # If the plan already exists, reuse it.
    # This prevents a rerun from changing the four trials.
    if planned_path.exists():

        planned_df = pd.read_csv(
            planned_path
        )

        required_columns = {
            "trial",
            "learning_rate",
            "batch_size",
            "weight_decay",
        }

        missing = (
            required_columns -
            set(planned_df.columns)
        )

        if missing:
            raise ValueError(
                "Existing planned_trials.csv is missing "
                f"columns: {sorted(missing)}"
            )

        if len(planned_df) != NUMBER_OF_TRIALS:
            raise ValueError(
                f"Expected {NUMBER_OF_TRIALS} planned trials, "
                f"but found {len(planned_df)}."
            )

        planned_df = (
            planned_df
            .sort_values("trial")
            .reset_index(drop=True)
        )

        # Verify saved values remain in range.
        for _, row in planned_df.iterrows():

            assert (
                LR_MIN
                <= float(row["learning_rate"])
                <= LR_MAX
            )

            assert (
                int(row["batch_size"])
                in BATCH_OPTIONS
            )

            assert (
                WD_MIN
                <= float(row["weight_decay"])
                <= WD_MAX
            )

        trial_configs = []

        for _, row in planned_df.iterrows():

            trial_configs.append(
                (
                    int(row["trial"]),
                    {
                        "learning_rate": float(
                            row["learning_rate"]
                        ),
                        "batch_size": int(
                            row["batch_size"]
                        ),
                        "weight_decay": float(
                            row["weight_decay"]
                        ),
                    }
                )
            )

        print(
            "[PASS] Existing official trial plan reused:"
        )

        print(
            f"{planned_path}\n"
        )

        return trial_configs

    # Otherwise generate the plan once.
    trial_configs = []

    for trial_number in range(
        1,
        NUMBER_OF_TRIALS + 1
    ):

        config = generate_trial()

        trial_configs.append(
            (
                trial_number,
                config
            )
        )

    planned_df = pd.DataFrame(
        [
            {
                "trial": trial_number,
                "learning_rate": config[
                    "learning_rate"
                ],
                "batch_size": config[
                    "batch_size"
                ],
                "weight_decay": config[
                    "weight_decay"
                ],
            }
            for trial_number, config
            in trial_configs
        ]
    )

    planned_df.to_csv(
        planned_path,
        index=False
    )

    print(
        f"[SAVED] {planned_path}\n"
    )

    return trial_configs


# ============================================================
# SAVE TRIAL CONFIGURATION
# ============================================================
# IMPORTANT:
# The configuration is stored outside the actual trial output
# folder. Therefore trial_01/trial_02/etc. are NOT created
# before Ultralytics starts training.
# ============================================================

def save_trial_configuration(
    trial_number,
    config
):

    config_path = (
        SEARCH_DIR /
        "trial_configs" /
        f"trial_{trial_number:02d}_configuration.txt"
    )

    config_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        config_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "STAGE 2 HYPERPARAMETER TRIAL\n"
        )

        f.write(
            "=============================\n\n"
        )

        f.write(
            f"Trial Number : {trial_number}\n"
        )

        f.write(
            f"Random Seed  : {RANDOM_SEED}\n"
        )

        f.write(
            f"Learning Rate: "
            f"{config['learning_rate']}\n"
        )

        f.write(
            f"Batch Size   : "
            f"{config['batch_size']}\n"
        )

        f.write(
            f"Weight Decay : "
            f"{config['weight_decay']}\n"
        )

        f.write(
            "Dropout      : None "
            "(not used for detector)\n"
        )

        f.write(
            "\nFIXED SETTINGS\n"
        )

        f.write(
            f"Model        : {MODEL_PATH.name}\n"
        )

        f.write(
            f"Epochs       : {MAX_EPOCHS}\n"
        )

        f.write(
            f"Image Size   : {IMAGE_SIZE} x {IMAGE_SIZE}\n"
        )

        f.write(
            "Optimizer    : AdamW\n"
        )

        f.write(
            f"Device       : {DEVICE}\n"
        )

        f.write(
            "Workers      : 0\n"
        )

        f.write(
            "AMP          : False\n"
        )

        f.write(
            "Deterministic: True\n"
        )

        f.write(
            "\nFIXED INPUT / PREPROCESSING\n"
        )

        f.write(
            "Original image: 4608 x 2592\n"
        )

        f.write(
            "Model frame  : LetterBox 640 x 640\n"
        )

        f.write(
            "Aspect ratio : Preserved\n"
        )

        f.write(
            "\nFIXED TRAINING AUGMENTATIONS\n"
        )

        f.write(
            "Horizontal Flip : p = 0.5\n"
        )

        f.write(
            "Vertical Flip   : p = 0.5\n"
        )

        f.write(
            "Brightness      : +/-10%, p = 0.5\n"
        )

        f.write(
            "Contrast        : +/-10%, p = 0.5\n"
        )

        f.write(
            "Gaussian Noise  : sigma = 5/255, p = 0.5\n"
        )

        f.write(
            "\nEXCLUDED AUGMENTATIONS\n"
        )

        f.write(
            "Rotation        : None\n"
        )

        f.write(
            "Translation     : None\n"
        )

        f.write(
            "Random Crop     : None\n"
        )

        f.write(
            "Random Scaling  : None\n"
        )

        f.write(
            "Blur            : None\n"
        )

        f.write(
            "Perspective     : None\n"
        )

        f.write(
            "Shear           : None\n"
        )

        f.write(
            "Mosaic          : None\n"
        )

        f.write(
            "MixUp           : None\n"
        )

        f.write(
            "CutMix          : None\n"
        )

        f.write(
            "CopyPaste       : None\n"
        )

        f.write(
            "HSV             : None\n"
        )

        f.write(
            "Channel swap    : None\n"
        )


# ============================================================
# FIND COMPLETED TRIAL DIRECTORY
# ============================================================

def find_completed_trial_dir(
    trial_number
):

    pattern = (
        f"trial_{trial_number:02d}*"
    )

    candidates = []

    for directory in SEARCH_DIR.glob(
        pattern
    ):

        if not directory.is_dir():
            continue

        results_csv = (
            directory /
            "results.csv"
        )

        best_checkpoint = (
            directory /
            "weights" /
            "best.pt"
        )

        if (
            results_csv.exists()
            and
            best_checkpoint.exists()
        ):

            candidates.append(
                directory
            )

    if not candidates:
        return None

    # Most recently completed result first.
    candidates.sort(
        key=lambda path: (
            path /
            "results.csv"
        ).stat().st_mtime,
        reverse=True
    )

    return candidates[0]


# ============================================================
# ANALYZE ONE TRIAL
# ============================================================

def analyze_trial(
    results_csv
):

    if not results_csv.exists():
        raise FileNotFoundError(
            f"results.csv not found:\n{results_csv}"
        )

    df = pd.read_csv(
        results_csv
    )

    df.columns = (
        df.columns
        .str.strip()
    )

    required_columns = {
        "epoch",
        "metrics/precision(B)",
        "metrics/recall(B)",
        "metrics/mAP50(B)",
        "metrics/mAP50-95(B)",
    }

    missing = (
        required_columns -
        set(df.columns)
    )

    if missing:
        raise KeyError(
            "Stage 2 results.csv is missing required "
            f"columns: {sorted(missing)}\n"
            f"Available columns: {list(df.columns)}"
        )

    # Primary selection metric:
    # best validation mAP@0.50:0.95.
    best_index = (
        df[
            "metrics/mAP50-95(B)"
        ]
        .idxmax()
    )

    best_row = (
        df.loc[best_index]
    )

    return {
        "best_map50_95": float(
            best_row[
                "metrics/mAP50-95(B)"
            ]
        ),

        "best_map50": float(
            best_row[
                "metrics/mAP50(B)"
            ]
        ),

        "best_precision": float(
            best_row[
                "metrics/precision(B)"
            ]
        ),

        "best_recall": float(
            best_row[
                "metrics/recall(B)"
            ]
        ),

        "best_epoch": int(
            best_row["epoch"]
        ),

        "final_map50_95": float(
            df.iloc[-1][
                "metrics/mAP50-95(B)"
            ]
        ),

        "final_map50": float(
            df.iloc[-1][
                "metrics/mAP50(B)"
            ]
        ),

        "final_precision": float(
            df.iloc[-1][
                "metrics/precision(B)"
            ]
        ),

        "final_recall": float(
            df.iloc[-1][
                "metrics/recall(B)"
            ]
        ),
    }


# ============================================================
# RUN A NEW TRIAL
# ============================================================

def run_new_trial(
    trial_number,
    config
):

    trial_name = (
        f"trial_{trial_number:02d}"
    )

    print()
    print("=" * 70)
    print(
        f"STAGE 2 HYPERPARAMETER TRIAL "
        f"{trial_number}/{NUMBER_OF_TRIALS}"
    )
    print("=" * 70)

    print(
        f"Learning Rate : "
        f"{config['learning_rate']}"
    )

    print(
        f"Batch Size    : "
        f"{config['batch_size']}"
    )

    print(
        f"Weight Decay  : "
        f"{config['weight_decay']}"
    )

    print(
        "Dropout       : None"
    )

    print()

    save_trial_configuration(
        trial_number,
        config
    )

    # Reset seeds before each trial.
    set_random_seeds()

    start_time = time.time()

    # Every trial starts from the same pretrained checkpoint.
    model = YOLO(
        str(MODEL_PATH)
    )

    print(
        "Starting training...\n"
    )

    model.train(

        data=str(
            DATA_YAML
        ),

        epochs=MAX_EPOCHS,

        imgsz=IMAGE_SIZE,

        batch=config[
            "batch_size"
        ],

        optimizer="AdamW",

        lr0=config[
            "learning_rate"
        ],

        weight_decay=config[
            "weight_decay"
        ],

        device=DEVICE,

        seed=RANDOM_SEED,

        deterministic=True,

        amp=False,

        workers=0,

        # ----------------------------------------------------
        # DISABLE ALL NATIVE AUGMENTATIONS THAT ARE NOT PART
        # OF THE APPROVED CUSTOM PIPELINE.
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Use EXACT verified custom Stage 2 trainer.
        # ----------------------------------------------------

        trainer=CustomDetectionTrainer,

        # ----------------------------------------------------
        # IMPORTANT:
        # Trial folders are NOT created beforehand.
        # ----------------------------------------------------

        project=str(
            SEARCH_DIR
        ),

        name=trial_name,

        exist_ok=False,

        plots=True,

        verbose=True,
    )

    elapsed_time = (
        time.time() -
        start_time
    )

    # Find the actual output folder created by Ultralytics.
    actual_trial_dir = (
        find_completed_trial_dir(
            trial_number
        )
    )

    if actual_trial_dir is None:
        raise FileNotFoundError(
            f"Could not locate a completed output "
            f"folder for Trial {trial_number:02d}."
        )

    return (
        model,
        actual_trial_dir,
        elapsed_time
    )


# ============================================================
# SAVE INTERMEDIATE SEARCH RESULTS
# ============================================================

def save_search_results(
    all_results
):

    results_path = (
        SEARCH_DIR /
        "stage2_hparam_results.csv"
    )

    pd.DataFrame(
        all_results
    ).to_csv(
        results_path,
        index=False
    )

    return results_path


# ============================================================
# MAIN
# ============================================================

def main():

    set_random_seeds()

    create_search_folder()

    print("=" * 70)
    print(
        "STAGE 2 HYPERPARAMETER OPTIMIZATION"
    )
    print("=" * 70)

    print(
        f"PyTorch Version : "
        f"{torch.__version__}"
    )

    print(
        f"CUDA Available  : "
        f"{torch.cuda.is_available()}"
    )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. "
            "Search aborted."
        )

    print(
        f"GPU             : "
        f"{torch.cuda.get_device_name(0)}"
    )

    print(
        f"Device          : {DEVICE}"
    )

    print(
        f"Number of Trials: "
        f"{NUMBER_OF_TRIALS}"
    )

    print(
        f"Random Seed     : "
        f"{RANDOM_SEED}"
    )

    print(
        f"Output Folder   : "
        f"{SEARCH_DIR}"
    )

    print()

    # --------------------------------------------------------
    # Dataset checks.
    # --------------------------------------------------------

    verify_dataset()

    count_split_files()

    verify_annotations()

    # --------------------------------------------------------
    # Reuse the exact official trial plan if already saved.
    # --------------------------------------------------------

    trial_configs = (
        load_or_create_trial_plan()
    )

    print("=" * 70)
    print(
        "PLANNED RANDOM SEARCH TRIALS"
    )
    print("=" * 70)

    for trial_number, config in (
        trial_configs
    ):

        print(
            f"Trial {trial_number:02d} | "
            f"LR={config['learning_rate']} | "
            f"Batch={config['batch_size']} | "
            f"WD={config['weight_decay']} | "
            f"Dropout=None"
        )

    print()

    input(
        "\n[PAUSED] Search plan verified.\n"
        "Press ENTER to begin/resume the search, "
        "or Ctrl+C to abort..."
    )

    all_results = []

    # ========================================================
    # RUN / RESUME ALL TRIALS
    # ========================================================

    for trial_number, config in (
        trial_configs
    ):

        # ----------------------------------------------------
        # If this trial already completed, DO NOT retrain it.
        # ----------------------------------------------------

        existing_trial_dir = (
            find_completed_trial_dir(
                trial_number
            )
        )

        if existing_trial_dir is not None:

            print()
            print("=" * 70)
            print(
                f"[RESUME] TRIAL {trial_number:02d} "
                "ALREADY COMPLETED"
            )
            print("=" * 70)

            print(
                "No retraining will be performed."
            )

            print(
                f"Using existing run:\n"
                f"{existing_trial_dir}"
            )

            results_csv = (
                existing_trial_dir /
                "results.csv"
            )

            metrics = analyze_trial(
                results_csv
            )

            record = {
                "trial": trial_number,

                "trial_name": (
                    f"trial_{trial_number:02d}"
                ),

                "actual_output_folder": str(
                    existing_trial_dir
                ),

                "resumed_existing_trial": True,

                "learning_rate": config[
                    "learning_rate"
                ],

                "batch_size": config[
                    "batch_size"
                ],

                "weight_decay": config[
                    "weight_decay"
                ],

                "dropout": None,

                "best_val_map50_95": metrics[
                    "best_map50_95"
                ],

                "best_val_map50": metrics[
                    "best_map50"
                ],

                "best_val_precision": metrics[
                    "best_precision"
                ],

                "best_val_recall": metrics[
                    "best_recall"
                ],

                "best_val_epoch": metrics[
                    "best_epoch"
                ],

                "final_val_map50_95": metrics[
                    "final_map50_95"
                ],

                "final_val_map50": metrics[
                    "final_map50"
                ],

                "final_val_precision": metrics[
                    "final_precision"
                ],

                "final_val_recall": metrics[
                    "final_recall"
                ],

                "training_time_seconds": np.nan,
            }

            all_results.append(
                record
            )

            results_path = (
                save_search_results(
                    all_results
                )
            )

            print()
            print(
                f"Best Val mAP@0.50:0.95 : "
                f"{record['best_val_map50_95']:.5f}"
            )

            print(
                f"Best Val mAP@0.50      : "
                f"{record['best_val_map50']:.5f}"
            )

            print(
                f"Best Val Precision     : "
                f"{record['best_val_precision']:.5f}"
            )

            print(
                f"Best Val Recall        : "
                f"{record['best_val_recall']:.5f}"
            )

            print(
                f"Best Val Epoch         : "
                f"{record['best_val_epoch']}"
            )

            print(
                f"[SAVED] {results_path}"
            )

            continue

        # ----------------------------------------------------
        # No completed run -> train this trial.
        # ----------------------------------------------------

        (
            model,
            actual_trial_dir,
            elapsed_time
        ) = run_new_trial(
            trial_number,
            config
        )

        results_csv = (
            actual_trial_dir /
            "results.csv"
        )

        metrics = analyze_trial(
            results_csv
        )

        best_checkpoint = (
            actual_trial_dir /
            "weights" /
            "best.pt"
        )

        if not best_checkpoint.exists():
            raise FileNotFoundError(
                f"Best checkpoint not found:\n"
                f"{best_checkpoint}"
            )

        record = {
            "trial": trial_number,

            "trial_name": (
                f"trial_{trial_number:02d}"
            ),

            "actual_output_folder": str(
                actual_trial_dir
            ),

            "resumed_existing_trial": False,

            "learning_rate": config[
                "learning_rate"
            ],

            "batch_size": config[
                "batch_size"
            ],

            "weight_decay": config[
                "weight_decay"
            ],

            "dropout": None,

            "best_val_map50_95": metrics[
                "best_map50_95"
            ],

            "best_val_map50": metrics[
                "best_map50"
            ],

            "best_val_precision": metrics[
                "best_precision"
            ],

            "best_val_recall": metrics[
                "best_recall"
            ],

            "best_val_epoch": metrics[
                "best_epoch"
            ],

            "final_val_map50_95": metrics[
                "final_map50_95"
            ],

            "final_val_map50": metrics[
                "final_map50"
            ],

            "final_val_precision": metrics[
                "final_precision"
            ],

            "final_val_recall": metrics[
                "final_recall"
            ],

            "training_time_seconds": (
                elapsed_time
            ),
        }

        all_results.append(
            record
        )

        results_path = (
            save_search_results(
                all_results
            )
        )

        print()
        print("=" * 70)
        print(
            f"TRIAL {trial_number} COMPLETE"
        )
        print("=" * 70)

        print(
            f"Actual Output Folder     : "
            f"{actual_trial_dir}"
        )

        print(
            f"Best Val mAP@0.50:0.95  : "
            f"{record['best_val_map50_95']:.5f}"
        )

        print(
            f"Best Val mAP@0.50       : "
            f"{record['best_val_map50']:.5f}"
        )

        print(
            f"Best Val Precision      : "
            f"{record['best_val_precision']:.5f}"
        )

        print(
            f"Best Val Recall         : "
            f"{record['best_val_recall']:.5f}"
        )

        print(
            f"Best Val Epoch          : "
            f"{record['best_val_epoch']}"
        )

        print(
            f"Training Time           : "
            f"{elapsed_time / 60:.2f} minutes"
        )

        print()
        print(
            f"[SAVED] {results_path}"
        )

        del model

        torch.cuda.empty_cache()

    # ========================================================
    # FINAL VALIDATION RANKING
    # ========================================================

    results_df = pd.DataFrame(
        all_results
    )

    # Primary:
    #   best validation mAP@0.50:0.95
    #
    # Tie-breakers:
    #   best validation mAP@0.50
    #   best validation precision
    #   best validation recall
    #
    # All based on validation data only.
    results_df = (
        results_df
        .sort_values(
            by=[
                "best_val_map50_95",
                "best_val_map50",
                "best_val_precision",
                "best_val_recall",
            ],
            ascending=False,
        )
        .reset_index(drop=True)
    )

    results_df[
        "selection_rank"
    ] = (
        results_df.index + 1
    )

    ranking_path = (
        SEARCH_DIR /
        "stage2_hparam_final_ranking.csv"
    )

    results_df.to_csv(
        ranking_path,
        index=False
    )

    # ========================================================
    # FINAL DISPLAY
    # ========================================================

    display_columns = [
        "selection_rank",
        "trial",
        "learning_rate",
        "batch_size",
        "weight_decay",
        "best_val_map50_95",
        "best_val_map50",
        "best_val_precision",
        "best_val_recall",
        "best_val_epoch",
        "final_val_map50_95",
        "training_time_seconds",
    ]

    print()
    print("=" * 70)
    print(
        "STAGE 2 HYPERPARAMETER SEARCH COMPLETE"
    )
    print("=" * 70)

    print(
        results_df[
            display_columns
        ].to_string(
            index=False
        )
    )

    print()
    print(
        f"[SAVED] Final ranking:\n"
        f"{ranking_path}"
    )

    # ========================================================
    # BEST CONFIGURATION
    # ========================================================

    best = results_df.iloc[0]

    print()
    print("=" * 70)
    print(
        "BEST VALIDATION CONFIGURATION"
    )
    print("=" * 70)

    print(
        f"Trial         : "
        f"{int(best['trial'])}"
    )

    print(
        f"Learning Rate : "
        f"{best['learning_rate']}"
    )

    print(
        f"Batch Size    : "
        f"{int(best['batch_size'])}"
    )

    print(
        f"Weight Decay  : "
        f"{best['weight_decay']}"
    )

    print(
        "Dropout       : None"
    )

    print(
        f"Best Val mAP50-95 : "
        f"{best['best_val_map50_95']:.5f}"
    )

    print(
        f"Best Val mAP50    : "
        f"{best['best_val_map50']:.5f}"
    )

    print(
        f"Best Val Precision: "
        f"{best['best_val_precision']:.5f}"
    )

    print(
        f"Best Val Recall   : "
        f"{best['best_val_recall']:.5f}"
    )

    print(
        f"Best Val Epoch    : "
        f"{int(best['best_val_epoch'])}"
    )

    print()
    print(
        "[IMPORTANT] The Stage 2 test set was NOT used "
        "for hyperparameter selection."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
