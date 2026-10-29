from pathlib import Path
import math
import json

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from ultralytics import YOLO


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(
    r"C:\Users\saiar\Pictures\cardava_thesis"
)

DATASET_DIR = (
    PROJECT_ROOT
    / "dataset"
    / "stage1_cls"
)

YOLO_RUN_DIR = (
    PROJECT_ROOT
    / "runs"
    / "stage1_final"
)

YOLO_CHECKPOINT = (
    YOLO_RUN_DIR
    / "weights"
    / "best.pt"
)

YOLO_OUTPUT_DIR = (
    YOLO_RUN_DIR
    / "test_evaluation"
)

SVM_OUTPUT_DIR = (
    PROJECT_ROOT
    / "runs"
    / "stage1_svm_baseline"
)

SVM_PREDICTIONS = (
    SVM_OUTPUT_DIR
    / "svm_test_predictions.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "runs"
    / "stage1_mcnemar"
)


# ============================================================
# MODEL SETTINGS
# ============================================================

IMAGE_SIZE = 224
BATCH_SIZE = 16
DEVICE = "cuda:0"


# ============================================================
# EXACT TEST TRANSFORM USED BY FINAL YOLO EVALUATION
# ============================================================

TEST_TRANSFORM = T.Compose([
    T.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),
    T.ToTensor(),
])


# ============================================================
# CLASS DEFINITIONS
# ============================================================

CLASS_NAMES = [
    "Healthy",
    "Unhealthy",
]


# ============================================================
# LOGIT EXTRACTION
# ============================================================

def extract_logits(output):

    if isinstance(
        output,
        torch.Tensor
    ):
        return output

    if isinstance(
        output,
        (tuple, list)
    ):
        for item in output:
            if isinstance(
                item,
                torch.Tensor
            ):
                return item

    if isinstance(
        output,
        dict
    ):
        for key in (
            "logits",
            "output",
            "pred",
            "preds",
        ):
            value = output.get(key)

            if isinstance(
                value,
                torch.Tensor
            ):
                return value

    raise RuntimeError(
        f"Unable to extract logits "
        f"from {type(output)}"
    )


# ============================================================
# VERIFY FILES
# ============================================================

def verify_files():

    if not YOLO_CHECKPOINT.exists():
        raise FileNotFoundError(
            f"YOLO checkpoint not found:\n"
            f"{YOLO_CHECKPOINT}"
        )

    if not SVM_PREDICTIONS.exists():
        raise FileNotFoundError(
            f"SVM predictions not found:\n"
            f"{SVM_PREDICTIONS}"
        )

    print(
        f"[PASS] YOLO checkpoint:\n"
        f"       {YOLO_CHECKPOINT}"
    )

    print(
        f"[PASS] SVM predictions:\n"
        f"       {SVM_PREDICTIONS}"
    )


# ============================================================
# LOAD TEST DATASET
# ============================================================

def load_yolo_test_dataset():

    test_dir = (
        DATASET_DIR
        / "test"
    )

    if not test_dir.exists():
        raise FileNotFoundError(
            f"Test directory not found:\n"
            f"{test_dir}"
        )

    dataset = ImageFolder(
        str(test_dir),
        transform=TEST_TRANSFORM
    )

    if dataset.classes != CLASS_NAMES:
        raise RuntimeError(
            f"Unexpected classes: "
            f"{dataset.classes}"
        )

    if len(dataset) != 225:
        raise RuntimeError(
            f"Expected 225 test images, "
            f"found {len(dataset)}"
        )

    healthy_count = sum(
        label == 0
        for _, label in dataset.samples
    )

    unhealthy_count = sum(
        label == 1
        for _, label in dataset.samples
    )

    if (
        healthy_count != 150
        or unhealthy_count != 75
    ):
        raise RuntimeError(
            "Expected 150 Healthy and "
            f"75 Unhealthy; found "
            f"{healthy_count} and "
            f"{unhealthy_count}"
        )

    print(
        "[PASS] YOLO test dataset: "
        "225 images"
    )

    print(
        "[PASS] Distribution: "
        "150 Healthy / 75 Unhealthy"
    )

    print(
        "[PASS] Transform: "
        "Resize(224,224) -> ToTensor()"
    )

    print(
        "[PASS] Test augmentation: NONE"
    )

    return dataset


# ============================================================
# GENERATE INDIVIDUAL YOLO PREDICTIONS
# ============================================================

def generate_yolo_predictions(dataset):

    print()
    print("=" * 70)
    print("GENERATING YOLO INDIVIDUAL TEST PREDICTIONS")
    print("=" * 70)

    model = YOLO(
        str(YOLO_CHECKPOINT)
    )

    names = model.names

    checkpoint_classes = [
        names[i]
        for i in sorted(names)
    ] if isinstance(
        names,
        dict
    ) else list(names)

    if checkpoint_classes != CLASS_NAMES:
        raise RuntimeError(
            f"Unexpected checkpoint classes: "
            f"{checkpoint_classes}"
        )

    net = model.model.to(
        DEVICE
    )

    net.eval()

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    rows = []

    with torch.inference_mode():

        sample_index = 0

        for images, labels in loader:

            images = images.to(
                DEVICE,
                non_blocking=True
            )

            output = extract_logits(
                net(images)
            )

            if (
                output.ndim != 2
                or output.shape[1] != 2
            ):
                raise RuntimeError(
                    f"Unexpected output shape: "
                    f"{tuple(output.shape)}"
                )

            predictions = (
                output.argmax(
                    dim=1
                )
                .cpu()
                .numpy()
            )

            labels_np = (
                labels
                .cpu()
                .numpy()
            )

            batch_size = len(
                labels_np
            )

            for i in range(
                batch_size
            ):

                path, true_index = (
                    dataset.samples[
                        sample_index
                    ]
                )

                pred_index = int(
                    predictions[i]
                )

                true_index = int(
                    true_index
                )

                rows.append({
                    "filename":
                        Path(path).name,

                    "true_label":
                        CLASS_NAMES[
                            true_index
                        ],

                    "yolo_predicted_label":
                        CLASS_NAMES[
                            pred_index
                        ],

                    "yolo_correct":
                        (
                            pred_index
                            == true_index
                        ),
                })

                sample_index += 1

    if len(rows) != 225:
        raise RuntimeError(
            f"Expected 225 YOLO predictions, "
            f"created {len(rows)}"
        )

    df = pd.DataFrame(
        rows
    )

    output_path = (
        YOLO_OUTPUT_DIR
        / "stage1_test_predictions.csv"
    )

    YOLO_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        output_path,
        index=False
    )

    print(
        f"[PASS] Generated "
        f"{len(df)} YOLO predictions."
    )

    print(
        f"[SAVED] {output_path}"
    )

    return df


# ============================================================
# LOAD SVM PREDICTIONS
# ============================================================

def load_svm_predictions():

    print()
    print("=" * 70)
    print("LOADING SVM TEST PREDICTIONS")
    print("=" * 70)

    df = pd.read_csv(
        SVM_PREDICTIONS
    )

    required_columns = {
        "filename",
        "true_label",
        "predicted_label",
        "correct",
    }

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:
        raise RuntimeError(
            "SVM prediction file is missing "
            f"columns: {sorted(missing)}"
        )

    if len(df) != 225:
        raise RuntimeError(
            f"Expected 225 SVM predictions, "
            f"found {len(df)}"
        )

    df = df.rename(
        columns={
            "predicted_label":
                "svm_predicted_label",

            "correct":
                "svm_correct",
        }
    )

    print(
        "[PASS] Loaded 225 SVM predictions."
    )

    return df[
        [
            "filename",
            "true_label",
            "svm_predicted_label",
            "svm_correct",
        ]
    ].copy()


# ============================================================
# MERGE PREDICTIONS
# ============================================================

def merge_predictions(
    yolo_df,
    svm_df
):

    print()
    print("=" * 70)
    print("MATCHING YOLO AND SVM PREDICTIONS")
    print("=" * 70)

    merged = pd.merge(
        yolo_df,
        svm_df,
        on="filename",
        how="inner",
        validate="one_to_one",
        suffixes=(
            "_yolo",
            "_svm"
        )
    )

    if len(merged) != 225:
        raise RuntimeError(
            f"Expected 225 matched images, "
            f"found {len(merged)}"
        )

    # Verify that both models use
    # identical ground-truth labels.
    if not (
        merged["true_label_yolo"]
        ==
        merged["true_label_svm"]
    ).all():
        raise RuntimeError(
            "YOLO and SVM ground-truth "
            "labels do not match."
        )

    # Create unified true label.
    merged["true_label"] = (
        merged["true_label_yolo"]
    )

    # Normalize correctness columns.
    merged["yolo_correct"] = (
        merged["yolo_correct"]
        .astype(bool)
    )

    merged["svm_correct"] = (
        merged["svm_correct"]
        .astype(bool)
    )

    print(
        "[PASS] 225/225 predictions "
        "matched by filename."
    )

    return merged


# ============================================================
# MCNEMAR TEST
# ============================================================

def exact_mcnemar(
    yolo_correct,
    svm_correct
):

    # b:
    # YOLO correct, SVM wrong
    b = int(
        np.sum(
            yolo_correct
            & ~svm_correct
        )
    )

    # c:
    # YOLO wrong, SVM correct
    c = int(
        np.sum(
            ~yolo_correct
            & svm_correct
        )
    )

    discordant = b + c

    if discordant == 0:
        p_value = 1.0

    else:

        k = min(
            b,
            c
        )

        # Exact two-sided binomial test:
        #
        # Under H0:
        # P(discordant pair favors either model)
        # = 0.5
        #
        # P(X <= k)
        # multiplied by 2.

        lower_probability = 0.0

        for i in range(
            k + 1
        ):

            combination = math.comb(
                discordant,
                i
            )

            lower_probability += (
                combination
                * (
                    0.5
                    ** discordant
                )
            )

        p_value = min(
            1.0,
            2.0
            * lower_probability
        )

    return (
        b,
        c,
        discordant,
        p_value
    )


# ============================================================
# CREATE MCNEMAR CONTINGENCY TABLE
# ============================================================

def create_mcnemar_table(
    merged
):

    # Rows = YOLO correctness
    # Columns = SVM correctness

    table = pd.DataFrame(
        [
            [
                int(
                    np.sum(
                        merged["yolo_correct"]
                        &
                        merged["svm_correct"]
                    )
                ),
                int(
                    np.sum(
                        merged["yolo_correct"]
                        &
                        ~merged["svm_correct"]
                    )
                ),
            ],
            [
                int(
                    np.sum(
                        ~merged["yolo_correct"]
                        &
                        merged["svm_correct"]
                    )
                ),
                int(
                    np.sum(
                        ~merged["yolo_correct"]
                        &
                        ~merged["svm_correct"]
                    )
                ),
            ],
        ],
        index=[
            "YOLO Correct",
            "YOLO Incorrect",
        ],
        columns=[
            "SVM Correct",
            "SVM Incorrect",
        ]
    )

    return table


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 70)
    print("STAGE 1 YOLO vs SVM MCNEMAR COMPARISON")
    print("=" * 70)

    print()
    print(
        "NO TRAINING"
    )

    print(
        "YOLO hyperparameters: LOCKED"
    )

    print(
        "SVM configuration: LOCKED"
    )

    print(
        "Test set: SAME 225 IMAGES"
    )

    verify_files()

    dataset = (
        load_yolo_test_dataset()
    )

    yolo_df = (
        generate_yolo_predictions(
            dataset
        )
    )

    svm_df = (
        load_svm_predictions()
    )

    merged = merge_predictions(
        yolo_df,
        svm_df
    )

    # --------------------------------------------------------
    # SAVE PAIRED PREDICTIONS
    # --------------------------------------------------------

    paired_path = (
        OUTPUT_DIR
        / "stage1_yolo_svm_paired_predictions.csv"
    )

    merged[
        [
            "filename",
            "true_label",
            "yolo_predicted_label",
            "yolo_correct",
            "svm_predicted_label",
            "svm_correct",
        ]
    ].to_csv(
        paired_path,
        index=False
    )

    # --------------------------------------------------------
    # MCNEMAR
    # --------------------------------------------------------

    yolo_correct = (
        merged["yolo_correct"]
        .to_numpy()
    )

    svm_correct = (
        merged["svm_correct"]
        .to_numpy()
    )

    (
        b,
        c,
        discordant,
        p_value
    ) = exact_mcnemar(
        yolo_correct,
        svm_correct
    )

    contingency = (
        create_mcnemar_table(
            merged
        )
    )

    contingency_path = (
        OUTPUT_DIR
        / "mcnemar_contingency_table.csv"
    )

    contingency.to_csv(
        contingency_path
    )

    # --------------------------------------------------------
    # MODEL ACCURACIES
    # --------------------------------------------------------

    yolo_accuracy = float(
        yolo_correct.mean()
    )

    svm_accuracy = float(
        svm_correct.mean()
    )

    alpha = 0.05

    significant = (
        p_value < alpha
    )

    # --------------------------------------------------------
    # SAVE SUMMARY
    # --------------------------------------------------------

    summary = {
        "test_images": 225,

        "yolo_accuracy":
            yolo_accuracy,

        "svm_accuracy":
            svm_accuracy,

        "yolo_correct":
            int(yolo_correct.sum()),

        "svm_correct":
            int(svm_correct.sum()),

        "mcnemar_b":
            b,

        "mcnemar_c":
            c,

        "discordant_pairs":
            discordant,

        "mcnemar_exact_p_value":
            p_value,

        "alpha":
            alpha,

        "significant_at_alpha_0.05":
            significant,
    }

    summary_path = (
        OUTPUT_DIR
        / "mcnemar_results.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=4
        ),
        encoding="utf-8"
    )

    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("MCNEMAR CONTINGENCY TABLE")
    print("=" * 70)

    print(
        contingency
    )

    print()
    print("=" * 70)
    print("MODEL TEST ACCURACIES")
    print("=" * 70)

    print(
        f"YOLO Accuracy : "
        f"{yolo_accuracy:.4f}"
    )

    print(
        f"SVM Accuracy  : "
        f"{svm_accuracy:.4f}"
    )

    print()
    print("=" * 70)
    print("MCNEMAR'S EXACT TEST")
    print("=" * 70)

    print(
        f"b = YOLO correct / SVM wrong : "
        f"{b}"
    )

    print(
        f"c = YOLO wrong / SVM correct : "
        f"{c}"
    )

    print(
        f"Discordant pairs             : "
        f"{discordant}"
    )

    print(
        f"p-value                      : "
        f"{p_value:.6f}"
    )

    print(
        f"Alpha                        : "
        f"{alpha:.2f}"
    )

    if significant:
        print(
            "Result                       : "
            "Statistically significant"
        )
    else:
        print(
            "Result                       : "
            "Not statistically significant"
        )

    print()
    print(
        f"[SAVED] {paired_path}"
    )

    print(
        f"[SAVED] {contingency_path}"
    )

    print(
        f"[SAVED] {summary_path}"
    )

    print()
    print(
        "=" * 70
    )

    print(
        "MCNEMAR COMPARISON COMPLETE"
    )


if __name__ == "__main__":
    main()