from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(
    r"C:\Users\saiar\Pictures\cardava_thesis"
)

VALIDATION_DIR = (
    PROJECT_ROOT
    / "runs"
    / "stage2_final"
    / "validation_grading"
)

PREDICTIONS_CSV = (
    VALIDATION_DIR
    / "stage2_validation_grading_predictions.csv"
)

OUT_CSV = (
    VALIDATION_DIR
    / "validation_threshold_search.csv"
)

BEST_JSON = (
    VALIDATION_DIR
    / "validation_selected_threshold.json"
)


# ============================================================
# CURRENT TRAINING-DERIVED THRESHOLD
# ============================================================

TRAINING_THRESHOLD = 2733.19


# ============================================================
# LOAD VALIDATION PREDICTIONS
# ============================================================

def load_data():

    if not PREDICTIONS_CSV.exists():
        raise FileNotFoundError(
            f"Validation predictions not found:\n"
            f"{PREDICTIONS_CSV}"
        )

    df = pd.read_csv(
        PREDICTIONS_CSV
    )

    required_columns = {
        "filename",
        "true_grade",
        "bbox_width_px",
    }

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            f"Missing required columns: "
            f"{sorted(missing)}"
        )

    df = df[
        df["bbox_width_px"].notna()
    ].copy()

    if len(df) != 150:
        raise ValueError(
            f"Expected 150 validation "
            f"predictions, found {len(df)}."
        )

    print(
        f"[PASS] Loaded {len(df)} "
        f"validation predictions."
    )

    return df


# ============================================================
# SEARCH THRESHOLDS
# ============================================================

def search_thresholds(df):

    widths = (
        df["bbox_width_px"]
        .astype(float)
        .to_numpy()
    )

    y_true = (
        df["true_grade"]
        .astype(str)
        .to_numpy()
    )

    # Search around all observed validation widths.
    # We use midpoints so each candidate represents a
    # distinct classification boundary.
    unique_widths = np.sort(
        np.unique(widths)
    )

    candidates = []

    # Include boundaries just below/above the observed range
    candidates.append(
        unique_widths[0] - 1e-6
    )

    for i in range(
        len(unique_widths) - 1
    ):
        left = unique_widths[i]
        right = unique_widths[i + 1]

        candidates.append(
            (left + right) / 2.0
        )

    candidates.append(
        unique_widths[-1] + 1e-6
    )

    rows = []

    for threshold in candidates:

        y_pred = np.where(
            widths >= threshold,
            "A",
            "B"
        )

        accuracy = accuracy_score(
            y_true,
            y_pred
        )

        precision, recall, f1, _ = (
            precision_recall_fscore_support(
                y_true,
                y_pred,
                labels=["A", "B"],
                zero_division=0,
            )
        )

        rows.append({
            "threshold_px": threshold,
            "accuracy": accuracy,
            "macro_precision":
                float(precision.mean()),
            "macro_recall":
                float(recall.mean()),
            "macro_f1":
                float(f1.mean()),
            "A_precision":
                float(precision[0]),
            "A_recall":
                float(recall[0]),
            "A_f1":
                float(f1[0]),
            "B_precision":
                float(precision[1]),
            "B_recall":
                float(recall[1]),
            "B_f1":
                float(f1[1]),
        })

    return pd.DataFrame(rows)


# ============================================================
# SELECT BEST THRESHOLD
# ============================================================

def select_best(results):

    # Primary selection criterion:
    # macro F1
    #
    # Secondary criterion:
    # accuracy
    #
    # Third criterion:
    # closeness to the original
    # training-derived threshold.
    #
    # The third criterion only resolves exact ties.

    results = results.copy()

    results["distance_from_training_threshold"] = (
        abs(
            results["threshold_px"]
            - TRAINING_THRESHOLD
        )
    )

    ranked = results.sort_values(
        by=[
            "macro_f1",
            "accuracy",
            "distance_from_training_threshold",
        ],
        ascending=[
            False,
            False,
            True,
        ],
    )

    best = ranked.iloc[0]

    return best, ranked


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("STAGE 2 VALIDATION THRESHOLD SELECTION")
    print("=" * 70)

    print(
        f"Original training threshold: "
        f"{TRAINING_THRESHOLD:.2f} px"
    )

    print(
        "Test set: NOT USED"
    )

    df = load_data()

    results = search_thresholds(
        df
    )

    best, ranked = select_best(
        results
    )

    VALIDATION_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save all threshold results
    ranked.to_csv(
        OUT_CSV,
        index=False
    )

    selected_threshold = (
        float(best["threshold_px"])
    )

    # Calculate final confusion matrix
    widths = (
        df["bbox_width_px"]
        .astype(float)
        .to_numpy()
    )

    y_true = (
        df["true_grade"]
        .astype(str)
        .to_numpy()
    )

    y_pred = np.where(
        widths >= selected_threshold,
        "A",
        "B"
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=["A", "B"]
    )

    summary = {
        "selection_set":
            "validation",
        "test_set_used":
            False,
        "training_derived_threshold_px":
            TRAINING_THRESHOLD,
        "selected_validation_threshold_px":
            selected_threshold,
        "validation_accuracy":
            float(best["accuracy"]),
        "validation_macro_precision":
            float(best["macro_precision"]),
        "validation_macro_recall":
            float(best["macro_recall"]),
        "validation_macro_f1":
            float(best["macro_f1"]),
        "class_A_precision":
            float(best["A_precision"]),
        "class_A_recall":
            float(best["A_recall"]),
        "class_A_f1":
            float(best["A_f1"]),
        "class_B_precision":
            float(best["B_precision"]),
        "class_B_recall":
            float(best["B_recall"]),
        "class_B_f1":
            float(best["B_f1"]),
        "confusion_matrix": {
            "actual_A_predicted_A":
                int(cm[0, 0]),
            "actual_A_predicted_B":
                int(cm[0, 1]),
            "actual_B_predicted_A":
                int(cm[1, 0]),
            "actual_B_predicted_B":
                int(cm[1, 1]),
        },
    }

    BEST_JSON.write_text(
        json.dumps(
            summary,
            indent=4
        ),
        encoding="utf-8"
    )

    print()
    print("=" * 70)
    print("SELECTED VALIDATION THRESHOLD")
    print("=" * 70)

    print(
        f"Threshold: "
        f"{selected_threshold:.2f} px"
    )

    print(
        f"Validation Accuracy: "
        f"{best['accuracy']:.4f}"
    )

    print(
        f"Validation Macro Precision: "
        f"{best['macro_precision']:.4f}"
    )

    print(
        f"Validation Macro Recall: "
        f"{best['macro_recall']:.4f}"
    )

    print(
        f"Validation Macro F1: "
        f"{best['macro_f1']:.4f}"
    )

    print()
    print(
        "Confusion Matrix"
    )

    print(
        "                 Predicted A   Predicted B"
    )

    print(
        f"Actual A         "
        f"{cm[0,0]:13d}"
        f"{cm[0,1]:14d}"
    )

    print(
        f"Actual B         "
        f"{cm[1,0]:13d}"
        f"{cm[1,1]:14d}"
    )

    print()
    print(
        f"[SAVED] {OUT_CSV}"
    )

    print(
        f"[SAVED] {BEST_JSON}"
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "The selected threshold is now the "
        "candidate FINAL A/B threshold."
    )

    print(
        "Do not inspect or tune it using the "
        "test set."
    )


if __name__ == "__main__":
    main()