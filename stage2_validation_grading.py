from pathlib import Path
import re
import math
import json

import pandas as pd
from ultralytics import YOLO
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(
    r"C:\Users\saiar\Pictures\cardava_thesis"
)

DATASET_DIR = (
    PROJECT_ROOT / "dataset" / "stage2_det"
)

VAL_IMAGES_DIR = (
    DATASET_DIR / "images" / "val"
)

VAL_LABELS_DIR = (
    DATASET_DIR / "labels" / "val"
)

FINAL_RUN_DIR = (
    PROJECT_ROOT / "runs" / "stage2_final"
)

CHECKPOINT = (
    FINAL_RUN_DIR / "weights" / "best.pt"
)

OUT_DIR = (
    FINAL_RUN_DIR / "validation_grading"
)


# ============================================================
# FINAL VALIDATION SETTINGS
# ============================================================

DEVICE = "cuda:0"
IMAGE_SIZE = 640

CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.70

# ============================================================
# FROZEN TRAINING-DERIVED A/B THRESHOLD
# ============================================================

WIDTH_THRESHOLD_PX = 2733.19


# ============================================================
# EXPECTED VALIDATION DATA
# ============================================================

EXPECTED_IMAGES = 150
EXPECTED_CLASS_A = 75
EXPECTED_CLASS_B = 75


# ============================================================
# IMAGE EXTENSIONS
# ============================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


# ============================================================
# GRADE HELPERS
# ============================================================

def grade_from_filename(filename):

    stem = Path(filename).stem.lower()

    patterns_a = [
        r"(^|[_\-\s])class[_\-\s]?a([_\-\s]|$)",
        r"(^|[_\-\s])grade[_\-\s]?a([_\-\s]|$)",
    ]

    patterns_b = [
        r"(^|[_\-\s])class[_\-\s]?b([_\-\s]|$)",
        r"(^|[_\-\s])grade[_\-\s]?b([_\-\s]|$)",
    ]

    found_a = any(
        re.search(pattern, stem)
        for pattern in patterns_a
    )

    found_b = any(
        re.search(pattern, stem)
        for pattern in patterns_b
    )

    if found_a and not found_b:
        return "A"

    if found_b and not found_a:
        return "B"

    return None


def grade_from_source_location(filename):

    candidates = list(
        PROJECT_ROOT.rglob(filename)
    )

    grades = set()

    for candidate in candidates:

        parts = [
            part.lower()
            for part in candidate.parts
        ]

        # Ignore generated Stage 2/evaluation copies
        if "stage2_det" in parts:
            continue

        if "stage2_final" in parts:
            continue

        if "runs" in parts:
            continue

        if (
            "class_a" in parts
            or "class a" in parts
            or "grade_a" in parts
            or "grade a" in parts
        ):
            grades.add("A")

        if (
            "class_b" in parts
            or "class b" in parts
            or "grade_b" in parts
            or "grade b" in parts
        ):
            grades.add("B")

    if len(grades) == 1:
        return grades.pop()

    return None


def determine_true_grade(image_path):

    grade = grade_from_source_location(
        image_path.name
    )

    if grade is not None:
        return grade

    grade = grade_from_filename(
        image_path.name
    )

    if grade is not None:
        return grade

    return None


# ============================================================
# GET VALIDATION IMAGES
# ============================================================

def get_validation_images():

    if not VAL_IMAGES_DIR.exists():
        raise FileNotFoundError(
            f"Validation directory not found:\n"
            f"{VAL_IMAGES_DIR}"
        )

    images = sorted([
        p
        for p in VAL_IMAGES_DIR.iterdir()
        if p.is_file()
        and p.suffix.lower()
        in IMAGE_EXTENSIONS
    ])

    if len(images) != EXPECTED_IMAGES:
        raise ValueError(
            f"Expected {EXPECTED_IMAGES} validation images, "
            f"found {len(images)}."
        )

    print(
        f"[PASS] Validation images: {len(images)}"
    )

    return images


# ============================================================
# RESOLVE GROUND TRUTH
# ============================================================

def resolve_ground_truth(images):

    mapping = {}
    unresolved = []

    for image in images:

        grade = determine_true_grade(
            image
        )

        if grade is None:
            unresolved.append(
                image.name
            )
        else:
            mapping[image.name] = grade

    if unresolved:

        unresolved_path = (
            OUT_DIR
            / "unresolved_validation_labels.csv"
        )

        OUT_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        pd.DataFrame({
            "filename": unresolved
        }).to_csv(
            unresolved_path,
            index=False
        )

        raise ValueError(
            f"Could not determine A/B grade for "
            f"{len(unresolved)} validation images.\n"
            f"Saved list to:\n{unresolved_path}"
        )

    a_count = sum(
        grade == "A"
        for grade in mapping.values()
    )

    b_count = sum(
        grade == "B"
        for grade in mapping.values()
    )

    print(
        f"[PASS] Validation ground truth:"
        f" A={a_count}, B={b_count}"
    )

    if (
        a_count != EXPECTED_CLASS_A
        or b_count != EXPECTED_CLASS_B
    ):
        raise ValueError(
            "Expected 75 A and 75 B validation images, "
            f"found A={a_count}, B={b_count}."
        )

    return mapping


# ============================================================
# PREDICT GRADING
# ============================================================

def predict_grades(
    model,
    images,
    ground_truth
):

    rows = []

    print()
    print("=" * 70)
    print("STAGE 2 VALIDATION SIZE-GRADING")
    print("=" * 70)

    print(
        f"Frozen width threshold: "
        f"{WIDTH_THRESHOLD_PX:.2f} px"
    )

    print(
        "Rule:"
    )

    print(
        f"  width >= {WIDTH_THRESHOLD_PX:.2f} px -> A"
    )

    print(
        f"  width <  {WIDTH_THRESHOLD_PX:.2f} px -> B"
    )

    print()

    for index, image_path in enumerate(
        images,
        start=1
    ):

        true_grade = ground_truth[
            image_path.name
        ]

        try:

            results = model.predict(
                source=str(image_path),
                imgsz=IMAGE_SIZE,
                conf=CONF_THRESHOLD,
                iou=IOU_THRESHOLD,
                device=DEVICE,
                verbose=False,
            )

            result = results[0]

            if (
                result.boxes is None
                or len(result.boxes) == 0
            ):

                rows.append({
                    "filename": image_path.name,
                    "true_grade": true_grade,
                    "detected": False,
                    "confidence": 0.0,
                    "bbox_width_px": math.nan,
                    "predicted_grade": "No Detection",
                    "correct": False,
                })

                print(
                    f"[{index:03d}/150] "
                    f"{image_path.name} -> NO DETECTION"
                )

                continue

            confidences = (
                result.boxes.conf
                .cpu()
                .numpy()
            )

            best_index = int(
                confidences.argmax()
            )

            xyxy = (
                result.boxes
                .xyxy[best_index]
                .cpu()
                .numpy()
            )

            confidence = float(
                confidences[best_index]
            )

            x1, y1, x2, y2 = map(
                float,
                xyxy
            )

            width_px = max(
                0.0,
                x2 - x1
            )

            if width_px >= WIDTH_THRESHOLD_PX:
                predicted_grade = "A"
            else:
                predicted_grade = "B"

            correct = (
                predicted_grade
                == true_grade
            )

            rows.append({
                "filename": image_path.name,
                "true_grade": true_grade,
                "detected": True,
                "confidence": confidence,
                "bbox_width_px": width_px,
                "predicted_grade": predicted_grade,
                "correct": correct,
            })

            print(
                f"[{index:03d}/150] "
                f"GT={true_grade} | "
                f"Width={width_px:.2f}px | "
                f"Pred={predicted_grade} | "
                f"{'CORRECT' if correct else 'WRONG'}"
            )

        except Exception as exc:

            rows.append({
                "filename": image_path.name,
                "true_grade": true_grade,
                "detected": False,
                "confidence": 0.0,
                "bbox_width_px": math.nan,
                "predicted_grade": "Error",
                "correct": False,
            })

            print(
                f"[{index:03d}/150] "
                f"{image_path.name} -> ERROR: {exc}"
            )

    return pd.DataFrame(rows)


# ============================================================
# CALCULATE METRICS
# ============================================================

def calculate_metrics(df):

    valid_predictions = df[
        df["predicted_grade"].isin(
            ["A", "B"]
        )
    ].copy()

    y_true = (
        valid_predictions[
            "true_grade"
        ].tolist()
    )

    y_pred = (
        valid_predictions[
            "predicted_grade"
        ].tolist()
    )

    accuracy = accuracy_score(
        df["true_grade"],
        df["predicted_grade"]
    )

    precision, recall, f1, support = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=["A", "B"],
            zero_division=0,
        )
    )

    macro_precision = float(
        precision.mean()
    )

    macro_recall = float(
        recall.mean()
    )

    macro_f1 = float(
        f1.mean()
    )

    cm = confusion_matrix(
        df["true_grade"],
        df["predicted_grade"],
        labels=["A", "B"]
    )

    cm_df = pd.DataFrame(
        cm,
        index=[
            "Actual A",
            "Actual B"
        ],
        columns=[
            "Predicted A",
            "Predicted B"
        ]
    )

    metrics_df = pd.DataFrame([
        [
            "validation_images",
            len(df)
        ],
        [
            "correct_predictions",
            int(df["correct"].sum())
        ],
        [
            "incorrect_predictions",
            int(
                (~df["correct"]).sum()
            )
        ],
        [
            "no_detection",
            int(
                (
                    df["predicted_grade"]
                    == "No Detection"
                ).sum()
            )
        ],
        [
            "errors",
            int(
                (
                    df["predicted_grade"]
                    == "Error"
                ).sum()
            )
        ],
        [
            "accuracy",
            accuracy
        ],
        [
            "class_A_precision",
            precision[0]
        ],
        [
            "class_A_recall",
            recall[0]
        ],
        [
            "class_A_f1",
            f1[0]
        ],
        [
            "class_B_precision",
            precision[1]
        ],
        [
            "class_B_recall",
            recall[1]
        ],
        [
            "class_B_f1",
            f1[1]
        ],
        [
            "macro_precision",
            macro_precision
        ],
        [
            "macro_recall",
            macro_recall
        ],
        [
            "macro_f1",
            macro_f1
        ],
        [
            "frozen_width_threshold_px",
            WIDTH_THRESHOLD_PX
        ],
    ],
        columns=[
            "metric",
            "value"
        ]
    )

    return metrics_df, cm_df


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    predictions,
    metrics,
    confusion
):

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    predictions_path = (
        OUT_DIR
        / "stage2_validation_grading_predictions.csv"
    )

    metrics_path = (
        OUT_DIR
        / "stage2_validation_grading_metrics.csv"
    )

    confusion_path = (
        OUT_DIR
        / "stage2_validation_grading_confusion_matrix.csv"
    )

    summary_path = (
        OUT_DIR
        / "stage2_validation_grading_summary.json"
    )

    predictions.to_csv(
        predictions_path,
        index=False
    )

    metrics.to_csv(
        metrics_path,
        index=False
    )

    confusion.to_csv(
        confusion_path
    )

    summary = {
        "validation_images": len(
            predictions
        ),
        "frozen_width_threshold_px":
            WIDTH_THRESHOLD_PX,
        "accuracy": float(
            metrics.loc[
                metrics["metric"]
                == "accuracy",
                "value"
            ].iloc[0]
        ),
        "macro_precision": float(
            metrics.loc[
                metrics["metric"]
                == "macro_precision",
                "value"
            ].iloc[0]
        ),
        "macro_recall": float(
            metrics.loc[
                metrics["metric"]
                == "macro_recall",
                "value"
            ].iloc[0]
        ),
        "macro_f1": float(
            metrics.loc[
                metrics["metric"]
                == "macro_f1",
                "value"
            ].iloc[0]
        ),
    }

    summary_path.write_text(
        json.dumps(
            summary,
            indent=4
        ),
        encoding="utf-8"
    )

    print()
    print(
        f"[SAVED] {predictions_path}"
    )

    print(
        f"[SAVED] {metrics_path}"
    )

    print(
        f"[SAVED] {confusion_path}"
    )

    print(
        f"[SAVED] {summary_path}"
    )


# ============================================================
# PRINT RESULTS
# ============================================================

def print_results(
    metrics,
    confusion
):

    def get_metric(name):

        return float(
            metrics.loc[
                metrics["metric"] == name,
                "value"
            ].iloc[0]
        )

    print()
    print("=" * 70)
    print("FINAL VALIDATION A/B GRADING RESULTS")
    print("=" * 70)

    print(
        f"Accuracy        : "
        f"{get_metric('accuracy'):.4f}"
    )

    print()
    print("CLASS A")

    print(
        f"Precision       : "
        f"{get_metric('class_A_precision'):.4f}"
    )

    print(
        f"Recall          : "
        f"{get_metric('class_A_recall'):.4f}"
    )

    print(
        f"F1              : "
        f"{get_metric('class_A_f1'):.4f}"
    )

    print()
    print("CLASS B")

    print(
        f"Precision       : "
        f"{get_metric('class_B_precision'):.4f}"
    )

    print(
        f"Recall          : "
        f"{get_metric('class_B_recall'):.4f}"
    )

    print(
        f"F1              : "
        f"{get_metric('class_B_f1'):.4f}"
    )

    print()
    print("MACRO")

    print(
        f"Precision       : "
        f"{get_metric('macro_precision'):.4f}"
    )

    print(
        f"Recall          : "
        f"{get_metric('macro_recall'):.4f}"
    )

    print(
        f"F1              : "
        f"{get_metric('macro_f1'):.4f}"
    )

    print()
    print("Confusion Matrix")
    print(confusion)

    print()
    print(
        "Validation grading evaluation complete."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("STAGE 2 VALIDATION GRADING EVALUATION")
    print("=" * 70)

    print(
        "TRAINING        : NO"
    )

    print(
        "TEST SET        : NOT USED"
    )

    print(
        f"THRESHOLD       : "
        f"{WIDTH_THRESHOLD_PX:.2f} px"
    )

    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Final checkpoint not found:\n"
            f"{CHECKPOINT}"
        )

    images = get_validation_images()

    ground_truth = resolve_ground_truth(
        images
    )

    print(
        f"[PASS] Ground truth resolved "
        f"for {len(ground_truth)} images"
    )

    model = YOLO(
        str(CHECKPOINT)
    )

    print(
        f"[PASS] Loaded final Stage 2 model:\n"
        f"{CHECKPOINT}"
    )

    predictions = predict_grades(
        model,
        images,
        ground_truth
    )

    metrics, confusion = (
        calculate_metrics(
            predictions
        )
    )

    save_results(
        predictions,
        metrics,
        confusion
    )

    print_results(
        metrics,
        confusion
    )


if __name__ == "__main__":
    main()