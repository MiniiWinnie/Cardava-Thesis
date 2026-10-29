from pathlib import Path
import json
import re
import math

import pandas as pd
from ultralytics import YOLO
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(r"C:\Users\saiar\Pictures\cardava_thesis")

DATASET_DIR = PROJECT_ROOT / "dataset" / "stage2_det"
TEST_IMAGES_DIR = DATASET_DIR / "images" / "test"

FINAL_RUN_DIR = PROJECT_ROOT / "runs" / "stage2_final"
CHECKPOINT = FINAL_RUN_DIR / "weights" / "best.pt"

OUT_DIR = FINAL_RUN_DIR / "grading_evaluation"


# ============================================================
# FINAL SYSTEM CONFIGURATION
# ============================================================

DEVICE = "cuda:0"
IMAGE_SIZE = 640
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.70

# Documented deployment calibration constant
CM_PER_PIXEL = 0.012658

# Physical grading thresholds
CLASS_B_MIN_CM = 8.0
CLASS_A_MIN_CM = 12.0

# Convert the physical thresholds back into pixel widths
CLASS_B_MIN_PX = CLASS_B_MIN_CM / CM_PER_PIXEL
CLASS_A_MIN_PX = CLASS_A_MIN_CM / CM_PER_PIXEL


# ============================================================
# OUTPUT FILES
# ============================================================

RESULTS_CSV = OUT_DIR / "stage2_grading_predictions.csv"
METRICS_CSV = OUT_DIR / "stage2_grading_metrics.csv"
CONFUSION_CSV = OUT_DIR / "stage2_grading_confusion_matrix.csv"
SUMMARY_JSON = OUT_DIR / "stage2_grading_summary.json"
GROUND_TRUTH_TEMPLATE = OUT_DIR / "stage2_ground_truth_template.csv"


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
# NORMALIZE GRADE
# ============================================================

def normalize_grade(value):
    """
    Converts common A/B representations into 'A' or 'B'.
    Returns None when the value cannot be safely interpreted.
    """

    if value is None:
        return None

    text = str(value).strip().lower()

    if text in {
        "a",
        "class a",
        "class_a",
        "grade a",
        "grade_a",
    }:
        return "A"

    if text in {
        "b",
        "class b",
        "class_b",
        "grade b",
        "grade_b",
    }:
        return "B"

    return None


# ============================================================
# FIND LABEL FROM FILENAME
# ============================================================

def grade_from_filename(filename):
    """
    Detect A/B only when the filename contains an explicit,
    reasonably safe class/grade token.

    Examples:
        Class_A_001.jpg -> A
        class-b-012.jpg -> B
        Grade_A_25.jpg -> A
    """

    stem = Path(filename).stem.lower()

    patterns_a = [
        r"(^|[_\-\s])class[_\-\s]?a([_\-\s]|$)",
        r"(^|[_\-\s])grade[_\-\s]?a([_\-\s]|$)",
    ]

    patterns_b = [
        r"(^|[_\-\s])class[_\-\s]?b([_\-\s]|$)",
        r"(^|[_\-\s])grade[_\-\s]?b([_\-\s]|$)",
    ]

    found_a = any(re.search(pattern, stem) for pattern in patterns_a)
    found_b = any(re.search(pattern, stem) for pattern in patterns_b)

    if found_a and not found_b:
        return "A"

    if found_b and not found_a:
        return "B"

    return None


# ============================================================
# TRY TO FIND ORIGINAL SOURCE GRADE
# ============================================================

def grade_from_source_location(image_name):
    """
    Searches the project directory for the same image filename.
    If it is found inside a folder explicitly named Class_A/Class_B,
    use that folder as the ground-truth source.

    This does NOT guess based on arbitrary folder names.
    """

    candidates = list(PROJECT_ROOT.rglob(image_name))

    valid_grades = set()

    for candidate in candidates:

        # Ignore generated Stage 2 detector copies
        try:
            relative = candidate.relative_to(DATASET_DIR)
            if relative:
                continue
        except ValueError:
            pass

        parts = [p.lower() for p in candidate.parts]

        has_a = any(
            part in {
                "class_a",
                "class a",
                "grade_a",
                "grade a",
            }
            for part in parts
        )

        has_b = any(
            part in {
                "class_b",
                "class b",
                "grade_b",
                "grade b",
            }
            for part in parts
        )

        if has_a and not has_b:
            valid_grades.add("A")

        elif has_b and not has_a:
            valid_grades.add("B")

    if len(valid_grades) == 1:
        return valid_grades.pop()

    # If multiple conflicting sources were found, do not guess.
    return None


# ============================================================
# LOAD OPTIONAL GROUND-TRUTH CSV
# ============================================================

def load_ground_truth_csv():
    """
    Optional file:
        runs/stage2_final/grading_evaluation/
        stage2_test_ground_truth.csv

    Required columns:
        filename
        true_grade
    """

    csv_path = OUT_DIR / "stage2_test_ground_truth.csv"

    if not csv_path.exists():
        return {}

    df = pd.read_csv(csv_path)

    required = {"filename", "true_grade"}

    if not required.issubset(df.columns):
        raise ValueError(
            f"{csv_path} must contain columns: filename,true_grade"
        )

    mapping = {}

    for _, row in df.iterrows():

        filename = Path(str(row["filename"])).name
        grade = normalize_grade(row["true_grade"])

        if grade is None:
            raise ValueError(
                f"Invalid true_grade for {filename}: "
                f"{row['true_grade']}"
            )

        mapping[filename] = grade

    print(f"[PASS] Loaded ground truth CSV: {csv_path}")

    return mapping


# ============================================================
# RESOLVE GROUND TRUTH
# ============================================================

def resolve_ground_truth(image_files):
    """
    Safely determine the physical A/B grade for each test image.
    """

    csv_mapping = load_ground_truth_csv()

    mapping = {}
    unresolved = []

    for image_path in image_files:

        filename = image_path.name

        # Priority 1: explicit CSV
        if filename in csv_mapping:
            mapping[filename] = csv_mapping[filename]
            continue

        # Priority 2: source folder
        source_grade = grade_from_source_location(filename)

        if source_grade is not None:
            mapping[filename] = source_grade
            continue

        # Priority 3: explicit filename token
        filename_grade = grade_from_filename(filename)

        if filename_grade is not None:
            mapping[filename] = filename_grade
            continue

        unresolved.append(filename)

    if unresolved:

        OUT_DIR.mkdir(parents=True, exist_ok=True)

        template = pd.DataFrame({
            "filename": unresolved,
            "true_grade": ["" for _ in unresolved],
        })

        template.to_csv(
            GROUND_TRUTH_TEMPLATE,
            index=False
        )

        raise ValueError(
            "\nCould not safely determine the physical A/B label "
            f"for {len(unresolved)} test images.\n\n"
            "A template has been created here:\n"
            f"{GROUND_TRUTH_TEMPLATE}\n\n"
            "Fill true_grade with A or B, save it as:\n"
            f"{OUT_DIR / 'stage2_test_ground_truth.csv'}\n"
            "then run this script again."
        )

    # Verify class counts
    a_count = sum(grade == "A" for grade in mapping.values())
    b_count = sum(grade == "B" for grade in mapping.values())

    print()
    print("Ground Truth Distribution")
    print(f"  Class A : {a_count}")
    print(f"  Class B : {b_count}")

    if a_count != 75 or b_count != 75:
        raise ValueError(
            "Expected 75 Class A and 75 Class B test images, "
            f"but found A={a_count}, B={b_count}."
        )

    print("[PASS] Test ground truth = 75 A / 75 B")

    return mapping


# ============================================================
# VERIFY TEST IMAGE COUNT
# ============================================================

def get_test_images():
    if not TEST_IMAGES_DIR.exists():
        raise FileNotFoundError(
            f"Test image directory not found:\n{TEST_IMAGES_DIR}"
        )

    images = sorted([
        p
        for p in TEST_IMAGES_DIR.iterdir()
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTENSIONS
    ])

    if len(images) != 150:
        raise ValueError(
            f"Expected 150 Stage 2 test images, "
            f"found {len(images)}."
        )

    print(f"[PASS] Stage 2 test images: {len(images)}")

    return images


# ============================================================
# CONVERT SIZE TO GRADE
# ============================================================

def grade_from_length(length_cm):
    if length_cm >= CLASS_A_MIN_CM:
        return "A"

    if length_cm >= CLASS_B_MIN_CM:
        return "B"

    return "Size Rejected"


# ============================================================
# RUN DETECTION + MEASUREMENT
# ============================================================

def evaluate_images(model, image_files, ground_truth):
    rows = []

    print()
    print("=" * 70)
    print("RUNNING FINAL STAGE 2 SIZE-GRADING EVALUATION")
    print("=" * 70)
    print(f"Checkpoint       : {CHECKPOINT}")
    print(f"Device           : {DEVICE}")
    print(f"Image size       : {IMAGE_SIZE}")
    print(f"Confidence       : {CONF_THRESHOLD}")
    print(f"Calibration      : {CM_PER_PIXEL} cm/px")
    print(f"Class B minimum  : {CLASS_B_MIN_CM} cm")
    print(f"Class A minimum  : {CLASS_A_MIN_CM} cm")
    print(f"B minimum pixels : {CLASS_B_MIN_PX:.2f} px")
    print(f"A minimum pixels : {CLASS_A_MIN_PX:.2f} px")
    print()

    for index, image_path in enumerate(image_files, start=1):

        true_grade = ground_truth[image_path.name]

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

            if result.boxes is None or len(result.boxes) == 0:

                rows.append({
                    "filename": image_path.name,
                    "true_grade": true_grade,
                    "detected": False,
                    "confidence": 0.0,
                    "x1_px": math.nan,
                    "y1_px": math.nan,
                    "x2_px": math.nan,
                    "y2_px": math.nan,
                    "bbox_width_px": math.nan,
                    "length_cm": math.nan,
                    "predicted_grade": "No Detection",
                    "correct": False,
                })

                print(
                    f"[{index:03d}/150] "
                    f"{image_path.name} -> NO DETECTION"
                )

                continue

            # Get the highest-confidence detection
            confidences = result.boxes.conf.cpu().numpy()

            best_index = int(confidences.argmax())

            xyxy = (
                result.boxes.xyxy[best_index]
                .cpu()
                .numpy()
            )

            confidence = float(confidences[best_index])

            x1, y1, x2, y2 = map(float, xyxy)

            bbox_width_px = max(0.0, x2 - x1)

            # Pixel-to-centimeter conversion
            length_cm = bbox_width_px * CM_PER_PIXEL

            predicted_grade = grade_from_length(length_cm)

            correct = predicted_grade == true_grade

            rows.append({
                "filename": image_path.name,
                "true_grade": true_grade,
                "detected": True,
                "confidence": confidence,
                "x1_px": x1,
                "y1_px": y1,
                "x2_px": x2,
                "y2_px": y2,
                "bbox_width_px": bbox_width_px,
                "length_cm": length_cm,
                "predicted_grade": predicted_grade,
                "correct": correct,
            })

            print(
                f"[{index:03d}/150] "
                f"{image_path.name} | "
                f"GT={true_grade} | "
                f"Width={bbox_width_px:.2f}px | "
                f"Length={length_cm:.2f}cm | "
                f"Pred={predicted_grade} | "
                f"{'CORRECT' if correct else 'WRONG'}"
            )

        except Exception as exc:

            rows.append({
                "filename": image_path.name,
                "true_grade": true_grade,
                "detected": False,
                "confidence": 0.0,
                "x1_px": math.nan,
                "y1_px": math.nan,
                "x2_px": math.nan,
                "y2_px": math.nan,
                "bbox_width_px": math.nan,
                "length_cm": math.nan,
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

    y_true = df["true_grade"].tolist()
    y_pred = df["predicted_grade"].tolist()

    # Exact A/B grading accuracy
    accuracy = accuracy_score(y_true, y_pred)

    # A/B metrics.
    # "Size Rejected", "No Detection", and "Error" count as
    # incorrect predictions and are NOT treated as A or B.
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=["A", "B"],
        zero_division=0,
    )

    macro_precision = float(precision.mean())
    macro_recall = float(recall.mean())
    macro_f1 = float(f1.mean())

    total = len(df)
    correct = int(df["correct"].sum())
    wrong = total - correct

    detected = int(df["detected"].sum())

    size_rejected = int(
        (df["predicted_grade"] == "Size Rejected").sum()
    )

    no_detection = int(
        (df["predicted_grade"] == "No Detection").sum()
    )

    errors = int(
        (df["predicted_grade"] == "Error").sum()
    )

    # Explicit 2x3 grading table
    cm = pd.DataFrame(
        0,
        index=["Actual A", "Actual B"],
        columns=[
            "Predicted A",
            "Predicted B",
            "Predicted Size Rejected",
        ],
    )

    for _, row in df.iterrows():

        true_grade = row["true_grade"]
        pred_grade = row["predicted_grade"]

        row_name = (
            "Actual A"
            if true_grade == "A"
            else "Actual B"
        )

        if pred_grade == "A":
            cm.loc[row_name, "Predicted A"] += 1

        elif pred_grade == "B":
            cm.loc[row_name, "Predicted B"] += 1

        elif pred_grade == "Size Rejected":
            cm.loc[row_name, "Predicted Size Rejected"] += 1

    metrics = pd.DataFrame([
        ["test_images", total],
        ["detected_images", detected],
        ["no_detection", no_detection],
        ["size_rejected_predictions", size_rejected],
        ["processing_errors", errors],
        ["correct_predictions", correct],
        ["incorrect_predictions", wrong],
        ["accuracy", accuracy],
        ["class_A_precision", precision[0]],
        ["class_A_recall", recall[0]],
        ["class_A_f1", f1[0]],
        ["class_B_precision", precision[1]],
        ["class_B_recall", recall[1]],
        ["class_B_f1", f1[1]],
        ["macro_precision", macro_precision],
        ["macro_recall", macro_recall],
        ["macro_f1", macro_f1],
        ["calibration_cm_per_pixel", CM_PER_PIXEL],
        ["class_B_min_cm", CLASS_B_MIN_CM],
        ["class_A_min_cm", CLASS_A_MIN_CM],
        ["class_B_min_px", CLASS_B_MIN_PX],
        ["class_A_min_px", CLASS_A_MIN_PX],
    ], columns=["metric", "value"])

    return metrics, cm


# ============================================================
# SAVE EVERYTHING
# ============================================================

def save_results(df, metrics, cm):

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df.to_csv(
        RESULTS_CSV,
        index=False
    )

    metrics.to_csv(
        METRICS_CSV,
        index=False
    )

    cm.to_csv(
        CONFUSION_CSV
    )

    summary = {
        "checkpoint": str(CHECKPOINT),
        "test_images": int(len(df)),
        "calibration_cm_per_pixel": CM_PER_PIXEL,
        "class_B_min_cm": CLASS_B_MIN_CM,
        "class_A_min_cm": CLASS_A_MIN_CM,
        "class_B_min_px": CLASS_B_MIN_PX,
        "class_A_min_px": CLASS_A_MIN_PX,
        "accuracy": float(
            metrics.loc[
                metrics["metric"] == "accuracy",
                "value"
            ].iloc[0]
        ),
        "macro_precision": float(
            metrics.loc[
                metrics["metric"] == "macro_precision",
                "value"
            ].iloc[0]
        ),
        "macro_recall": float(
            metrics.loc[
                metrics["metric"] == "macro_recall",
                "value"
            ].iloc[0]
        ),
        "macro_f1": float(
            metrics.loc[
                metrics["metric"] == "macro_f1",
                "value"
            ].iloc[0]
        ),
    }

    SUMMARY_JSON.write_text(
        json.dumps(summary, indent=4),
        encoding="utf-8"
    )


# ============================================================
# PRINT FINAL RESULTS
# ============================================================

def print_final_results(metrics, cm):

    def get_metric(name):
        return float(
            metrics.loc[
                metrics["metric"] == name,
                "value"
            ].iloc[0]
        )

    print()
    print("=" * 70)
    print("FINAL STAGE 2 SIZE-GRADING RESULTS")
    print("=" * 70)

    print(f"Test Images       : {int(get_metric('test_images'))}")
    print(f"Correct           : {int(get_metric('correct_predictions'))}")
    print(f"Incorrect         : {int(get_metric('incorrect_predictions'))}")
    print(
        f"Accuracy          : "
        f"{get_metric('accuracy'):.4f}"
    )

    print()
    print("Class A")
    print(
        f"  Precision       : "
        f"{get_metric('class_A_precision'):.4f}"
    )
    print(
        f"  Recall          : "
        f"{get_metric('class_A_recall'):.4f}"
    )
    print(
        f"  F1              : "
        f"{get_metric('class_A_f1'):.4f}"
    )

    print()
    print("Class B")
    print(
        f"  Precision       : "
        f"{get_metric('class_B_precision'):.4f}"
    )
    print(
        f"  Recall          : "
        f"{get_metric('class_B_recall'):.4f}"
    )
    print(
        f"  F1              : "
        f"{get_metric('class_B_f1'):.4f}"
    )

    print()
    print("Macro")
    print(
        f"  Precision       : "
        f"{get_metric('macro_precision'):.4f}"
    )
    print(
        f"  Recall          : "
        f"{get_metric('macro_recall'):.4f}"
    )
    print(
        f"  F1              : "
        f"{get_metric('macro_f1'):.4f}"
    )

    print()
    print("Grading Confusion Matrix")
    print(cm)

    print()
    print(f"[SAVED] {RESULTS_CSV}")
    print(f"[SAVED] {METRICS_CSV}")
    print(f"[SAVED] {CONFUSION_CSV}")
    print(f"[SAVED] {SUMMARY_JSON}")

    print()
    print("STAGE 2 SIZE-GRADING EVALUATION COMPLETE")


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("STAGE 2 SIZE-GRADING EVALUATION")
    print("=" * 70)

    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Final checkpoint not found:\n{CHECKPOINT}"
        )

    image_files = get_test_images()

    ground_truth = resolve_ground_truth(
        image_files
    )

    print()
    print("[PASS] Ground truth labels resolved")
    print("[PASS] Test set remains untouched for final evaluation")

    model = YOLO(str(CHECKPOINT))

    print(
        f"[PASS] Loaded final model: {CHECKPOINT}"
    )

    df = evaluate_images(
        model,
        image_files,
        ground_truth
    )

    metrics, cm = calculate_metrics(df)

    save_results(
        df,
        metrics,
        cm
    )

    print_final_results(
        metrics,
        cm
    )


if __name__ == "__main__":
    main()