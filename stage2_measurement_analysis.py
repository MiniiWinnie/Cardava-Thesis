from pathlib import Path
import re
import math

import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(
    r"C:\Users\saiar\Pictures\cardava_thesis"
)

DATASET_DIR = (
    PROJECT_ROOT / "dataset" / "stage2_det"
)

TRAIN_IMAGES_DIR = (
    DATASET_DIR / "images" / "train"
)

TRAIN_LABELS_DIR = (
    DATASET_DIR / "labels" / "train"
)

OUT_DIR = (
    PROJECT_ROOT / "runs"
    / "stage2_measurement_analysis"
)


# ============================================================
# EXPECTED DATASET
# ============================================================

EXPECTED_TRAIN_IMAGES = 700
EXPECTED_TRAIN_LABELS = 700

YOLO_CLASS_ID = 0


# ============================================================
# PHYSICAL CALIBRATION
# ============================================================
#
# This is documented in the deployment protocol.
#
# 0.012658 cm/px
#
# We include it here for reference, but the main purpose
# of this script is to inspect the image-space measurements.
#
# ============================================================

CM_PER_PIXEL = 0.012658


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
# GRADE DETECTION HELPERS
# ============================================================

A_FOLDER_NAMES = {
    "class_a",
    "class a",
    "grade_a",
    "grade a",
}

B_FOLDER_NAMES = {
    "class_b",
    "class b",
    "grade_b",
    "grade b",
}


def normalize_grade(value):
    """
    Convert common representations to A or B.
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


def grade_from_filename(filename):
    """
    Look for explicit A/B class tokens in the filename.
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


def grade_from_source_path(image_name):
    """
    Search PROJECT_ROOT for the same filename, excluding generated
    Stage 2 files and common generated folders.

    If the matching source image is inside a folder explicitly named
    Class_A/Class_B or equivalent, use that as the physical grade.
    """

    excluded_parts = {
        ".venv",
        "stage2_det",
        "runs",
        "__pycache__",
    }

    candidates = []

    for path in PROJECT_ROOT.rglob(image_name):

        lower_parts = {
            p.lower()
            for p in path.parts
        }

        if lower_parts.intersection(excluded_parts):
            continue

        candidates.append(path)

    grades_found = set()

    for candidate in candidates:

        parts = [
            p.lower()
            for p in candidate.parts
        ]

        has_a = any(
            part in A_FOLDER_NAMES
            for part in parts
        )

        has_b = any(
            part in B_FOLDER_NAMES
            for part in parts
        )

        if has_a and not has_b:
            grades_found.add("A")

        elif has_b and not has_a:
            grades_found.add("B")

    if len(grades_found) == 1:
        return grades_found.pop()

    return None


def determine_grade(image_path):
    """
    Try, in order:

    1. Source-folder grade
    2. Filename grade
    """

    filename = image_path.name

    grade = grade_from_source_path(filename)

    if grade is not None:
        return grade

    grade = grade_from_filename(filename)

    if grade is not None:
        return grade

    return None


# ============================================================
# CHECK DATASET
# ============================================================

def check_dataset():
    print("=" * 70)
    print("STAGE 2 TRAINING DATA MEASUREMENT ANALYSIS")
    print("=" * 70)

    if not TRAIN_IMAGES_DIR.exists():
        raise FileNotFoundError(
            f"Training image directory not found:\n"
            f"{TRAIN_IMAGES_DIR}"
        )

    if not TRAIN_LABELS_DIR.exists():
        raise FileNotFoundError(
            f"Training label directory not found:\n"
            f"{TRAIN_LABELS_DIR}"
        )

    image_files = sorted([
        p
        for p in TRAIN_IMAGES_DIR.iterdir()
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTENSIONS
    ])

    label_files = sorted([
        p
        for p in TRAIN_LABELS_DIR.iterdir()
        if p.is_file()
        and p.suffix.lower() == ".txt"
    ])

    print(f"Training images : {len(image_files)}")
    print(f"Training labels : {len(label_files)}")

    if len(image_files) != EXPECTED_TRAIN_IMAGES:
        raise ValueError(
            f"Expected {EXPECTED_TRAIN_IMAGES} images, "
            f"found {len(image_files)}."
        )

    if len(label_files) != EXPECTED_TRAIN_LABELS:
        raise ValueError(
            f"Expected {EXPECTED_TRAIN_LABELS} labels, "
            f"found {len(label_files)}."
        )

    print(
        "[PASS] Training set contains exactly "
        "700 images and 700 labels."
    )

    return image_files


# ============================================================
# READ YOLO BOUNDING BOX
# ============================================================

def read_yolo_box(label_path):
    """
    Read exactly one normalized YOLO bounding box.

    Format:
        class x_center y_center width height
    """

    lines = [
        line.strip()
        for line in label_path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    if len(lines) != 1:
        raise ValueError(
            f"{label_path.name}: expected exactly "
            f"one bounding box, found {len(lines)}."
        )

    parts = lines[0].split()

    if len(parts) != 5:
        raise ValueError(
            f"{label_path.name}: expected 5 YOLO values."
        )

    try:
        class_id = int(parts[0])

        x_center = float(parts[1])
        y_center = float(parts[2])
        width_norm = float(parts[3])
        height_norm = float(parts[4])

    except ValueError as exc:
        raise ValueError(
            f"{label_path.name}: invalid numeric values."
        ) from exc

    if class_id != YOLO_CLASS_ID:
        raise ValueError(
            f"{label_path.name}: expected class "
            f"{YOLO_CLASS_ID}, found {class_id}."
        )

    values = [
        x_center,
        y_center,
        width_norm,
        height_norm,
    ]

    if not all(
        0.0 <= value <= 1.0
        for value in values
    ):
        raise ValueError(
            f"{label_path.name}: normalized values "
            f"must be between 0 and 1."
        )

    if width_norm <= 0 or height_norm <= 0:
        raise ValueError(
            f"{label_path.name}: bounding box "
            f"width/height must be > 0."
        )

    return (
        x_center,
        y_center,
        width_norm,
        height_norm,
    )


# ============================================================
# BUILD MEASUREMENT DATASET
# ============================================================

def build_measurement_dataframe(image_files):

    records = []
    unresolved = []

    for index, image_path in enumerate(
        image_files,
        start=1
    ):

        label_path = (
            TRAIN_LABELS_DIR
            / f"{image_path.stem}.txt"
        )

        if not label_path.exists():
            raise FileNotFoundError(
                f"Missing label for {image_path.name}"
            )

        true_grade = determine_grade(image_path)

        if true_grade is None:
            unresolved.append(
                image_path.name
            )
            continue

        try:
            with Image.open(image_path) as img:
                image_width, image_height = (
                    img.size
                )

        except Exception as exc:
            raise RuntimeError(
                f"Could not read image "
                f"{image_path.name}"
            ) from exc

        (
            x_center,
            y_center,
            width_norm,
            height_norm,
        ) = read_yolo_box(label_path)

        # Convert normalized YOLO dimensions
        # back to the original image dimensions.
        bbox_width_px = (
            width_norm * image_width
        )

        bbox_height_px = (
            height_norm * image_height
        )

        max_dimension_px = max(
            bbox_width_px,
            bbox_height_px
        )

        min_dimension_px = min(
            bbox_width_px,
            bbox_height_px
        )

        aspect_ratio = (
            bbox_width_px / bbox_height_px
        )

        # Optional calibration reference
        width_cm = (
            bbox_width_px
            * CM_PER_PIXEL
        )

        height_cm = (
            bbox_height_px
            * CM_PER_PIXEL
        )

        max_dimension_cm = (
            max_dimension_px
            * CM_PER_PIXEL
        )

        records.append({
            "filename": image_path.name,
            "true_grade": true_grade,
            "image_width_px": image_width,
            "image_height_px": image_height,
            "bbox_width_px": bbox_width_px,
            "bbox_height_px": bbox_height_px,
            "max_dimension_px": max_dimension_px,
            "min_dimension_px": min_dimension_px,
            "aspect_ratio_width_height": aspect_ratio,
            "bbox_width_cm_reference": width_cm,
            "bbox_height_cm_reference": height_cm,
            "max_dimension_cm_reference": max_dimension_cm,
        })

        if index % 50 == 0:
            print(
                f"Processed {index}/"
                f"{len(image_files)} images..."
            )

    if unresolved:
        unresolved_path = (
            OUT_DIR
            / "unresolved_ground_truth.csv"
        )

        pd.DataFrame({
            "filename": unresolved
        }).to_csv(
            unresolved_path,
            index=False
        )

        raise ValueError(
            "\nCould not determine A/B ground truth "
            f"for {len(unresolved)} images.\n\n"
            f"Saved unresolved list to:\n"
            f"{unresolved_path}\n\n"
            "The script stopped rather than guessing "
            "the labels."
        )

    df = pd.DataFrame(records)

    return df


# ============================================================
# SUMMARY STATISTICS
# ============================================================

def calculate_statistics(df):

    measurement_columns = [
        "bbox_width_px",
        "bbox_height_px",
        "max_dimension_px",
        "min_dimension_px",
        "aspect_ratio_width_height",
    ]

    records = []

    for grade in ["A", "B"]:

        subset = df[
            df["true_grade"] == grade
        ]

        for column in measurement_columns:

            values = subset[column]

            records.append({
                "grade": grade,
                "measurement": column,
                "count": len(values),
                "min": values.min(),
                "q1": values.quantile(0.25),
                "median": values.median(),
                "mean": values.mean(),
                "q3": values.quantile(0.75),
                "max": values.max(),
                "std": values.std(),
                "iqr": (
                    values.quantile(0.75)
                    - values.quantile(0.25)
                ),
            })

    return pd.DataFrame(records)


# ============================================================
# OVERLAP INFORMATION
# ============================================================

def calculate_overlap(df):

    measurement_columns = [
        "bbox_width_px",
        "bbox_height_px",
        "max_dimension_px",
        "min_dimension_px",
    ]

    records = []

    for column in measurement_columns:

        a = df[
            df["true_grade"] == "A"
        ][column]

        b = df[
            df["true_grade"] == "B"
        ][column]

        a_min = a.min()
        a_max = a.max()

        b_min = b.min()
        b_max = b.max()

        overlap_low = max(
            a_min,
            b_min
        )

        overlap_high = min(
            a_max,
            b_max
        )

        overlap_exists = (
            overlap_low <= overlap_high
        )

        records.append({
            "measurement": column,
            "A_min": a_min,
            "A_max": a_max,
            "B_min": b_min,
            "B_max": b_max,
            "overlap_exists": overlap_exists,
            "overlap_low": (
                overlap_low
                if overlap_exists
                else math.nan
            ),
            "overlap_high": (
                overlap_high
                if overlap_exists
                else math.nan
            ),
        })

    return pd.DataFrame(records)


# ============================================================
# FIND BEST TRAINING THRESHOLD
# ============================================================

def evaluate_thresholds(
    df,
    measurement_column,
    number_of_thresholds=1000,
):
    """
    Evaluate thresholds using TRAINING DATA ONLY.

    Rule:

        measurement >= threshold -> A
        measurement <  threshold -> B
    """

    values = df[
        measurement_column
    ].astype(float)

    min_value = values.min()
    max_value = values.max()

    thresholds = np.linspace(
        min_value,
        max_value,
        number_of_thresholds
    )

    rows = []

    y_true = df[
        "true_grade"
    ].to_numpy()

    for threshold in thresholds:

        predictions = np.where(
            values.to_numpy() >= threshold,
            "A",
            "B"
        )

        accuracy = np.mean(
            predictions == y_true
        )

        # Confusion matrix counts
        tp_a = np.sum(
            (y_true == "A")
            & (predictions == "A")
        )

        fn_a = np.sum(
            (y_true == "A")
            & (predictions == "B")
        )

        tp_b = np.sum(
            (y_true == "B")
            & (predictions == "B")
        )

        fn_b = np.sum(
            (y_true == "B")
            & (predictions == "A")
        )

        precision_a = (
            tp_a / (
                tp_a
                + np.sum(
                    (y_true == "B")
                    & (predictions == "A")
                )
            )
            if (
                tp_a
                + np.sum(
                    (y_true == "B")
                    & (predictions == "A")
                )
            ) > 0
            else 0.0
        )

        recall_a = (
            tp_a
            / (tp_a + fn_a)
            if (tp_a + fn_a) > 0
            else 0.0
        )

        precision_b = (
            tp_b / (
                tp_b
                + np.sum(
                    (y_true == "A")
                    & (predictions == "B")
                )
            )
            if (
                tp_b
                + np.sum(
                    (y_true == "A")
                    & (predictions == "B")
                )
            ) > 0
            else 0.0
        )

        recall_b = (
            tp_b
            / (tp_b + fn_b)
            if (tp_b + fn_b) > 0
            else 0.0
        )

        f1_a = (
            2
            * precision_a
            * recall_a
            / (precision_a + recall_a)
            if (
                precision_a + recall_a
            ) > 0
            else 0.0
        )

        f1_b = (
            2
            * precision_b
            * recall_b
            / (precision_b + recall_b)
            if (
                precision_b + recall_b
            ) > 0
            else 0.0
        )

        macro_f1 = (
            f1_a + f1_b
        ) / 2

        rows.append({
            "measurement": measurement_column,
            "threshold_px": threshold,
            "accuracy": accuracy,
            "A_precision": precision_a,
            "A_recall": recall_a,
            "A_f1": f1_a,
            "B_precision": precision_b,
            "B_recall": recall_b,
            "B_f1": f1_b,
            "macro_f1": macro_f1,
        })

    return pd.DataFrame(rows)


# ============================================================
# PLOT DISTRIBUTIONS
# ============================================================

def create_distribution_plot(
    df,
    column,
    title,
    xlabel,
    output_name,
):
    plt.figure(
        figsize=(10, 6)
    )

    a_values = df[
        df["true_grade"] == "A"
    ][column]

    b_values = df[
        df["true_grade"] == "B"
    ][column]

    plt.hist(
        a_values,
        bins=30,
        alpha=0.6,
        label="Class A",
    )

    plt.hist(
        b_values,
        bins=30,
        alpha=0.6,
        label="Class B",
    )

    plt.xlabel(xlabel)
    plt.ylabel("Number of Images")
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        OUT_DIR / output_name,
        dpi=200
    )

    plt.close()


# ============================================================
# BOXPLOT
# ============================================================

def create_boxplot(
    df,
    column,
    title,
    ylabel,
    output_name,
):

    a_values = df[
        df["true_grade"] == "A"
    ][column]

    b_values = df[
        df["true_grade"] == "B"
    ][column]

    plt.figure(
        figsize=(8, 6)
    )

    plt.boxplot(
        [a_values, b_values],
        tick_labels=["Class A", "Class B"],
    )

    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()

    plt.savefig(
        OUT_DIR / output_name,
        dpi=200
    )

    plt.close()


# ============================================================
# THRESHOLD PLOTS
# ============================================================

def create_threshold_plot(
    threshold_df,
    metric,
    title,
    output_name,
):

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        threshold_df["threshold_px"],
        threshold_df[metric],
    )

    plt.xlabel(
        "Candidate Threshold (pixels)"
    )

    plt.ylabel(metric)
    plt.title(title)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        OUT_DIR / output_name,
        dpi=200
    )

    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print()
    image_files = check_dataset()

    print()
    print("=" * 70)
    print("DETERMINING PHYSICAL A/B LABELS")
    print("=" * 70)

    df = build_measurement_dataframe(
        image_files
    )

    print()
    print(
        f"[PASS] Measurement rows created: "
        f"{len(df)}"
    )

    a_count = int(
        (df["true_grade"] == "A").sum()
    )

    b_count = int(
        (df["true_grade"] == "B").sum()
    )

    print(f"Class A: {a_count}")
    print(f"Class B: {b_count}")

    if a_count != 350 or b_count != 350:
        raise ValueError(
            "Expected Stage 2 training ground truth "
            "to contain 350 A and 350 B images."
        )

    print(
        "[PASS] Training distribution = "
        "350 A / 350 B"
    )

    # --------------------------------------------------------
    # SAVE RAW MEASUREMENTS
    # --------------------------------------------------------

    measurements_path = (
        OUT_DIR
        / "stage2_training_measurements.csv"
    )

    df.to_csv(
        measurements_path,
        index=False
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    stats_df = calculate_statistics(
        df
    )

    stats_path = (
        OUT_DIR
        / "stage2_training_statistics.csv"
    )

    stats_df.to_csv(
        stats_path,
        index=False
    )

    # --------------------------------------------------------
    # OVERLAP
    # --------------------------------------------------------

    overlap_df = calculate_overlap(
        df
    )

    overlap_path = (
        OUT_DIR
        / "stage2_training_overlap.csv"
    )

    overlap_df.to_csv(
        overlap_path,
        index=False
    )

    # --------------------------------------------------------
    # PRINT SUMMARY
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("TRAINING-SET MEASUREMENT SUMMARY")
    print("=" * 70)

    for measurement in [
        "bbox_width_px",
        "bbox_height_px",
        "max_dimension_px",
        "min_dimension_px",
        "aspect_ratio_width_height",
    ]:

        print()
        print(measurement)

        for grade in ["A", "B"]:

            values = df[
                df["true_grade"] == grade
            ][measurement]

            print(
                f"  Class {grade}: "
                f"min={values.min():.2f}, "
                f"median={values.median():.2f}, "
                f"mean={values.mean():.2f}, "
                f"max={values.max():.2f}, "
                f"std={values.std():.2f}"
            )

    # --------------------------------------------------------
    # THRESHOLD SEARCH
    # --------------------------------------------------------

    all_threshold_results = []

    for measurement in [
        "bbox_width_px",
        "bbox_height_px",
        "max_dimension_px",
    ]:

        result = evaluate_thresholds(
            df,
            measurement
        )

        all_threshold_results.append(
            result
        )

        # Best by training accuracy
        best_accuracy = result.sort_values(
            [
                "accuracy",
                "macro_f1",
            ],
            ascending=False
        ).iloc[0]

        # Best by macro F1
        best_f1 = result.sort_values(
            [
                "macro_f1",
                "accuracy",
            ],
            ascending=False
        ).iloc[0]

        print()
        print(
            "=" * 70
        )
        print(
            f"THRESHOLD CANDIDATES: {measurement}"
        )
        print(
            "=" * 70
        )

        print(
            "Best training accuracy:"
        )

        print(
            f"  Threshold = "
            f"{best_accuracy['threshold_px']:.2f} px"
        )

        print(
            f"  Accuracy = "
            f"{best_accuracy['accuracy']:.4f}"
        )

        print(
            f"  Macro F1 = "
            f"{best_accuracy['macro_f1']:.4f}"
        )

        print()
        print(
            "Best training macro F1:"
        )

        print(
            f"  Threshold = "
            f"{best_f1['threshold_px']:.2f} px"
        )

        print(
            f"  Accuracy = "
            f"{best_f1['accuracy']:.4f}"
        )

        print(
            f"  Macro F1 = "
            f"{best_f1['macro_f1']:.4f}"
        )

        # Save individual threshold results
        result.to_csv(
            OUT_DIR
            / f"{measurement}_thresholds.csv",
            index=False
        )

        # Plot accuracy
        create_threshold_plot(
            result,
            "accuracy",
            (
                f"Training Accuracy vs "
                f"Threshold: {measurement}"
            ),
            f"{measurement}_accuracy_vs_threshold.png",
        )

        # Plot macro F1
        create_threshold_plot(
            result,
            "macro_f1",
            (
                f"Training Macro F1 vs "
                f"Threshold: {measurement}"
            ),
            f"{measurement}_f1_vs_threshold.png",
        )

    # Combined threshold table
    threshold_all_df = pd.concat(
        all_threshold_results,
        ignore_index=True
    )

    threshold_all_df.to_csv(
        OUT_DIR
        / "all_threshold_candidates.csv",
        index=False
    )

    # --------------------------------------------------------
    # PLOTS
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("CREATING DISTRIBUTION PLOTS")
    print("=" * 70)

    create_distribution_plot(
        df,
        "bbox_width_px",
        "Class A vs Class B Bounding-Box Width",
        "Bounding-Box Width (pixels)",
        "width_distribution.png",
    )

    create_distribution_plot(
        df,
        "bbox_height_px",
        "Class A vs Class B Bounding-Box Height",
        "Bounding-Box Height (pixels)",
        "height_distribution.png",
    )

    create_distribution_plot(
        df,
        "max_dimension_px",
        "Class A vs Class B Maximum Bounding-Box Dimension",
        "Maximum Dimension (pixels)",
        "max_dimension_distribution.png",
    )

    create_boxplot(
        df,
        "bbox_width_px",
        "Class A vs Class B Bounding-Box Width",
        "Bounding-Box Width (pixels)",
        "width_boxplot.png",
    )

    create_boxplot(
        df,
        "bbox_height_px",
        "Class A vs Class B Bounding-Box Height",
        "Bounding-Box Height (pixels)",
        "height_boxplot.png",
    )

    # --------------------------------------------------------
    # FINAL MESSAGE
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)

    print()
    print("Generated files:")
    print(
        f"  {measurements_path}"
    )
    print(
        f"  {stats_path}"
    )
    print(
        f"  {overlap_path}"
    )
    print(
        f"  {OUT_DIR / 'all_threshold_candidates.csv'}"
    )

    print()
    print(
        "Important:"
    )
    print(
        "  The 150-image test set was NOT used."
    )
    print(
        "  No model training was performed."
    )
    print(
        "  No final threshold has been applied to "
        "the test set."
    )

    print()
    print(
        "Next step: inspect the A/B measurement "
        "distributions and determine the justified "
        "measurement definition and threshold."
    )


if __name__ == "__main__":
    main()