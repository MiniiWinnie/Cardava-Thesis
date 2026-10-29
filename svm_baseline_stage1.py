from pathlib import Path
import json
import math

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline


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

TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "val"
TEST_DIR = DATASET_DIR / "test"

OUT_DIR = (
    PROJECT_ROOT
    / "runs"
    / "stage1_svm_baseline"
)


# ============================================================
# DATASET SETTINGS
# ============================================================

EXPECTED_COUNTS = {
    "train": {
        "Healthy": 700,
        "Unhealthy": 350,
    },
    "val": {
        "Healthy": 150,
        "Unhealthy": 75,
    },
    "test": {
        "Healthy": 150,
        "Unhealthy": 75,
    },
}

IMAGE_SIZE = (224, 224)

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


# ============================================================
# SVM SEARCH SPACE
# ============================================================

C_VALUES = [
    0.1,
    1.0,
    10.0,
    100.0,
]

KERNELS = [
    "linear",
    "rbf",
]

GAMMA_VALUES = [
    "scale",
    0.01,
]

CLASS_WEIGHT_VALUES = [
    None,
    "balanced",
]


# ============================================================
# FEATURE EXTRACTION SETTINGS
# ============================================================

FOREGROUND_THRESHOLD = 8

LBP_BINS = 256


# ============================================================
# HELPERS
# ============================================================

def get_images(directory):
    if not directory.exists():
        raise FileNotFoundError(
            f"Directory not found:\n{directory}"
        )

    return sorted([
        p
        for p in directory.rglob("*")
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTENSIONS
    ])


def verify_split(split_name, split_dir):

    print()
    print("=" * 70)
    print(f"VERIFYING {split_name.upper()} SET")
    print("=" * 70)

    for class_name in ["Healthy", "Unhealthy"]:

        class_dir = (
            split_dir / class_name
        )

        if not class_dir.exists():
            raise FileNotFoundError(
                f"Missing class directory:\n"
                f"{class_dir}"
            )

        images = get_images(
            class_dir
        )

        expected = EXPECTED_COUNTS[
            split_name
        ][class_name]

        print(
            f"{class_name:10s}: "
            f"{len(images)} images"
        )

        if len(images) != expected:
            raise ValueError(
                f"{split_name}/{class_name}: "
                f"expected {expected}, "
                f"found {len(images)}"
            )

    print(
        f"[PASS] {split_name} split verified."
    )


# ============================================================
# LBP FEATURE
# ============================================================

def calculate_lbp(gray):

    """
    Basic 8-neighbor Local Binary Pattern.
    Produces a 256-bin normalized histogram.
    """

    padded = cv2.copyMakeBorder(
        gray,
        1,
        1,
        1,
        1,
        cv2.BORDER_REFLECT
    )

    center = padded[
        1:-1,
        1:-1
    ]

    neighbors = [
        padded[0:-2, 0:-2],
        padded[0:-2, 1:-1],
        padded[0:-2, 2:],
        padded[1:-1, 2:],
        padded[2:, 2:],
        padded[2:, 1:-1],
        padded[2:, 0:-2],
        padded[1:-1, 0:-2],
    ]

    lbp = np.zeros_like(
        gray,
        dtype=np.uint8
    )

    for bit, neighbor in enumerate(
        neighbors
    ):
        lbp |= (
            (neighbor >= center).astype(
                np.uint8
            )
            << bit
        )

    histogram, _ = np.histogram(
        lbp.ravel(),
        bins=256,
        range=(0, 256)
    )

    histogram = (
        histogram.astype(np.float32)
        / max(
            histogram.sum(),
            1
        )
    )

    return histogram


# ============================================================
# HANDCRAFTED COLOR + TEXTURE FEATURES
# ============================================================

def extract_features(image_path):

    image = cv2.imread(
        str(image_path)
    )

    if image is None:
        raise ValueError(
            f"Could not read image:\n"
            f"{image_path}"
        )

    # Resize to the same spatial size used by Stage 1.
    image = cv2.resize(
        image,
        IMAGE_SIZE,
        interpolation=cv2.INTER_AREA
    )

    # OpenCV BGR -> RGB
    rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    # RGB -> HSV
    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # --------------------------------------------------------
    # Foreground mask
    #
    # Stage 1 images are masked RGB images with a black
    # background. Exclude near-black background pixels.
    # --------------------------------------------------------

    foreground = (
        np.max(rgb, axis=2)
        > FOREGROUND_THRESHOLD
    )

    if foreground.sum() < 10:
        foreground = np.ones(
            gray.shape,
            dtype=bool
        )

    # ========================================================
    # COLOR FEATURES
    # ========================================================

    features = []

    for channel_index in range(3):

        values = rgb[
            :, :, channel_index
        ][foreground].astype(
            np.float32
        )

        features.append(
            float(values.mean())
        )

        features.append(
            float(values.std())
        )

    for channel_index in range(3):

        values = hsv[
            :, :, channel_index
        ][foreground].astype(
            np.float32
        )

        features.append(
            float(values.mean())
        )

        features.append(
            float(values.std())
        )

    # HSV histogram
    histogram_bins = 16

    for channel_index in range(3):

        values = hsv[
            :, :, channel_index
        ][foreground]

        if channel_index == 0:
            hist_range = (0, 180)
        else:
            hist_range = (0, 256)

        histogram, _ = np.histogram(
            values,
            bins=histogram_bins,
            range=hist_range
        )

        histogram = (
            histogram.astype(
                np.float32
            )
            / max(
                histogram.sum(),
                1
            )
        )

        features.extend(
            histogram.tolist()
        )

    # ========================================================
    # TEXTURE FEATURES
    # ========================================================

    gray_values = gray[
        foreground
    ].astype(
        np.float32
    )

    # Basic grayscale statistics
    features.append(
        float(gray_values.mean())
    )

    features.append(
        float(gray_values.std())
    )

    # Sobel edge magnitude
    sobel_x = cv2.Sobel(
        gray,
        cv2.CV_32F,
        1,
        0,
        ksize=3
    )

    sobel_y = cv2.Sobel(
        gray,
        cv2.CV_32F,
        0,
        1,
        ksize=3
    )

    magnitude = cv2.magnitude(
        sobel_x,
        sobel_y
    )

    edge_values = magnitude[
        foreground
    ]

    features.append(
        float(edge_values.mean())
    )

    features.append(
        float(edge_values.std())
    )

    # Laplacian variance
    laplacian = cv2.Laplacian(
        gray,
        cv2.CV_32F
    )

    lap_values = laplacian[
        foreground
    ]

    features.append(
        float(lap_values.var())
    )

    # LBP texture histogram
    lbp_hist = calculate_lbp(
        gray
    )

    features.extend(
        lbp_hist.tolist()
    )

    return np.asarray(
        features,
        dtype=np.float32
    )


# ============================================================
# EXTRACT SPLIT
# ============================================================

def extract_split(split_name, split_dir):

    X = []
    y = []
    filenames = []

    for class_name, label in [
        ("Healthy", 0),
        ("Unhealthy", 1),
    ]:

        class_dir = (
            split_dir / class_name
        )

        images = get_images(
            class_dir
        )

        for index, image_path in enumerate(
            images,
            start=1
        ):

            features = extract_features(
                image_path
            )

            X.append(features)
            y.append(label)
            filenames.append(
                image_path.name
            )

        print(
            f"[PASS] Extracted {class_name} "
            f"features: {len(images)}"
        )

    X = np.asarray(
        X,
        dtype=np.float32
    )

    y = np.asarray(
        y,
        dtype=np.int64
    )

    return X, y, filenames


# ============================================================
# SAVE FEATURE CACHE
# ============================================================

def save_feature_cache(
    name,
    X,
    y,
    filenames
):

    np.savez_compressed(
        OUT_DIR / f"{name}_features.npz",
        X=X,
        y=y,
        filenames=np.asarray(
            filenames
        )
    )


# ============================================================
# VALIDATION SEARCH
# ============================================================

def run_validation_search(
    X_train,
    y_train,
    X_val,
    y_val
):

    rows = []

    trial_number = 0

    total_trials = (
        len(C_VALUES)
        * len(KERNELS)
        * len(GAMMA_VALUES)
        * len(CLASS_WEIGHT_VALUES)
    )

    print()
    print("=" * 70)
    print("SVM VALIDATION SEARCH")
    print("=" * 70)

    print(
        f"Planned configurations: "
        f"{total_trials}"
    )

    for kernel in KERNELS:

        for C in C_VALUES:

            for gamma in GAMMA_VALUES:

                # Gamma is only meaningful for RBF.
                if kernel == "linear":
                    effective_gamma = "scale"
                else:
                    effective_gamma = gamma

                for class_weight in (
                    CLASS_WEIGHT_VALUES
                ):

                    trial_number += 1

                    pipeline = Pipeline([
                        (
                            "scaler",
                            StandardScaler()
                        ),
                        (
                            "svm",
                            SVC(
                                kernel=kernel,
                                C=C,
                                gamma=effective_gamma,
                                class_weight=class_weight,
                            )
                        )
                    ])

                    pipeline.fit(
                        X_train,
                        y_train
                    )

                    y_pred = pipeline.predict(
                        X_val
                    )

                    accuracy = (
                        accuracy_score(
                            y_val,
                            y_pred
                        )
                    )

                    precision, recall, f1, _ = (
                        precision_recall_fscore_support(
                            y_val,
                            y_pred,
                            labels=[0, 1],
                            average=None,
                            zero_division=0
                        )
                    )

                    macro_f1 = float(
                        f1.mean()
                    )

                    macro_precision = float(
                        precision.mean()
                    )

                    macro_recall = float(
                        recall.mean()
                    )

                    rows.append({
                        "trial": trial_number,
                        "kernel": kernel,
                        "C": C,
                        "gamma": effective_gamma,
                        "class_weight":
                            class_weight,
                        "validation_accuracy":
                            accuracy,
                        "validation_macro_precision":
                            macro_precision,
                        "validation_macro_recall":
                            macro_recall,
                        "validation_macro_f1":
                            macro_f1,
                    })

                    print(
                        f"[{trial_number:02d}/"
                        f"{total_trials}] "
                        f"kernel={kernel}, "
                        f"C={C}, "
                        f"gamma={effective_gamma}, "
                        f"class_weight="
                        f"{class_weight} | "
                        f"acc={accuracy:.4f} | "
                        f"macroF1="
                        f"{macro_f1:.4f}"
                    )

    results = pd.DataFrame(
        rows
    )

    # Primary criterion:
    # Macro F1.
    #
    # Secondary:
    # Accuracy.
    #
    # This prevents the 2:1 class imbalance from making
    # accuracy the only criterion.

    ranked = results.sort_values(
        by=[
            "validation_macro_f1",
            "validation_accuracy",
        ],
        ascending=[
            False,
            False,
        ]
    ).reset_index(
        drop=True
    )

    best = ranked.iloc[0]

    return ranked, best


# ============================================================
# FINAL TEST EVALUATION
# ============================================================

def evaluate_final_test(
    X_train_val,
    y_train_val,
    X_test,
    y_test,
    filenames_test,
    best_config
):

    pipeline = Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "svm",
            SVC(
                kernel=best_config["kernel"],
                C=float(
                    best_config["C"]
                ),
                gamma=best_config["gamma"],
                class_weight=(
                    None
                    if pd.isna(
                        best_config[
                            "class_weight"
                        ]
                    )
                    else best_config[
                        "class_weight"
                    ]
                ),
            )
        )
    ])

    print()
    print("=" * 70)
    print("TRAINING FINAL SVM ON TRAIN + VALIDATION")
    print("=" * 70)

    pipeline.fit(
        X_train_val,
        y_train_val
    )

    print(
        "[PASS] Final SVM trained."
    )

    y_pred = pipeline.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision, recall, f1, support = (
        precision_recall_fscore_support(
            y_test,
            y_pred,
            labels=[0, 1],
            average=None,
            zero_division=0
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
        y_test,
        y_pred,
        labels=[0, 1]
    )

    metrics = pd.DataFrame([
        [
            "test_images",
            len(y_test)
        ],
        [
            "accuracy",
            accuracy
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
            "healthy_precision",
            precision[0]
        ],
        [
            "healthy_recall",
            recall[0]
        ],
        [
            "healthy_f1",
            f1[0]
        ],
        [
            "unhealthy_precision",
            precision[1]
        ],
        [
            "unhealthy_recall",
            recall[1]
        ],
        [
            "unhealthy_f1",
            f1[1]
        ],
        [
            "healthy_support",
            support[0]
        ],
        [
            "unhealthy_support",
            support[1]
        ],
    ],
        columns=[
            "metric",
            "value"
        ]
    )

    predictions = pd.DataFrame({
        "filename":
            filenames_test,
        "true_label":
            np.where(
                y_test == 0,
                "Healthy",
                "Unhealthy"
            ),
        "predicted_label":
            np.where(
                y_pred == 0,
                "Healthy",
                "Unhealthy"
            ),
        "correct":
            y_test == y_pred
    })

    confusion = pd.DataFrame(
        cm,
        index=[
            "Actual Healthy",
            "Actual Unhealthy",
        ],
        columns=[
            "Predicted Healthy",
            "Predicted Unhealthy",
        ]
    )

    return (
        pipeline,
        metrics,
        predictions,
        confusion
    )


# ============================================================
# PLOT CONFUSION MATRIX
# ============================================================

def save_confusion_plot(cm):

    plt.figure(
        figsize=(7, 6)
    )

    plt.imshow(
        cm.values
    )

    plt.title(
        "SVM Baseline Confusion Matrix"
    )

    plt.xlabel(
        "Predicted Label"
    )

    plt.ylabel(
        "Actual Label"
    )

    plt.xticks(
        [0, 1],
        ["Healthy", "Unhealthy"]
    )

    plt.yticks(
        [0, 1],
        ["Healthy", "Unhealthy"]
    )

    for i in range(2):
        for j in range(2):

            plt.text(
                j,
                i,
                int(cm.values[i, j]),
                ha="center",
                va="center"
            )

    plt.tight_layout()

    plt.savefig(
        OUT_DIR
        / "svm_test_confusion_matrix.png",
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

    print("=" * 70)
    print("SVM BASELINE FOR STAGE 1 HEALTH SCREENING")
    print("=" * 70)

    print(
        "Training model: YES"
    )

    print(
        "Test set: FINAL EVALUATION ONLY"
    )

    print(
        "Features: handcrafted color + texture"
    )

    print(
        "Classifier: Support Vector Machine"
    )

    print()

    # --------------------------------------------------------
    # VERIFY SPLITS
    # --------------------------------------------------------

    verify_split(
        "train",
        TRAIN_DIR
    )

    verify_split(
        "val",
        VAL_DIR
    )

    verify_split(
        "test",
        TEST_DIR
    )

    # --------------------------------------------------------
    # FEATURE EXTRACTION
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("EXTRACTING TRAINING FEATURES")
    print("=" * 70)

    X_train, y_train, files_train = (
        extract_split(
            "train",
            TRAIN_DIR
        )
    )

    save_feature_cache(
        "train",
        X_train,
        y_train,
        files_train
    )

    print(
        f"[PASS] Training feature matrix: "
        f"{X_train.shape}"
    )

    print()
    print("=" * 70)
    print("EXTRACTING VALIDATION FEATURES")
    print("=" * 70)

    X_val, y_val, files_val = (
        extract_split(
            "val",
            VAL_DIR
        )
    )

    save_feature_cache(
        "val",
        X_val,
        y_val,
        files_val
    )

    print(
        f"[PASS] Validation feature matrix: "
        f"{X_val.shape}"
    )

    print()
    print("=" * 70)
    print("EXTRACTING TEST FEATURES")
    print("=" * 70)

    X_test, y_test, files_test = (
        extract_split(
            "test",
            TEST_DIR
        )
    )

    save_feature_cache(
        "test",
        X_test,
        y_test,
        files_test
    )

    print(
        f"[PASS] Test feature matrix: "
        f"{X_test.shape}"
    )

    # --------------------------------------------------------
    # SVM VALIDATION SEARCH
    # --------------------------------------------------------

    validation_results, best = (
        run_validation_search(
            X_train,
            y_train,
            X_val,
            y_val
        )
    )

    validation_results.to_csv(
        OUT_DIR
        / "svm_validation_trials.csv",
        index=False
    )

    # --------------------------------------------------------
    # BEST CONFIG
    # --------------------------------------------------------

    best_config = {
        "kernel":
            best["kernel"],
        "C":
            float(best["C"]),
        "gamma":
            best["gamma"],
        "class_weight":
            (
                None
                if pd.isna(
                    best["class_weight"]
                )
                else best["class_weight"]
            ),
        "validation_accuracy":
            float(
                best[
                    "validation_accuracy"
                ]
            ),
        "validation_macro_precision":
            float(
                best[
                    "validation_macro_precision"
                ]
            ),
        "validation_macro_recall":
            float(
                best[
                    "validation_macro_recall"
                ]
            ),
        "validation_macro_f1":
            float(
                best[
                    "validation_macro_f1"
                ]
            ),
    }

    (
        OUT_DIR
        / "svm_selected_configuration.json"
    ).write_text(
        json.dumps(
            best_config,
            indent=4
        ),
        encoding="utf-8"
    )

    print()
    print("=" * 70)
    print("SELECTED SVM CONFIGURATION")
    print("=" * 70)

    print(
        f"Kernel       : "
        f"{best_config['kernel']}"
    )

    print(
        f"C            : "
        f"{best_config['C']}"
    )

    print(
        f"Gamma        : "
        f"{best_config['gamma']}"
    )

    print(
        f"Class weight : "
        f"{best_config['class_weight']}"
    )

    print(
        f"Validation Accuracy : "
        f"{best_config['validation_accuracy']:.4f}"
    )

    print(
        f"Validation Macro F1 : "
        f"{best_config['validation_macro_f1']:.4f}"
    )

    # --------------------------------------------------------
    # TRAIN + VALIDATION
    # --------------------------------------------------------

    X_train_val = np.vstack([
        X_train,
        X_val
    ])

    y_train_val = np.concatenate([
        y_train,
        y_val
    ])

    # --------------------------------------------------------
    # FINAL TEST
    # --------------------------------------------------------

    (
        final_model,
        test_metrics,
        test_predictions,
        test_confusion
    ) = evaluate_final_test(
        X_train_val,
        y_train_val,
        X_test,
        y_test,
        files_test,
        best_config
    )

    # --------------------------------------------------------
    # SAVE TEST RESULTS
    # --------------------------------------------------------

    test_metrics.to_csv(
        OUT_DIR
        / "svm_test_metrics.csv",
        index=False
    )

    test_predictions.to_csv(
        OUT_DIR
        / "svm_test_predictions.csv",
        index=False
    )

    test_confusion.to_csv(
        OUT_DIR
        / "svm_test_confusion_matrix.csv"
    )

    save_confusion_plot(
        test_confusion
    )

    # --------------------------------------------------------
    # FINAL PRINT
    # --------------------------------------------------------

    def get_metric(name):
        return float(
            test_metrics.loc[
                test_metrics["metric"]
                == name,
                "value"
            ].iloc[0]
        )

    print()
    print("=" * 70)
    print("FINAL SVM BASELINE TEST RESULTS")
    print("=" * 70)

    print(
        f"Accuracy          : "
        f"{get_metric('accuracy'):.4f}"
    )

    print()
    print("HEALTHY")

    print(
        f"Precision         : "
        f"{get_metric('healthy_precision'):.4f}"
    )

    print(
        f"Recall            : "
        f"{get_metric('healthy_recall'):.4f}"
    )

    print(
        f"F1                : "
        f"{get_metric('healthy_f1'):.4f}"
    )

    print()
    print("UNHEALTHY")

    print(
        f"Precision         : "
        f"{get_metric('unhealthy_precision'):.4f}"
    )

    print(
        f"Recall            : "
        f"{get_metric('unhealthy_recall'):.4f}"
    )

    print(
        f"F1                : "
        f"{get_metric('unhealthy_f1'):.4f}"
    )

    print()
    print("MACRO")

    print(
        f"Precision         : "
        f"{get_metric('macro_precision'):.4f}"
    )

    print(
        f"Recall            : "
        f"{get_metric('macro_recall'):.4f}"
    )

    print(
        f"F1                : "
        f"{get_metric('macro_f1'):.4f}"
    )

    print()
    print("CONFUSION MATRIX")
    print(
        test_confusion
    )

    print()
    print(
        "Saved outputs to:"
    )

    print(
        OUT_DIR
    )

    print()
    print(
        "SVM BASELINE COMPLETE"
    )


if __name__ == "__main__":
    main()