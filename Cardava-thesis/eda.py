import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# SETTINGS
# ============================================================

DATASET_DIR = r"C:\Users\saiar\Pictures\dataset\raw"


# ============================================================
# FAST MEDIAN FUNCTION
# ============================================================

def fast_median_uint8(image_channel):
    """
    Calculates the exact median of an 8-bit image channel
    using a histogram.

    This is faster than np.median() for large images because
    an 8-bit channel only contains values from 0 to 255.
    """

    histogram = cv2.calcHist(
        [image_channel],
        [0],
        None,
        [256],
        [0, 256]
    ).flatten()

    cumulative = np.cumsum(histogram)
    total_pixels = image_channel.size

    median_position = (total_pixels + 1) / 2

    median_value = np.searchsorted(
        cumulative,
        median_position
    )

    return float(median_value)


# ============================================================
# CALCULATE STATISTICS
# ============================================================

def calculate_statistics(series):
    """
    Calculates the statistical values required for EDA.
    """

    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)

    return {
        "Mean": series.mean(),
        "Median": series.median(),
        "Standard Deviation": series.std(),
        "Q1": q1,
        "Q3": q3,
        "IQR": q3 - q1
    }


# ============================================================
# CHECK DATASET DIRECTORY
# ============================================================

if not os.path.exists(DATASET_DIR):

    print("ERROR:")
    print(f"Dataset directory does not exist:")
    print(DATASET_DIR)

    raise SystemExit


# ============================================================
# READ DATASET
# ============================================================

records = []

print("=" * 60)
print("SCANNING DATASET")
print("=" * 60)

print(f"Dataset location:")
print(DATASET_DIR)
print()


for root, dirs, files in os.walk(DATASET_DIR):

    for filename in files:

        # ----------------------------------------------------
        # Accept image file types
        # ----------------------------------------------------

        if not filename.lower().endswith(
            (
                ".png",
                ".jpg",
                ".jpeg",
                ".bmp",
                ".webp",
                ".tiff"
            )
        ):
            continue

        img_path = os.path.join(root, filename)

        # ----------------------------------------------------
        # Folder name is treated as the class name
        # ----------------------------------------------------

        class_name = os.path.basename(root)

        # ----------------------------------------------------
        # Read image
        # ----------------------------------------------------

        img = cv2.imread(img_path)

        if img is None:

            print("WARNING: Could not read image:")
            print(img_path)

            continue

        # ----------------------------------------------------
        # Get image dimensions
        # ----------------------------------------------------

        height, width = img.shape[:2]

        # ----------------------------------------------------
        # Convert image to HSV
        # ----------------------------------------------------
        # IMPORTANT:
        # This is ONLY for EDA measurements.
        # We are NOT replacing the original RGB images.
        # ----------------------------------------------------

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        saturation = hsv[:, :, 1]
        brightness = hsv[:, :, 2]

        # ----------------------------------------------------
        # Saturation statistics
        # ----------------------------------------------------

        mean_saturation = float(
            np.mean(saturation)
        )

        median_saturation = fast_median_uint8(
            saturation
        )

        # ----------------------------------------------------
        # Brightness statistics
        # ----------------------------------------------------

        mean_brightness = float(
            np.mean(brightness)
        )

        median_brightness = fast_median_uint8(
            brightness
        )

        # ----------------------------------------------------
        # Save record
        # ----------------------------------------------------

        records.append({

            "class": class_name,

            "filename": filename,

            "filepath": img_path,

            "width": width,

            "height": height,

            "mean_saturation": mean_saturation,

            "median_saturation": median_saturation,

            "mean_brightness": mean_brightness,

            "median_brightness": median_brightness
        })

        print(
            f"Processed: {class_name} / {filename}"
        )


# ============================================================
# CREATE DATAFRAME
# ============================================================

df = pd.DataFrame(records)


# ============================================================
# DATASET SUMMARY
# ============================================================

print("\n")
print("=" * 60)
print("DATASET SUMMARY")
print("=" * 60)

print(
    f"Total images: {len(df)}"
)


print("\nClass distribution:")

class_counts = df["class"].value_counts()

print(class_counts)


# ============================================================
# CHECK EXPECTED CLASSES
# ============================================================

expected_classes = {
    "Class_A",
    "Class_B",
    "Rejected"
}

actual_classes = set(df["class"].unique())

missing_classes = expected_classes - actual_classes

unexpected_classes = actual_classes - expected_classes


if missing_classes:

    print("\nWARNING:")
    print("Missing expected classes:")
    print(missing_classes)


if unexpected_classes:

    print("\nWARNING:")
    print("Unexpected class names found:")
    print(unexpected_classes)


# ============================================================
# IMAGE DIMENSIONS
# ============================================================

print("\n")
print("=" * 60)
print("IMAGE DIMENSIONS")
print("=" * 60)

dimension_summary = (
    df
    .groupby("class")[["width", "height"]]
    .describe()
)

print(dimension_summary)


# ============================================================
# REQUIRED EDA STATISTICS
# ============================================================

features = [

    "mean_brightness",

    "mean_saturation"
]


statistics_records = []


for feature in features:

    print("\n")
    print("=" * 60)
    print(feature.upper())
    print("=" * 60)

    for class_name in sorted(df["class"].unique()):

        subset = df[
            df["class"] == class_name
        ][feature]

        stats = calculate_statistics(
            subset
        )

        print(
            f"\nClass: {class_name}"
        )

        print(
            f"Mean: {stats['Mean']:.2f}"
        )

        print(
            f"Median: {stats['Median']:.2f}"
        )

        print(
            f"Standard Deviation: "
            f"{stats['Standard Deviation']:.2f}"
        )

        print(
            f"Q1: {stats['Q1']:.2f}"
        )

        print(
            f"Q3: {stats['Q3']:.2f}"
        )

        print(
            f"IQR: {stats['IQR']:.2f}"
        )

        # Save for CSV
        statistics_records.append({

            "Feature": feature,

            "Class": class_name,

            "Mean": stats["Mean"],

            "Median": stats["Median"],

            "Standard Deviation":
                stats["Standard Deviation"],

            "Q1": stats["Q1"],

            "Q3": stats["Q3"],

            "IQR": stats["IQR"]
        })


# ============================================================
# STAGE 1 CLASS MAPPING
# ============================================================
#
# Class A + Class B = Healthy
# Rejected = Unhealthy
#
# Based on your current thesis formulation.
# ============================================================

df["stage1_class"] = df["class"].replace({

    "Class_A": "Healthy",

    "Class_B": "Healthy",

    "Rejected": "Unhealthy"
})


print("\n")
print("=" * 60)
print("STAGE 1 CLASS DISTRIBUTION")
print("=" * 60)

stage1_counts = (
    df["stage1_class"]
    .value_counts()
)

print(stage1_counts)


# ============================================================
# STAGE 2 CLASS MAPPING
# ============================================================
#
# Stage 2 operates on the healthy classes only.
#
# Class A vs Class B
# ============================================================

stage2_df = df[
    df["class"].isin(
        ["Class_A", "Class_B"]
    )
].copy()


print("\n")
print("=" * 60)
print("STAGE 2 CLASS DISTRIBUTION")
print("=" * 60)

stage2_counts = (
    stage2_df["class"]
    .value_counts()
)

print(stage2_counts)


# ============================================================
# BOXPLOT — MEAN BRIGHTNESS
# ============================================================
#
# NOTE:
# There is intentionally NO plt.figure() before df.boxplot().
# This prevents the blank Figure 1 problem.
# ============================================================

df.boxplot(
    column="mean_brightness",
    by="class"
)

plt.title(
    "Mean Brightness by Class"
)

plt.suptitle("")

plt.xlabel("Class")

plt.ylabel(
    "Mean Brightness"
)

plt.tight_layout()

plt.show()


# ============================================================
# BOXplot — MEAN SATURATION
# ============================================================

df.boxplot(
    column="mean_saturation",
    by="class"
)

plt.title(
    "Mean Saturation by Class"
)

plt.suptitle("")

plt.xlabel("Class")

plt.ylabel(
    "Mean Saturation"
)

plt.tight_layout()

plt.show()


# ============================================================
# HISTOGRAM — MEAN BRIGHTNESS
# ============================================================

plt.figure()

plt.hist(
    df["mean_brightness"],
    bins=30
)

plt.xlabel(
    "Mean Brightness"
)

plt.ylabel(
    "Number of Images"
)

plt.title(
    "Distribution of Mean Image Brightness"
)

plt.tight_layout()

plt.show()


# ============================================================
# HISTOGRAM — MEAN SATURATION
# ============================================================

plt.figure()

plt.hist(
    df["mean_saturation"],
    bins=30
)

plt.xlabel(
    "Mean Saturation"
)

plt.ylabel(
    "Number of Images"
)

plt.title(
    "Distribution of Mean Image Saturation"
)

plt.tight_layout()

plt.show()


# ============================================================
# SAVE IMAGE-LEVEL EDA RESULTS
# ============================================================

df.to_csv(
    "eda_results.csv",
    index=False
)


# ============================================================
# SAVE STATISTICAL SUMMARY
# ============================================================

statistics_df = pd.DataFrame(
    statistics_records
)

statistics_df.to_csv(
    "eda_statistics.csv",
    index=False
)


# ============================================================
# SAVE CLASS DISTRIBUTIONS
# ============================================================

class_summary = pd.DataFrame({

    "Original Class": [
        "Class_A",
        "Class_B",
        "Rejected"
    ],

    "Count": [

        int(
            class_counts.get(
                "Class_A",
                0
            )
        ),

        int(
            class_counts.get(
                "Class_B",
                0
            )
        ),

        int(
            class_counts.get(
                "Rejected",
                0
            )
        )
    ]
})


class_summary.to_csv(
    "class_distribution.csv",
    index=False
)


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n")
print("=" * 60)
print("EDA COMPLETE")
print("=" * 60)

print(
    "Saved files:"
)

print(
    "1. eda_results.csv"
)

print(
    "2. eda_statistics.csv"
)

print(
    "3. class_distribution.csv"
)