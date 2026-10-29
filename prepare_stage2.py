import os
import shutil
import cv2
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split


# ============================================================
# SETTINGS
# ============================================================

RAW_DIR = r"dataset/raw"
CLEANED_DIR = r"dataset/cleaned"

OUTPUT_DIR = r"dataset/stage2_det"
PREVIEW_DIR = r"stage2_annotation_preview"

RANDOM_STATE = 42

# Black-background threshold
VALUE_THRESHOLD = 15

# Ignore tiny foreground artifacts
MIN_AREA_RATIO = 0.001


# ============================================================
# CLEANED -> RAW FOLDER MAPPING
# ============================================================

# IMPORTANT:
# These are the actual cleaned folder names you currently have.
#
# Cleaned images are used ONLY to generate bounding boxes.
# Raw images are used as the actual Stage 2 training images.

CLASS_FOLDER_MAP = {
    "Cleaned_Class_A_Final": "Class_A",
    "Cleaned_Class_B_S": "Class_B",
}


# ============================================================
# IMAGE EXTENSIONS
# ============================================================

IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tiff"
)


def is_image(filename):
    return filename.lower().endswith(IMAGE_EXTENSIONS)


# ============================================================
# FILENAME NORMALIZATION
# ============================================================

def normalize_filename(filename):
    """
    Simplifies filenames so cleaned images can be matched
    with their corresponding raw images.

    Example:
        Cleaned_Class_A_003.jpg
        -> classa003

        Class_A_003.jpg
        -> classa003
    """

    name = os.path.splitext(filename)[0].lower()

    prefixes = [
        "cleaned_",
        "edited_",
        "final_"
    ]

    changed = True

    while changed:

        changed = False

        for prefix in prefixes:

            if name.startswith(prefix):

                name = name[len(prefix):]

                changed = True

    name = (
        name
        .replace(" ", "")
        .replace("-", "")
        .replace("_", "")
    )

    return name


# ============================================================
# FIND MATCHING RAW IMAGE
# ============================================================

def find_matching_raw(cleaned_filename, raw_class_dir):
    """
    Finds the raw image corresponding to a cleaned image.
    """

    # --------------------------------------------------------
    # Try exact filename first
    # --------------------------------------------------------

    exact_path = os.path.join(
        raw_class_dir,
        cleaned_filename
    )

    if os.path.exists(exact_path):
        return exact_path


    # --------------------------------------------------------
    # Try normalized filename
    # --------------------------------------------------------

    target = normalize_filename(
        cleaned_filename
    )

    candidates = []

    for filename in os.listdir(raw_class_dir):

        if not is_image(filename):
            continue

        normalized = normalize_filename(
            filename
        )

        if normalized == target:

            candidates.append(
                os.path.join(
                    raw_class_dir,
                    filename
                )
            )


    if len(candidates) == 1:
        return candidates[0]


    if len(candidates) > 1:

        print(
            f"WARNING: Multiple raw matches for "
            f"{cleaned_filename}"
        )

        for candidate in candidates:
            print("   ", candidate)

        return None


    return None


# ============================================================
# CREATE FOREGROUND MASK
# ============================================================

def create_bunch_mask(image):
    """
    Uses the black background of the cleaned image
    to isolate the banana bunch.
    """

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    value = hsv[:, :, 2]

    # Pixels brighter than the black background
    mask = np.where(
        value > VALUE_THRESHOLD,
        255,
        0
    ).astype(np.uint8)


    # --------------------------------------------------------
    # Remove small isolated noise
    # --------------------------------------------------------

    open_kernel = np.ones(
        (5, 5),
        np.uint8
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        open_kernel
    )


    # --------------------------------------------------------
    # Fill small gaps
    # --------------------------------------------------------

    close_kernel = np.ones(
        (11, 11),
        np.uint8
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        close_kernel
    )


    return mask


# ============================================================
# FIND WHOLE-BUNCH BOUNDING BOX
# ============================================================

def get_whole_bunch_bbox(image):
    """
    Finds ONE bounding box around the whole banana bunch.
    """

    mask = create_bunch_mask(
        image
    )

    image_height, image_width = image.shape[:2]

    minimum_area = (
        image_height *
        image_width *
        MIN_AREA_RATIO
    )


    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )


    valid_contours = []

    for contour in contours:

        area = cv2.contourArea(
            contour
        )

        if area >= minimum_area:

            valid_contours.append(
                contour
            )


    if not valid_contours:
        return None, mask


    # Combine all valid foreground regions.
    # The target is the WHOLE banana bunch.
    all_points = np.vstack(
        valid_contours
    )


    x, y, w, h = cv2.boundingRect(
        all_points
    )


    return (
        x,
        y,
        w,
        h
    ), mask


# ============================================================
# CONVERT BOUNDING BOX TO YOLO FORMAT
# ============================================================

def bbox_to_yolo(
    x,
    y,
    w,
    h,
    image_width,
    image_height
):
    """
    YOLO format:

    class_id center_x center_y width height

    Coordinates are normalized from 0 to 1.
    """

    center_x = (
        x + (w / 2)
    ) / image_width

    center_y = (
        y + (h / 2)
    ) / image_height

    normalized_width = (
        w / image_width
    )

    normalized_height = (
        h / image_height
    )

    return (
        0,
        center_x,
        center_y,
        normalized_width,
        normalized_height
    )


# ============================================================
# LOAD ALL CLEANED IMAGES AND BUILD ANNOTATION RECORDS
# ============================================================

records = []

processed_count = 0
failed_count = 0


print()
print("=" * 70)
print("STAGE 2 — BUILDING BOUNDING-BOX ANNOTATIONS")
print("=" * 70)


for cleaned_class_name, raw_class_name in CLASS_FOLDER_MAP.items():

    print()
    print("-" * 70)
    print(
        f"Processing cleaned folder: "
        f"{cleaned_class_name}"
    )

    print(
        f"Matching raw folder: "
        f"{raw_class_name}"
    )

    print("-" * 70)


    cleaned_class_dir = os.path.join(
        CLEANED_DIR,
        cleaned_class_name
    )


    raw_class_dir = os.path.join(
        RAW_DIR,
        raw_class_name
    )


    if not os.path.exists(
        cleaned_class_dir
    ):

        print(
            "ERROR: Cleaned folder not found:"
        )

        print(
            os.path.abspath(
                cleaned_class_dir
            )
        )

        continue


    if not os.path.exists(
        raw_class_dir
    ):

        print(
            "ERROR: Raw folder not found:"
        )

        print(
            os.path.abspath(
                raw_class_dir
            )
        )

        continue


    cleaned_files = sorted(
        [
            filename
            for filename in os.listdir(
                cleaned_class_dir
            )
            if is_image(filename)
        ]
    )


    print(
        f"Cleaned images found: "
        f"{len(cleaned_files)}"
    )


    for cleaned_filename in cleaned_files:

        cleaned_path = os.path.join(
            cleaned_class_dir,
            cleaned_filename
        )


        # ----------------------------------------------------
        # Find matching raw image
        # ----------------------------------------------------

        raw_path = find_matching_raw(
            cleaned_filename,
            raw_class_dir
        )


        if raw_path is None:

            print(
                "FAILED — no raw match:",
                cleaned_filename
            )

            failed_count += 1

            continue


        # ----------------------------------------------------
        # Read cleaned image
        # ----------------------------------------------------

        cleaned_image = cv2.imread(
            cleaned_path
        )


        if cleaned_image is None:

            print(
                "FAILED — cannot read cleaned image:",
                cleaned_path
            )

            failed_count += 1

            continue


        # ----------------------------------------------------
        # Read raw image
        # ----------------------------------------------------

        raw_image = cv2.imread(
            raw_path
        )


        if raw_image is None:

            print(
                "FAILED — cannot read raw image:",
                raw_path
            )

            failed_count += 1

            continue


        # ----------------------------------------------------
        # Verify dimensions
        # ----------------------------------------------------

        if (
            cleaned_image.shape[:2]
            != raw_image.shape[:2]
        ):

            print(
                "FAILED — dimension mismatch:"
            )

            print(
                "  Cleaned:",
                cleaned_image.shape[:2]
            )

            print(
                "  Raw:",
                raw_image.shape[:2]
            )

            failed_count += 1

            continue


        # ----------------------------------------------------
        # Generate box FROM CLEANED IMAGE
        # ----------------------------------------------------

        bbox, mask = get_whole_bunch_bbox(
            cleaned_image
        )


        if bbox is None:

            print(
                "FAILED — no bounding box:",
                cleaned_filename
            )

            failed_count += 1

            continue


        x, y, w, h = bbox


        image_height, image_width = (
            raw_image.shape[:2]
        )


        # ----------------------------------------------------
        # Convert box to YOLO coordinates
        # ----------------------------------------------------

        yolo_values = bbox_to_yolo(
            x,
            y,
            w,
            h,
            image_width,
            image_height
        )


        # ----------------------------------------------------
        # Save annotation information
        # ----------------------------------------------------

        raw_filename = os.path.basename(
            raw_path
        )


        records.append({

            "raw_filename":
                raw_filename,

            "cleaned_filename":
                cleaned_filename,

            "original_class":
                raw_class_name,

            "x":
                x,

            "y":
                y,

            "width":
                w,

            "height":
                h,

            "image_width":
                image_width,

            "image_height":
                image_height,

            "yolo_class":
                0,

            "yolo_center_x":
                yolo_values[1],

            "yolo_center_y":
                yolo_values[2],

            "yolo_width":
                yolo_values[3],

            "yolo_height":
                yolo_values[4]
        })


        processed_count += 1


print()
print("=" * 70)
print("ANNOTATION GENERATION FINISHED")
print("=" * 70)

print(
    f"Successfully processed: "
    f"{processed_count}"
)

print(
    f"Failed/skipped: "
    f"{failed_count}"
)


# ============================================================
# CREATE DATAFRAME
# ============================================================

annotations_df = pd.DataFrame(
    records
)


if annotations_df.empty:

    print()
    print(
        "ERROR: No annotations were generated."
    )

    raise SystemExit


# ============================================================
# SAVE FULL ANNOTATION RECORD
# ============================================================

os.makedirs(
    PREVIEW_DIR,
    exist_ok=True
)


annotation_record_path = os.path.join(
    PREVIEW_DIR,
    "stage2_annotation_record.csv"
)


annotations_df.to_csv(
    annotation_record_path,
    index=False
)


print()
print(
    "Annotation record saved to:"
)

print(
    os.path.abspath(
        annotation_record_path
    )
)


# ============================================================
# STRATIFIED STAGE 2 SPLIT
# ============================================================
#
# Stage 2:
# Class A = 500
# Class B = 500
#
# 70% = 700
# 15% = 150
# 15% = 150
# ============================================================

train_df, temp_df = train_test_split(
    annotations_df,
    test_size=0.30,
    stratify=annotations_df["original_class"],
    random_state=RANDOM_STATE
)


val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df["original_class"],
    random_state=RANDOM_STATE
)


# ============================================================
# PRINT SPLIT COUNTS
# ============================================================

print()
print("=" * 70)
print("STAGE 2 STRATIFIED SPLIT")
print("=" * 70)


print("\nTRAIN:")

print(
    train_df["original_class"]
    .value_counts()
)


print("\nVALIDATION:")

print(
    val_df["original_class"]
    .value_counts()
)


print("\nTEST:")

print(
    test_df["original_class"]
    .value_counts()
)


# ============================================================
# CREATE YOLO DATASET FOLDERS
# ============================================================

for split_name in [
    "train",
    "val",
    "test"
]:

    os.makedirs(
        os.path.join(
            OUTPUT_DIR,
            "images",
            split_name
        ),
        exist_ok=True
    )

    os.makedirs(
        os.path.join(
            OUTPUT_DIR,
            "labels",
            split_name
        ),
        exist_ok=True
    )


# ============================================================
# COPY RAW IMAGES + WRITE YOLO LABELS
# ============================================================

def export_split(
    split_df,
    split_name
):

    for _, row in split_df.iterrows():

        # ----------------------------------------------------
        # Original RAW image
        # ----------------------------------------------------

        source_image = os.path.join(
            RAW_DIR,
            row["original_class"],
            row["raw_filename"]
        )


        # ----------------------------------------------------
        # Destination image
        # ----------------------------------------------------

        destination_image = os.path.join(
            OUTPUT_DIR,
            "images",
            split_name,
            row["raw_filename"]
        )


        shutil.copy2(
            source_image,
            destination_image
        )


        # ----------------------------------------------------
        # Matching YOLO label
        # ----------------------------------------------------

        base_name = os.path.splitext(
            row["raw_filename"]
        )[0]


        label_path = os.path.join(
            OUTPUT_DIR,
            "labels",
            split_name,
            base_name + ".txt"
        )


        with open(
            label_path,
            "w",
            encoding="utf-8"
        ) as file:

            file.write(
                "0 "
                f"{row['yolo_center_x']:.6f} "
                f"{row['yolo_center_y']:.6f} "
                f"{row['yolo_width']:.6f} "
                f"{row['yolo_height']:.6f}\n"
            )


# ============================================================
# EXPORT ALL SPLITS
# ============================================================

export_split(
    train_df,
    "train"
)

export_split(
    val_df,
    "val"
)

export_split(
    test_df,
    "test"
)


# ============================================================
# SAVE SPLIT RECORD
# ============================================================

train_df = train_df.copy()
train_df["split"] = "train"

val_df = val_df.copy()
val_df["split"] = "val"

test_df = test_df.copy()
test_df["split"] = "test"


final_split_df = pd.concat(
    [
        train_df,
        val_df,
        test_df
    ],
    ignore_index=True
)


split_record_path = os.path.join(
    OUTPUT_DIR,
    "stage2_split_record.csv"
)


final_split_df.to_csv(
    split_record_path,
    index=False
)


# ============================================================
# CREATE data.yaml
# ============================================================

yaml_path = os.path.join(
    OUTPUT_DIR,
    "data.yaml"
)


yaml_content = f"""path: {os.path.abspath(OUTPUT_DIR).replace(os.sep, '/')}
train: images/train
val: images/val
test: images/test

names:
  0: banana_bunch
"""


with open(
    yaml_path,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        yaml_content
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("STAGE 2 DATASET PREPARATION COMPLETE")
print("=" * 70)

print(
    f"Total annotated images: "
    f"{len(annotations_df)}"
)

print(
    f"Training images: "
    f"{len(train_df)}"
)

print(
    f"Validation images: "
    f"{len(val_df)}"
)

print(
    f"Testing images: "
    f"{len(test_df)}"
)

print()
print(
    "Dataset location:"
)

print(
    os.path.abspath(
        OUTPUT_DIR
    )
)

print()
print(
    "YOLO configuration:"
)

print(
    os.path.abspath(
        yaml_path
    )
)

print()
print(
    "Split record:"
)

print(
    os.path.abspath(
        split_record_path
    )

)

print()
print(
    "Annotation record:"
)

print(
    os.path.abspath(
        annotation_record_path
    )
)