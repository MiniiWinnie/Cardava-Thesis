from pathlib import Path
from copy import copy
from types import SimpleNamespace
import random

import cv2
import numpy as np
import torch

from ultralytics import YOLO
from ultralytics.data.dataset import YOLODataset
from ultralytics.data.augment import (
    Compose,
    Format,
    LetterBox,
    RandomFlip,
    BaseTransform,
)
from ultralytics.models.yolo.detect import (
    DetectionTrainer,
    DetectionValidator,
)


# ============================================================
# PATHS
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


# ============================================================
# FIXED TRAINING CONFIGURATION
# ============================================================

EPOCHS = 50

BATCH_SIZE = 4

IMAGE_SIZE = 640

LEARNING_RATE = 0.001

WEIGHT_DECAY = 0.0005

RANDOM_SEED = 42

DEVICE = "cuda:0"


# ============================================================
# AUGMENTATION CONFIGURATION
# ============================================================

HORIZONTAL_FLIP_P = 0.5

VERTICAL_FLIP_P = 0.5

BRIGHTNESS_LIMIT = 0.10

CONTRAST_LIMIT = 0.10

APPEARANCE_P = 0.5

NOISE_SIGMA_8BIT = 5.0

NOISE_P = 0.5


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_random_seeds():

    random.seed(RANDOM_SEED)

    np.random.seed(RANDOM_SEED)

    torch.manual_seed(RANDOM_SEED)

    if torch.cuda.is_available():

        torch.cuda.manual_seed(
            RANDOM_SEED
        )

        torch.cuda.manual_seed_all(
            RANDOM_SEED
        )


# ============================================================
# CUSTOM BRIGHTNESS + CONTRAST
# ============================================================

class RandomBrightnessContrast(
    BaseTransform
):
    """
    Apply brightness and contrast adjustment
    to the image only.

    Brightness:
        ±10%

    Contrast:
        ±10%

    Probability:
        0.5

    Bounding boxes are unchanged because these are
    appearance-only transformations.
    """

    def __init__(
        self,
        brightness=0.10,
        contrast=0.10,
        p=0.5,
    ):

        self.brightness = brightness

        self.contrast = contrast

        self.p = p


    def __call__(self, labels):

        if random.random() >= self.p:

            return labels


        image = labels["img"]


        # ----------------------------------------------------
        # Random brightness factor
        # 0.90 to 1.10
        # ----------------------------------------------------

        brightness_factor = random.uniform(

            1.0 - self.brightness,

            1.0 + self.brightness,
        )


        # ----------------------------------------------------
        # Random contrast factor
        # 0.90 to 1.10
        # ----------------------------------------------------

        contrast_factor = random.uniform(

            1.0 - self.contrast,

            1.0 + self.contrast,
        )


        # ----------------------------------------------------
        # Convert to float for adjustment
        # ----------------------------------------------------

        image_float = (
            image.astype(
                np.float32
            )
        )


        # ----------------------------------------------------
        # Contrast adjustment around the image mean
        # ----------------------------------------------------

        mean = image_float.mean(

            axis=(0, 1),

            keepdims=True
        )


        image_float = (

            (
                image_float - mean
            )

            * contrast_factor

            + mean
        )


        # ----------------------------------------------------
        # Brightness adjustment
        # ----------------------------------------------------

        image_float *= (
            brightness_factor
        )


        # ----------------------------------------------------
        # Keep valid uint8 range
        # ----------------------------------------------------

        image_float = np.clip(

            image_float,

            0,

            255
        )


        labels["img"] = (
            image_float.astype(
                np.uint8
            )
        )


        return labels


# ============================================================
# CUSTOM GAUSSIAN NOISE
# ============================================================

class GaussianNoise(
    BaseTransform
):
    """
    Add Gaussian noise to the image.

    sigma = 5 intensity levels
    on the 0-255 image scale.

    Equivalent normalized sigma:

        5 / 255

    Bounding boxes are not modified.
    """

    def __init__(
        self,
        sigma=5.0,
        p=0.5,
    ):

        self.sigma = sigma

        self.p = p


    def __call__(self, labels):

        if random.random() >= self.p:

            return labels


        image = labels["img"]


        # ----------------------------------------------------
        # Generate Gaussian noise
        # ----------------------------------------------------

        noise = np.random.normal(

            loc=0.0,

            scale=self.sigma,

            size=image.shape
        )


        # ----------------------------------------------------
        # Apply noise
        # ----------------------------------------------------

        noisy = (

            image.astype(
                np.float32
            )

            + noise
        )


        # ----------------------------------------------------
        # Clip to image range
        # ----------------------------------------------------

        noisy = np.clip(

            noisy,

            0,

            255
        )


        labels["img"] = (
            noisy.astype(
                np.uint8
            )
        )


        return labels


# ============================================================
# CUSTOM YOLO DETECTION DATASET
# ============================================================

class CustomDetectionDataset(
    YOLODataset
):
    """
    Stage 2 detection dataset.

    Training augmentations:
        1. Horizontal flip
        2. Vertical flip
        3. Brightness adjustment
        4. Contrast adjustment
        5. Gaussian noise

    No:
        - Rotation
        - Translation
        - Random crop
        - Random scaling augmentation
        - Blur
        - Perspective
        - Shear
        - Mosaic
        - MixUp
        - CutMix
        - CopyPaste
        - HSV augmentation
        - Channel swapping

    Validation:
        No augmentation.
    """

    def build_transforms(
        self,
        hyp=None
    ):

        # ====================================================
        # VALIDATION / TEST PIPELINE
        # ====================================================

        if not self.augment:

            transforms = Compose(

                [
                    # ----------------------------------------
                    # Resize/downscale while preserving the
                    # complete image and aspect ratio.
                    # ----------------------------------------

                    LetterBox(

                        new_shape=(

                            self.imgsz,

                            self.imgsz
                        ),

                        scaleup=False,
                    )
                ]
            )


        # ====================================================
        # TRAINING PIPELINE
        # ====================================================

        else:

            transforms = Compose(

                [
                    # ----------------------------------------
                    # MODEL INPUT PREPARATION
                    #
                    # This is not random scaling augmentation.
                    # It converts the original 4608 × 2592
                    # image into the 640 × 640 model frame
                    # while preserving aspect ratio.
                    # ----------------------------------------

                    LetterBox(

                        new_shape=(

                            self.imgsz,

                            self.imgsz
                        ),

                        scaleup=False,
                    ),


                    # ----------------------------------------
                    # AUGMENTATION 1
                    #
                    # Bounding-box aware horizontal flip.
                    # ----------------------------------------

                    RandomFlip(

                        p=HORIZONTAL_FLIP_P,

                        direction="horizontal",
                    ),


                    # ----------------------------------------
                    # AUGMENTATION 2
                    #
                    # Bounding-box aware vertical flip.
                    # ----------------------------------------

                    RandomFlip(

                        p=VERTICAL_FLIP_P,

                        direction="vertical",
                    ),


                    # ----------------------------------------
                    # AUGMENTATIONS 3 + 4
                    #
                    # Brightness ±10%
                    # Contrast ±10%
                    # ----------------------------------------

                    RandomBrightnessContrast(

                        brightness=BRIGHTNESS_LIMIT,

                        contrast=CONTRAST_LIMIT,

                        p=APPEARANCE_P,
                    ),


                    # ----------------------------------------
                    # AUGMENTATION 5
                    #
                    # Gaussian noise σ = 5/255.
                    # ----------------------------------------

                    GaussianNoise(

                        sigma=NOISE_SIGMA_8BIT,

                        p=NOISE_P,
                    ),
                ]
            )


        # ====================================================
        # FINAL YOLO FORMATTING
        # ====================================================

        transforms.append(

            self.format_class(

                bbox_format="xywh",

                normalize=True,

                return_mask=self.use_segments,

                return_keypoint=self.use_keypoints,

                return_obb=self.use_obb,

                batch_idx=True,

                mask_ratio=hyp.mask_ratio,

                mask_overlap=hyp.overlap_mask,

                # No channel-swapping augmentation.
                bgr=(
                    hyp.bgr
                    if self.augment
                    else 0.0
                ),
            )
        )


        return transforms


# ============================================================
# CUSTOM DETECTION TRAINER
# ============================================================

class CustomDetectionTrainer(
    DetectionTrainer
):
    """
    DetectionTrainer using the custom Stage 2 dataset.

    The validation dataloader is also built through this
    trainer, but because mode='val', the dataset receives
    augment=False and therefore uses NO augmentation.
    """

    def build_dataset(
        self,
        img_path,
        mode="train",
        batch=None,
    ):

        gs = 32


        if (
            hasattr(self, "model")
            and self.model is not None
        ):

            try:

                gs = max(

                    int(
                        self.model
                        .stride
                        .max()
                    ),

                    32
                )

            except Exception:

                gs = 32


        return CustomDetectionDataset(

            img_path=img_path,

            imgsz=self.args.imgsz,

            batch_size=batch,

            augment=(mode == "train"),

            hyp=self.args,

            rect=(mode == "val"),

            stride=gs,

            pad=(
                0.0
                if mode == "train"
                else 0.5
            ),

            prefix=f"{mode}: ",

            task="detect",

            classes=self.args.classes,

            data=self.data,

            fraction=(
                self.args.fraction
                if mode == "train"
                else 1.0
            ),
        )


# ============================================================
# DATASET DIRECTORY CHECK
# ============================================================

def check_dataset():

    required_paths = [

        DATASET_DIR / "images" / "train",

        DATASET_DIR / "images" / "val",

        DATASET_DIR / "images" / "test",

        DATASET_DIR / "labels" / "train",

        DATASET_DIR / "labels" / "val",

        DATASET_DIR / "labels" / "test",

        DATA_YAML,
    ]


    print(
        "=" * 70
    )

    print(
        "STAGE 2 DATASET VERIFICATION"
    )

    print(
        "=" * 70
    )


    for path in required_paths:

        if not path.exists():

            raise FileNotFoundError(

                f"Required dataset path "
                f"not found:\n{path}"
            )


        print(
            f"[PASS] {path}"
        )


    print()


# ============================================================
# COUNT IMAGES AND LABELS
# ============================================================

def count_files():

    print(
        "=" * 70
    )

    print(
        "STAGE 2 SPLIT CHECK"
    )

    print(
        "=" * 70
    )


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


    for split, expected_count in (
        expected.items()
    ):

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

            for p in image_dir.iterdir()

            if (
                p.is_file()
                and
                p.suffix.lower()
                in extensions
            )
        )


        label_count = sum(

            1

            for p in label_dir.iterdir()

            if (
                p.is_file()
                and
                p.suffix.lower()
                == ".txt"
            )
        )


        print(

            f"{split.upper():5s} | "
            f"Images: {image_count:3d} | "
            f"Labels: {label_count:3d}"
        )


        if image_count != expected_count:

            raise ValueError(

                f"{split} image count "
                f"is {image_count}, "
                f"expected {expected_count}."
            )


        if label_count != expected_count:

            raise ValueError(

                f"{split} label count "
                f"is {label_count}, "
                f"expected {expected_count}."
            )


    print()

    print(
        "[PASS] All Stage 2 split counts "
        "match the methodology."
    )

    print()


# ============================================================
# VERIFY ONE BOX PER IMAGE
# ============================================================

def verify_annotations():

    print(
        "=" * 70
    )

    print(
        "STAGE 2 BOUNDING-BOX VERIFICATION"
    )

    print(
        "=" * 70
    )


    total_images = 0

    total_labels = 0

    invalid = 0

    wrong_box_count = 0


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


            # -----------------------------------------------
            # One whole-bunch bounding box per image.
            # -----------------------------------------------

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
                    float(v)
                    for v in parts[1:]
                ]

            except ValueError:

                invalid += 1

                continue


            # Stage 2 has one detector class:
            # class 0 = banana_bunch
            if cls != 0:

                invalid += 1

                continue


            # YOLO normalized box values
            if not all(
                0.0 <= value <= 1.0
                for value in values
            ):

                invalid += 1

                continue


            # Width and height cannot be zero.
            if (
                values[2] <= 0
                or
                values[3] <= 0
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

            "At least one image does not "
            "contain exactly one bounding box."
        )


    if total_labels != total_images:

        raise ValueError(

            "Total label count does not "
            "match total image count."
        )


    print(
        "[PASS] Exactly one valid "
        "whole-bunch bounding box "
        "exists for every image."
    )

    print()


# ============================================================
# NATIVE AUGMENTATION VERIFICATION
# ============================================================

def verify_training_arguments(
    training_kwargs
):

    print(
        "=" * 70
    )

    print(
        "STRICT M4 METHODOLOGY & "
        "AUGMENTATION VERIFICATION"
    )

    print(
        "=" * 70
    )


    # ========================================================
    # NATIVE DETECTION AUGMENTATIONS
    # ========================================================

    native_disabled = {

        "degrees": 0.0,

        "translate": 0.0,

        "scale": 0.0,

        "shear": 0.0,

        "perspective": 0.0,

        "mosaic": 0.0,

        "mixup": 0.0,

        "cutmix": 0.0,

        "copy_paste": 0.0,

        "fliplr": 0.0,

        "flipud": 0.0,

        "hsv_h": 0.0,

        "hsv_s": 0.0,

        "hsv_v": 0.0,

        "bgr": 0.0,

        "multi_scale": 0.0,
    }


    for arg, expected in (
        native_disabled.items()
    ):

        actual = (
            training_kwargs
            .get(arg)
        )


        if actual != expected:

            raise ValueError(

                f"THESIS RULE VIOLATION: "
                f"'{arg}' = {actual}, "
                f"expected {expected}."
            )


    print(
        "[PASS] All native detection "
        "augmentations are disabled."
    )


    # ========================================================
    # VERIFY OUR EXACT CONFIGURATION
    # ========================================================

    assert (
        HORIZONTAL_FLIP_P == 0.5
    )

    assert (
        VERTICAL_FLIP_P == 0.5
    )

    assert (
        BRIGHTNESS_LIMIT == 0.10
    )

    assert (
        CONTRAST_LIMIT == 0.10
    )

    assert (
        NOISE_SIGMA_8BIT == 5.0
    )

    assert (
        APPEARANCE_P == 0.5
    )

    assert (
        NOISE_P == 0.5
    )


    print(
        "[PASS] EXACT Stage 2 "
        "augmentation configuration:"
    )

    print(
        "       1. Horizontal Flip "
        "(p = 0.5)"
    )

    print(
        "       2. Vertical Flip "
        "(p = 0.5)"
    )

    print(
        "       3. Brightness "
        "(±10%, p = 0.5)"
    )

    print(
        "       4. Contrast "
        "(±10%, p = 0.5)"
    )

    print(
        "       5. Gaussian Noise "
        "(σ = 5/255, p = 0.5)"
    )


    print(
        "[PASS] Excluded:"
    )

    print(
        "       Rotation"
    )

    print(
        "       Translation"
    )

    print(
        "       Random Crop"
    )

    print(
        "       Random Scaling"
    )

    print(
        "       Gaussian Blur"
    )

    print(
        "       Perspective / Shear"
    )

    print(
        "       Mosaic / MixUp / CutMix"
    )

    print(
        "       CopyPaste / HSV augmentation"
    )


    print(
        "[PASS] Preprocessing:"
    )

    print(
        "       Original raw image"
    )

    print(
        "       LetterBox → 640 × 640"
    )

    print(
        "       Aspect ratio preserved"
    )

    print(
        "[PASS] Pre-flight verification complete."
    )

    print()


# ============================================================
# VERIFY CUSTOM TRANSFORM STRUCTURE
# ============================================================

def verify_custom_pipeline():

    print(
        "=" * 70
    )

    print(
        "CUSTOM PIPELINE STRUCTURE CHECK"
    )

    print(
        "=" * 70
    )


    # --------------------------------------------------------
    # Build a lightweight dummy dataset object without
    # scanning the real dataset.
    # --------------------------------------------------------

    dummy = (
        CustomDetectionDataset.__new__(
            CustomDetectionDataset
        )
    )


    dummy.augment = True

    dummy.rect = False

    dummy.imgsz = IMAGE_SIZE

    dummy.use_segments = False

    dummy.use_keypoints = False

    dummy.use_obb = False

    dummy.format_class = Format


    dummy_hyp = SimpleNamespace(

        bgr=0.0,

        mask_ratio=4,

        overlap_mask=True,
    )


    train_pipeline = (
        dummy.build_transforms(
            dummy_hyp
        )
    )


    train_transforms = (
        train_pipeline.transforms
    )


    # ----------------------------------------
    # Expected:
    #
    # 0 LetterBox
    # 1 H Flip
    # 2 V Flip
    # 3 B/C
    # 4 Gaussian Noise
    # 5 Format
    # ----------------------------------------

    if len(train_transforms) != 6:

        raise ValueError(

            "Unexpected training pipeline "
            f"length: {len(train_transforms)}"
        )


    assert isinstance(

        train_transforms[0],

        LetterBox
    )


    assert isinstance(

        train_transforms[1],

        RandomFlip
    )


    assert isinstance(

        train_transforms[2],

        RandomFlip
    )


    assert isinstance(

        train_transforms[3],

        RandomBrightnessContrast
    )


    assert isinstance(

        train_transforms[4],

        GaussianNoise
    )


    print(
        "[PASS] Training pipeline contains "
        "exactly the intended transformations."
    )


    # ========================================================
    # VALIDATION PIPELINE
    # ========================================================

    dummy.augment = False

    dummy.rect = True


    val_pipeline = (
        dummy.build_transforms(
            dummy_hyp
        )
    )


    val_transforms = (
        val_pipeline.transforms
    )


    # Validation should contain only:
    #
    #   LetterBox
    #   Format
    #

    if len(val_transforms) != 2:

        raise ValueError(

            "Validation pipeline contains "
            f"{len(val_transforms)} steps; "
            "expected 2."
        )


    assert isinstance(

        val_transforms[0],

        LetterBox
    )


    print(
        "[PASS] Validation pipeline contains "
        "NO augmentation."
    )


    print(
        "[PASS] Training/validation "
        "pipeline separation verified."
    )


    print()


# ============================================================
# ENVIRONMENT CHECK
# ============================================================

def check_environment():

    print(
        "=" * 70
    )

    print(
        "STAGE 2 INITIAL TRAINING"
    )

    print(
        "=" * 70
    )


    print(
        f"PyTorch Version  : "
        f"{torch.__version__}"
    )


    print(
        f"CUDA Available   : "
        f"{torch.cuda.is_available()}"
    )


    if not torch.cuda.is_available():

        raise RuntimeError(

            "CUDA is not available. "
            "Training aborted."
        )


    torch.cuda.empty_cache()


    print(
        f"GPU Model        : "
        f"{torch.cuda.get_device_name(0)}"
    )


    print(
        f"Target Device    : "
        f"{DEVICE}"
    )


    print(
        f"VRAM Allocated   : "
        f"{torch.cuda.memory_allocated(0) / 1024**2:.2f} MB"
    )


    print()


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------

    set_random_seeds()


    # --------------------------------------------------------
    # Environment
    # --------------------------------------------------------

    check_environment()


    # --------------------------------------------------------
    # Dataset checks
    # --------------------------------------------------------

    check_dataset()

    count_files()

    verify_annotations()


    # --------------------------------------------------------
    # Training arguments
    # --------------------------------------------------------

    training_kwargs = {

        # ====================================================
        # DATA
        # ====================================================

        "data": str(
            DATA_YAML
        ),


        # ====================================================
        # TRAINING
        # ====================================================

        "epochs": EPOCHS,

        "imgsz": IMAGE_SIZE,

        "batch": BATCH_SIZE,

        "optimizer": "AdamW",

        "lr0": LEARNING_RATE,

        "weight_decay": WEIGHT_DECAY,


        # ====================================================
        # HARDWARE
        # ====================================================

        "device": DEVICE,

        "seed": RANDOM_SEED,

        "deterministic": True,

        "amp": False,

        "workers": 0,


        # ====================================================
        # NO UNINTENDED GEOMETRIC AUGMENTATION
        # ====================================================

        "degrees": 0.0,

        "translate": 0.0,

        "scale": 0.0,

        "shear": 0.0,

        "perspective": 0.0,


        # ====================================================
        # NO COMPOSITE AUGMENTATION
        # ====================================================

        "mosaic": 0.0,

        "mixup": 0.0,

        "cutmix": 0.0,

        "copy_paste": 0.0,


        # ====================================================
        # NO NATIVE FLIPS
        #
        # We use our own bbox-aware RandomFlip.
        # ====================================================

        "fliplr": 0.0,

        "flipud": 0.0,


        # ====================================================
        # NO HSV / CHANNEL AUGMENTATION
        # ====================================================

        "hsv_h": 0.0,

        "hsv_s": 0.0,

        "hsv_v": 0.0,

        "bgr": 0.0,


        # ====================================================
        # NO RANDOM MULTI-SCALE TRAINING
        # ====================================================

        "multi_scale": 0.0,


        # ====================================================
        # USE CUSTOM DETECTION TRAINER
        # ====================================================

        "trainer": CustomDetectionTrainer,


        # ====================================================
        # OUTPUT
        # ====================================================

        "project": str(
            RUNS_DIR
        ),

        "name": "stage2_initial",

        "exist_ok": False,

        "plots": True,

        "verbose": True,
    }


    # --------------------------------------------------------
    # Verify every rule before training
    # --------------------------------------------------------

    verify_training_arguments(
        training_kwargs
    )


    verify_custom_pipeline()


    # --------------------------------------------------------
    # Human confirmation
    # --------------------------------------------------------

    input(

        "\n[PAUSED] Stage 2 pre-flight "
        "checks passed!\n"
        "Press ENTER to start training, "
        "or Ctrl+C to abort..."
    )


    # --------------------------------------------------------
    # Load pretrained YOLO11n
    # --------------------------------------------------------

    model = YOLO(
        str(MODEL_PATH)
    )


    print(
        f"\nLoaded weights from: "
        f"{MODEL_PATH}\n"
    )


    # --------------------------------------------------------
    # Explicit model-device check
    # --------------------------------------------------------

    print(
        "[GPU VERIFICATION]"
    )


    # The model is transferred to the target device
    # by the Ultralytics trainer.
    print(
        "Target training device: "
        f"{DEVICE}"
    )


    print()


    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    results = model.train(
        **training_kwargs
    )


    # --------------------------------------------------------
    # Complete
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "STAGE 2 INITIAL TRAINING COMPLETE"
    )

    print(
        "=" * 70
    )


    print(
        f"Results saved under: "
        f"{RUNS_DIR / 'stage2_initial'}"
    )

    print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()