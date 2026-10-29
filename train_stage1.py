from pathlib import Path
from copy import copy

import torch
import torchvision.transforms as T

from ultralytics import YOLO
from ultralytics.data.dataset import ClassificationDataset
from ultralytics.models.yolo.classify import (
    ClassificationTrainer,
    ClassificationValidator,
)
from ultralytics.models import yolo


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(
    r"C:\Users\saiar\Pictures\cardava_thesis"
)

DATASET_DIR = (
    PROJECT_ROOT /
    "dataset" /
    "stage1_cls"
)

MODEL_PATH = (
    PROJECT_ROOT /
    "yolo11n-cls.pt"
)

RUNS_DIR = (
    PROJECT_ROOT /
    "runs"
)


# ============================================================
# FIXED TRAINING CONFIGURATION
# ============================================================

EPOCHS = 50

BATCH_SIZE = 16

IMAGE_SIZE = 224

LEARNING_RATE = 0.001

WEIGHT_DECAY = 0.0005

RANDOM_SEED = 42

DEVICE = "cuda:0"


# ============================================================
# CUSTOM GAUSSIAN NOISE
# ============================================================

class GaussianNoise:
    """
    Add Gaussian noise to a tensor image.

    Tensor values are expected to be in the [0, 1] range.

    sigma = 5 / 255 converts 5 intensity levels on the
    0-255 image scale into normalized tensor units.
    """

    def __init__(
        self,
        sigma=5 / 255.0,
        p=0.5
    ):
        self.sigma = sigma
        self.p = p

    def __call__(self, tensor):

        if torch.rand(1).item() > self.p:
            return tensor

        noise = (
            torch.randn_like(tensor) *
            self.sigma
        )

        return torch.clamp(
            tensor + noise,
            0.0,
            1.0
        )


# ============================================================
# CUSTOM CLASSIFICATION DATASET
# ============================================================

class CustomClassificationDataset(
    ClassificationDataset
):
    """
    Classification dataset with the exact augmentation
    configuration defined for the Stage 1 experiment.

    Training augmentations:
        1. Horizontal flip
        2. Vertical flip
        3. Brightness adjustment
        4. Contrast adjustment
        5. Gaussian noise

    Validation:
        No augmentation.
    """

    def __init__(
        self,
        root,
        args,
        augment=False,
        prefix=""
    ):

        # ----------------------------------------------------
        # IMPORTANT
        # ----------------------------------------------------
        #
        # Initialize the parent dataset with augment=False.
        #
        # This prevents the native Ultralytics classification
        # augmentation pipeline from being constructed.
        #
        # Our custom pipeline is assigned immediately below.
        #

        super().__init__(
            root,
            args,
            augment=False,
            prefix=prefix,
        )


        # ====================================================
        # TRAINING TRANSFORMS
        # ====================================================

        train_transforms = T.Compose(

            [
                # ------------------------------------------------
                # PREPROCESSING 1
                # Resize to the required Stage 1 input size.
                #
                # Chapter 3 specifies 224 × 224.
                #
                # This is input preparation, NOT random scaling.
                # ------------------------------------------------

                T.Resize(
                    (
                        args.imgsz,
                        args.imgsz
                    )
                ),


                # ------------------------------------------------
                # AUGMENTATION 1
                # Horizontal flip
                # ------------------------------------------------

                T.RandomHorizontalFlip(
                    p=0.5
                ),


                # ------------------------------------------------
                # AUGMENTATION 2
                # Vertical flip
                # ------------------------------------------------

                T.RandomVerticalFlip(
                    p=0.5
                ),


                # ------------------------------------------------
                # AUGMENTATIONS 3 & 4
                #
                # Brightness ±10%
                # Contrast   ±10%
                #
                # Applied together with probability 0.5.
                # ------------------------------------------------

                T.RandomApply(

                    [
                        T.ColorJitter(

                            brightness=0.10,

                            contrast=0.10,

                            saturation=0.0,

                            hue=0.0,
                        )
                    ],

                    p=0.5,
                ),


                # ------------------------------------------------
                # PREPROCESSING 2
                #
                # Convert image to tensor and normalize pixel
                # values from [0,255] to [0,1].
                # ------------------------------------------------

                T.ToTensor(),


                # ------------------------------------------------
                # AUGMENTATION 5
                #
                # Gaussian noise:
                # sigma = 5/255 normalized
                # p = 0.5
                # ------------------------------------------------

                GaussianNoise(

                    sigma=5 / 255.0,

                    p=0.5,
                ),
            ]
        )


        # ====================================================
        # VALIDATION TRANSFORMS
        # ====================================================
        #
        # Validation receives NO augmentation.
        #
        # Only the preprocessing steps defined for Stage 1
        # are retained.
        #

        val_transforms = T.Compose(

            [
                T.Resize(
                    (
                        args.imgsz,
                        args.imgsz
                    )
                ),

                T.ToTensor(),
            ]
        )


        # ====================================================
        # SELECT TRANSFORM PIPELINE
        # ====================================================

        self.torch_transforms = (

            train_transforms
            if augment
            else val_transforms
        )


# ============================================================
# CUSTOM TRAINER
# ============================================================

class CustomClassificationTrainer(
    ClassificationTrainer
):
    """
    Custom YOLO classification trainer using the
    explicitly controlled Stage 1 dataset.
    """

    def build_dataset(
        self,
        img_path,
        mode="train",
        batch=None,
    ):

        return CustomClassificationDataset(

            root=img_path,

            args=self.args,

            augment=(mode == "train"),

            prefix=mode,
        )


    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Force training-time validation to use our custom
    # validator, which uses the no-augmentation validation
    # pipeline.
    # --------------------------------------------------------

    def get_validator(self):

        return CustomClassificationValidator(

            self.test_loader,

            self.save_dir,

            args=copy(self.args),

            _callbacks=self.callbacks,
        )


# ============================================================
# CUSTOM VALIDATOR
# ============================================================

class CustomClassificationValidator(
    ClassificationValidator
):
    """
    Validation dataset for Stage 1.

    No training augmentation is applied.
    """

    def build_dataset(
        self,
        img_path
    ):

        return CustomClassificationDataset(

            root=img_path,

            args=self.args,

            augment=False,

            prefix=self.args.split,
        )


# ============================================================
# ENVIRONMENT VERIFICATION
# ============================================================

def check_environment():

    print(
        "=" * 70
    )

    print(
        "STAGE 1 INITIAL TRAINING"
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


    if torch.cuda.is_available():

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

    else:

        raise RuntimeError(
            "CUDA is not available. "
            "Training aborted."
        )

    print()


# ============================================================
# DATASET CHECK
# ============================================================

def check_dataset():

    train_dir = DATASET_DIR / "train"

    val_dir = DATASET_DIR / "val"

    test_dir = DATASET_DIR / "test"


    print(
        "Dataset directories:"
    )

    print(
        f"Train : {train_dir}"
    )

    print(
        f"Val   : {val_dir}"
    )

    print(
        f"Test  : {test_dir}"
    )

    print()


    required_dirs = [

        train_dir,

        val_dir,

        test_dir,
    ]


    for directory in required_dirs:

        if not directory.exists():

            raise FileNotFoundError(

                f"Dataset directory not found: "
                f"{directory}"
            )


    print(
        "[PASS] Dataset directories verified."
    )

    print()


# ============================================================
# METHODOLOGY VERIFICATION
# ============================================================

def verify_thesis_rules(
    training_args
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
    # 1. VERIFY NATIVE ULTRALYTICS AUGMENTATIONS ARE OFF
    # ========================================================

    native_disabled_args = {

        "auto_augment": None,

        "erasing": 0.0,

        "degrees": 0.0,

        "translate": 0.0,

        "scale": 0.0,

        "shear": 0.0,

        "perspective": 0.0,

        "mosaic": 0.0,

        "mixup": 0.0,

        "copy_paste": 0.0,

        "fliplr": 0.0,

        "flipud": 0.0,
    }


    for arg, expected in (
        native_disabled_args.items()
    ):

        actual = training_args.get(
            arg
        )

        if actual != expected:

            raise ValueError(

                "THESIS RULE VIOLATION: "
                f"Native argument '{arg}' "
                f"is set to {actual}, "
                f"but must be {expected}."
            )


    print(
        "[PASS] All native Ultralytics "
        "augmentations verified DISABLED "
        "(0.0 / None)."
    )


    # ========================================================
    # 2. BUILD DUMMY DATASET
    #    TO VERIFY CUSTOM TRANSFORMS
    # ========================================================

    class DummyArgs:

        imgsz = IMAGE_SIZE

        split = "train"

        fraction = 1.0

        cache = False

        scale = 0.0

        hsv_h = 0.0

        hsv_s = 0.0

        hsv_v = 0.0

        auto_augment = None

        erasing = 0.0

        fliplr = 0.0

        flipud = 0.0

        single_cls = False


        def __getattr__(
            self,
            item
        ):

            return None


    dummy_dataset = (
        CustomClassificationDataset(

            root=DATASET_DIR / "train",

            args=DummyArgs(),

            augment=True,
        )
    )


    transforms_list = (
        dummy_dataset
        .torch_transforms
        .transforms
    )


    # ========================================================
    # 3. EXPECT EXACTLY SIX TRANSFORM ITEMS
    #
    # 2 preprocessing:
    #   Resize
    #   ToTensor
    #
    # 4 transform blocks:
    #   Horizontal Flip
    #   Vertical Flip
    #   Brightness + Contrast
    #   Gaussian Noise
    #
    # These contain exactly 5 augmentations.
    # ========================================================

    if len(transforms_list) != 6:

        raise ValueError(

            "THESIS RULE VIOLATION: "
            "Expected exactly 6 pipeline items "
            f"but found {len(transforms_list)}."
        )


    # ========================================================
    # 4. VERIFY TRANSFORM ORDER AND TYPES
    # ========================================================

    assert isinstance(
        transforms_list[0],
        T.Resize
    ), (
        "Step 1 must be T.Resize."
    )


    assert isinstance(
        transforms_list[1],
        T.RandomHorizontalFlip
    ), (
        "Step 2 must be "
        "RandomHorizontalFlip."
    )


    assert isinstance(
        transforms_list[2],
        T.RandomVerticalFlip
    ), (
        "Step 3 must be "
        "RandomVerticalFlip."
    )


    assert isinstance(
        transforms_list[3],
        T.RandomApply
    ), (
        "Step 4 must be "
        "RandomApply containing ColorJitter."
    )


    assert isinstance(
        transforms_list[4],
        T.ToTensor
    ), (
        "Step 5 must be T.ToTensor."
    )


    assert isinstance(
        transforms_list[5],
        GaussianNoise
    ), (
        "Step 6 must be GaussianNoise."
    )


    # ========================================================
    # 5. VERIFY HORIZONTAL FLIP
    # ========================================================

    assert (
        transforms_list[1].p
        == 0.5
    ), (
        "Horizontal flip probability "
        "must be 0.5."
    )


    # ========================================================
    # 6. VERIFY VERTICAL FLIP
    # ========================================================

    assert (
        transforms_list[2].p
        == 0.5
    ), (
        "Vertical flip probability "
        "must be 0.5."
    )


    # ========================================================
    # 7. VERIFY BRIGHTNESS AND CONTRAST
    # ========================================================

    jitter = (
        transforms_list[3]
        .transforms[0]
    )


    assert (
        jitter.brightness
        == (0.9, 1.1)
    ), (
        "Brightness must be exactly "
        "±10%."
    )


    assert (
        jitter.contrast
        == (0.9, 1.1)
    ), (
        "Contrast must be exactly "
        "±10%."
    )


    # ========================================================
    # 8. VERIFY GAUSSIAN NOISE
    # ========================================================

    assert (
        transforms_list[5].sigma
        == 5 / 255.0
    ), (
        "Gaussian noise sigma must "
        "be exactly 5/255."
    )


    assert (
        transforms_list[5].p
        == 0.5
    ), (
        "Gaussian noise probability "
        "must be 0.5."
    )


    # ========================================================
    # 9. CONFIRM FINAL CONFIGURATION
    # ========================================================

    print(
        "[PASS] Custom pipeline verified."
    )

    print(
        "       EXACTLY and ONLY these "
        "5 augmentations are active:"
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
        "(±10%)"
    )

    print(
        "       4. Contrast "
        "(±10%)"
    )

    print(
        "       5. Gaussian Noise "
        "(σ = 5/255, p = 0.5)"
    )


    print(
        "[PASS] Preprocessing verified:"
    )

    print(
        "       Resize (224 × 224)"
    )

    print(
        "       ToTensor ([0, 1])"
    )


    print(
        "[PASS] Zero extra augmentations "
        "detected."
    )

    print(
        "[PASS] Pre-flight check complete."
    )

    print()


# ============================================================
# MAIN TRAINING PIPELINE
# ============================================================

def main():

    # --------------------------------------------------------
    # ENVIRONMENT
    # --------------------------------------------------------

    check_environment()


    # --------------------------------------------------------
    # DATASET
    # --------------------------------------------------------

    check_dataset()


    # ========================================================
    # CENTRALIZED TRAINING ARGUMENTS
    # ========================================================

    training_kwargs = {

        # ----------------------------------------------------
        # DATASET
        # ----------------------------------------------------

        "data": str(
            DATASET_DIR
        ),


        # ----------------------------------------------------
        # TRAINING
        # ----------------------------------------------------

        "epochs": EPOCHS,

        "imgsz": IMAGE_SIZE,

        "batch": BATCH_SIZE,

        "optimizer": "AdamW",

        "lr0": LEARNING_RATE,

        "weight_decay": WEIGHT_DECAY,


        # ----------------------------------------------------
        # HARDWARE
        # ----------------------------------------------------

        "device": DEVICE,

        "seed": RANDOM_SEED,


        # ----------------------------------------------------
        # REPRODUCIBILITY / STABILITY
        # ----------------------------------------------------

        "deterministic": True,

        "amp": False,

        "workers": 0,


        # ----------------------------------------------------
        # NATIVE ULTRALYTICS AUGMENTATIONS DISABLED
        # ----------------------------------------------------

        "auto_augment": None,

        "erasing": 0.0,

        "degrees": 0.0,

        "translate": 0.0,

        "scale": 0.0,

        "shear": 0.0,

        "perspective": 0.0,

        "mosaic": 0.0,

        "mixup": 0.0,

        "copy_paste": 0.0,

        "fliplr": 0.0,

        "flipud": 0.0,


        # ----------------------------------------------------
        # CUSTOM TRAINER
        # ----------------------------------------------------

        "trainer": CustomClassificationTrainer,


        # ----------------------------------------------------
        # OUTPUT
        # ----------------------------------------------------

        "project": str(
            RUNS_DIR
        ),

        "name": "stage1_initial",

        "exist_ok": False,

        "plots": True,

        "verbose": True,
    }


    # ========================================================
    # PREFLIGHT VERIFICATION
    # ========================================================

    verify_thesis_rules(
        training_kwargs
    )


    # ========================================================
    # HUMAN CONFIRMATION
    # ========================================================

    input(
        "\n[PAUSED] Pre-flight checks passed! "
        "Press ENTER to start training, "
        "or Ctrl+C to abort..."
    )


    # ========================================================
    # LOAD PRETRAINED YOLO11N-CLS
    # ========================================================

    model = YOLO(
        str(MODEL_PATH)
    )


    print(
        f"\nLoaded weights from: "
        f"{MODEL_PATH}\n"
    )


    # ========================================================
    # START TRAINING
    # ========================================================

    results = model.train(
        **training_kwargs
    )


    # ========================================================
    # COMPLETE
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "STAGE 1 INITIAL TRAINING COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        f"Results saved under: "
        f"{RUNS_DIR / 'stage1_initial'}"
    )

    print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()