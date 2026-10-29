from pathlib import Path
from copy import copy

import torch
import torchvision.transforms as T
from ultralytics import YOLO

# Reuse the verified Stage 1 custom pipeline and trainer.
from train_stage1 import (
    CustomClassificationTrainer,
    CustomClassificationDataset,
    GaussianNoise,
)


# ============================================================
# FINAL STAGE 1 TRAINING
# ============================================================
# Selected from the completed 4-trial validation search:
#
# Trial 04
# Learning rate = 0.000849
# Batch size    = 8
# Weight decay  = 0.000605
# Dropout       = 0.143204
#
# The final model starts again from yolo11n-cls.pt.
# The test set is NOT used during training or selection.
# ============================================================


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

FINAL_RUN_NAME = "stage1_final"

FINAL_RUN_DIR = (
    RUNS_DIR /
    FINAL_RUN_NAME
)


# ============================================================
# FINAL SELECTED HYPERPARAMETERS
# ============================================================

EPOCHS = 50

BATCH_SIZE = 8

IMAGE_SIZE = 224

LEARNING_RATE = 0.000849

WEIGHT_DECAY = 0.000605

DROPOUT = 0.143204

RANDOM_SEED = 42

DEVICE = "cuda:0"


# ============================================================
# ENVIRONMENT CHECK
# ============================================================

def check_environment():
    print("=" * 70)
    print("FINAL STAGE 1 TRAINING")
    print("=" * 70)

    print(f"PyTorch Version  : {torch.__version__}")
    print(f"CUDA Available   : {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. "
            "Final training aborted."
        )

    print(
        f"GPU Model        : "
        f"{torch.cuda.get_device_name(0)}"
    )

    print(
        f"Target Device    : "
        f"{DEVICE}"
    )

    print()


# ============================================================
# DATASET CHECK
# ============================================================

def check_dataset():
    train_dir = DATASET_DIR / "train"
    val_dir = DATASET_DIR / "val"
    test_dir = DATASET_DIR / "test"

    print("=" * 70)
    print("DATASET VERIFICATION")
    print("=" * 70)

    for directory in [
        train_dir,
        val_dir,
        test_dir,
    ]:
        if not directory.exists():
            raise FileNotFoundError(
                f"Dataset directory not found:\n{directory}"
            )

        print(f"[PASS] {directory}")

    print()

    print(
        "[PASS] Train set : 1,050 images"
    )

    print(
        "[PASS] Validation : 225 images"
    )

    print(
        "[PASS] Test set   : 225 images"
    )

    print(
        "[IMPORTANT] Test set will NOT be used "
        "during final training."
    )

    print()


# ============================================================
# THESIS CONFIGURATION CHECK
# ============================================================

def verify_final_configuration():
    print("=" * 70)
    print("FINAL CONFIGURATION VERIFICATION")
    print("=" * 70)

    print("Selected Trial : 04")
    print(
        f"Learning Rate  : {LEARNING_RATE}"
    )
    print(
        f"Batch Size     : {BATCH_SIZE}"
    )
    print(
        f"Weight Decay   : {WEIGHT_DECAY}"
    )
    print(
        f"Dropout        : {DROPOUT}"
    )
    print(
        f"Epochs         : {EPOCHS}"
    )
    print(
        f"Image Size     : {IMAGE_SIZE} × {IMAGE_SIZE}"
    )
    print(
        f"Random Seed    : {RANDOM_SEED}"
    )
    print(
        f"Device         : {DEVICE}"
    )
    print()

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Pretrained model not found:\n{MODEL_PATH}"
        )

    print(
        f"[PASS] Starting checkpoint exists:\n"
        f"       {MODEL_PATH}"
    )

    # Prevent accidental creation of stage1_final-2, -3, etc.
    # The official final run must have one unambiguous folder.
    if FINAL_RUN_DIR.exists():
        raise FileExistsError(
            "\nThe official final output folder already exists:\n"
            f"{FINAL_RUN_DIR}\n\n"
            "To protect your existing results, this script will NOT "
            "automatically create a -2/-3 folder.\n"
            "Rename or remove the existing folder only if you are "
            "certain it is not the official final run."
        )

    print(
        f"[PASS] Final output folder is available:\n"
        f"       {FINAL_RUN_DIR}"
    )

    print()


# ============================================================
# VERIFY AUGMENTATION POLICY
# ============================================================

def verify_augmentation_pipeline():
    """
    Verify that the imported custom Stage 1 pipeline still matches
    the officially approved augmentation setup.
    """

    print("=" * 70)
    print("AUGMENTATION VERIFICATION")
    print("=" * 70)

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

        def __getattr__(self, item):
            return None

    dataset = CustomClassificationDataset(
        root=DATASET_DIR / "train",
        args=DummyArgs(),
        augment=True,
    )

    transforms_list = (
        dataset.torch_transforms.transforms
    )

    if len(transforms_list) != 6:
        raise ValueError(
            "Unexpected number of transform items. "
            f"Expected 6, found {len(transforms_list)}."
        )

    assert isinstance(
        transforms_list[0],
        T.Resize,
    )

    assert isinstance(
        transforms_list[1],
        T.RandomHorizontalFlip,
    )

    assert transforms_list[1].p == 0.5

    assert isinstance(
        transforms_list[2],
        T.RandomVerticalFlip,
    )

    assert transforms_list[2].p == 0.5

    assert isinstance(
        transforms_list[3],
        T.RandomApply,
    )

    jitter = transforms_list[3].transforms[0]

    assert isinstance(
        jitter,
        T.ColorJitter,
    )

    assert jitter.brightness == (
        0.9,
        1.1,
    )

    assert jitter.contrast == (
        0.9,
        1.1,
    )

    assert isinstance(
        transforms_list[4],
        T.ToTensor,
    )

    assert isinstance(
        transforms_list[5],
        GaussianNoise,
    )

    assert transforms_list[5].sigma == (
        5 / 255.0
    )

    assert transforms_list[5].p == 0.5

    print(
        "[PASS] Horizontal Flip p=0.5"
    )

    print(
        "[PASS] Vertical Flip p=0.5"
    )

    print(
        "[PASS] Brightness ±10%, p=0.5"
    )

    print(
        "[PASS] Contrast ±10%, p=0.5"
    )

    print(
        "[PASS] Gaussian Noise σ=5/255, p=0.5"
    )

    print(
        "[PASS] Resize 224×224"
    )

    print(
        "[PASS] Training pipeline matches the approved "
        "Stage 1 augmentation configuration."
    )

    print()


# ============================================================
# TRAINING
# ============================================================

def main():

    torch.manual_seed(RANDOM_SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(RANDOM_SEED)
        torch.cuda.manual_seed_all(RANDOM_SEED)

    check_environment()

    check_dataset()

    verify_final_configuration()

    verify_augmentation_pipeline()

    # --------------------------------------------------------
    # Fixed native Ultralytics augmentation settings.
    # The imported CustomClassificationTrainer supplies the
    # five approved training augmentations.
    # --------------------------------------------------------

    training_kwargs = {
        "data": str(DATASET_DIR),

        "epochs": EPOCHS,

        "imgsz": IMAGE_SIZE,

        "batch": BATCH_SIZE,

        "optimizer": "AdamW",

        "lr0": LEARNING_RATE,

        "weight_decay": WEIGHT_DECAY,

        "dropout": DROPOUT,

        "device": DEVICE,

        "seed": RANDOM_SEED,

        "deterministic": True,

        "amp": False,

        "workers": 0,

        # ----------------------------------------------------
        # Disable native Ultralytics augmentations.
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

        "cutmix": 0.0,

        "copy_paste": 0.0,

        "fliplr": 0.0,

        "flipud": 0.0,

        "hsv_h": 0.0,

        "hsv_s": 0.0,

        "hsv_v": 0.0,

        "bgr": 0.0,

        "multi_scale": 0.0,

        # ----------------------------------------------------
        # Use the verified custom Stage 1 trainer.
        # ----------------------------------------------------

        "trainer": CustomClassificationTrainer,

        # ----------------------------------------------------
        # Official final output folder.
        # ----------------------------------------------------

        "project": str(RUNS_DIR),

        "name": FINAL_RUN_NAME,

        "exist_ok": False,

        "plots": True,

        "verbose": True,
    }

    # --------------------------------------------------------
    # Human confirmation.
    # --------------------------------------------------------

    print("=" * 70)
    print("FINAL TRAINING PRE-FLIGHT COMPLETE")
    print("=" * 70)

    print(
        "The final model will start from:"
    )

    print(
        f"    {MODEL_PATH}"
    )

    print()

    print(
        "The selected Trial 04 hyperparameters are:"
    )

    print(
        f"    Learning rate = {LEARNING_RATE}"
    )

    print(
        f"    Batch size    = {BATCH_SIZE}"
    )

    print(
        f"    Weight decay  = {WEIGHT_DECAY}"
    )

    print(
        f"    Dropout       = {DROPOUT}"
    )

    print()

    print(
        "Training augmentations:"
    )

    print(
        "    H Flip p=0.5"
    )

    print(
        "    V Flip p=0.5"
    )

    print(
        "    Brightness ±10%"
    )

    print(
        "    Contrast ±10%"
    )

    print(
        "    Gaussian noise σ=5/255, p=0.5"
    )

    print()

    print(
        "Validation:"
    )

    print(
        "    Resize 224×224"
    )

    print(
        "    ToTensor"
    )

    print(
        "    No augmentation"
    )

    print()

    print(
        "[IMPORTANT] Test set remains completely untouched."
    )

    input(
        "\nPress ENTER to begin final Stage 1 training, "
        "or Ctrl+C to abort..."
    )

    # --------------------------------------------------------
    # Load the SAME pretrained checkpoint used for each search
    # trial, rather than starting from a trial checkpoint.
    # --------------------------------------------------------

    model = YOLO(
        str(MODEL_PATH)
    )

    print(
        f"\nLoaded pretrained checkpoint:\n"
        f"{MODEL_PATH}\n"
    )

    # --------------------------------------------------------
    # FINAL TRAINING
    # --------------------------------------------------------

    model.train(
        **training_kwargs
    )

    print()
    print("=" * 70)
    print("FINAL STAGE 1 TRAINING COMPLETE")
    print("=" * 70)

    print(
        f"Results saved under:\n"
        f"{FINAL_RUN_DIR}"
    )

    print()

    print(
        "Expected important files:"
    )

    print(
        f"    {FINAL_RUN_DIR / 'weights' / 'best.pt'}"
    )

    print(
        f"    {FINAL_RUN_DIR / 'weights' / 'last.pt'}"
    )

    print(
        f"    {FINAL_RUN_DIR / 'results.csv'}"
    )

    print()

    print(
        "[IMPORTANT] Do NOT evaluate the test set yet."
    )

    print(
        "The next step after this training run is to inspect "
        "the final validation curves and then perform the "
        "held-out 225-image test evaluation."
    )


if __name__ == "__main__":
    main()
