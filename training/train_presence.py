#!/usr/bin/env python3
"""Fine-tune a one-class YOLO11n presence detector using the project's Stage 2 augmentation pipeline."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=ROOT / "dataset/presence_det/data.yaml")
    p.add_argument("--weights", type=Path, default=ROOT / "yolo11n.pt",
                   help="Default is general pretrained yolo11n.pt; Stage 2 best.pt can be tested as an alternate experiment.")
    p.add_argument("--project", type=Path, default=ROOT / "runs")
    p.add_argument("--name", default="presence_final")
    p.add_argument("--device", default="0", help="Ultralytics device, e.g. 0 for first CUDA GPU or cpu.")
    p.add_argument("--exist-ok", action="store_true")
    return p.parse_args()


def main() -> int:
    a = parse_args()
    if not a.data.is_file(): raise SystemExit(f"Dataset YAML not found: {a.data}. Run build_presence_dataset.py first.")
    if not a.weights.is_file(): raise SystemExit(f"Initial checkpoint not found: {a.weights}")
    try:
        from ultralytics import YOLO
        from train_stage2_final import CustomDetectionTrainer
    except Exception as exc:
        raise SystemExit(
            "Could not import Ultralytics or the existing custom Stage 2 trainer. "
            "Run this script from a compatible project environment and keep train_stage2_final.py in the repo root. "
            f"Details: {exc}"
        ) from exc

    print(f"Project root     : {ROOT}")
    print(f"Dataset YAML     : {a.data.resolve()}")
    print(f"Initial weights  : {a.weights.resolve()}")
    print(f"Output run       : {a.project / a.name}")
    print("Task             : one class, banana_bunch (presence only)")
    print("Settings         : 50 epochs, 640 px, batch 4, AdamW, lr0=0.000867, weight_decay=0.000763, seed=42")
    print("Augmentations    : reuses the custom Stage 2 pipeline (horizontal/vertical flips, brightness/contrast ±10%, Gaussian noise σ=5/255)")
    print("No Stage 1 or Stage 2 checkpoint will be overwritten by this script.")
    input("\nPre-flight summary above. Press ENTER to start training, or Ctrl+C to abort... ")

    model = YOLO(str(a.weights))
    model.train(
        data=str(a.data.resolve()),
        epochs=50,
        imgsz=640,
        batch=4,
        optimizer="AdamW",
        lr0=0.000867,
        weight_decay=0.000763,
        device=a.device,
        seed=42,
        deterministic=True,
        amp=False,
        workers=0,
        degrees=0.0,
        translate=0.0,
        scale=0.0,
        shear=0.0,
        perspective=0.0,
        mosaic=0.0,
        mixup=0.0,
        cutmix=0.0,
        copy_paste=0.0,
        fliplr=0.0,
        flipud=0.0,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        bgr=0.0,
        multi_scale=0.0,
        trainer=CustomDetectionTrainer,
        project=str(a.project.resolve()),
        name=a.name,
        exist_ok=a.exist_ok,
        plots=True,
        verbose=True,
    )
    best = a.project / a.name / "weights" / "best.pt"
    print(f"\nTraining command finished. Expected best checkpoint: {best}")
    if best.is_file(): print(f"Checkpoint exists ({best.stat().st_size:,} bytes). Run evaluate_presence.py next.")
    else: print("WARNING: best.pt was not found at the expected path; inspect the training output before proceeding.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
