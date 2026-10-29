# Dedicated YOLO11n-Presence Model — Implementation Guide

## Purpose

The presence detector has one class, `banana_bunch`. Its only job is to trigger a full-resolution snapshot when a bunch is stably present on the table. It does **not** decide Healthy/Unhealthy or Class A/Class B. The existing Stage 1 classifier and Stage 2 size detector remain unchanged.

This guide was prepared from the supplied thesis project archive. The supplied project ZIP contains `train_stage2_final.py`, `yolo11n.pt`, `runs/stage2_final/weights/best.pt`, `stage1_split_record.csv`, and `stage2_annotation_preview/stage2_annotation_record.csv`, but not the original photo files. Your local project root is `C:\Users\saiar\Pictures\cardava_thesis`. The source folders currently are `dataset/raw/Class_A/`, `dataset/raw/Class_B/`, and `dataset/raw/Rejected/Rejected/`. The dataset builder scans `dataset/raw/` recursively, and the Rejected-box proposal script is configured to read from the nested `Rejected/Rejected` directory. Background photos are recommended; a preliminary banana-only run is possible with `--allow-no-background`, but cannot validate false detections on empty scenes.

## 0. Expected repository layout

Run commands from the thesis repository root (the folder containing `train_stage2_final.py`):

```text
thesis/
  train_stage2_final.py
  yolo11n.pt
  stage1_split_record.csv
  stage2_annotation_preview/stage2_annotation_record.csv
  dataset/
    raw/
      Class_A/       # original unmasked images
      Class_B/       # original unmasked images
      Rejected/
        Rejected/    # original unmasked Rejected images (your actual nested folder)
    backgrounds/     # Background_001.jpg ... Background_150.jpg
  training/
    propose_rejected_boxes.py
    merge_rejected_labels.py
    build_presence_dataset.py
    train_presence.py
    evaluate_presence.py
  docs/PRESENCE_MODEL.md
```

The scripts default to these locations but accept command-line path overrides. Do not put the black-background masked copies in `dataset/raw`; the detector should be trained on the original camera-like RGB images.

## 1. Capture 150 background images (recommended before final deployment)

Take the images at the real Raspberry Pi Camera Module 3 rig, with the same 45 cm camera height, black velvet, and ring light used operationally. Include these filename ranges so the dataset builder can infer scene categories and distribute them across splits:

- `Background_001.jpg`–`Background_040.jpg`: empty velvet, ring light on.
- `Background_041.jpg`–`Background_060.jpg`: empty velvet with changed room/ring lighting.
- `Background_061.jpg`–`Background_100.jpg`: hand or arm present, no whole bunch.
- `Background_101.jpg`–`Background_130.jpg`: planned random objects.
- `Background_131.jpg`–`Background_150.jpg`: loose fingers/debris, no whole bunch.

Do not take all 150 by holding the camera absolutely still in one burst if they are near-identical. Vary realistic frames modestly and preserve the scene category for each file. A custom `--background-manifest` may be supplied with `filename,scene_type` columns when names don't follow the ranges above. If the system's allowable operating conditions do not include an item such as a knife, don't add it merely to create a negative example.

## 2. Generate proposed boxes for Rejected bunches

Check that the Stage 2 checkpoint exists at `runs/stage2_final/weights/best.pt` and the 500 original Rejected images exist at `dataset/raw/Rejected/Rejected/` (absolute path: `C:\Users\saiar\Pictures\cardava_thesis\dataset\raw\Rejected\Rejected`).

```bash
python training/propose_rejected_boxes.py
```

The script uses Stage 2 only to suggest boxes, at a deliberately low proposal threshold of 0.05. It writes contact sheets, one TXT proposal per image, and `runs/presence_annotation_review/rejected_review.csv`. The low proposal threshold is **not** the eventual application trigger threshold.

Inspect every Rejected image. The box must include the complete bunch—crown and every finger—and must not be widened to include the shadow. In the CSV, mark each row:

- `ok` when the proposal covers the whole bunch.
- `fix` when the box is absent or incorrect.

For each `fix` row, draw a single box around the entire bunch in CVAT, Label Studio, or makesense.ai and export YOLO detection label TXT files with class ID `0`. Put the corrected TXT files in `runs/presence_annotation_review/corrected_labels/` using the source image stem (for example, `Rejected_038.txt`). Every file marked `fix` must have one corrected box.

Then merge and validate the reviewed labels:

```bash
python training/merge_rejected_labels.py
```

The script stops if any review status is missing, if a box file is missing, if a file contains anything other than one box, or if the normalized box is invalid. Output is `runs/presence_annotation_review/verified_labels/`. Review is not optional: generated boxes are not ground truth.

## 3. Build the presence dataset

```bash
python training/build_presence_dataset.py
```

The script:

1. Uses `stage1_split_record.csv` as the source of train/validation/test membership for all 1,500 banana images.
2. Reconstructs Class A/B YOLO labels from `stage2_annotation_preview/stage2_annotation_record.csv`.
3. Requires one verified whole-bunch label for every Rejected image.
4. Adds each background image with an empty TXT label.
5. Splits background photos approximately 70/15/15, stratified by scene category when possible.
6. Writes `dataset/presence_det/data.yaml` and `presence_split_manifest.csv`.

Expected banana-only counts are train 1,050, validation 225, and test 225. The held-out banana test subset should contain 67 Class A, 83 Class B, and 75 Rejected images. Background images are added to each split, not substituted for banana images. The expected total with 150 backgrounds is 1,650 images: 1,155 train, 247 validation, 248 test (the background split is 105/22/23).

The builder refuses to proceed when raw images, metadata records, or Rejected labels are missing or inconsistent. If you correct annotations after building the dataset, rerun the build script with `--overwrite`.

### If you do not have background images yet

For a preliminary development run only, after all 1,500 banana images and all verified labels are ready, run:

```bash
python training/build_presence_dataset.py --allow-no-background
```

This builds a banana-only detection dataset using the existing train/validation/test split. It allows you to start training and examine banana-bunch detection, but it does **not** teach the model with full empty-scene frames. It is not a complete evaluation of a real presence trigger. The evaluator will report background false-trigger performance as **NOT EVALUATED** and will not declare all requested targets passed. Add real rig background images later, rebuild the dataset, retrain or fine-tune under a documented experiment, and test those backgrounds before production integration.

## 4. Train the one-class detector

Use the same thesis environment and GPU workstation used for Stage 2. Check the paths before starting. The default initialization is the general pretrained `yolo11n.pt`, not the previously trained Stage 2 checkpoint. The training script imports `CustomDetectionTrainer` from the existing `train_stage2_final.py`, which preserves that code's bbox-aware flips, ±10% brightness/contrast, Gaussian noise σ=5/255, and disabled native/composite augmentations.

```bash
python training/train_presence.py
```

Default settings: 50 epochs; 640 input size; batch 4; AdamW; learning rate 0.000867; weight decay 0.000763; seed 42; deterministic mode; AMP disabled; workers 0; CUDA device `0`. Use `--device cpu` only if you intentionally need CPU training; it will be much slower.

Expected checkpoint:

```text
runs/presence_final/weights/best.pt
```

The model is a new checkpoint. Do not overwrite `runs/stage1_final/weights/best.pt` or `runs/stage2_final/weights/best.pt`.

Reusing Stage 2's tuned hyperparameters is a defensible starting configuration because both models localize whole bunches, but it is not evidence that those settings are optimal for presence detection. The new detector must be evaluated independently. If comparing initialization from `yolo11n.pt` against `runs/stage2_final/weights/best.pt`, treat it as a separate experiment, select using validation results only, and do not run both alternatives against the held-out test set for repeated tuning.

## 5. Evaluate before app integration

```bash
python training/evaluate_presence.py
```

The evaluation has two distinct parts. If the dataset was built with `--allow-no-background`, the first part can still report held-out banana detection metrics, but the empty-scene false-trigger part is not evaluated and the overall target status cannot be PASS:

- Standard held-out test detection metrics, including mAP@0.50 and mAP@0.50:0.95, using 640-pixel YOLO validation at a low confidence threshold suitable for calculating detection metrics.
- An application-like trigger test that resizes each test frame to 640 × 360, predicts at `imgsz=640` and confidence `0.80`, and reports whether each of the three banana grades triggers a detection plus how many background test images cause a false trigger.

Pass targets are:

- Class A: 67/67 test bunches detected.
- Class B: 83/83 test bunches detected.
- Rejected: 75/75 test bunches detected.
- Background: zero false detections on the held-out background test photos.

These are acceptance targets, not guaranteed results. Review the individual miss list and background false triggers if the model fails. When background images are absent, zero false detections must be reported as not evaluated—not as zero. Change confidence based on validation data, not the held-out test set, then evaluate the selected settings once. Report mAP, per-grade trigger recall, false detections, and preview-trigger latency distinctly; detection mAP alone does not prove reliable capture triggering.

The script writes `presence_operational_test.csv`, `presence_map_metrics.json`, and `presence_evaluation_summary.json` under `runs/presence_final/evaluation/`. The app-like script reports per-image outcomes; it is not a complete measurement of real-time multi-frame stability, lighting errors, or GPIO timing. Those still require live Raspberry Pi tests.

## 6. Integrate only after passing review

The supplied ZIP contains the thesis/training project but not the separate Banagrade application source, so this kit cannot verify or wire the app's checkpoint path. After evaluating the model, copy the selected checkpoint to the application's intended path, such as `models/presence/presence_best.pt`, and change the acquisition trigger to use only this one-class detector. Keep Stage 1 and Stage 2 grading code unchanged.

Verify the application logs the exact presence checkpoint path at startup. Test all three banana grades, empty table, hands without a bunch, and object-only scenes on the Pi. Confirm that capture happens only after a stable whole-bunch detection and that any fallback to Stage 2 is clearly logged—not silent. Do not replace the current backup-enabled trigger until the new model has passed the real-rig checks.

## Thesis reporting guardrails

- Chapter 3 should document the new capture-trigger purpose, source images, box annotation procedure, the negative/background examples, split protocol, training settings, and trigger evaluation protocol.
- Chapter 4 should report the measured presence-model results and compare operational capture recall with the current Stage 2-only trigger (reported in your tests as 66/75 Rejected detections at confidence 0.80). Keep the exact test context and confidence threshold with the comparison.
- Existing Stage 1/Stage 2 model-specific results can remain unchanged only if their weights, datasets, splits, thresholds, and evaluation procedures remain unchanged. System-level capture results and total latency should be updated because the trigger logic has changed.
- Do not state that the new detector passed 67/67, 83/83, 75/75, or achieved zero false triggers until the held-out evaluation actually shows those numbers.
