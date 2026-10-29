# Cardava-Thesis: model development and evaluation

Training and evaluation code for the thesis **"A Two-Stage Deep Learning System for Cardava Banana Surface Defect Screening and Size-Based Grading with an Edge-Based Operational Analytics Dashboard"** (Ateneo de Davao University).

The deployed application (Raspberry Pi 5, FastAPI, Vue 3 dashboard) is in a separate repository: https://github.com/mokichino/banagrade

## Models

| Model | Task | Checkpoint |
|---|---|---|
| Stage 1: YOLO11n-cls | Healthy / Unhealthy health screening (224 × 224, raw image) | `runs/stage1_final/weights/best.pt` |
| Stage 2: YOLO11n | Whole-bunch detection (one class, `banana_bunch`, 640 × 640); Class A if box width ≥ 2730.07 px, otherwise Class B | `runs/stage2_final/weights/best.pt` |
| Presence detector: YOLO11n | Detects a bunch before capture (one class, `banana_bunch`) | `runs/presence_final/weights/best.pt` |

## Dataset

1,500 self-collected Cardava banana bunch images (500 Class A, 500 Class B, 500 Rejected), 4608 × 2592 px, captured at 45 cm on black velvet in Davao del Norte. The images are not included in this repository. They are stored by the researchers and may be made available upon reasonable request, subject to the research team's approval.

Splits (70 / 15 / 15, stratified, `random_state=42`):
- Stage 1 and the presence detector: `stage1_split_record.csv` (1,050 / 225 / 225)
- Stage 2: generated in `prepare_stage2.py` from `stage2_annotation_preview/stage2_annotation_record.csv` (700 / 150 / 150, Class A and Class B only)

## Pipeline and scripts

| Step | Script(s) | Outputs |
|---|---|---|
| Exploratory data analysis | `eda.py` | `eda_results.csv`, `eda_statistics.csv` |
| Dataset preparation | `prepare_stage1.py`, `prepare_stage2.py` | split record, Stage 2 annotations |
| Initial training | `train_stage1.py`, `train_stage2.py` | `runs/stage1_initial-2/`, `runs/stage2_initial/` |
| Hyperparameter search (4 random trials per stage, seed 42) | `stage1_hparam_search.py`, `stage2_hparam_search.py`, `evaluate_stage1_hparam_search.py`, `plot_stage*_hparam_results.py` | `runs/stage1_hparam_search_final/`, `runs/stage2_hparam_search_final/` |
| Final training | `train_stage1_final.py`, `train_stage2_final.py` | `runs/stage1_final/`, `runs/stage2_final/` |
| Held-out test evaluation | `test_stage1_final.py`, `test_stage2_final.py` | `runs/*/test_evaluation/` |
| A/B threshold selection (validation only) and test grading | `stage2_measurement_analysis.py`, `stage2_validation_threshold_tuning.py`, `stage2_validation_grading.py`, `stage2_test_grading_final.py` | `runs/stage2_final/validation_grading/`, `runs/stage2_final/final_grading_evaluation/` |
| SVM baseline and McNemar test | `svm_baseline_stage1.py`, `stage1_mcnemar_comparison.py` | `runs/stage1_svm_baseline/`, `runs/stage1_mcnemar/` |
| End-to-end evaluation (106 images unseen by both stages) | `prepare_end_to_end_heldout.py`, `run_end_to_end_heldout.py` | `evaluation/end_to_end_heldout_106/` |
| Presence detector | `training/propose_rejected_boxes.py`, `training/merge_rejected_labels.py`, `training/build_presence_dataset.py`, `training/train_presence.py`, `training/evaluate_presence.py` (guide: `docs/PRESENCE_MODEL.md`) | `runs/presence_final/` |

## Key held-out results

| Evaluation | Result |
|---|---|
| Stage 1 (225 test images) | 99.56% accuracy, macro F1 99.50% |
| SVM baseline (same 225 images) | 94.67% accuracy; exact McNemar p = 0.003418 |
| Stage 2 detection (150 test images) | mAP@0.50 99.50%, mAP@0.50:0.95 98.57% |
| Class A/B grading (150 test images) | 84.67% accuracy, macro F1 84.55% |
| End-to-end (106 images) | 94.34% accuracy, balanced accuracy 89.14% |
| Presence detector (225 test images) | mAP@0.50 99.50%, mAP@0.50:0.95 98.74%; 225/225 detected at confidence 0.80 |

The presence detector was trained and tested on banana images only. No empty-table images were used, so its false-trigger rate on empty scenes has not been evaluated.

## Environment

Training: Windows 11, Python 3.13.5, Ultralytics 8.4.153, PyTorch 2.11.0+cu126, NVIDIA GeForce GTX 1650 (4 GB), ASUS TUF 15. All training runs use seed 42.

## Reproducing

Place the original images under `dataset/raw/Class_A/`, `dataset/raw/Class_B/` and `dataset/raw/Rejected/`, then run the scripts in the order of the table above. Paths inside some scripts point to the original project folder and may need adjusting.
