# Guide: Test the complete two-stage model pipeline on the 106-image holdout

This runs the saved Stage 1 classifier followed by the saved Stage 2 detector and width-based A/B grade rule. It does **not** train models or adjust thresholds.

## Files in this kit

- `prepare_end_to_end_heldout.py` — picks and copies the 106 images from your original `dataset/raw` folders using the recorded split CSVs.
- `run_end_to_end_heldout.py` — runs inference sequentially through Stage 1 and Stage 2 and saves the predictions and metrics.
- `end_to_end_heldout_106_manifest.csv` — the current filename list and ground-truth labels, for reference.

## 1. Put the scripts in your project folder

Extract your project so that the root contains folders such as `dataset`, `runs`, and the CSV files. For example:

```text
C:\Users\saiar\Pictures\cardava_thesis\
  dataset\raw\
  runs\stage1_final\weights\best.pt
  runs\stage2_final\weights\best.pt
  runs\stage1_final\test_evaluation\stage1_test_predictions.csv
  runs\stage2_final\final_grading_evaluation\stage2_final_grading_predictions.csv
```

Copy both `.py` scripts into this project root, beside `dataset` and `runs`.

**Important:** `thesis.zip` includes the model weights and evaluation CSVs, but it does not include all of the original raw photographs. Make sure your existing original images remain under `dataset\raw\` in the project folder.

## 2. Open a terminal in the project root

In Windows File Explorer, open the `cardava_thesis` folder, click the address bar, type `cmd`, then press Enter. Confirm you are in the correct folder by running:

```bat
 dir runs\stage1_final\weights\best.pt
 dir runs\stage2_final\weights\best.pt
 dir dataset\raw
```

Use the same Python/virtual environment where you trained or ran the project. If necessary, install the packages in that environment:

```bat
python -m pip install ultralytics pandas
```

## 3. Prepare the 106 images

Run:

```bat
python prepare_end_to_end_heldout.py
```

The script reads the Stage 1 and Stage 2 test prediction CSVs and copies:

- 75 `Rejected` images from the Stage 1 held-out test set;
- 15 `Class_A` images also in the Stage 2 held-out test set;
- 16 `Class_B` images also in the Stage 2 held-out test set.

Expected total: **106 images** in `evaluation\end_to_end_heldout_106\images\`.

The preparation script recreates only the generated `evaluation\end_to_end_heldout_106` output folder. If you already have results inside that folder, copy them elsewhere before running the preparation script again.

## 4. Run the full image-level pipeline

Run on the machine's automatically selected device:

```bat
python run_end_to_end_heldout.py
```

If you want to explicitly use your NVIDIA GPU, run:

```bat
python run_end_to_end_heldout.py --device 0
```

For CPU inference instead:

```bat
python run_end_to_end_heldout.py --device cpu
```

By default, the runner uses Stage 1 image size 224, Stage 2 image size 640, a Stage 2 post-capture detection confidence of 0.25, and the validation-selected grading threshold of **2730.073639 pixels**. Do not change the width threshold after looking at results. A missing Stage 2 box is recorded as `NO_DETECTION` and counts as an end-to-end error; it is not dropped from the denominator.

If your installed application uses a *different post-capture detector confidence setting*, you may run that setting explicitly, e.g. `--det-conf 0.50`. Keep the setting documented and do not choose it based on which one scores best on these 106 images. The camera's pre-capture placement-trigger threshold (previously discussed as 0.80) is a separate operational step and is **not** measured by this image-level script.

## 5. Get the results

The runner writes these files to `evaluation\end_to_end_heldout_106\results\`:

- `end_to_end_predictions.csv` — one row per image, ground truth, Stage 1 output, whether Stage 2 ran, detection confidence/bounding box width, final class, correctness, and inference time.
- `end_to_end_metrics_by_class.csv` — precision, recall, F1, and support for Class A, Class B, and Rejected.
- `end_to_end_confusion_matrix.csv` — actual grades by final predicted grade, including a `NO_DETECTION` column.
- `end_to_end_summary.json` — overall accuracy, balanced accuracy, macro precision/F1, counts, thresholds, and checkpoint paths.

The terminal also prints a per-image pass/miss line and the final summary. If a row is wrong, find it in `end_to_end_predictions.csv` to see whether the error arose at Stage 1 or during Stage 2 detection/grading.

## What this evaluation does and does not establish

- It tests the image-level sequence: **Stage 1 Healthy/Unhealthy classification → Stage 2 whole-bunch detection → pixel-width-based Class A/B grading**.
- All 106 images were held out from both stages' model-fitting/validation split assignments, according to the recorded split audit.
- These are still images from the original 1,500-image collection, and this exact subset has already been reported in Section 4.4.4. This is a repeatable execution/check of the stored pipeline, not a newly collected external dataset.
- As all inputs are already-captured banana photos, this does not assess the camera placement trigger, empty-table false positives, or failures to capture an item in the first place.
- Do not train, tune the threshold, or choose confidence settings on these results. Save the CSV outputs unchanged for reporting.
