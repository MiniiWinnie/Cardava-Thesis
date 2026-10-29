from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from ultralytics import YOLO


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent if HERE.name.lower() == "training" else HERE

HELDOUT_DIR = ROOT / "evaluation" / "end_to_end_heldout_106"
IMAGE_DIR = HELDOUT_DIR / "images"
MANIFEST_CSV = HELDOUT_DIR / "end_to_end_heldout_manifest.csv"

STAGE1_WEIGHTS = (
    ROOT / "runs" / "stage1_final" / "weights" / "best.pt"
)
STAGE2_WEIGHTS = (
    ROOT / "runs" / "stage2_final" / "weights" / "best.pt"
)
OUTPUT_DIR = HELDOUT_DIR / "results"

CLASS_ORDER = ["Class_A", "Class_B", "Rejected"]
PREDICTION_ORDER = [
    "Class_A",
    "Class_B",
    "Rejected",
    "NO_DETECTION",
]

DEFAULT_WIDTH_THRESHOLD_PX = 2730.0736389160156

# Must match test_stage1_final.py and the custom Stage 1
# evaluation preprocessing.
#
# Do NOT replace this with YOLO.predict() for Stage 1:
# Ultralytics' default classification preprocessing differs
# from this custom transform.
STAGE1_TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
])


def extract_logits(output: Any) -> torch.Tensor:
    """Extract logits from the Stage 1 model output."""

    if isinstance(output, torch.Tensor):
        return output

    if isinstance(output, (tuple, list)):
        for item in output:
            if isinstance(item, torch.Tensor):
                return item

    if isinstance(output, dict):
        for key in ("logits", "output", "pred", "preds"):
            value = output.get(key)
            if isinstance(value, torch.Tensor):
                return value

    raise RuntimeError(
        f"Unable to extract Stage 1 logits from {type(output)}"
    )


def resolve_torch_device(
    requested: str | None,
) -> torch.device:
    """Resolve the requested PyTorch device."""

    if requested is None or requested.lower() == "auto":
        device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )
    elif requested.isdigit():
        device = torch.device(f"cuda:{requested}")
    else:
        device = torch.device(requested)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA device {device} was requested, but CUDA is "
            "unavailable in this Python environment. "
            "Use --device cpu or activate the correct environment."
        )

    return device


def normalize_class_name(value: Any) -> str:
    return (
        str(value)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )


def stage1_class_name(
    names: Any,
    class_index: int,
) -> str:
    """Map the Stage 1 class index to Healthy or Unhealthy."""

    if isinstance(names, dict):
        raw = names.get(
            class_index,
            names.get(str(class_index), str(class_index)),
        )
    elif isinstance(names, (list, tuple)) and (
        0 <= class_index < len(names)
    ):
        raw = names[class_index]
    else:
        raw = str(class_index)

    label = normalize_class_name(raw)

    if label in {"healthy", "class_a", "class_b"}:
        return "Healthy"

    if label in {"unhealthy", "rejected"}:
        return "Unhealthy"

    raise ValueError(
        f"Unexpected Stage 1 class name for index "
        f"{class_index}: {raw!r}. "
        "Expected Healthy and Unhealthy."
    )


def predict_stage1_custom(
    model: YOLO,
    image_path: Path,
    device: torch.device,
) -> tuple[str, float]:
    """
    Run Stage 1 using custom preprocessing:

    RGB -> Resize((224, 224)) -> ToTensor()
        -> direct model forward -> softmax
    """

    with Image.open(image_path) as opened:
        image = opened.convert("RGB")
        tensor = STAGE1_TRANSFORM(image)
        tensor = tensor.unsqueeze(0).to(device)

    with torch.inference_mode():
        output = model.model(tensor)
        logits = extract_logits(output)

        if (
            logits.ndim != 2
            or logits.shape[0] != 1
            or logits.shape[1] != 2
        ):
            raise RuntimeError(
                f"Unexpected Stage 1 output shape for "
                f"{image_path.name}: {tuple(logits.shape)}"
            )

        probabilities = torch.softmax(logits, dim=1)[0]
        confidence, index = torch.max(probabilities, dim=0)

    class_index = int(index.item())
    label = stage1_class_name(model.names, class_index)

    return label, float(confidence.item())


def true_stage1_label(grade: str) -> str:
    """Convert the true grade to the Stage 1 binary label."""

    return "Unhealthy" if grade == "Rejected" else "Healthy"


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {label}:\n  {path}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the saved Stage 1 classifier and Stage 2 "
            "detector/grader sequentially on the 106-image "
            "cross-stage held-out subset."
        )
    )

    parser.add_argument(
        "--device",
        default=None,
        help=(
            "Device for inference, e.g. 0 or cpu. "
            "Default: automatically select a device."
        ),
    )

    parser.add_argument(
        "--det-conf",
        type=float,
        default=0.25,
        help=(
            "Minimum Stage 2 box confidence for post-capture "
            "grading. Default: 0.25."
        ),
    )

    parser.add_argument(
        "--width-threshold",
        type=float,
        default=DEFAULT_WIDTH_THRESHOLD_PX,
        help=(
            "Fixed bbox-width threshold in original-image pixels."
        ),
    )

    parser.add_argument(
        "--stage1-imgsz",
        type=int,
        default=224,
    )

    parser.add_argument(
        "--stage2-imgsz",
        type=int,
        default=640,
    )

    return parser.parse_args()


def calculate_metrics(
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Calculate final multiclass metrics and confusion matrix."""

    matrix = pd.crosstab(
        pd.Categorical(
            predictions["ground_truth_grade"],
            categories=CLASS_ORDER,
            ordered=True,
        ),
        pd.Categorical(
            predictions["final_predicted_grade"],
            categories=PREDICTION_ORDER,
            ordered=True,
        ),
        dropna=False,
    )

    matrix.index.name = "actual_grade"
    matrix.columns.name = "predicted_grade"

    matrix = matrix.reindex(
        index=CLASS_ORDER,
        columns=PREDICTION_ORDER,
        fill_value=0,
    )

    rows = []

    for label in CLASS_ORDER:
        tp = int(matrix.loc[label, label])

        support = int(
            (predictions["ground_truth_grade"] == label).sum()
        )

        predicted_count = int(
            (predictions["final_predicted_grade"] == label).sum()
        )

        recall = tp / support if support else 0.0
        precision = (
            tp / predicted_count if predicted_count else 0.0
        )

        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )

        rows.append({
            "class": label,
            "support": support,
            "correct": tp,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })

    metrics = pd.DataFrame(rows)

    accuracy = (
        float(predictions["correct"].mean())
        if len(predictions)
        else 0.0
    )

    balanced_accuracy = (
        float(metrics["recall"].mean())
        if len(metrics)
        else 0.0
    )

    macro_precision = (
        float(metrics["precision"].mean())
        if len(metrics)
        else 0.0
    )

    macro_f1 = (
        float(metrics["f1"].mean())
        if len(metrics)
        else 0.0
    )

    summary = {
        "total_images": int(len(predictions)),
        "correct_images": int(predictions["correct"].sum()),
        "incorrect_images": int(
            (~predictions["correct"]).sum()
        ),
        "end_to_end_accuracy": accuracy,
        "balanced_accuracy_macro_recall": balanced_accuracy,
        "macro_precision": macro_precision,
        "macro_f1": macro_f1,
        "stage1_correct": int(
            predictions["stage1_correct"].sum()
        ),
        "stage1_accuracy_on_106": (
            float(predictions["stage1_correct"].mean())
            if len(predictions)
            else 0.0
        ),
        "stage2_invoked_images": int(
            predictions["stage2_applied"].sum()
        ),
        "stage2_detection_successes": int(
            predictions["stage2_detected"].sum()
        ),
        "stage2_no_detection_count": int(
            (
                predictions["stage2_applied"]
                & ~predictions["stage2_detected"]
            ).sum()
        ),
        "final_prediction_counts": {
            str(key): int(value)
            for key, value in predictions[
                "final_predicted_grade"
            ].value_counts(dropna=False).items()
        },
        "ground_truth_counts": {
            str(key): int(value)
            for key, value in predictions[
                "ground_truth_grade"
            ].value_counts(dropna=False).items()
        },
    }

    return metrics, matrix, summary


def main() -> int:
    args = parse_args()

    if not 0.0 <= args.det_conf <= 1.0:
        raise ValueError(
            "--det-conf must be between 0 and 1."
        )

    if args.width_threshold <= 0:
        raise ValueError(
            "--width-threshold must be positive."
        )

    if args.stage1_imgsz != 224:
        raise ValueError(
            "Stage 1 uses Resize((224, 224)). "
            "Use the default --stage1-imgsz 224."
        )

    require_file(STAGE1_WEIGHTS, "Stage 1 checkpoint")
    require_file(STAGE2_WEIGHTS, "Stage 2 checkpoint")
    require_file(
        MANIFEST_CSV,
        "106-image manifest; run prepare_end_to_end_heldout.py first",
    )

    if not IMAGE_DIR.is_dir():
        raise FileNotFoundError(
            f"Held-out image directory not found: {IMAGE_DIR}"
        )

    manifest = pd.read_csv(MANIFEST_CSV)

    required_columns = {
        "filename",
        "ground_truth_grade",
    }

    if not required_columns.issubset(manifest.columns):
        missing = required_columns - set(manifest.columns)
        raise ValueError(
            f"Manifest missing columns: {sorted(missing)}"
        )

    manifest["filename"] = (
        manifest["filename"].astype(str).str.strip()
    )
    manifest["ground_truth_grade"] = (
        manifest["ground_truth_grade"].astype(str).str.strip()
    )

    expected_counts = {
        "Rejected": 75,
        "Class_A": 15,
        "Class_B": 16,
    }

    if len(manifest) != 106:
        raise ValueError(
            f"Expected exactly 106 manifest rows; "
            f"found {len(manifest)}."
        )

    observed_counts = (
        manifest["ground_truth_grade"].value_counts().to_dict()
    )

    if observed_counts != expected_counts:
        raise ValueError(
            f"Unexpected manifest class counts: {observed_counts}; "
            f"expected {expected_counts}."
        )

    missing_images = [
        name
        for name in manifest["filename"]
        if not (IMAGE_DIR / name).is_file()
    ]

    if missing_images:
        raise FileNotFoundError(
            f"Missing {len(missing_images)} held-out image(s) "
            f"under {IMAGE_DIR}. "
            f"Examples: {missing_images[:10]}. "
            "Run the preparation script again if necessary."
        )

    print("=" * 72)
    print("END-TO-END EVALUATION — 106 CROSS-STAGE HELD-OUT IMAGES")
    print("=" * 72)
    print("Training / fine-tuning: NO")
    print("Threshold tuning:       NO (locked threshold)")
    print(f"Stage 1 checkpoint:     {STAGE1_WEIGHTS}")
    print(f"Stage 2 checkpoint:     {STAGE2_WEIGHTS}")
    print(f"Stage 2 detector conf:  {args.det_conf:.3f}")
    print(f"A/B width threshold:    {args.width_threshold:.6f} px")

    torch_device = resolve_torch_device(args.device)
    print(f"Device:                 {torch_device}")
    print(
        "Stage 1 preprocessing: "
        "RGB -> Resize((224,224)) -> ToTensor() -> direct forward"
    )
    print()

    # Load Stage 1 and verify its class ordering.
    stage1_model = YOLO(str(STAGE1_WEIGHTS))
    stage1_names = stage1_model.names

    if isinstance(stage1_names, dict):
        stage1_class_names = [
            stage1_names[i] for i in sorted(stage1_names)
        ]
    else:
        stage1_class_names = list(stage1_names)

    if stage1_class_names != ["Healthy", "Unhealthy"]:
        raise RuntimeError(
            "Stage 1 checkpoint classes do not match the expected "
            "class order. Expected ['Healthy', 'Unhealthy']; "
            f"found {stage1_class_names}."
        )

    stage1_model.model.to(torch_device)
    stage1_model.model.eval()

    # Load Stage 2 detector.
    stage2_model = YOLO(str(STAGE2_WEIGHTS))

    print(f"Stage 1 class map: {stage1_model.names}")
    print(f"Stage 2 class map: {stage2_model.names}")

    records = []

    for count, row in enumerate(
        manifest.to_dict(orient="records"),
        start=1,
    ):
        filename = row["filename"]
        truth = row["ground_truth_grade"]
        image_path = IMAGE_DIR / filename

        total_start = time.perf_counter()

        # --------------------------
        # Stage 1: binary classifier
        # --------------------------
        s1_start = time.perf_counter()

        s1_label, s1_confidence = predict_stage1_custom(
            stage1_model,
            image_path,
            torch_device,
        )

        stage1_ms = (
            time.perf_counter() - s1_start
        ) * 1000.0

        final_prediction = ""
        s2_applied = False
        s2_detected = False

        s2_confidence = None
        bbox_x1 = None
        bbox_y1 = None
        bbox_x2 = None
        bbox_y2 = None
        bbox_width = None

        s2_prediction = "NOT_RUN_STAGE1_REJECTED"
        stage2_ms = 0.0

        # Stage 1 says Unhealthy => Rejected.
        if s1_label == "Unhealthy":
            final_prediction = "Rejected"

        # Stage 1 says Healthy => run Stage 2.
        else:
            s2_applied = True
            s2_start = time.perf_counter()

            s2_result = stage2_model.predict(
                source=str(image_path),
                imgsz=args.stage2_imgsz,
                conf=args.det_conf,
                device=args.device,
                augment=False,
                verbose=False,
            )[0]

            stage2_ms = (
                time.perf_counter() - s2_start
            ) * 1000.0

            boxes = s2_result.boxes

            if boxes is not None and len(boxes) > 0:
                confidences = (
                    boxes.conf.detach().cpu().tolist()
                )

                # Select the highest-confidence whole-bunch box.
                best_index = max(
                    range(len(confidences)),
                    key=lambda idx: confidences[idx],
                )

                coords = (
                    boxes.xyxy[best_index]
                    .detach()
                    .cpu()
                    .tolist()
                )

                s2_confidence = float(
                    confidences[best_index]
                )

                bbox_x1, bbox_y1, bbox_x2, bbox_y2 = [
                    float(value) for value in coords
                ]

                bbox_width = max(
                    0.0,
                    bbox_x2 - bbox_x1,
                )

                s2_detected = True

                # Fixed width threshold; do not tune on this set.
                if bbox_width > args.width_threshold:
                    s2_prediction = "Class_A"
                else:
                    s2_prediction = "Class_B"

                final_prediction = s2_prediction

            else:
                final_prediction = "NO_DETECTION"
                s2_prediction = "NO_DETECTION"

        total_ms = (
            time.perf_counter() - total_start
        ) * 1000.0

        expected_s1 = true_stage1_label(truth)

        records.append({
            "filename": filename,
            "ground_truth_grade": truth,
            "stage1_true_label": expected_s1,
            "stage1_predicted_label": s1_label,
            "stage1_confidence": s1_confidence,
            "stage1_correct": (
                s1_label == expected_s1
            ),
            "stage2_applied": s2_applied,
            "stage2_detected": s2_detected,
            "stage2_detection_confidence": s2_confidence,
            "bbox_x1_px": bbox_x1,
            "bbox_y1_px": bbox_y1,
            "bbox_x2_px": bbox_x2,
            "bbox_y2_px": bbox_y2,
            "bbox_width_px": bbox_width,
            "stage2_predicted_grade": s2_prediction,
            "final_predicted_grade": final_prediction,
            "correct": final_prediction == truth,
            "stage1_inference_ms": stage1_ms,
            "stage2_inference_ms": (
                stage2_ms if s2_applied else None
            ),
            "total_inference_ms": total_ms,
        })

        status = (
            "PASS" if final_prediction == truth else "MISS"
        )

        print(
            f"[{count:03d}/106] {filename} | "
            f"actual={truth:<8} "
            f"stage1={s1_label:<8} "
            f"final={final_prediction:<12} {status}"
        )

    # --------------------------
    # Save results and metrics
    # --------------------------
    predictions = pd.DataFrame(records)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    predictions_path = (
        OUTPUT_DIR / "end_to_end_predictions.csv"
    )
    metrics_path = (
        OUTPUT_DIR / "end_to_end_metrics_by_class.csv"
    )
    matrix_path = (
        OUTPUT_DIR / "end_to_end_confusion_matrix.csv"
    )
    summary_path = (
        OUTPUT_DIR / "end_to_end_summary.json"
    )

    predictions.to_csv(predictions_path, index=False)

    metrics, matrix, summary = calculate_metrics(predictions)

    metrics.to_csv(metrics_path, index=False)
    matrix.to_csv(matrix_path)

    summary.update({
        "stage1_checkpoint": str(STAGE1_WEIGHTS),
        "stage2_checkpoint": str(STAGE2_WEIGHTS),
        "stage1_imgsz": args.stage1_imgsz,
        "stage1_preprocessing": (
            "PIL RGB -> torchvision Resize((224,224)) "
            "-> ToTensor() -> direct model forward"
        ),
        "stage2_imgsz": args.stage2_imgsz,
        "stage2_detector_confidence": args.det_conf,
        "width_threshold_px": args.width_threshold,
        "width_rule": (
            "Class_A if bbox_width_px > width_threshold_px; "
            "otherwise Class_B"
        ),
        "test_subset": (
            "106 cross-stage held-out images "
            "from original 1500-image dataset"
        ),
        "external_new_dataset": False,
        "training_or_threshold_tuning_performed": False,
    })

    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 72)
    print("END-TO-END RESULTS")
    print("=" * 72)

    print(
        f"Overall accuracy:      "
        f"{summary['end_to_end_accuracy']:.4%} "
        f"({summary['correct_images']}/"
        f"{summary['total_images']})"
    )

    print(
        f"Balanced accuracy:     "
        f"{summary['balanced_accuracy_macro_recall']:.4%}"
    )

    print(
        f"Macro precision:       "
        f"{summary['macro_precision']:.4%}"
    )

    print(
        f"Macro F1:              "
        f"{summary['macro_f1']:.4%}"
    )

    print(
        f"Stage 1 correct:       "
        f"{summary['stage1_correct']}/"
        f"{summary['total_images']}"
    )

    print(
        f"Stage 2 invoked:       "
        f"{summary['stage2_invoked_images']}/"
        f"{summary['total_images']}"
    )

    print(
        f"Stage 2 detections:    "
        f"{summary['stage2_detection_successes']}/"
        f"{summary['stage2_invoked_images']}"
    )

    print(
        f"No detection:          "
        f"{summary['stage2_no_detection_count']}"
    )

    print("\nPer-class metrics:")
    print(
        metrics.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print(
        "\nConfusion matrix "
        "(rows = actual, columns = final prediction):"
    )
    print(matrix.to_string())

    print("\nSaved:")
    print(f"  {predictions_path}")
    print(f"  {metrics_path}")
    print(f"  {matrix_path}")
    print(f"  {summary_path}")

    print(
        "\nReminder: these 106 photos are an internal "
        "cross-stage holdout, not a newly collected external test set."
    )
    print(
        "This evaluates already-captured photos; it does not test "
        "the camera placement trigger or empty-table false positives."
    )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)