#!/usr/bin/env python3
"""Evaluate mAP on held-out test images and app-like trigger recall at confidence 0.80."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import pandas as pd
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=ROOT / "dataset/presence_det/data.yaml")
    p.add_argument("--weights", type=Path, default=ROOT / "runs/presence_final/weights/best.pt")
    p.add_argument("--device", default="0", help="CUDA index such as 0, or cpu.")
    p.add_argument("--conf", type=float, default=0.80, help="Trigger threshold used for operational counts.")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--preview-width", type=int, default=640)
    p.add_argument("--preview-height", type=int, default=360)
    p.add_argument("--out", type=Path, default=ROOT / "runs/presence_final/evaluation")
    return p.parse_args()


def main() -> int:
    a = parse_args()
    if not a.data.is_file(): raise SystemExit(f"Data YAML not found: {a.data}")
    if not a.weights.is_file(): raise SystemExit(f"Model checkpoint not found: {a.weights}")
    dataset_root = a.data.parent
    manifest_path = dataset_root / "presence_split_manifest.csv"
    if not manifest_path.is_file(): raise SystemExit(f"Split manifest not found: {manifest_path}. Rebuild the dataset.")
    manifest = pd.read_csv(manifest_path)
    test_df = manifest.loc[manifest["split"].astype(str).str.lower() == "test"].copy()
    if test_df.empty: raise SystemExit("No test records in presence split manifest.")
    model = YOLO(str(a.weights))
    a.out.mkdir(parents=True, exist_ok=True)

    # Standard held-out detection metrics use Ultralytics validation at the standard
    # evaluation confidence; the separate app-like test below uses conf=0.80.
    print("Running standard held-out detection metrics (mAP uses low confidence, not trigger conf=0.80)...")
    metrics = model.val(
        data=str(a.data.resolve()), split="test", imgsz=a.imgsz, batch=4,
        conf=0.001, iou=0.70, augment=False, device=a.device, workers=0,
        project=str(a.out), name="map_test", exist_ok=True, plots=True, verbose=True,
    )
    box_metrics = getattr(metrics, "box", None)
    map50 = float(box_metrics.map50) if box_metrics is not None else None
    map5095 = float(box_metrics.map) if box_metrics is not None else None
    standard_metrics = {"test_images_in_dataset": int(len(test_df)), "mAP50": map50, "mAP50_95": map5095,
                        "validation_conf_for_map": 0.001, "imgsz": a.imgsz}
    (a.out / "presence_map_metrics.json").write_text(json.dumps(standard_metrics, indent=2), encoding="utf-8")

    print(f"Running app-like trigger evaluation: resize to {a.preview_width}x{a.preview_height}, predict imgsz={a.imgsz}, conf={a.conf:.2f}...")
    rows = []
    missing_test_images = []
    for row in test_df.itertuples(index=False):
        image_path = dataset_root / "images" / "test" / str(row.filename)
        image = cv2.imread(str(image_path))
        if image is None:
            missing_test_images.append(str(image_path)); continue
        preview = cv2.resize(image, (a.preview_width, a.preview_height), interpolation=cv2.INTER_AREA)
        result = model.predict(source=preview, imgsz=a.imgsz, conf=a.conf, iou=0.70,
                               augment=False, device=a.device, verbose=False)[0]
        num_boxes = int(len(result.boxes)) if result.boxes is not None else 0
        max_conf = float(result.boxes.conf.max().item()) if num_boxes else None
        detected = num_boxes > 0
        src_type = str(row.source_type)
        expected = str(row.expected_class)
        rows.append({"filename": str(row.filename), "expected_class": expected, "source_type": src_type,
                     "scene_type": str(row.scene_type), "preview_width": a.preview_width,
                     "preview_height": a.preview_height, "imgsz": a.imgsz, "confidence_threshold": a.conf,
                     "detected": detected, "detection_count": num_boxes,
                     "max_confidence": max_conf if max_conf is not None else ""})
    if missing_test_images:
        raise SystemExit(f"{len(missing_test_images)} test images could not be read. First: {missing_test_images[:5]}")
    result_df = pd.DataFrame(rows)
    csv_path = a.out / "presence_operational_test.csv"
    result_df.to_csv(csv_path, index=False)

    banana = result_df[result_df.source_type == "banana"]
    background = result_df[result_df.source_type == "background"]
    expected_order = ["Class_A", "Class_B", "Rejected"]
    per_grade = {}
    for grade in expected_order:
        group = banana[banana.expected_class == grade]
        total = int(len(group)); count = int(group.detected.sum()) if total else 0
        per_grade[grade] = {"detected": count, "total": total, "recall": count / total if total else None,
                            "misses": group.loc[~group.detected, "filename"].tolist()}
    background_total = int(len(background))
    false_triggers = int(background.detected.sum()) if background_total > 0 else None
    expected_targets = {"Class_A": 67, "Class_B": 83, "Rejected": 75}
    targets_match_counts = all(per_grade[g]["total"] == expected_targets[g] for g in expected_order)
    all_grade_targets_met = all(per_grade[g]["total"] > 0 and per_grade[g]["detected"] == per_grade[g]["total"] for g in expected_order)
    summary = {
        "weights": str(a.weights.resolve()), "trigger_confidence": a.conf,
        "preview_size": [a.preview_width, a.preview_height], "model_imgsz": a.imgsz,
        "mAP50": map50, "mAP50_95": map5095,
        "per_grade_trigger_recall": per_grade,
        "background_false_detections": false_triggers, "background_test_images": background_total,
        "background_false_trigger_evaluation_available": background_total > 0,
        "target_test_counts_match_plan": targets_match_counts,
        "all_banana_test_images_detected": all_grade_targets_met,
        "zero_background_false_detections": (false_triggers == 0) if background_total > 0 else None,
        "pass_all_requested_targets": bool(all_grade_targets_met and background_total > 0 and false_triggers == 0 and targets_match_counts),
        "warning": "These metrics reflect this dataset and one-frame preview simulation. If no background test images are present, false-trigger performance is NOT EVALUATED and the model cannot pass all targets. Verify live camera stability and trigger timing on the Raspberry Pi."
    }
    (a.out / "presence_evaluation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n=== PRESENCE MODEL EVALUATION ===")
    print(f"mAP@0.50      : {map50:.6f}" if map50 is not None else "mAP@0.50      : unavailable")
    print(f"mAP@0.50:0.95 : {map5095:.6f}" if map5095 is not None else "mAP@0.50:0.95 : unavailable")
    for grade in expected_order:
        g = per_grade[grade]
        recall = (f"{100*g['recall']:.2f}%" if g['recall'] is not None else "n/a")
        print(f"{grade:10s}: {g['detected']}/{g['total']} detected ({recall})")
        if g["misses"]: print("  Misses:", ", ".join(g["misses"]))
    if background_total > 0:
        print(f"Background: {false_triggers}/{background_total} false detections")
    else:
        print("Background: NOT EVALUATED (no background images in the test split)")
    print("PASS" if summary["pass_all_requested_targets"] else "NOT YET PASSING ALL TARGETS")
    print(f"Operational predictions: {csv_path}")
    print(f"Summary JSON           : {a.out / 'presence_evaluation_summary.json'}")
    print("Do not modify thresholds based on this held-out test set; use validation data for threshold selection.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
