#!/usr/bin/env python
"""Diagnostic for the final Stage 2 YOLO model and Rejected_038.jpg.

Run from PowerShell in the Cardava-Thesis project directory:
    python check_rejected038.py

The script does not train models or modify source images, checkpoints, or CSVs.
It checks Ultralytics, re-runs inference on a Class A test image, compares the
result with the saved grading CSV when possible, tests Rejected_038.jpg, and
prints the relevant grading logic from stage2_test_grading_final.py.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import pandas as pd
    from PIL import Image
    import ultralytics
    from ultralytics import YOLO
except ImportError as exc:
    print(f"[ERROR] Missing required package: {exc}")
    print("Activate the project environment before running this script.")
    raise SystemExit(2)

EXPECTED_ULTRALYTICS = "8.4.153"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MASKED_HINTS = ("mask", "black_background", "black-background", "cleaned", "reference")
IGNORE_DIRS = {".git", "venv", ".venv", "__pycache__", "node_modules"}
THRESHOLD_PX = 2730.07


def args_parser() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Check a Stage 2 Class A sample and Rejected_038.jpg")
    p.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parent)
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--predictions-csv", type=Path, default=None)
    p.add_argument("--class-a-image", type=Path, default=None)
    p.add_argument("--rejected-image", type=Path, default=None)
    p.add_argument("--source-script", type=Path, default=None)
    p.add_argument("--width-tolerance", type=float, default=1.0)
    return p.parse_args()


def masked_path(path: Path) -> bool:
    return any(hint in str(part).lower() for part in path.parts for hint in MASKED_HINTS)


def find_files(root: Path, filename: str) -> list[Path]:
    found = []
    for path in root.rglob("*"):
        if path.is_file() and path.name.lower() == filename.lower():
            if not any(part.lower() in IGNORE_DIRS for part in path.parts):
                found.append(path)
    return sorted(found, key=lambda x: (masked_path(x), len(str(x))))


def ask_existing_path(prompt: str) -> Path:
    while True:
        value = input(prompt).strip().strip('"')
        if value:
            p = Path(value).expanduser()
            if p.is_file():
                return p
            print(f"[NOT FOUND] {p}")
        else:
            print("Please enter a file path.")


def find_csv(root: Path, given: Path | None) -> Path:
    if given:
        path = given.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Predictions CSV not found: {path}")
        return path
    preferred = [
        root / "runs" / "stage2_final" / "final_grading_evaluation" / "stage2_final_grading_predictions.csv",
        root / "stage2_final_grading_predictions(1).csv",
        root / "stage2_final_grading_predictions.csv",
    ]
    for path in preferred:
        if path.is_file():
            return path
    candidates = [p for p in root.rglob("stage2_final_grading_predictions*.csv") if p.is_file()]
    candidates = [p for p in candidates if not any(part.lower() in IGNORE_DIRS for part in p.parts)]
    if candidates:
        candidates.sort(key=lambda p: ("final_grading_evaluation" not in str(p).lower(), "(1)" in p.name, len(str(p))))
        return candidates[0]
    return ask_existing_path("Could not find the final grading CSV. Paste its full path: ")


def get_col(df: pd.DataFrame, options: tuple[str, ...]) -> str | None:
    cols = {str(c).strip().lower(): str(c) for c in df.columns}
    for name in options:
        if name.lower() in cols:
            return cols[name.lower()]
    return None


def find_raw_rejected(root: Path, explicit: Path | None) -> Path:
    if explicit:
        p = explicit.expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f"Rejected image not found: {p}")
        return p
    candidates = find_files(root, "Rejected_038.jpg")
    if not candidates:
        return ask_existing_path("Could not find Rejected_038.jpg. Paste the full path to the RAW image: ")
    print("\n[SEARCH] Rejected_038.jpg candidates:")
    for i, path in enumerate(candidates, 1):
        try:
            with Image.open(path) as im:
                size = im.size
        except Exception:
            size = (0, 0)
        tag = " [LIKELY MASK/REFERENCE]" if masked_path(path) else " [candidate raw image]"
        print(f"  {i}. {path} | size={size[0]}x{size[1]}{tag}")
    raw = [p for p in candidates if not masked_path(p)]
    if len(raw) == 1:
        print(f"[AUTO-SELECT] {raw[0]}")
        return raw[0]
    while True:
        ans = input(f"Choose the original/raw image (1-{len(candidates)}) or paste a full path: ").strip().strip('"')
        if ans.isdigit() and 1 <= int(ans) <= len(candidates):
            return candidates[int(ans) - 1]
        p = Path(ans).expanduser()
        if p.is_file():
            return p
        print("Invalid choice or file path.")


def resolve_class_a(root: Path, df: pd.DataFrame, explicit: Path | None):
    if explicit:
        path = explicit.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Class A test image not found: {path}")
        return path, None

    filename_col = get_col(df, ("filename", "image", "image_name"))
    grade_col = get_col(df, ("predicted_grade", "pred_grade", "prediction", "grade"))
    detected_col = get_col(df, ("detected",))
    width_col = get_col(df, ("bbox_width_px", "width_px", "width"))
    conf_col = get_col(df, ("confidence", "conf"))
    if not filename_col or not grade_col:
        raise ValueError(f"Could not identify filename and predicted-grade columns. CSV columns: {list(df.columns)}")

    candidates = df.copy()
    candidates[grade_col] = candidates[grade_col].astype(str).str.strip().str.lower()
    candidates = candidates[candidates[grade_col].isin({"a", "class a"})]
    if detected_col:
        candidates = candidates[candidates[detected_col].astype(str).str.lower().isin({"true", "1", "yes"})]

    for _, row in candidates.iterrows():
        filename = str(row[filename_col])
        likely = root / "dataset" / "stage2_det" / "images" / "test" / filename
        if likely.is_file():
            expected = {
                "filename": filename,
                "grade": str(row[grade_col]),
                "width": float(row[width_col]) if width_col and pd.notna(row[width_col]) else None,
                "confidence": float(row[conf_col]) if conf_col and pd.notna(row[conf_col]) else None,
            }
            return likely, expected
        matches = find_files(root, filename)
        matches = [m for m in matches if "stage2_det" in str(m).lower() and "test" in [x.lower() for x in m.parts]]
        if matches:
            expected = {
                "filename": filename,
                "grade": str(row[grade_col]),
                "width": float(row[width_col]) if width_col and pd.notna(row[width_col]) else None,
                "confidence": float(row[conf_col]) if conf_col and pd.notna(row[conf_col]) else None,
            }
            return matches[0], expected
    print("[WARN] No Class A CSV row could be matched to an image path automatically.")
    return ask_existing_path("Paste the full path to a Class A image included in the final test CSV: "), None


def run_inference(model: YOLO, path: Path, label: str) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    with Image.open(path) as im:
        size = im.size
    print("\n" + "=" * 76)
    print(label)
    print("=" * 76)
    print(f"Image path : {path}")
    print(f"Image size : {size[0]} x {size[1]} pixels")
    if size != (4608, 2592):
        print("[NOTE] Expected raw-image dimensions were 4608 x 2592; verify this is the intended raw file.")
    result = model.predict(source=str(path), imgsz=640, conf=0.25, iou=0.70, augment=False, verbose=False)[0]
    boxes = result.boxes
    count = 0 if boxes is None else len(boxes)
    print(f"Detections : {count}")
    if count == 0:
        print("Result     : NO DETECTION (this diagnostic does not treat it as Rejected automatically)")
        return {"path": path, "size": size, "detections": 0, "width": None, "confidence": None, "grade": "No Detection"}

    xyxys = boxes.xyxy.cpu().tolist()  # Ultralytics returns coords scaled to the original image.
    confs = boxes.conf.cpu().tolist()
    for idx, (xyxy, conf) in enumerate(zip(xyxys, confs)):
        width = max(0.0, float(xyxy[2]) - float(xyxy[0]))
        print(f"Detection {idx}: xyxy={[round(v, 2) for v in xyxy]}, confidence={conf:.4f}, width={width:.2f}px")
    selected = max(range(count), key=lambda i: confs[i])
    xyxy = xyxys[selected]
    width = max(0.0, float(xyxy[2]) - float(xyxy[0]))
    confidence = float(confs[selected])
    grade = "Class A" if width >= THRESHOLD_PX else "Class B"
    print(f"\nSelected box       : {selected} (highest confidence)")
    print(f"Selected width     : {width:.2f} px")
    print(f"Selected confidence: {confidence:.4f}")
    print(f"Stage 2 grade      : {grade}")
    return {"path": path, "size": size, "detections": count, "width": width, "confidence": confidence, "grade": grade}


def compare_csv(result: dict, expected: dict | None, tolerance: float) -> None:
    print("\n[CSV COMPARISON]")
    if not expected:
        print("Could not match the rerun sample to a CSV row automatically.")
        return
    grade_normalize = lambda value: str(value).lower().replace("class", "").strip()
    grade_match = grade_normalize(result["grade"]) == grade_normalize(expected["grade"])
    print(f"CSV filename          : {expected['filename']}")
    print(f"CSV predicted grade   : {expected['grade']}")
    print(f"Rerun predicted grade : {result['grade']}")
    print(f"Grade match           : {'PASS' if grade_match else 'MISMATCH'}")
    if expected["width"] is not None and result["width"] is not None:
        diff = abs(result["width"] - expected["width"])
        print(f"CSV width             : {expected['width']:.4f}px")
        print(f"Rerun width           : {result['width']:.4f}px")
        print(f"Absolute difference   : {diff:.4f}px")
        print(f"Width match (<= {tolerance:.2f}px): {'PASS' if diff <= tolerance else 'MISMATCH'}")
    if expected["confidence"] is not None and result["confidence"] is not None:
        print(f"CSV confidence        : {expected['confidence']:.6f}")
        print(f"Rerun confidence      : {result['confidence']:.6f}")


def print_rule_lines(script_path: Path | None) -> None:
    print("\n" + "=" * 76)
    print("ORIGINAL EVALUATION SCRIPT RULES")
    print("=" * 76)
    if not script_path or not script_path.is_file():
        print("Could not find stage2_test_grading_final.py automatically.")
        print("Expected path: project root\\stage2_test_grading_final.py")
        return
    lines = script_path.read_text(encoding="utf-8", errors="replace").splitlines()
    keywords = ("WIDTH_THRESHOLD_PX", "CONF_THRESHOLD", "IOU_THRESHOLD", "argmax", "boxes.conf", "boxes.xyxy", "x2 - x1", "width_px >=", "No Detection", 'predicted_grade =')
    selected = set()
    for i, line in enumerate(lines):
        if any(k.lower() in line.lower() for k in keywords):
            selected.update(range(max(0, i - 1), min(len(lines), i + 2)))
    if not selected:
        print("No matching grading-rule lines found.")
    else:
        for i in sorted(selected):
            print(f"{i + 1:04d}: {lines[i]}")


def main() -> int:
    args = args_parser()
    root = args.project_root.expanduser().resolve()
    if not root.is_dir():
        print(f"[ERROR] Project root does not exist: {root}")
        return 2

    print("=" * 76)
    print("STAGE 2 DIAGNOSTIC: CLASS A CHECK + REJECTED_038.jpg")
    print("=" * 76)
    print(f"Python executable : {sys.executable}")
    print(f"Ultralytics       : {ultralytics.__version__}")
    if ultralytics.__version__ == EXPECTED_ULTRALYTICS:
        print(f"[PASS] Expected Ultralytics version {EXPECTED_ULTRALYTICS} is active.")
    else:
        print(f"[WARN] Expected {EXPECTED_ULTRALYTICS}; confirm this matches the environment used for the official evaluation.")
    print("Inference settings: imgsz=640, conf=0.25, iou=0.70, augment=False")
    print(f"Grading rule      : width >= {THRESHOLD_PX} px -> Class A; width < {THRESHOLD_PX} px -> Class B")

    checkpoint = (args.checkpoint or (root / "runs" / "stage2_final" / "weights" / "best.pt")).expanduser().resolve()
    csv_path = find_csv(root, args.predictions_csv).resolve()
    print(f"\n[CSV] {csv_path}")
    try:
        df = pd.read_csv(csv_path)
        class_a_path, expected = resolve_class_a(root, df, args.class_a_image)
        rejected_path = find_raw_rejected(root, args.rejected_image)
        source_script = args.source_script or (root / "stage2_test_grading_final.py")
        model = YOLO(str(checkpoint)) if checkpoint.is_file() else None
        if model is None:
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        print(f"\n[MODEL] Loaded {checkpoint}")
        class_a_result = run_inference(model, class_a_path, "STEP 4 — VERIFY A KNOWN CLASS A TEST IMAGE")
        compare_csv(class_a_result, expected, args.width_tolerance)
        rejected_result = run_inference(model, rejected_path, "STEP 5 — TEST Rejected_038.jpg")
        print_rule_lines(source_script)
        print("\n" + "=" * 76)
        print("SUMMARY TO SEND BACK")
        print("=" * 76)
        print(f"Predictions CSV   : {csv_path}")
        print(f"Checkpoint        : {checkpoint}")
        print(f"Class A image     : {class_a_path}")
        print(f"Class A grade     : {class_a_result['grade']}")
        print(f"Class A width     : {class_a_result['width']}")
        print(f"Rejected image    : {rejected_path}")
        print(f"Rejected size     : {rejected_result['size'][0]} x {rejected_result['size'][1]}")
        print(f"Rejected detections: {rejected_result['detections']}")
        print(f"Rejected_038 result: {rejected_result['grade']}")
        print("\nCopy the console output and paste it back into the chat.")
        print("Diagnostic only: this script does not calculate an overall end-to-end accuracy.")
        return 0
    except Exception as exc:
        print(f"\n[ERROR] {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
