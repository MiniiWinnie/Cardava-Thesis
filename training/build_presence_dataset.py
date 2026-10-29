#!/usr/bin/env python3
"""Build a single-class YOLO detection dataset using the thesis split manifest.

Expected raw image folders (recursive search): dataset/raw/Class_A, dataset/raw/Class_B, and dataset/raw/Rejected/Rejected.
Backgrounds: dataset/backgrounds/Background_001.jpg ...
A/B box metadata: stage2_annotation_preview/stage2_annotation_record.csv
Original split membership: stage1_split_record.csv
Reviewed Rejected labels: runs/presence_annotation_review/verified_labels
"""
from __future__ import annotations

import argparse
import csv
import math
import re
import shutil
from collections import Counter
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-dir", type=Path, default=ROOT / "dataset/raw",
                   help="Root scanned recursively for the Class_A, Class_B, and Rejected source images.")
    p.add_argument("--background-dir", type=Path, default=ROOT / "dataset/backgrounds")
    p.add_argument("--split-csv", type=Path, default=ROOT / "stage1_split_record.csv")
    p.add_argument("--annotations-csv", type=Path, default=ROOT / "stage2_annotation_preview/stage2_annotation_record.csv")
    p.add_argument("--rejected-labels", type=Path, default=ROOT / "runs/presence_annotation_review/verified_labels")
    p.add_argument("--background-manifest", type=Path, default=None,
                   help="Optional CSV with filename,scene_type columns. If omitted, infer scene type from Background_### filename ranges.")
    p.add_argument("--out", type=Path, default=ROOT / "dataset/presence_det")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--allow-no-background", action="store_true",
                   help="Build a preliminary banana-only dataset when no empty-scene images are available. Background false-trigger performance cannot be evaluated.")
    return p.parse_args()


def norm_class(value: str) -> str:
    value = str(value).strip().lower().replace(" ", "_")
    aliases = {"class_a": "Class_A", "class_b": "Class_B", "rejected": "Rejected"}
    if value not in aliases:
        raise ValueError(f"Unexpected original class: {value!r}")
    return aliases[value]


def file_map(folder: Path) -> dict[str, Path]:
    if not folder.exists():
        raise FileNotFoundError(f"Required directory does not exist: {folder}")
    found: dict[str, Path] = {}
    duplicates: dict[str, list[Path]] = {}
    for path in folder.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
            continue
        key = path.name.lower()
        if key in found:
            duplicates.setdefault(key, [found[key]]).append(path)
        else:
            found[key] = path
    if duplicates:
        example = next(iter(duplicates.items()))
        raise ValueError(f"Duplicate image basenames in {folder}: {example[0]} -> {example[1]}")
    return found


def box_line_from_row(row: pd.Series) -> str:
    if all(col in row.index and pd.notna(row[col]) for col in ("yolo_center_x", "yolo_center_y", "yolo_width", "yolo_height")):
        cx, cy, bw, bh = [float(row[c]) for c in ("yolo_center_x", "yolo_center_y", "yolo_width", "yolo_height")]
    else:
        x, y, bw_px, bh_px = [float(row[c]) for c in ("x", "y", "width", "height")]
        iw, ih = float(row["image_width"]), float(row["image_height"])
        cx, cy, bw, bh = (x + bw_px/2) / iw, (y + bh_px/2) / ih, bw_px / iw, bh_px / ih
    vals = (cx, cy, bw, bh)
    if not all(math.isfinite(v) and 0 <= v <= 1 for v in vals) or bw <= 0 or bh <= 0:
        raise ValueError(f"Invalid normalized box in annotation row for {row.get('raw_filename')}: {vals}")
    if cx - bw/2 < -1e-4 or cx + bw/2 > 1.0001 or cy - bh/2 < -1e-4 or cy + bh/2 > 1.0001:
        raise ValueError(f"Box extends outside image for {row.get('raw_filename')}: {vals}")
    cls = int(row.get("yolo_class", 0))
    if cls != 0:
        raise ValueError(f"Expected annotation class ID 0 for {row.get('raw_filename')}, got {cls}")
    return f"0 {cx:.8f} {cy:.8f} {bw:.8f} {bh:.8f}\n"


def infer_scene(filename: str) -> str:
    match = re.search(r"Background[_-](\d+)", filename, re.IGNORECASE)
    if not match:
        return "unknown"
    n = int(match.group(1))
    if 1 <= n <= 40: return "empty_velvet_ring_on"
    if 41 <= n <= 60: return "empty_velvet_changed_lighting"
    if 61 <= n <= 100: return "hand_or_arm_no_bunch"
    if 101 <= n <= 130: return "random_objects"
    if 131 <= n <= 150: return "loose_fingers_or_debris"
    return "unknown"


def split_backgrounds(paths: list[Path], manifest_path: Path | None, seed: int) -> tuple[dict[str, str], dict[str, str]]:
    if not paths:
        raise ValueError("No background images found.")
    type_map = {p.name: infer_scene(p.name) for p in paths}
    if manifest_path and manifest_path.is_file():
        df = pd.read_csv(manifest_path)
        if not {"filename", "scene_type"}.issubset(df.columns):
            raise ValueError(f"Background manifest must contain filename,scene_type columns: {manifest_path}")
        type_map.update({str(r.filename): str(r.scene_type) for r in df.itertuples(index=False)})
    missing_types = [p.name for p in paths if type_map.get(p.name, "unknown") == "unknown"]
    stratify_labels = [type_map[p.name] for p in paths]
    n = len(paths)
    n_train = int(math.floor(0.70 * n))
    n_test = int(math.ceil(0.15 * n))
    n_val = n - n_train - n_test
    names = [p.name for p in paths]
    num_categories = len(set(stratify_labels))
    can_stratify = (
        all(Counter(stratify_labels)[lab] >= 3 for lab in set(stratify_labels))
        and not missing_types
        and n_val >= num_categories
        and n_test >= num_categories
        and (n - n_train) >= 2 * num_categories
    )
    try:
        train_names, temp_names = train_test_split(
            names, train_size=n_train, random_state=seed,
            stratify=stratify_labels if can_stratify else None,
        )
        temp_labels = [type_map[name] for name in temp_names]
        val_names, test_names = train_test_split(
            temp_names, test_size=n_test, random_state=seed + 1,
            stratify=temp_labels if can_stratify else None,
        )
    except ValueError as exc:
        raise ValueError(f"Could not split background photos safely: {exc}. Use more examples per scene category or a complete background manifest.") from exc
    split_map = {}
    for split, group in (("train", train_names), ("val", val_names), ("test", test_names)):
        for name in group: split_map[name] = split
    return split_map, type_map


def main() -> int:
    a = parse_args()
    for needed in (a.split_csv, a.annotations_csv):
        if not needed.is_file(): raise SystemExit(f"Required metadata file not found: {needed}")
    if a.out.exists() and any(a.out.iterdir()):
        if not a.overwrite: raise SystemExit(f"Output directory is not empty: {a.out}. Use --overwrite to rebuild it.")
        shutil.rmtree(a.out)

    split_df = pd.read_csv(a.split_csv)
    required_split_cols = {"filename", "original_class", "split"}
    if not required_split_cols.issubset(split_df.columns):
        raise SystemExit(f"Split CSV needs columns: {sorted(required_split_cols)}")
    if split_df.filename.duplicated().any(): raise SystemExit("Duplicate filename(s) in Stage 1 split record.")
    split_df["original_class"] = split_df["original_class"].map(norm_class)
    split_df["split"] = split_df["split"].astype(str).str.lower().str.strip()
    if not set(split_df.split).issubset(SPLITS): raise SystemExit("Split CSV contains split names other than train/val/test.")
    split_counts = split_df.split.value_counts().to_dict()
    if split_counts != {"train": 1050, "val": 225, "test": 225}:
        print(f"WARNING: split counts differ from expected thesis Stage 1 counts: {split_counts}")

    raw_files = file_map(a.raw_dir)
    annotation_df = pd.read_csv(a.annotations_csv)
    if not {"raw_filename", "original_class"}.issubset(annotation_df.columns):
        raise SystemExit("Stage 2 annotation record must include raw_filename and original_class.")
    if annotation_df.raw_filename.duplicated().any(): raise SystemExit("Duplicate raw_filename rows in Stage 2 annotation record.")
    annotation_by_name = {str(r.raw_filename).lower(): r for _, r in annotation_df.iterrows()}

    backgrounds = sorted((p for p in a.background_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS), key=lambda p: p.name.lower()) if a.background_dir.exists() else []
    if not backgrounds and not a.allow_no_background:
        raise SystemExit(
            f"No background images found in {a.background_dir}. Add empty-scene photos, "
            "or pass --allow-no-background for a preliminary banana-only training run."
        )
    if backgrounds and len(backgrounds) != 150:
        print(f"WARNING: found {len(backgrounds)} background photos; the plan described 150.")
    if backgrounds:
        background_map, scene_map = split_backgrounds(backgrounds, a.background_manifest, a.seed)
    else:
        background_map, scene_map = {}, {}
        print("\nWARNING: NO BACKGROUND IMAGES INCLUDED.")
        print("This is a preliminary banana-only dataset; empty-table false-trigger performance cannot be evaluated.")

    rows_out = []
    missing = []
    for row in split_df.itertuples(index=False):
        filename, cls, split = str(row.filename), row.original_class, row.split
        source = raw_files.get(filename.lower())
        if source is None:
            missing.append(f"raw image missing: {filename} (expected class {cls})")
            continue
        out_label = None
        if cls in {"Class_A", "Class_B"}:
            record = annotation_by_name.get(filename.lower())
            if record is None:
                missing.append(f"A/B annotation missing: {filename}")
                continue
            if norm_class(record["original_class"]) != cls:
                missing.append(f"class mismatch for annotation: {filename}")
                continue
            try: label_content = box_line_from_row(record)
            except Exception as exc:
                missing.append(f"invalid A/B annotation for {filename}: {exc}")
                continue
        else:
            rejected_label_path = a.rejected_labels / f"{Path(filename).stem}.txt"
            if not rejected_label_path.is_file():
                missing.append(f"reviewed Rejected label missing: {rejected_label_path}")
                continue
            label_lines = [line.strip() for line in rejected_label_path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
            if len(label_lines) != 1:
                missing.append(f"Rejected label must contain exactly one box: {rejected_label_path}")
                continue
            try:
                fields = label_lines[0].split()
                if len(fields) != 5 or int(fields[0]) != 0: raise ValueError("expected class 0 and four box values")
                vals = [float(x) for x in fields[1:]]
                if not all(math.isfinite(v) and 0 <= v <= 1 for v in vals) or vals[2] <= 0 or vals[3] <= 0: raise ValueError("invalid normalized box")
                if vals[0]-vals[2]/2 < -1e-4 or vals[0]+vals[2]/2 > 1.0001 or vals[1]-vals[3]/2 < -1e-4 or vals[1]+vals[3]/2 > 1.0001: raise ValueError("box outside image")
                label_content = f"0 {vals[0]:.8f} {vals[1]:.8f} {vals[2]:.8f} {vals[3]:.8f}\n"
            except Exception as exc:
                missing.append(f"invalid Rejected annotation for {filename}: {exc}")
                continue
        rows_out.append({"filename": filename, "expected_class": cls, "split": split,
                         "source_type": "banana", "scene_type": "banana_bunch", "source_path": str(source),
                         "label_content": label_content})

    for path in backgrounds:
        rows_out.append({"filename": path.name, "expected_class": "Background", "split": background_map[path.name],
                         "source_type": "background", "scene_type": scene_map[path.name], "source_path": str(path),
                         "label_content": ""})

    # Each banana must have exactly one manifest row. We expect exactly one A/B annotation per original sample.
    annotation_names = set(annotation_df.raw_filename.astype(str).str.lower())
    ab_names = set(split_df.loc[split_df.original_class.isin(["Class_A", "Class_B"]), "filename"].astype(str).str.lower())
    missing_ab_annotations = sorted(ab_names - annotation_names)
    extra_annotations = sorted(annotation_names - ab_names)
    if missing_ab_annotations: missing.extend(f"A/B record absent from annotation CSV: {name}" for name in missing_ab_annotations[:20])
    if extra_annotations: missing.extend(f"annotation record not in expected A/B split: {name}" for name in extra_annotations[:20])
    if missing:
        print(f"Dataset build stopped: {len(missing)} issue(s). First 100:")
        for m in missing[:100]: print(" -", m)
        return 2

    a.out.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    for row in rows_out:
        split, filename = row["split"], row["filename"]
        image_out_dir = a.out / "images" / split
        labels_out_dir = a.out / "labels" / split
        image_out_dir.mkdir(parents=True, exist_ok=True); labels_out_dir.mkdir(parents=True, exist_ok=True)
        target_image = image_out_dir / filename
        shutil.copy2(row["source_path"], target_image)
        target_label = labels_out_dir / f"{Path(filename).stem}.txt"
        target_label.write_text(row["label_content"], encoding="utf-8")
        manifest_rows.append({k: row[k] for k in ("filename", "expected_class", "split", "source_type", "scene_type", "source_path")})

    # Quote absolute POSIX-style path so YAML paths with Windows drive letters/spaces are safe.
    base_path = a.out.resolve().as_posix().replace('"', '\\"')
    (a.out / "data.yaml").write_text(
        f'path: "{base_path}"\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: banana_bunch\n',
        encoding="utf-8",
    )
    manifest_path = a.out / "presence_split_manifest.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)

    print("Presence dataset created successfully.")
    print(f"Output: {a.out}")
    print("Counts by source and split:")
    print(pd.DataFrame(manifest_rows).groupby(["source_type", "split"]).size().to_string())
    print("Banana class counts by split:")
    print(pd.DataFrame(manifest_rows).query("source_type == 'banana'").groupby(["split", "expected_class"]).size().to_string())
    print(f"Data YAML: {a.out / 'data.yaml'}")
    print(f"Split manifest: {manifest_path}")
    print("Note: original banana split membership is preserved.")
    if backgrounds:
        print("Background photos were split separately by scene category where possible.")
    if not backgrounds:
        print("IMPORTANT: no backgrounds were included. Treat training as preliminary and do not report zero false triggers on empty scenes.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
