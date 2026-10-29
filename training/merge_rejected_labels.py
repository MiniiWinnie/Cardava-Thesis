#!/usr/bin/env python3
"""Merge human-reviewed Stage 2 proposals and corrected Rejected YOLO labels."""
from __future__ import annotations

import argparse
import csv
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REVIEW = ROOT / "runs/presence_annotation_review/rejected_review.csv"
DEFAULT_PROPOSALS = ROOT / "runs/presence_annotation_review/proposed_labels"
DEFAULT_CORRECTED = ROOT / "runs/presence_annotation_review/corrected_labels"
DEFAULT_OUTPUT = ROOT / "runs/presence_annotation_review/verified_labels"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--review-csv", type=Path, default=DEFAULT_REVIEW)
    p.add_argument("--proposals", type=Path, default=DEFAULT_PROPOSALS)
    p.add_argument("--corrected", type=Path, default=DEFAULT_CORRECTED,
                   help="Directory of corrected YOLO TXT files exported by the annotation tool.")
    p.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def read_label(path: Path) -> list[str]:
    if not path.exists():
        raise ValueError(f"Missing label file: {path}")
    lines = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if len(lines) != 1:
        raise ValueError(f"Expected exactly one whole-bunch box in {path}, found {len(lines)}")
    fields = lines[0].split()
    if len(fields) != 5:
        raise ValueError(f"Expected 5 YOLO fields in {path}, got {len(fields)}")
    try:
        cls = int(fields[0]); vals = [float(x) for x in fields[1:]]
    except ValueError as e:
        raise ValueError(f"Non-numeric label fields in {path}") from e
    if cls != 0:
        raise ValueError(f"Expected one-class ID 0 (banana_bunch) in {path}; found {cls}. Configure the annotation export for one class.")
    cx, cy, bw, bh = vals
    if not all(0 <= x <= 1 for x in vals) or bw <= 0 or bh <= 0:
        raise ValueError(f"Invalid normalized YOLO box in {path}: {lines[0]}")
    if cx - bw/2 < -1e-4 or cx + bw/2 > 1.0001 or cy - bh/2 < -1e-4 or cy + bh/2 > 1.0001:
        raise ValueError(f"Box extends outside image in {path}: {lines[0]}")
    return [f"0 {cx:.8f} {cy:.8f} {bw:.8f} {bh:.8f}"]


def find_label(folder: Path, stem: str) -> Path | None:
    candidates = list(folder.rglob(f"{stem}.txt")) if folder.exists() else []
    if len(candidates) > 1:
        raise ValueError(f"Multiple correction label files found for {stem}: {candidates}")
    return candidates[0] if candidates else None


def main() -> int:
    a = parse_args()
    if not a.review_csv.is_file():
        raise SystemExit(f"Review CSV not found: {a.review_csv}")
    with a.review_csv.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows or "filename" not in rows[0] or "status" not in rows[0]:
        raise SystemExit("Review CSV must have 'filename' and 'status' columns.")
    if a.out.exists() and any(a.out.iterdir()):
        if not a.overwrite:
            raise SystemExit(f"Output already exists and is not empty: {a.out}. Use --overwrite to replace it.")
        shutil.rmtree(a.out)
    a.out.mkdir(parents=True, exist_ok=True)

    errors, used, fixed = [], 0, 0
    for row in rows:
        filename = Path(row["filename"]).name
        stem = Path(filename).stem
        status = (row.get("status") or "").strip().lower()
        if status == "ok":
            source = a.proposals / f"{stem}.txt"
            try:
                lines = read_label(source)
            except Exception as e:
                errors.append(f"{filename}: status=ok but proposal is invalid: {e}")
                continue
            used += 1
        elif status in {"fix", "fixed", "no_box", "no box"}:
            source = find_label(a.corrected, stem)
            if source is None:
                errors.append(f"{filename}: status={status}, but corrected label {stem}.txt is missing")
                continue
            try:
                lines = read_label(source)
            except Exception as e:
                errors.append(f"{filename}: corrected label invalid: {e}")
                continue
            fixed += 1
        else:
            errors.append(f"{filename}: status must be 'ok' or 'fix'; current status is {status!r}")
            continue
        (a.out / f"{stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    if errors:
        shutil.rmtree(a.out, ignore_errors=True)
        print(f"Could not build verified label set; {len(errors)} issue(s):")
        for error in errors[:100]: print(" -", error)
        if len(errors) > 100: print(f" - ... plus {len(errors) - 100} more")
        return 2

    print(f"Verified labels created: {len(rows)}")
    print(f"Accepted proposed boxes: {used}")
    print(f"Used manually corrected boxes: {fixed}")
    print(f"Output directory: {a.out}")
    print("Before training, visually spot-check the exported labels and confirm every box includes the complete bunch, not the shadow.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
