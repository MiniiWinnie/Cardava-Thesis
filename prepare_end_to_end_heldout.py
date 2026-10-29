from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pandas as pd


# Save this script in the project root OR in the project's training/ folder.
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent if HERE.name.lower() == "training" else HERE

STAGE1_TEST_CSV = ROOT / "runs" / "stage1_final" / "test_evaluation" / "stage1_test_predictions.csv"
STAGE2_TEST_CSV = ROOT / "runs" / "stage2_final" / "final_grading_evaluation" / "stage2_final_grading_predictions.csv"
RAW_ROOT = ROOT / "dataset" / "raw"
OUTPUT_DIR = ROOT / "evaluation" / "end_to_end_heldout_106"
OUTPUT_IMAGES = OUTPUT_DIR / "images"


def require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            f"Cannot find {description}:\n  {path}\n"
            "Check that the thesis ZIP has been extracted into your project root and that this script is in the project root or training/ folder."
        )


def truthy_label(value: object) -> str:
    return str(value).strip().lower()


def find_image(filename: str, true_class: str) -> Path:
    """Find a unique raw image by filename. Fail rather than silently pick a duplicate."""
    candidates = [p for p in RAW_ROOT.rglob(filename) if p.is_file()]
    if not candidates:
        raise FileNotFoundError(
            f"Image not found under {RAW_ROOT}: {filename}\n"
            "Check that the original dataset is inside dataset/raw and filenames match the CSV records."
        )

    class_folder = {
        "Class_A": "Class_A",
        "Class_B": "Class_B",
        "Rejected": "Rejected",
    }.get(true_class)
    if class_folder:
        preferred = [p for p in candidates if class_folder.lower() in [part.lower() for part in p.parts]]
        if len(preferred) == 1:
            return preferred[0]
        if preferred:
            candidates = preferred

    if len(candidates) != 1:
        listing = "\n".join(f"  {p}" for p in candidates)
        raise RuntimeError(
            f"Found multiple files named {filename}; refusing to guess which is the original:\n{listing}"
        )
    return candidates[0]


def main() -> int:
    require_file(STAGE1_TEST_CSV, "Stage 1 held-out test predictions CSV")
    require_file(STAGE2_TEST_CSV, "Stage 2 held-out test predictions CSV")
    if not RAW_ROOT.is_dir():
        raise FileNotFoundError(f"Original images folder not found: {RAW_ROOT}")

    stage1 = pd.read_csv(STAGE1_TEST_CSV)
    stage2 = pd.read_csv(STAGE2_TEST_CSV)

    required_s1 = {"filename", "true_label", "yolo_predicted_label"}
    required_s2 = {"filename", "true_grade", "predicted_grade"}
    if not required_s1.issubset(stage1.columns):
        raise ValueError(f"Stage 1 CSV is missing columns: {sorted(required_s1 - set(stage1.columns))}")
    if not required_s2.issubset(stage2.columns):
        raise ValueError(f"Stage 2 CSV is missing columns: {sorted(required_s2 - set(stage2.columns))}")

    stage1["filename"] = stage1["filename"].astype(str).str.strip()
    stage2["filename"] = stage2["filename"].astype(str).str.strip()
    stage2_names = set(stage2["filename"])

    # All 75 Rejected images from Stage 1's held-out test set, plus the 31 Healthy
    # images which are ALSO members of Stage 2's separate held-out test set.
    s1_unhealthy = stage1[stage1["true_label"].map(truthy_label).isin({"unhealthy", "rejected"})].copy()
    s1_healthy = stage1[stage1["true_label"].map(truthy_label).eq("healthy")].copy()
    shared_healthy = s1_healthy[s1_healthy["filename"].isin(stage2_names)].copy()

    if len(s1_unhealthy) != 75:
        raise ValueError(f"Expected 75 Unhealthy/Rejected Stage 1 test images, found {len(s1_unhealthy)}.")
    if len(shared_healthy) != 31:
        raise ValueError(f"Expected 31 Healthy images shared by both test sets, found {len(shared_healthy)}.")

    stage2_index = stage2.set_index("filename", drop=False)
    records: list[dict[str, object]] = []
    selected = [(row, "Rejected") for _, row in s1_unhealthy.iterrows()]

    for _, row in shared_healthy.iterrows():
        name = row["filename"]
        s2_row = stage2_index.loc[name]
        grade_value = str(s2_row["true_grade"]).strip().upper()
        if grade_value not in {"A", "B"}:
            raise ValueError(f"Unexpected Stage 2 true_grade for {name}: {grade_value!r}")
        selected.append((row, f"Class_{grade_value}"))

    # Recreate only this generated output folder to avoid mixing in stale copies.
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_IMAGES.mkdir(parents=True, exist_ok=True)

    for s1_row, true_class in selected:
        filename = str(s1_row["filename"])
        source = find_image(filename, true_class)
        dest = OUTPUT_IMAGES / filename
        shutil.copy2(source, dest)

        rec: dict[str, object] = {
            "filename": filename,
            "ground_truth_grade": true_class,
            "stage1_true_label": str(s1_row["true_label"]),
            "stage1_prediction_recorded": str(s1_row["yolo_predicted_label"]),
            "source_path": str(source),
            "copied_path": str(dest),
        }
        if true_class in {"Class_A", "Class_B"}:
            s2_row = stage2_index.loc[filename]
            rec["stage2_true_grade"] = str(s2_row["true_grade"])
            rec["stage2_prediction_recorded"] = str(s2_row["predicted_grade"])
        else:
            rec["stage2_true_grade"] = "not_applicable_rejected_at_stage1"
            rec["stage2_prediction_recorded"] = "not_applicable_rejected_at_stage1"
        records.append(rec)

    manifest = pd.DataFrame(records).sort_values(["ground_truth_grade", "filename"])
    manifest.to_csv(OUTPUT_DIR / "end_to_end_heldout_manifest.csv", index=False)

    counts = manifest["ground_truth_grade"].value_counts().to_dict()
    expected = {"Rejected": 75, "Class_A": 15, "Class_B": 16}
    if counts != expected:
        raise ValueError(f"Unexpected selected counts: {counts}; expected {expected}")

    print("Created the existing 106-image end-to-end held-out subset.")
    print(f"Images:   {OUTPUT_IMAGES}")
    print(f"Manifest: {OUTPUT_DIR / 'end_to_end_heldout_manifest.csv'}")
    print("Counts:")
    for label in ("Rejected", "Class_A", "Class_B"):
        print(f"  {label}: {counts[label]}")
    print("\nNote: these are held out from Stage 1 and Stage 2 model fitting/selection per the paper's split analysis; they are not a newly collected external dataset.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
