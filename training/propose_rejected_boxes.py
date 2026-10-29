#!/usr/bin/env python3
"""Propose whole-bunch boxes for Rejected images using the existing Stage 2 model.

Every proposal is only a suggestion. Review all contact sheets and manually fix
bad/missing boxes before running merge_rejected_labels.py.
"""
from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--images", type=Path, default=ROOT / "dataset/raw/Rejected/Rejected",
                   help="Folder containing the 500 raw Rejected images (your current layout nests them under dataset/raw/Rejected/Rejected).")
    p.add_argument("--weights", type=Path, default=ROOT / "runs/stage2_final/weights/best.pt",
                   help="Existing Stage 2 checkpoint used only to propose boxes.")
    p.add_argument("--out", type=Path, default=ROOT / "runs/presence_annotation_review",
                   help="Output folder for proposed labels, contact sheets, and review CSV.")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.05,
                   help="Low proposal threshold to reduce missed suggestions; not the app threshold.")
    p.add_argument("--iou", type=float, default=0.70)
    p.add_argument("--per-page", type=int, default=12)
    return p.parse_args()


def image_files(folder: Path) -> list[Path]:
    return sorted((x for x in folder.iterdir() if x.is_file() and x.suffix.lower() in IMAGE_EXTS),
                  key=lambda x: x.name.lower())


def to_yolo_line(x1: float, y1: float, x2: float, y2: float, width: int, height: int) -> str:
    x1 = max(0.0, min(float(width), x1)); x2 = max(0.0, min(float(width), x2))
    y1 = max(0.0, min(float(height), y1)); y2 = max(0.0, min(float(height), y2))
    bw, bh = x2 - x1, y2 - y1
    if bw <= 0 or bh <= 0:
        raise ValueError("Predicted bounding box has zero/negative area after clipping")
    cx, cy = (x1 + x2) / 2 / width, (y1 + y2) / 2 / height
    return f"0 {cx:.8f} {cy:.8f} {bw / width:.8f} {bh / height:.8f}\n"


def draw_sheet_cell(image_path: Path, box: tuple[float, float, float, float] | None,
                    conf: float | None, cell_w: int = 360, cell_h: int = 280) -> np.ndarray:
    image = cv2.imread(str(image_path))
    if image is None:
        image = np.zeros((720, 1280, 3), np.uint8)
        cv2.putText(image, "UNREADABLE IMAGE", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 2)
    h, w = image.shape[:2]
    canvas = np.full((cell_h, cell_w, 3), 245, dtype=np.uint8)
    header = f"{image_path.name}"
    cv2.putText(canvas, header[:42], (7, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (20, 20, 20), 1, cv2.LINE_AA)
    avail_w, avail_h = cell_w - 12, cell_h - 52
    scale = min(avail_w / w, avail_h / h)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)
    ox, oy = (cell_w - nw) // 2, 31 + (avail_h - nh) // 2
    if box is not None:
        x1, y1, x2, y2 = box
        pts = [int(round(x * scale)) for x in (x1, x2)]
        pty = [int(round(y * scale)) for y in (y1, y2)]
        cv2.rectangle(resized, (max(0, pts[0]), max(0, pty[0])),
                      (min(nw - 1, pts[1]), min(nh - 1, pty[1])), (0, 180, 0), 3)
    canvas[oy:oy + nh, ox:ox + nw] = resized
    label = "PROPOSAL - REVIEW" if box is not None else "NO BOX - FIX MANUALLY"
    if conf is not None:
        label += f" ({conf:.2f})"
    color = (0, 120, 0) if box is not None else (0, 0, 220)
    cv2.putText(canvas, label, (7, cell_h - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 1, cv2.LINE_AA)
    return canvas


def main() -> int:
    a = parse_args()
    images = image_files(a.images)
    if not images:
        raise SystemExit(f"No images found in {a.images}. Check --images.")
    if not a.weights.is_file():
        raise SystemExit(f"Stage 2 checkpoint not found: {a.weights}")
    labels_dir = a.out / "proposed_labels"
    sheets_dir = a.out / "contact_sheets"
    labels_dir.mkdir(parents=True, exist_ok=True)
    sheets_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(a.weights))
    rows: list[dict[str, object]] = []
    preview_cells: list[np.ndarray] = []
    page_num = 1

    for index, path in enumerate(images, start=1):
        image = cv2.imread(str(path))
        if image is None:
            raise SystemExit(f"Cannot read image: {path}")
        h, w = image.shape[:2]
        result = model.predict(source=image, imgsz=a.imgsz, conf=a.conf, iou=a.iou,
                               augment=False, verbose=False)[0]
        candidates = []
        if result.boxes is not None and len(result.boxes) > 0:
            for b in result.boxes:
                xyxy = b.xyxy[0].detach().cpu().numpy().astype(float).tolist()
                score = float(b.conf[0].detach().cpu().item())
                x1, y1, x2, y2 = xyxy
                area = max(0, x2 - x1) * max(0, y2 - y1)
                candidates.append((score, area, (x1, y1, x2, y2)))
        candidates.sort(key=lambda q: (q[0], q[1]), reverse=True)
        selected = candidates[0] if candidates else None
        label_path = labels_dir / f"{path.stem}.txt"
        box = selected[2] if selected else None
        conf = selected[0] if selected else None
        if selected:
            label_path.write_text(to_yolo_line(*box, w, h), encoding="utf-8")
            status = ""  # human must set to ok or fix
        else:
            label_path.write_text("", encoding="utf-8")
            status = "fix"
        rows.append({
            "filename": path.name,
            "status": status,
            "proposal_count": len(candidates),
            "selected_confidence": f"{conf:.6f}" if conf is not None else "",
            "x1": f"{box[0]:.3f}" if box else "",
            "y1": f"{box[1]:.3f}" if box else "",
            "x2": f"{box[2]:.3f}" if box else "",
            "y2": f"{box[3]:.3f}" if box else "",
            "proposal_label": str(label_path.relative_to(a.out)),
        })
        preview_cells.append(draw_sheet_cell(path, box, conf))
        if len(preview_cells) == a.per_page or index == len(images):
            cols, rows_n = 3, math.ceil(len(preview_cells) / 3)
            cell_h, cell_w = preview_cells[0].shape[:2]
            sheet = np.full((rows_n * cell_h, cols * cell_w, 3), 255, np.uint8)
            for j, cell in enumerate(preview_cells):
                rr, cc = divmod(j, cols)
                sheet[rr * cell_h:(rr + 1) * cell_h, cc * cell_w:(cc + 1) * cell_w] = cell
            out_path = sheets_dir / f"contact_sheet_{page_num:03d}.jpg"
            cv2.imwrite(str(out_path), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            page_num += 1
            preview_cells = []

    review_csv = a.out / "rejected_review.csv"
    with review_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)

    misses = sum(1 for row in rows if not row["selected_confidence"])
    print(f"Images processed: {len(images)}")
    print(f"No-proposal images needing manual annotation: {misses}")
    print(f"Review every box against the whole bunch (all fingers + crown; exclude the shadow).")
    print(f"1) Inspect contact sheets: {sheets_dir}")
    print(f"2) In {review_csv}, mark each row status as 'ok' or 'fix'.")
    print(f"3) Export corrected YOLO label TXT files for 'fix' rows to: {a.out / 'corrected_labels'}")
    print(f"4) Run merge_rejected_labels.py after all rows have been reviewed.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
