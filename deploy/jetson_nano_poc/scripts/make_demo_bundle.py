"""Generate a synthetic demo bundle in the AICS generic_csv layout.

Used by the Jetson Nano POC so the demo works without a live AOI feed.
The synthetic data is designed to look realistic on stage:

  - ~10 components per board
  - mixed defect types (TOMBSTONE, MISALIGNMENT, SHORT, MISSING, INSUFFICIENT_SOLDER)
  - the FIRST 60% of components are obvious false calls (so the model
    "appears to learn fast" in the first minute of the demo)
  - the next 30% are obvious true defects
  - the last 10% are subtle — the kind only experienced operators catch

Output:
    <out>/board_<NNNN>.csv      one row per component
    <out>/boards/<board_id>/<ref_des>.jpg
"""
from __future__ import annotations

import argparse
import csv
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


DEFECT_TYPES = [
    "TOMBSTONE",
    "MISALIGNMENT",
    "SHORT",
    "MISSING",
    "INSUFFICIENT_SOLDER",
    "EXCESS_SOLDER",
]


def _synth_roi(label: str, defect_type: str, size: int = 224) -> Image.Image:
    """Render a plausible-looking ROI image."""
    rng = np.random.default_rng()

    # Background — green PCB-ish gradient
    bg = np.zeros((size, size, 3), dtype=np.uint8)
    bg[..., 1] = rng.integers(80, 130, size=(size, size))   # green
    bg[..., 0] = rng.integers(20, 50, size=(size, size))    # red
    bg[..., 2] = rng.integers(20, 60, size=(size, size))    # blue

    img = Image.fromarray(bg, mode="RGB")
    d = ImageDraw.Draw(img)

    # Component body — a metallic rectangle in the centre
    cx, cy = size // 2, size // 2
    w, h = rng.integers(50, 90), rng.integers(30, 60)
    if label == "TRUE_DEFECT":
        # Tilt or shift to simulate defects
        if defect_type == "TOMBSTONE":
            d.rectangle([cx - w // 2, cy - 10, cx + w // 2, cy + h], fill=(180, 180, 200))
            d.line([cx - w // 2, cy + h, cx + w // 2, cy + h], fill=(40, 30, 30), width=3)
        elif defect_type == "MISALIGNMENT":
            d.rectangle([cx - w // 2 + 25, cy - h // 2, cx + w // 2 + 25, cy + h // 2], fill=(180, 180, 200))
        elif defect_type == "SHORT":
            d.rectangle([cx - w // 2, cy - h // 2, cx + w // 2, cy + h // 2], fill=(180, 180, 200))
            d.ellipse([cx + w // 2 - 5, cy - 5, cx + w // 2 + 30, cy + 5], fill=(220, 220, 240))
        elif defect_type == "MISSING":
            pass  # nothing — just bare PCB
        else:  # solder issues
            d.rectangle([cx - w // 2, cy - h // 2, cx + w // 2, cy + h // 2], fill=(180, 180, 200))
            d.ellipse([cx - 10, cy + h // 2 - 5, cx + 10, cy + h // 2 + 15], fill=(60, 60, 80))
    else:
        # Clean component
        d.rectangle([cx - w // 2, cy - h // 2, cx + w // 2, cy + h // 2], fill=(180, 180, 200))

    # Add subtle noise so it doesn't look stamp-perfect
    noise = rng.normal(0, 8, (size, size, 3))
    arr = np.clip(np.asarray(img).astype(np.int32) + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n_boards", type=int, default=50)
    p.add_argument("--components_per_board", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    images_root = out / "boards"
    images_root.mkdir(exist_ok=True)

    t0 = datetime.now(timezone.utc) - timedelta(hours=1)

    for b in range(args.n_boards):
        board_id = f"BD-DEMO-{b:04d}"
        ts = (t0 + timedelta(minutes=b)).isoformat()
        line_id = "DEMO-L1"
        lot = "DEMO-LOT-A"

        rows = []
        # Allocate labels — front-loaded false calls for theatrical effect
        progress = b / max(args.n_boards - 1, 1)
        if progress < 0.6:
            true_defect_prob = 0.10
        elif progress < 0.9:
            true_defect_prob = 0.45
        else:
            true_defect_prob = 0.75   # the "subtle" stretch — model gets challenged

        for k in range(args.components_per_board):
            ref_des = f"{rng.choice(['C','R','U','Q','D'])}{rng.randint(1, 99)}"
            label = "TRUE_DEFECT" if rng.random() < true_defect_prob else "FALSE_CALL"
            defect_type = rng.choice(DEFECT_TYPES)

            img = _synth_roi(label, defect_type)
            board_dir = images_root / board_id
            board_dir.mkdir(exist_ok=True)
            img_path = board_dir / f"{ref_des}.jpg"
            img.save(img_path, quality=88)

            rows.append({
                "schema_version": "0.1",
                "board_id": board_id,
                "timestamp": ts,
                "vendor": "generic_csv",
                "line_id": line_id,
                "lot": lot,
                "ref_des": ref_des,
                "bbox_x1": 0, "bbox_y1": 0, "bbox_x2": 224, "bbox_y2": 224,
                "image_path": f"boards/{board_id}/{ref_des}.jpg",
                "height_map_path": "",
                "vendor_call": "DEFECT",   # all rows are AOI flags by definition
                "vendor_defect_type": defect_type,
                # Ground-truth label travels in the bundle as a side column so
                # the demo can simulate the operator click without manual labelling.
                "_demo_truth": label,
            })

        with (out / f"{board_id}.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    print(f"wrote {args.n_boards} boards × {args.components_per_board} components to {out}")


if __name__ == "__main__":
    main()
