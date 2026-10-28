"""Step 1 - Explore the dataset and measure train/test leakage.

Outputs (in results/data/):
    summary_original.json   per-split image, tray and box counts
    leakage_original.json   how close val/test frames are to training frames
    class_distribution.png  boxes per class and split
    box_sizes.png           defect box area distribution
    samples.png             example trays with ground-truth boxes

Usage:
    python scripts/01_explore_data.py
"""
import _setup  # noqa: F401

import cv2
import matplotlib.pyplot as plt
import numpy as np

from porkqc.config import CLASS_NAMES, load_config
from porkqc.data import (SPLITS, dataset_summary, label_path_for, leakage_report,
                         list_images, read_label_file)
from porkqc.reporting import draw_boxes, save_json


def main() -> None:
    cfg = load_config()
    raw, out = cfg["paths"]["raw_data"], cfg["paths"]["results"] / "data"
    if not (raw / "train" / "images").exists():
        raise SystemExit(f"Dataset not found in {raw}. Unzip the Roboflow export there (see README).")

    summary = dataset_summary(raw)
    save_json(summary, out / "summary_original.json")
    leak = leakage_report(raw)
    save_json(leak, out / "leakage_original.json")

    print("=== Dataset summary (original Roboflow split) ===")
    for split, s in summary.items():
        print(f"{split:6s} images={s['images']:5d} unique_frames={s['unique_frames']:5d} "
              f"FAIL={s['fail_trays']:5d} PASS={s['pass_trays']:5d} boxes={s['boxes_per_class']}")
    print("\n=== Leakage check: gap (frames) to nearest training frame ===")
    for split, r in leak.items():
        print(f"{split:6s} {r}")

    # Class distribution per split (log scale: classes differ by ~100x).
    fig, ax = plt.subplots(figsize=(7, 3.5))
    width = 0.27
    x = np.arange(len(CLASS_NAMES))
    for i, split in enumerate(SPLITS):
        counts = [summary[split]["boxes_per_class"][c] for c in CLASS_NAMES]
        ax.bar(x + (i - 1) * width, np.maximum(counts, 0.8), width, label=split)
    ax.set_xticks(x, CLASS_NAMES)
    ax.set_yscale("log")
    ax.set_ylabel("Annotated boxes (log scale)")
    ax.set_title("Defect class distribution - original split")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "class_distribution.png", dpi=150)

    # Box area as a fraction of the image: tells us whether small-object
    # detection (and hence input resolution) matters.
    areas = {c: [] for c in range(len(CLASS_NAMES))}
    for img in list_images(raw / "train" / "images"):
        for b in read_label_file(label_path_for(img)):
            areas[b.cls].append(b.area * 100)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.boxplot([areas[c] or [np.nan] for c in areas], labels=CLASS_NAMES)
    ax.set_yscale("log")
    ax.set_ylabel("Box area (% of image)")
    ax.set_title("Defect size distribution (training set)")
    fig.tight_layout()
    fig.savefig(out / "box_sizes.png", dpi=150)

    # One example image per class plus one PASS tray.
    examples, found = [], set()
    for img in list_images(raw / "train" / "images"):
        boxes = read_label_file(label_path_for(img))
        key = boxes[0].cls if boxes else "pass"
        if key not in found:
            found.add(key)
            examples.append((img, boxes, CLASS_NAMES[key] if boxes else "PASS (no defect)"))
        if len(found) == len(CLASS_NAMES) + 1:
            break
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5))
    for ax, (img, boxes, title) in zip(axes.ravel(), examples):
        im = cv2.imread(str(img))
        h, w = im.shape[:2]
        im = draw_boxes(im, [(b.cls, None, b.to_xyxy(w, h)) for b in boxes])
        ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        ax.set_title(title)
        ax.axis("off")
    for ax in axes.ravel()[len(examples):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out / "samples.png", dpi=120)
    print(f"\nFigures and JSON written to {out}")


if __name__ == "__main__":
    main()
