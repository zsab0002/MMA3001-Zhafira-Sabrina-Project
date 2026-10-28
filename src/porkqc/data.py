"""Dataset reading, validation, statistics and the leakage-free split.

The Roboflow export stores one ``.txt`` label file per image in YOLO format::

    <class_id> <x_centre> <y_centre> <width> <height>

with all four box values normalised to [0, 1] by the image width/height.
An empty label file means the tray has no annotated defect (a PASS tray).

The images are frames of a production-line video, named e.g.
``frame_with_red_10049_jpg.rf.<hash>.jpg``. Because neighbouring frames are
almost identical, a random split puts near-duplicates in both training and
test sets ("data leakage"). This module measures that leakage and builds a
split that keeps whole video sequences together.
"""

from __future__ import annotations

import random
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from porkqc.config import CLASS_NAMES, IMAGE_EXTENSIONS

SPLITS: tuple[str, ...] = ("train", "valid", "test")

_FRAME_RE = re.compile(r"frame_with_red_(\d+)_jpg")


@dataclass(frozen=True)
class Box:
    """One ground-truth bounding box in normalised YOLO format.

    Attributes:
        cls: Integer class id (index into `porkqc.config.CLASS_NAMES`).
        xc: Box centre x, fraction of image width, in [0, 1].
        yc: Box centre y, fraction of image height, in [0, 1].
        w: Box width, fraction of image width, in (0, 1].
        h: Box height, fraction of image height, in (0, 1].
    """

    cls: int
    xc: float
    yc: float
    w: float
    h: float

    @property
    def area(self) -> float:
        """Box area as a fraction of the image area (dimensionless, 0-1)."""
        return self.w * self.h

    def to_xyxy(self, width: int, height: int) -> tuple[float, float, float, float]:
        """Convert to pixel corner coordinates ``(x1, y1, x2, y2)``.

        Args:
            width: Image width in pixels.
            height: Image height in pixels.

        Returns:
            Tuple of pixel coordinates (top-left and bottom-right corners).
        """
        x1 = (self.xc - self.w / 2) * width
        y1 = (self.yc - self.h / 2) * height
        return (x1, y1, x1 + self.w * width, y1 + self.h * height)


def read_label_file(path: str | Path, n_classes: int = len(CLASS_NAMES)) -> list[Box]:
    """Read and validate one YOLO label file.

    Args:
        path: Path to the ``.txt`` label file.
        n_classes: Number of valid classes; ids must be in ``[0, n_classes)``.

    Returns:
        List of `Box` objects (empty list for a defect-free tray).

    Raises:
        FileNotFoundError: If the label file is missing.
        ValueError: If a line is malformed, a class id is out of range or a
            coordinate lies outside [0, 1]. Failing loudly here stops corrupt
            annotations from silently entering training or validation.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Label file missing: {path}")
    boxes: list[Box] = []
    for line_no, line in enumerate(path.read_text().splitlines(), start=1):
        parts = line.split()
        if not parts:
            continue  # blank lines are harmless
        if len(parts) != 5:
            raise ValueError(f"{path.name}:{line_no}: expected 5 values, got {len(parts)}")
        cls = int(float(parts[0]))
        xc, yc, w, h = (float(v) for v in parts[1:])
        if not 0 <= cls < n_classes:
            raise ValueError(f"{path.name}:{line_no}: class id {cls} out of range")
        # Small tolerance: Roboflow occasionally writes 1.0000000002.
        if not all(-1e-6 <= v <= 1 + 1e-6 for v in (xc, yc, w, h)) or w <= 0 or h <= 0:
            raise ValueError(f"{path.name}:{line_no}: coordinates outside [0, 1]")
        boxes.append(Box(cls, xc, yc, w, h))
    return boxes


def label_path_for(image_path: str | Path) -> Path:
    """Return the label path matching an image (``.../images/x.jpg`` -> ``.../labels/x.txt``)."""
    image_path = Path(image_path)
    return image_path.parent.parent / "labels" / (image_path.stem + ".txt")


def list_images(folder: str | Path) -> list[Path]:
    """List image files in a folder, sorted for reproducibility.

    Args:
        folder: Directory to search (non-recursive).

    Returns:
        Sorted list of image paths with an extension in ``IMAGE_EXTENSIONS``.
    """
    folder = Path(folder)
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)


def frame_number(image_path: str | Path) -> int | None:
    """Extract the video frame number from a Roboflow file name.

    Args:
        image_path: Path or file name such as ``frame_with_red_10049_jpg.rf.abc.jpg``.

    Returns:
        The frame number, or ``None`` if the name does not follow the pattern.
    """
    m = _FRAME_RE.search(Path(image_path).name)
    return int(m.group(1)) if m else None


def tray_label(boxes: list[Box]) -> int:
    """Ground-truth tray decision: 1 = FAIL (any defect present), 0 = PASS."""
    return int(len(boxes) > 0)


def dataset_summary(root: str | Path) -> dict:
    """Compute per-split statistics for a YOLO dataset.

    Args:
        root: Folder containing ``train/``, ``valid/`` and ``test/`` sub-folders.

    Returns:
        Nested dict ``{split: {...}}`` with image count, number of FAIL/PASS
        trays, box count per class and box-area statistics (fraction of image).
    """
    root = Path(root)
    summary: dict = {}
    for split in SPLITS:
        img_dir = root / split / "images"
        if not img_dir.exists():
            continue
        images = list_images(img_dir)
        cls_counts: Counter = Counter()
        areas: list[float] = []
        n_fail = 0
        for img in images:
            boxes = read_label_file(label_path_for(img))
            n_fail += tray_label(boxes)
            cls_counts.update(b.cls for b in boxes)
            areas.extend(b.area for b in boxes)
        summary[split] = {
            "images": len(images),
            "unique_frames": len({frame_number(i) for i in images}),
            "fail_trays": n_fail,
            "pass_trays": len(images) - n_fail,
            "boxes_per_class": {CLASS_NAMES[c]: cls_counts.get(c, 0) for c in range(len(CLASS_NAMES))},
            "box_area_median": float(np.median(areas)) if areas else 0.0,
            "box_area_p05": float(np.percentile(areas, 5)) if areas else 0.0,
            "box_area_p95": float(np.percentile(areas, 95)) if areas else 0.0,
        }
    return summary


def leakage_report(root: str | Path, near: int = 5) -> dict:
    """Quantify how close each val/test frame is to the nearest training frame.

    Consecutive video frames are visually near-identical, so a small gap means
    the "unseen" image was effectively seen during training.

    Args:
        root: Dataset folder with ``train/``, ``valid/`` and ``test/``.
        near: Gap (in frames) at or below which a frame counts as leaked.

    Returns:
        Dict per evaluation split with the number of exact duplicate frames,
        the median gap and the fraction of frames within ``near`` frames.
    """
    root = Path(root)
    train = np.array(sorted({frame_number(p) for p in list_images(root / "train" / "images")}))
    report = {}
    for split in ("valid", "test"):
        frames = [frame_number(p) for p in list_images(root / split / "images")]
        gaps = np.array([np.min(np.abs(train - f)) for f in frames])
        report[split] = {
            "frames": len(frames),
            "exact_duplicates": int(np.sum(gaps == 0)),
            "median_gap": float(np.median(gaps)),
            f"fraction_within_{near}": float(np.mean(gaps <= near)),
        }
    return report


def group_into_sequences(frames: list[int], gap: int) -> list[list[int]]:
    """Group sorted frame numbers into contiguous video sequences.

    A new sequence starts whenever two consecutive (sorted) frame numbers
    differ by more than ``gap``.

    Args:
        frames: Frame numbers (any order, duplicates allowed).
        gap: Maximum frame difference that keeps two frames in one sequence.

    Returns:
        List of sequences, each a sorted list of unique frame numbers.

    Raises:
        ValueError: If ``gap`` is negative.
    """
    if gap < 0:
        raise ValueError("gap must be >= 0")
    uniq = sorted(set(frames))
    sequences: list[list[int]] = []
    for f in uniq:
        if sequences and f - sequences[-1][-1] <= gap:
            sequences[-1].append(f)
        else:
            sequences.append([f])
    return sequences


def assign_sequences(
    sequences: list[list[int]], fractions: tuple[float, float, float], seed: int = 42
) -> dict[int, str]:
    """Randomly assign whole sequences to train/valid/test.

    Sequences are shuffled and then filled greedily so each split receives
    approximately the requested share of *frames*.

    Args:
        sequences: Output of `group_into_sequences`.
        fractions: Target (train, valid, test) fractions; must sum to 1.
        seed: Random seed, for a reproducible split.

    Returns:
        Mapping ``frame_number -> split name``.

    Raises:
        ValueError: If the fractions do not sum to 1.
    """
    if abs(sum(fractions) - 1) > 1e-6:
        raise ValueError("fractions must sum to 1")
    order = list(range(len(sequences)))
    random.Random(seed).shuffle(order)
    total = sum(len(s) for s in sequences)
    targets = {name: frac * total for name, frac in zip(SPLITS, fractions)}
    filled = defaultdict(int)
    mapping: dict[int, str] = {}
    for idx in order:
        seq = sequences[idx]
        # Put the sequence in the split that is furthest below its target share.
        split = max(SPLITS, key=lambda s: (targets[s] - filled[s]) / max(targets[s], 1e-9))
        filled[split] += len(seq)
        for f in seq:
            mapping[f] = split
    return mapping


def make_grouped_split(
    src_root: str | Path,
    dst_root: str | Path,
    gap: int = 20,
    fractions: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
) -> dict:
    """Create a leakage-free copy of the dataset, split by video sequence.

    All images from all original splits are pooled, grouped into sequences and
    re-assigned. Roboflow's exposure-augmented copies of a frame always follow
    their frame, so augmented twins cannot straddle the train/test boundary.
    For ``valid``/``test`` only ONE copy per frame is kept, so evaluation
    images are unique and un-augmented in spirit.

    Args:
        src_root: Original export folder (contains ``train/valid/test``).
        dst_root: Output folder; it is deleted and recreated.
        gap: Sequence gap, see `group_into_sequences`.
        fractions: Target split fractions.
        seed: Random seed.

    Returns:
        Dict with the number of images and sequences per split.
    """
    src_root, dst_root = Path(src_root), Path(dst_root)
    all_images = [p for s in SPLITS for p in list_images(src_root / s / "images")]
    frames = [frame_number(p) for p in all_images]
    if any(f is None for f in frames):
        raise ValueError("Some image names do not contain a frame number")
    sequences = group_into_sequences(frames, gap)
    mapping = assign_sequences(sequences, fractions, seed)

    if dst_root.exists():
        shutil.rmtree(dst_root)
    seen_eval_frames: set[int] = set()
    counts: Counter = Counter()
    for img, f in sorted(zip(all_images, frames), key=lambda t: t[0].name):
        split = mapping[f]
        if split != "train":
            if f in seen_eval_frames:
                continue  # keep one copy per frame in val/test
            seen_eval_frames.add(f)
        for kind, src in (("images", img), ("labels", label_path_for(img))):
            out = dst_root / split / kind
            out.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, out / src.name)
        counts[split] += 1

    write_data_yaml(dst_root)
    seq_counts = Counter(mapping[s[0]] for s in sequences)
    return {s: {"images": counts[s], "sequences": seq_counts[s]} for s in SPLITS}


def write_data_yaml(root: str | Path) -> Path:
    """Write an Ultralytics ``data.yaml`` with absolute paths.

    Absolute paths avoid the common Ultralytics error where relative paths are
    resolved against its own ``datasets_dir`` setting instead of the project.

    Args:
        root: Dataset root containing ``train/valid/test``.

    Returns:
        Path to the written YAML file.
    """
    root = Path(root).resolve()
    content = {
        "path": str(root),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(CLASS_NAMES),
        "names": CLASS_NAMES,
    }
    out = root / "data_abs.yaml"
    out.write_text(yaml.safe_dump(content, sort_keys=False))
    return out
