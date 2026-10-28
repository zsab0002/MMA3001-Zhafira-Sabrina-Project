"""Saving results and drawing figures/annotated images."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from porkqc.config import CLASS_NAMES

#: One BGR colour per class, for consistent annotation across all figures.
CLASS_COLOURS: list[tuple[int, int, int]] = [
    (60, 180, 75), (0, 130, 245), (200, 50, 200), (40, 40, 230), (200, 160, 0)
]


def save_json(obj, path: str | Path) -> Path:
    """Write ``obj`` as pretty-printed JSON, creating parent folders.

    Args:
        obj: JSON-serialisable object (NumPy scalars are converted).
        path: Output path.

    Returns:
        The output path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def default(o):
        if isinstance(o, (np.integer, np.floating)):
            return o.item()
        if isinstance(o, Path):
            return str(o)
        raise TypeError(f"not serialisable: {type(o)}")

    path.write_text(json.dumps(obj, indent=2, default=default))
    return path


def draw_boxes(img: np.ndarray, boxes: list[tuple], header: str = "",
               header_colour: tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
    """Draw labelled boxes and an optional header on a copy of an image.

    Args:
        img: BGR image.
        boxes: Tuples ``(cls, conf_or_None, (x1, y1, x2, y2))`` in pixels.
            ``conf=None`` marks a ground-truth box (drawn dashed-style thinner).
        header: Text written at the top-left (e.g. ``"FAIL 0.87"``).
        header_colour: BGR colour of the header text.

    Returns:
        Annotated copy of the image.
    """
    out = img.copy()
    for cls, conf, (x1, y1, x2, y2) in boxes:
        colour = CLASS_COLOURS[cls % len(CLASS_COLOURS)]
        thick = 1 if conf is None else 2
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), colour, thick)
        label = CLASS_NAMES[cls] + ("" if conf is None else f" {conf:.2f}")
        cv2.putText(out, label, (int(x1), max(12, int(y1) - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, colour, 1, cv2.LINE_AA)
    if header:
        cv2.rectangle(out, (0, 0), (out.shape[1], 26), (0, 0, 0), -1)
        cv2.putText(out, header, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, header_colour, 1, cv2.LINE_AA)
    return out


def markdown_table(rows: list[dict], columns: list[str], floatfmt: str = ".3f") -> str:
    """Render a list of dicts as a Markdown table (for pasting into the report).

    Args:
        rows: Row dicts.
        columns: Keys to include, in order.
        floatfmt: Format spec for floats.

    Returns:
        Markdown table as a string.
    """
    def fmt(v):
        return format(v, floatfmt) if isinstance(v, float) else str(v)

    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    lines += ["| " + " | ".join(fmt(r.get(c, "")) for c in columns) + " |" for r in rows]
    return "\n".join(lines)
