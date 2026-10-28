"""Turn raw detections into a tray inspection decision and report.

This is the engineering "decision layer" that sits on top of the detector.
It is deliberately simple and transparent so a line operator can understand
why a tray was rejected:

* a tray FAILS if any defect is detected with confidence >= ``conf_threshold``;
* severity is graded from the fraction of the image covered by defect boxes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

from porkqc.config import CLASS_NAMES, IMAGE_EXTENSIONS


@dataclass
class Detection:
    """One predicted defect.

    Attributes:
        cls: Class id (index into ``CLASS_NAMES``).
        conf: Detector confidence in [0, 1].
        xyxy: Box corners ``(x1, y1, x2, y2)`` in pixels.
    """

    cls: int
    conf: float
    xyxy: tuple[float, float, float, float]

    @property
    def name(self) -> str:
        """Human-readable class name."""
        return CLASS_NAMES[self.cls]

    def area_px(self) -> float:
        """Box area in square pixels."""
        x1, y1, x2, y2 = self.xyxy
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


@dataclass
class TrayReport:
    """Inspection result for one tray image.

    Attributes:
        image: File name of the image.
        decision: ``"PASS"``, ``"FAIL"`` or ``"INVALID"`` (unreadable input).
        fail_score: Highest defect confidence (0 if none) - the quantity
            compared against the threshold.
        severity: ``"none"``, ``"minor"``, ``"major"`` or ``"critical"``.
        defect_area_fraction: Total defect box area / image area (0-1, boxes
            may overlap so this is an upper bound on the true affected area).
        defects: Per-class count of accepted detections.
        detections: Accepted detections (as dicts).
        message: Explanation or error message.
    """

    image: str
    decision: str
    fail_score: float = 0.0
    severity: str = "none"
    defect_area_fraction: float = 0.0
    defects: dict = field(default_factory=dict)
    detections: list = field(default_factory=list)
    message: str = ""

    def to_dict(self) -> dict:
        """Return the report as a JSON-serialisable dict."""
        return asdict(self)


def validate_image(path: str | Path, expected_size: tuple[int, int] | None = None) -> tuple[bool, str]:
    """Check that an input file is a readable RGB image.

    Args:
        path: Image path.
        expected_size: Optional ``(width, height)``. A different size is
            accepted (YOLO letterboxes internally) but flagged in the message,
            because it is outside the validated input domain.

    Returns:
        ``(ok, message)``. ``ok`` is False for missing, unsupported or corrupt
        files; such images are reported as ``INVALID`` rather than crashing
        the whole batch.
    """
    path = Path(path)
    if not path.exists():
        return False, "file not found"
    if path.suffix.lower() not in IMAGE_EXTENSIONS:
        return False, f"unsupported extension {path.suffix}"
    try:
        with Image.open(path) as im:
            im.verify()  # detects truncated/corrupt files
        with Image.open(path) as im:
            size, mode = im.size, im.mode
    except (UnidentifiedImageError, OSError) as exc:
        return False, f"unreadable image ({exc.__class__.__name__})"
    msg = "ok"
    if mode not in ("RGB", "L", "RGBA"):
        msg = f"unusual colour mode {mode}"
    if expected_size and size != tuple(expected_size):
        msg = f"size {size} differs from validated {tuple(expected_size)}; results less certain"
    return True, msg


def severity_from_area(fraction: float, bands: dict[str, float]) -> str:
    """Map a defect area fraction to a severity grade.

    Args:
        fraction: Total defect area / image area.
        bands: ``{"minor": a, "major": b}`` with ``a < b``. Below ``a`` the
            defect is minor, between is major, above ``b`` critical.

    Returns:
        ``"none"`` if fraction is 0, else ``"minor"``, ``"major"`` or ``"critical"``.
    """
    if fraction <= 0:
        return "none"
    if fraction < bands["minor"]:
        return "minor"
    if fraction < bands["major"]:
        return "major"
    return "critical"


def inspect_tray(
    image_name: str,
    detections: list[Detection],
    image_size: tuple[int, int],
    conf_threshold: float = 0.25,
    severity_bands: dict[str, float] | None = None,
) -> TrayReport:
    """Make the PASS/FAIL decision for one tray.

    Args:
        image_name: Name used in the report.
        detections: All detector outputs for this image (any confidence).
        image_size: ``(width, height)`` in pixels, for the area fraction.
        conf_threshold: Minimum confidence for a detection to count.
        severity_bands: See `severity_from_area`. Defaults to 1 % / 5 %.

    Returns:
        A `TrayReport`.

    Raises:
        ValueError: If ``conf_threshold`` is outside [0, 1].
    """
    if not 0 <= conf_threshold <= 1:
        raise ValueError("conf_threshold must be in [0, 1]")
    bands = severity_bands or {"minor": 0.01, "major": 0.05}
    accepted = [d for d in detections if d.conf >= conf_threshold]
    fail_score = max((d.conf for d in detections), default=0.0)
    area = sum(d.area_px() for d in accepted) / float(image_size[0] * image_size[1])
    counts = {CLASS_NAMES[c]: sum(d.cls == c for d in accepted) for c in sorted({d.cls for d in accepted})}
    decision = "FAIL" if accepted else "PASS"
    msg = (
        "defects: " + ", ".join(f"{n} x{k}" for n, k in counts.items())
        if accepted else f"no defect above confidence {conf_threshold:.2f}"
    )
    return TrayReport(
        image=image_name,
        decision=decision,
        fail_score=round(float(fail_score), 4),
        severity=severity_from_area(area, bands),
        defect_area_fraction=round(float(min(area, 1.0)), 5),
        defects=counts,
        detections=[{"class": d.name, "conf": round(d.conf, 4),
                     "xyxy": [round(v, 1) for v in d.xyxy]} for d in accepted],
        message=msg,
    )


def tray_scores(predictions: dict[str, list[Detection]], names: list[str]) -> np.ndarray:
    """Per-image FAIL score (max confidence, 0 when nothing detected).

    Args:
        predictions: ``{image_name: [Detection, ...]}``.
        names: Image names in the order wanted.

    Returns:
        Array of scores aligned with ``names``.
    """
    return np.array([max((d.conf for d in predictions.get(n, [])), default=0.0) for n in names])
