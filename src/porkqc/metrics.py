"""Metrics for tray-level (image-level) decisions and box overlap.

Object-detection metrics (mAP) are computed by Ultralytics. This module covers
what the *factory* cares about: of the trays that should be rejected, how many
did we reject (recall), and how many good trays did we waste (false alarms).
"""

from __future__ import annotations

import numpy as np


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise Intersection-over-Union between two sets of boxes.

    Args:
        a: Array of shape (N, 4) with ``x1, y1, x2, y2`` corners.
        b: Array of shape (M, 4) with ``x1, y1, x2, y2`` corners.

    Returns:
        Array of shape (N, M) with IoU values in [0, 1].
    """
    a = np.asarray(a, dtype=float).reshape(-1, 4)
    b = np.asarray(b, dtype=float).reshape(-1, 4)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    # Guard against zero-area boxes (union = 0) to keep the result finite.
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def binary_metrics(y_true, y_pred) -> dict[str, float]:
    """Confusion matrix and summary metrics for PASS(0)/FAIL(1) decisions.

    "Positive" means FAIL (a defective tray), because missing a defect is the
    costly error for the manufacturer.

    Args:
        y_true: Ground-truth labels (0/1), length N.
        y_pred: Predicted labels (0/1), length N.

    Returns:
        Dict with ``tp, fp, tn, fn``, ``accuracy``, ``precision``, ``recall``
        (defect detection rate), ``specificity`` (good trays passed),
        ``f1`` and ``balanced_accuracy``. Undefined ratios are reported as 0.

    Raises:
        ValueError: If lengths differ or labels are not 0/1.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same length")
    if not set(np.unique(np.concatenate([y_true, y_pred]))) <= {0, 1}:
        raise ValueError("labels must be 0 or 1")

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    def ratio(n: float, d: float) -> float:
        return float(n / d) if d else 0.0

    precision = ratio(tp, tp + fp)
    recall = ratio(tp, tp + fn)
    specificity = ratio(tn, tn + fp)
    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "accuracy": ratio(tp + tn, len(y_true)),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": ratio(2 * precision * recall, precision + recall),
        "balanced_accuracy": (recall + specificity) / 2,
    }


def threshold_sweep(y_true, scores, thresholds) -> list[dict[str, float]]:
    """Evaluate tray decisions over a range of score thresholds.

    Args:
        y_true: Ground-truth labels (0/1).
        scores: Per-image FAIL score, e.g. the highest defect confidence
            (0 if nothing was detected).
        thresholds: Iterable of thresholds; FAIL if ``score >= threshold``.

    Returns:
        One metrics dict (see `binary_metrics`) per threshold, each with a
        ``threshold`` key added.
    """
    scores = np.asarray(scores, dtype=float)
    rows = []
    for t in thresholds:
        m = binary_metrics(y_true, (scores >= t).astype(int))
        m["threshold"] = float(t)
        rows.append(m)
    return rows


def bootstrap_ci(y_true, y_pred, metric: str = "recall", n: int = 1000,
                 alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    """Bootstrap confidence interval for one tray-level metric.

    The test sets are small (tens to hundreds of trays), so a point estimate
    alone overstates certainty. Resampling trays with replacement shows how
    much the metric could move with a different sample.

    Args:
        y_true: Ground-truth labels (0/1).
        y_pred: Predicted labels (0/1).
        metric: Key of `binary_metrics` to bootstrap.
        n: Number of bootstrap resamples.
        alpha: 1 - confidence level (0.05 -> 95 % interval).
        seed: Random seed.

    Returns:
        ``(lower, upper)`` percentile interval.
    """
    rng = np.random.default_rng(seed)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(y_true), len(y_true))
        vals.append(binary_metrics(y_true[idx], y_pred[idx])[metric])
    return float(np.percentile(vals, 100 * alpha / 2)), float(np.percentile(vals, 100 * (1 - alpha / 2)))
