"""Tests for porkqc.metrics, checked against hand-calculated answers."""
import numpy as np
import pytest

from porkqc.metrics import binary_metrics, bootstrap_ci, iou_xyxy, threshold_sweep


def test_iou_known_values():
    a = np.array([[0, 0, 10, 10]])
    b = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]])
    # identical -> 1; half overlap -> 50/150 = 1/3; disjoint -> 0
    assert iou_xyxy(a, b) == pytest.approx(np.array([[1.0, 1 / 3, 0.0]]))


def test_iou_zero_area_is_finite():
    assert np.isfinite(iou_xyxy([[0, 0, 0, 0]], [[0, 0, 0, 0]])).all()


def test_binary_metrics_hand_example():
    y = [1, 1, 1, 0, 0]
    p = [1, 1, 0, 1, 0]
    m = binary_metrics(y, p)
    assert (m["tp"], m["fn"], m["fp"], m["tn"]) == (2, 1, 1, 1)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["specificity"] == pytest.approx(0.5)
    assert m["balanced_accuracy"] == pytest.approx((2 / 3 + 0.5) / 2)


def test_always_fail_baseline_has_half_balanced_accuracy():
    y = np.array([1] * 80 + [0] * 20)
    m = binary_metrics(y, np.ones_like(y))
    assert m["accuracy"] == pytest.approx(0.8) and m["balanced_accuracy"] == pytest.approx(0.5)


def test_binary_metrics_input_checks():
    with pytest.raises(ValueError):
        binary_metrics([0, 1], [1])
    with pytest.raises(ValueError):
        binary_metrics([0, 2], [0, 1])


def test_threshold_sweep_monotonic_recall():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    s = np.clip(y * 0.5 + rng.random(200) * 0.6, 0, 1)
    rec = [r["recall"] for r in threshold_sweep(y, s, np.linspace(0, 1, 21))]
    assert all(a >= b for a, b in zip(rec, rec[1:]))  # raising threshold never increases recall
    assert rec[0] == 1.0


def test_bootstrap_ci_contains_point_estimate():
    y = np.array([1] * 50 + [0] * 50)
    p = y.copy()
    p[:10] = 0
    lo, hi = bootstrap_ci(y, p, "recall", n=300)
    assert lo <= 0.8 <= hi
