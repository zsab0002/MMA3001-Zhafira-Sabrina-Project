"""Tests for the baseline features, perturbations and profiling helpers."""
import numpy as np
import pytest

from porkqc.baseline import build_baseline, extract_features
from porkqc.profiling import scale_flops, time_function
from porkqc.robustness import PERTURBATIONS, blur, brightness, glare, noise

IMG = (np.random.default_rng(0).random((540, 720, 3)) * 255).astype(np.uint8)


def test_feature_vector_shape_and_determinism():
    f1, f2 = extract_features(IMG), extract_features(IMG)
    assert f1.ndim == 1 and np.array_equal(f1, f2)
    assert f1[:128].sum() == pytest.approx(1.0, abs=1e-4)  # colour histogram normalised


def test_feature_rejects_grayscale():
    with pytest.raises(ValueError):
        extract_features(IMG[..., 0])


def test_baseline_learns_separable_problem():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0, 1, (50, 5)), rng.normal(3, 1, (50, 5))])
    y = np.array([0] * 50 + [1] * 50)
    assert build_baseline(C=1.0).fit(X, y).score(X, y) > 0.95


@pytest.mark.parametrize("name", list(PERTURBATIONS))
def test_perturbations_keep_shape_and_clean_level_is_identity(name):
    fn, levels = PERTURBATIONS[name]
    out = fn(IMG, levels[0])
    assert out.shape == IMG.shape and out.dtype == np.uint8
    assert np.array_equal(out, IMG)  # first level is the clean reference


def test_perturbations_change_image():
    assert brightness(IMG, 0.5).mean() < IMG.mean()
    assert blur(IMG, 9).std() < IMG.std()
    assert not np.array_equal(noise(IMG, 20), IMG)
    assert glare(IMG, 0.8).mean() > IMG.mean()


def test_scale_flops_quadratic():
    assert scale_flops(8.0, 320) == pytest.approx(2.0)


def test_time_function_reports():
    r = time_function(lambda: sum(range(100)), n_repeat=5, n_warmup=1)
    assert r["median_ms"] >= 0 and r["fps"] > 0
