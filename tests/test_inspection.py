"""Tests for the PASS/FAIL decision layer and input validation."""
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from porkqc.inspection import Detection, inspect_tray, severity_from_area, tray_scores, validate_image

BANDS = {"minor": 0.01, "major": 0.05}


def test_no_detections_pass():
    r = inspect_tray("a.jpg", [], (720, 540), 0.25)
    assert r.decision == "PASS" and r.severity == "none" and r.fail_score == 0


def test_low_confidence_ignored_but_scored():
    r = inspect_tray("a.jpg", [Detection(1, 0.1, (0, 0, 10, 10))], (720, 540), 0.25)
    assert r.decision == "PASS" and r.fail_score == pytest.approx(0.1)


def test_fail_with_counts_and_area():
    dets = [Detection(1, 0.9, (0, 0, 72, 54)), Detection(3, 0.5, (100, 100, 172, 154))]
    r = inspect_tray("a.jpg", dets, (720, 540), 0.25, BANDS)
    assert r.decision == "FAIL"
    assert r.defects == {"packaging-error": 1, "unsealed": 1}
    assert r.defect_area_fraction == pytest.approx(0.02)  # 2 boxes of 1 % each
    assert r.severity == "major"


def test_threshold_validation():
    with pytest.raises(ValueError):
        inspect_tray("a.jpg", [], (720, 540), 1.5)


@pytest.mark.parametrize("frac,expected", [(0, "none"), (0.005, "minor"), (0.03, "major"), (0.2, "critical")])
def test_severity_bands(frac, expected):
    assert severity_from_area(frac, BANDS) == expected


def test_tray_scores():
    preds = {"a": [Detection(0, 0.3, (0, 0, 1, 1)), Detection(1, 0.7, (0, 0, 1, 1))]}
    assert tray_scores(preds, ["a", "b"]).tolist() == [0.7, 0.0]


def test_validate_image(tmp_path: Path):
    good = tmp_path / "g.jpg"
    Image.fromarray(np.zeros((540, 720, 3), np.uint8)).save(good)
    assert validate_image(good, (720, 540)) == (True, "ok")
    ok, msg = validate_image(good, (640, 480))
    assert ok and "differs" in msg
    bad = tmp_path / "b.jpg"
    bad.write_bytes(b"not an image")
    assert validate_image(bad)[0] is False
    assert validate_image(tmp_path / "x.gif")[0] is False
    assert validate_image(tmp_path / "missing.jpg") == (False, "file not found")
