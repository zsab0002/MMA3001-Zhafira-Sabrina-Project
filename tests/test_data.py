"""Tests for porkqc.data: label parsing, statistics, leakage and grouped split."""
from pathlib import Path

import pytest

from porkqc.data import (Box, assign_sequences, dataset_summary, frame_number,
                         group_into_sequences, leakage_report, make_grouped_split,
                         read_label_file, tray_label)


def test_read_valid_label(tmp_path: Path):
    f = tmp_path / "a.txt"
    f.write_text("1 0.5 0.5 0.2 0.1\n\n3 0.1 0.9 0.05 0.05\n")
    boxes = read_label_file(f)
    assert len(boxes) == 2 and boxes[0].cls == 1
    assert boxes[0].area == pytest.approx(0.02)


def test_empty_label_is_pass(tmp_path: Path):
    f = tmp_path / "a.txt"
    f.write_text("")
    assert tray_label(read_label_file(f)) == 0


@pytest.mark.parametrize("line", ["1 0.5 0.5 0.2", "9 0.5 0.5 0.2 0.2", "1 1.5 0.5 0.2 0.2", "1 0.5 0.5 0 0.2"])
def test_invalid_labels_raise(tmp_path: Path, line: str):
    f = tmp_path / "a.txt"
    f.write_text(line)
    with pytest.raises(ValueError):
        read_label_file(f)


def test_missing_label_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        read_label_file(tmp_path / "nope.txt")


def test_box_to_xyxy():
    # Known answer: centred box of half width/height on a 720x540 image.
    assert Box(0, 0.5, 0.5, 0.5, 0.5).to_xyxy(720, 540) == pytest.approx((180, 135, 540, 405))


def test_frame_number():
    assert frame_number("frame_with_red_10049_jpg.rf.cd78.jpg") == 10049
    assert frame_number("random.jpg") is None


def test_group_into_sequences():
    assert group_into_sequences([1, 2, 3, 50, 52, 200], gap=5) == [[1, 2, 3], [50, 52], [200]]
    with pytest.raises(ValueError):
        group_into_sequences([1], gap=-1)


def test_assign_sequences_keeps_sequences_whole():
    seqs = [[i * 100 + k for k in range(5)] for i in range(20)]
    m = assign_sequences(seqs, (0.7, 0.15, 0.15), seed=1)
    for s in seqs:
        assert len({m[f] for f in s}) == 1  # never split across sets
    assert set(m.values()) == {"train", "valid", "test"}
    with pytest.raises(ValueError):
        assign_sequences(seqs, (0.5, 0.5, 0.5))


def test_assign_sequences_reproducible():
    seqs = [[i * 100] for i in range(30)]
    assert assign_sequences(seqs, (0.7, 0.15, 0.15), 3) == assign_sequences(seqs, (0.7, 0.15, 0.15), 3)


def test_summary_and_leakage(tiny_dataset: Path):
    s = dataset_summary(tiny_dataset)
    assert s["train"]["images"] == 10 and s["train"]["unique_frames"] == 5
    assert s["test"]["fail_trays"] == 1 and s["test"]["pass_trays"] == 1
    leak = leakage_report(tiny_dataset)
    assert leak["test"]["exact_duplicates"] == 1     # frame 900 is in train AND test
    assert leak["test"]["median_gap"] == 0.5         # gaps are 1 and 0


def test_grouped_split_removes_leakage(tiny_dataset: Path, tmp_path: Path):
    info = make_grouped_split(tiny_dataset, tmp_path / "g", gap=20, fractions=(0.4, 0.3, 0.3), seed=0)
    assert sum(v["images"] for v in info.values()) <= 13
    assert (tmp_path / "g" / "data_abs.yaml").exists()
    leak = leakage_report(tmp_path / "g")
    for split in ("valid", "test"):
        if leak[split]["frames"]:
            assert leak[split]["exact_duplicates"] == 0
            assert leak[split]["median_gap"] > 20
