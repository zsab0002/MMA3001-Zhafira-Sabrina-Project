"""Shared pytest fixtures: tiny synthetic YOLO datasets built in a temp folder,
so tests run in seconds and do not need the real 240 MB dataset."""
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _write(root: Path, split: str, frame: int, copy: str, labels: str) -> None:
    for kind in ("images", "labels"):
        (root / split / kind).mkdir(parents=True, exist_ok=True)
    name = f"frame_with_red_{frame}_jpg.rf.{copy}"
    Image.fromarray(np.full((54, 72, 3), 120, np.uint8)).save(root / split / "images" / f"{name}.jpg")
    (root / split / "labels" / f"{name}.txt").write_text(labels)


@pytest.fixture
def tiny_dataset(tmp_path: Path) -> Path:
    """Dataset with deliberate leakage: test frame 101 is adjacent to train frame 100."""
    root = tmp_path / "raw"
    for f in (100, 102, 500, 502, 900):
        for c in ("a", "b"):  # two augmented copies per training frame
            _write(root, "train", f, c, "1 0.5 0.5 0.2 0.2\n" if f % 200 else "")
    _write(root, "valid", 503, "a", "3 0.4 0.4 0.1 0.1\n")
    _write(root, "test", 101, "a", "")
    _write(root, "test", 900, "a", "0 0.5 0.5 0.3 0.3\n")
    return root
