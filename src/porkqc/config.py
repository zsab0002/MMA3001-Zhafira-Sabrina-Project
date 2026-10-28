"""Project-wide constants and configuration loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

#: Root of the repository (two levels above this file: src/porkqc/config.py).
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

#: Class names in the order used by the Roboflow YOLOv8 export (``data.yaml``).
CLASS_NAMES: list[str] = [
    "loose-meat",
    "packaging-error",
    "twisted-meat",
    "unsealed",
    "wrinkle",
]

#: Image size produced by the Roboflow pre-processing (width, height) in pixels.
IMAGE_SIZE: tuple[int, int] = (720, 540)

#: Image file extensions accepted as input.
IMAGE_EXTENSIONS: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp")


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load the YAML configuration file.

    Args:
        path: Path to a YAML file. Defaults to ``configs/config.yaml``.

    Returns:
        The configuration as a nested dictionary. Relative paths under the
        ``paths`` key are converted to absolute ``Path`` objects relative to
        the project root, so scripts work from any working directory.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
    """
    path = Path(path) if path else PROJECT_ROOT / "configs" / "config.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg["paths"] = {k: (PROJECT_ROOT / v).resolve() for k, v in cfg["paths"].items()}
    return cfg
