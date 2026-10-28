"""Classical computer-vision baseline for the PASS/FAIL decision.

Purpose: answer the question "is a deep object detector actually needed?".
The baseline uses hand-crafted features that a pre-deep-learning inspection
system would use, and a linear classifier:

* HSV colour histogram  - captures meat/packaging colour distribution;
* HOG (histogram of oriented gradients) - captures edges such as wrinkles,
  folded rashers and open seals.

It predicts only the tray decision (it cannot localise defects), which is
itself a limitation worth reporting.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from skimage.feature import hog
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

#: Working resolution for feature extraction (width, height). Small on purpose:
#: HOG cost scales with pixel count and the baseline must stay cheap.
FEATURE_SIZE: tuple[int, int] = (160, 120)


def extract_features(image_bgr: np.ndarray) -> np.ndarray:
    """Compute the hand-crafted feature vector for one image.

    Args:
        image_bgr: Image as a ``uint8`` array (H, W, 3) in OpenCV BGR order.

    Returns:
        1-D float32 feature vector (colour histogram followed by HOG).

    Raises:
        ValueError: If the image is not a 3-channel array.
    """
    if image_bgr is None or image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("expected an (H, W, 3) BGR image")
    small = cv2.resize(image_bgr, FEATURE_SIZE, interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    # 8 hue x 4 saturation x 4 value bins = 128 colour features, normalised to sum 1.
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 4, 4], [0, 180, 0, 256, 0, 256]).ravel()
    hist /= hist.sum() + 1e-9
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    hog_vec = hog(gray, orientations=9, pixels_per_cell=(16, 16),
                  cells_per_block=(2, 2), feature_vector=True)
    return np.concatenate([hist, hog_vec]).astype(np.float32)


def load_features(image_paths: list[Path]) -> np.ndarray:
    """Extract features for a list of image files.

    Args:
        image_paths: Image paths.

    Returns:
        Array (N, D) of features.

    Raises:
        IOError: If an image cannot be read.
    """
    feats = []
    for p in image_paths:
        img = cv2.imread(str(p))
        if img is None:
            raise IOError(f"cannot read {p}")
        feats.append(extract_features(img))
    return np.vstack(feats)


def build_baseline(C: float = 0.1, seed: int = 42) -> Pipeline:
    """Create the (untrained) baseline classifier.

    Standardisation is needed because HOG and histogram features have very
    different scales. ``class_weight='balanced'`` compensates for ~80 % of
    trays being FAIL. A small ``C`` (strong L2 regularisation) is used because
    there are thousands of features but only ~1,000 distinct frames.

    Args:
        C: Inverse regularisation strength.
        seed: Random seed.

    Returns:
        A scikit-learn pipeline.
    """
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=C, class_weight="balanced", max_iter=2000, random_state=seed),
    )
