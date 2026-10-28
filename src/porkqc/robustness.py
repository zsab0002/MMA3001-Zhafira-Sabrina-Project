"""Image perturbations that mimic real production-line disturbances.

Each function takes and returns a ``uint8`` BGR image of the same shape, so
they can be applied before any model. Severity parameters are chosen to be
physically plausible, not adversarial:

* ``brightness`` - lighting drift / camera exposure change;
* ``blur``       - conveyor motion or slightly out-of-focus camera;
* ``noise``      - sensor noise in low light;
* ``glare``      - specular reflection from the plastic film, which the
  dataset description highlights as a known error source.
"""

from __future__ import annotations

import cv2
import numpy as np


def brightness(img: np.ndarray, factor: float) -> np.ndarray:
    """Scale pixel intensities by ``factor`` (e.g. 0.6 = 40 % darker)."""
    return np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def blur(img: np.ndarray, ksize: int) -> np.ndarray:
    """Horizontal motion blur with a kernel of ``ksize`` pixels (1 = no blur)."""
    if ksize <= 1:
        return img.copy()
    kernel = np.zeros((ksize, ksize), np.float32)
    kernel[ksize // 2, :] = 1.0 / ksize  # conveyor moves horizontally
    return cv2.filter2D(img, -1, kernel)


def noise(img: np.ndarray, sigma: float, seed: int = 0) -> np.ndarray:
    """Add zero-mean Gaussian noise with standard deviation ``sigma`` (grey levels)."""
    rng = np.random.default_rng(seed)
    return np.clip(img.astype(np.float32) + rng.normal(0, sigma, img.shape), 0, 255).astype(np.uint8)


def glare(img: np.ndarray, strength: float, seed: int = 0) -> np.ndarray:
    """Overlay a soft elliptical highlight imitating film reflection.

    Args:
        img: Input image.
        strength: Peak added brightness as a fraction of 255 (0 = none).
        seed: Controls the highlight position, for repeatability.

    Returns:
        Perturbed image.
    """
    if strength <= 0:
        return img.copy()
    h, w = img.shape[:2]
    rng = np.random.default_rng(seed)
    cx, cy = rng.uniform(0.25, 0.75) * w, rng.uniform(0.25, 0.75) * h
    yy, xx = np.mgrid[0:h, 0:w]
    mask = np.exp(-(((xx - cx) / (0.18 * w)) ** 2 + ((yy - cy) / (0.08 * h)) ** 2))
    out = img.astype(np.float32) + strength * 255 * mask[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


#: Perturbations and severity levels used by ``scripts/06_sensitivity.py``.
#: The first level of each is the "clean" reference.
PERTURBATIONS: dict = {
    "brightness": (brightness, [1.0, 0.8, 0.6, 0.4, 1.3, 1.6]),
    "blur": (blur, [1, 5, 9, 15]),
    "noise": (noise, [0, 10, 20, 35]),
    "glare": (glare, [0.0, 0.3, 0.6, 0.9]),
}
