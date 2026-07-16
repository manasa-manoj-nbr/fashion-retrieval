"""Cheap dominant-colour descriptor for a garment crop.

CLIP embeddings capture colour, but a small explicit colour signal is robust and
interpretable, and lets the retriever apply a light colour prior when a query
names a colour. We take the median RGB of the central patch (avoids background
bleed at the crop edges) and snap it to the nearest named colour anchor.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from PIL import Image

from common.config import CONFIG

_COLOR_ANCHORS: Dict[str, np.ndarray] = {
    name: np.array(rgb, dtype="float32") for name, rgb in CONFIG["colors"].items()
}


def dominant_color(image: Image.Image) -> Tuple[str, list]:
    """Return (nearest_named_colour, median_rgb) for a crop."""
    im = image.convert("RGB")
    arr = np.asarray(im, dtype="float32")
    h, w, _ = arr.shape
    # Central 60% patch to reduce background contamination at crop borders.
    y0, y1 = int(0.2 * h), int(0.8 * h) or h
    x0, x1 = int(0.2 * w), int(0.8 * w) or w
    patch = arr[y0:y1, x0:x1].reshape(-1, 3)
    if patch.size == 0:
        patch = arr.reshape(-1, 3)
    med = np.median(patch, axis=0)
    # Nearest anchor by Euclidean distance in RGB.
    names = list(_COLOR_ANCHORS.keys())
    dists = [float(np.linalg.norm(med - _COLOR_ANCHORS[n])) for n in names]
    return names[int(np.argmin(dists))], med.round().astype(int).tolist()
