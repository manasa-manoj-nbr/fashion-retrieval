"""Garment localisation via human-parsing segmentation.

This is the heart of the compositionality fix. Vanilla CLIP encodes the whole
image into one vector, so "red tie + white shirt" and "white tie + red shirt"
look almost identical to it. By segmenting the image into garment regions and
embedding each region separately, we can later bind an attribute (red) to a
specific garment (tie/upper) instead of to the image as a whole.

Model: ``mattmdjaga/segformer_b2_clothes`` (SegFormer trained on ATR human
parsing, 18 classes). It is coarse -- it exposes "Upper-clothes", "Pants",
"Skirt", "Dress", "Scarf", etc., but does NOT separate a tie from a shirt.
That is an accepted limitation: coarse regions already enable upper-vs-lower
colour binding, and the VQA re-rank stage (retriever/rerank.py) recovers the
fine-grained cases. This trade-off is documented in the write-up.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForSemanticSegmentation, SegformerImageProcessor

from common.config import CONFIG

# ATR label id -> our coarse region class (see config.garment_map keys).
# Ids come from the segformer_b2_clothes label set.
_ATR_TO_REGION: Dict[int, str] = {
    1: "hat",
    4: "upper",   # Upper-clothes
    5: "lower",   # Skirt
    6: "lower",   # Pants
    7: "dress",   # Dress
    9: "shoes",   # Left-shoe
    10: "shoes",  # Right-shoe
    16: "bag",    # Bag
    17: "scarf",  # Scarf
}


@dataclass
class Region:
    """A localised garment crop plus its coarse class and bbox."""
    region_class: str
    image: Image.Image           # tight RGB crop of the garment
    bbox: List[int]              # [x0, y0, x1, y1] in original pixels
    area_frac: float             # crop area / image area


class GarmentSegmenter:
    def __init__(self, model_name: str | None = None, device: str | None = None):
        self.model_name = model_name or CONFIG["models"]["segmenter"]
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = SegformerImageProcessor.from_pretrained(self.model_name)
        self.model = (
            AutoModelForSemanticSegmentation.from_pretrained(self.model_name)
            .to(self.device)
            .eval()
        )
        self.min_area = float(CONFIG["index"]["min_region_area_frac"])

    @torch.no_grad()
    def _label_map(self, image: Image.Image) -> np.ndarray:
        """Return a per-pixel ATR class-id map upsampled to the image size."""
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        logits = self.model(**inputs).logits  # (1, C, h, w)
        upsampled = torch.nn.functional.interpolate(
            logits,
            size=image.size[::-1],  # (H, W)
            mode="bilinear",
            align_corners=False,
        )
        return upsampled.argmax(dim=1)[0].cpu().numpy()

    def segment(self, image: Image.Image) -> List[Region]:
        """Split one image into garment regions (merging duplicate classes)."""
        image = image.convert("RGB")
        W, H = image.size
        img_area = float(W * H)
        label_map = self._label_map(image)

        # Merge same-region labels (e.g. left/right shoe -> "shoes") by union
        # of their bounding boxes.
        boxes: Dict[str, List[int]] = {}
        for atr_id, region_class in _ATR_TO_REGION.items():
            mask = label_map == atr_id
            if not mask.any():
                continue
            ys, xs = np.where(mask)
            x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
            if region_class in boxes:
                bx = boxes[region_class]
                boxes[region_class] = [
                    min(bx[0], x0), min(bx[1], y0), max(bx[2], x1), max(bx[3], y1)
                ]
            else:
                boxes[region_class] = [x0, y0, x1, y1]

        regions: List[Region] = []
        for region_class, (x0, y0, x1, y1) in boxes.items():
            area_frac = ((x1 - x0 + 1) * (y1 - y0 + 1)) / img_area
            if area_frac < self.min_area:
                continue
            crop = image.crop((x0, y0, x1 + 1, y1 + 1))
            regions.append(
                Region(region_class=region_class, image=crop,
                       bbox=[x0, y0, x1, y1], area_frac=area_frac)
            )
        return regions
