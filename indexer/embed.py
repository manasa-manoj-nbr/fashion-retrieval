"""Per-image representation builder.

For each image we produce:
  * one GLOBAL embedding  (whole image  -> scene / vibe / location)
  * K   REGION embeddings (garment crops -> attribute binding)
plus a dominant-colour tag per region. This is the multi-vector representation
that the retriever consumes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np
from PIL import Image

from indexer.colors import dominant_color
from indexer.encoder import FashionEncoder
from indexer.segmenter import GarmentSegmenter


@dataclass
class RegionRecord:
    region_id: str
    region_class: str
    embedding: np.ndarray
    color_name: str
    bbox: list
    area_frac: float


@dataclass
class ImageRecord:
    image_id: str
    path: str
    global_embedding: np.ndarray
    regions: List[RegionRecord] = field(default_factory=list)


class RepresentationBuilder:
    def __init__(self, encoder: FashionEncoder | None = None,
                segmenter: GarmentSegmenter | None = None):
        self.encoder = encoder or FashionEncoder()
        self.segmenter = segmenter or GarmentSegmenter()

    def build(self, image_id: str, path: str) -> ImageRecord:
        image = Image.open(path).convert("RGB")

        # Global (whole-scene) vector.
        global_emb = self.encoder.encode_images([image])[0]

        # Garment regions -> per-region vectors + colour.
        regions = self.segmenter.segment(image)
        region_records: List[RegionRecord] = []
        if regions:
            crops = [r.image for r in regions]
            crop_embs = self.encoder.encode_images(crops)
            for i, r in enumerate(regions):
                color_name, _ = dominant_color(r.image)
                region_records.append(
                    RegionRecord(
                        region_id=f"{image_id}::{r.region_class}::{i}",
                        region_class=r.region_class,
                        embedding=crop_embs[i],
                        color_name=color_name,
                        bbox=r.bbox,
                        area_frac=r.area_frac,
                    )
                )
        return ImageRecord(image_id=image_id, path=path,
                          global_embedding=global_emb, regions=region_records)
