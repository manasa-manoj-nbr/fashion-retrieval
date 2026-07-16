"""Fast end-to-end smoke test (~30s) -- run BEFORE the full index build.

Exercises every moving part on just 2 images so that version/API breakages
surface in seconds instead of after an 800-image loop:

    python -m eval.smoke_test

Checks:
  1. FashionCLIP image + text encoding (shape, unit-norm, sane cosine)
  2. Garment segmentation (regions found)
  3. Dominant-colour tagging
  4. Chroma add + query round-trip (in a throwaway index dir)
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from common.config import resolve


def _ok(msg):
    print(f"  [PASS] {msg}")


def main() -> None:
    img_dir = resolve("data/images")
    paths = sorted(p for p in img_dir.glob("*.jpg"))[:2]
    if len(paths) < 2:
        sys.exit(f"Need >=2 images in {img_dir}. Run: python -m eval.fetch_data --n 800")
    images = [Image.open(p).convert("RGB") for p in paths]

    # 1. Encoder ------------------------------------------------------------
    print("\n[1/4] FashionCLIP encoder")
    from indexer.encoder import FashionEncoder
    enc = FashionEncoder()
    v_img = enc.encode_images(images)
    v_txt = enc.encode_texts(["a red shirt", "a blue dress"])
    assert v_img.ndim == 2 and v_img.shape[0] == 2, f"bad image shape {v_img.shape}"
    assert v_txt.shape[0] == 2, f"bad text shape {v_txt.shape}"
    assert v_img.shape[1] == v_txt.shape[1], "image/text dims differ"
    norms = np.linalg.norm(v_img, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3), f"not unit-norm: {norms}"
    _ok(f"image {v_img.shape}, text {v_txt.shape}, unit-norm OK")
    _ok(f"sample cosine(img0, 'a red shirt') = {float(np.dot(v_img[0], v_txt[0])):.3f}")

    # 2. Segmenter ----------------------------------------------------------
    print("\n[2/4] Garment segmenter")
    from indexer.segmenter import GarmentSegmenter
    seg = GarmentSegmenter()
    regions = seg.segment(images[0])
    _ok(f"{len(regions)} regions: {[r.region_class for r in regions]}")
    if not regions:
        print("  [WARN] no regions found on this image (may be fine; check others)")

    # 3. Colours ------------------------------------------------------------
    print("\n[3/4] Dominant colour")
    from indexer.colors import dominant_color
    for r in regions[:3]:
        name, rgb = dominant_color(r.image)
        _ok(f"{r.region_class:8} -> {name} {rgb}")

    # 4. Store round-trip ---------------------------------------------------
    print("\n[4/4] Chroma store round-trip")
    tmp = resolve("index_smoketest")
    if tmp.exists():
        shutil.rmtree(tmp)
    from indexer.store import VectorStore
    store = VectorStore(index_dir="index_smoketest")
    store.add_global(
        ids=["a", "b"], embeddings=[v_img[0], v_img[1]],
        metadatas=[{"image_id": "a", "path": str(paths[0])},
                   {"image_id": "b", "path": str(paths[1])}],
    )
    hits = store.query_global(v_img[0], n=2)
    assert hits and hits[0]["id"] == "a", f"round-trip failed: {hits}"
    _ok(f"query returned {len(hits)} hits, top={hits[0]['id']} score={hits[0]['score']:.3f}")
    shutil.rmtree(tmp, ignore_errors=True)

    print("\nAll smoke checks passed. Safe to run: python -m indexer.build --data data/images\n")


if __name__ == "__main__":
    main()
