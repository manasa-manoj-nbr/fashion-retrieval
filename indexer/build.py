"""CLI entry point for Part A: build the searchable index from raw images.

    python -m indexer.build --data data/images --limit 1000

Walks an image directory, builds the multi-vector representation for each image,
and writes it to the Chroma store. Idempotent-ish: rebuilding re-adds ids, so
delete the ``index/`` dir for a clean rebuild.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from tqdm import tqdm

from common.config import CONFIG, resolve
from indexer.embed import RepresentationBuilder
from indexer.store import VectorStore

_IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def iter_images(data_dir: Path, limit: int | None):
    files = sorted(p for p in data_dir.rglob("*") if p.suffix.lower() in _IMG_EXT)
    if limit:
        files = files[:limit]
    return files


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the fashion retrieval index.")
    ap.add_argument("--data", default=CONFIG["paths"]["data_dir"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch", type=int, default=50, help="images per store flush")
    args = ap.parse_args()

    data_dir = resolve(args.data)
    files = iter_images(data_dir, args.limit)
    if not files:
        raise SystemExit(f"No images found under {data_dir}")

    builder = RepresentationBuilder()
    store = VectorStore()

    g_ids, g_emb, g_meta = [], [], []
    r_ids, r_emb, r_meta = [], [], []

    def flush():
        store.add_global(g_ids, g_emb, g_meta)
        store.add_regions(r_ids, r_emb, r_meta)
        for lst in (g_ids, g_emb, g_meta, r_ids, r_emb, r_meta):
            lst.clear()

    for path in tqdm(files, desc="Indexing"):
        image_id = path.stem
        rec = builder.build(image_id, str(path))
        g_ids.append(image_id)
        g_emb.append(rec.global_embedding)
        g_meta.append({"image_id": image_id, "path": str(path)})
        for rr in rec.regions:
            r_ids.append(rr.region_id)
            r_emb.append(rr.embedding)
            r_meta.append({
                "image_id": image_id,
                "region_class": rr.region_class,
                "color_name": rr.color_name,
                "area_frac": float(rr.area_frac),
                "path": str(path),
            })
        if len(g_ids) >= args.batch:
            flush()
    flush()

    counts = store.count()
    print(f"Done. Indexed {counts['global']} images, {counts['region']} garment regions.")


if __name__ == "__main__":
    main()
