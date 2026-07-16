"""Download a Fashionpedia image subset for indexing.

Data is fetched AT RUNTIME by this script -- images are never committed to git
(see .gitignore). Run this inside Colab, not on your laptop.

Default source is the OFFICIAL Fashionpedia S3 bucket (cvdfoundation), val/test
split: 236 MB (verified), which comfortably covers the assignment's 500-1,000
image requirement without touching the 3.3 GB train split.

Verified endpoints (HEAD-checked):
  images/val_test2020.zip                       236 MB   <- default
  images/train2020.zip                         3344 MB
  annotations/instances_attributes_val2020.json  15 MB   <- eval only

    python -m eval.fetch_data --n 800                  # official S3 (default)
    python -m eval.fetch_data --n 800 --with-annotations
    python -m eval.fetch_data --n 800 --source hf      # HuggingFace fallback

Design note: the raw *images* are all the indexer uses. Annotations (optional)
are consumed only by the evaluation harness -- the indexer/retriever never see
them, which keeps the system zero-shot and dataset-agnostic.
"""
from __future__ import annotations

import argparse
import io
import urllib.request
import zipfile
from pathlib import Path

from common.config import resolve

S3_ROOT = "https://s3.amazonaws.com/ifashionist-dataset"
IMAGES_VAL_TEST = f"{S3_ROOT}/images/val_test2020.zip"
IMAGES_TRAIN = f"{S3_ROOT}/images/train2020.zip"
ANNOTATIONS_VAL = f"{S3_ROOT}/annotations/instances_attributes_val2020.json"

_IMG_EXT = {".jpg", ".jpeg", ".png"}


def _download(url: str, dest: Path) -> Path:
    """Stream a URL to disk with a simple progress readout."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[cached] {dest.name} ({dest.stat().st_size / 1e6:.0f} MB)")
        return dest
    print(f"[download] {url}")
    with urllib.request.urlopen(url) as resp, open(dest, "wb") as fh:
        total = int(resp.headers.get("Content-Length", 0))
        done = 0
        while chunk := resp.read(1 << 20):  # 1 MB chunks
            fh.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done/1e6:7.0f} / {total/1e6:.0f} MB", end="")
        print()
    return dest


def fetch_from_s3(n: int, out_dir: Path, cache_dir: Path, split: str) -> int:
    url = IMAGES_TRAIN if split == "train" else IMAGES_VAL_TEST
    zip_path = _download(url, cache_dir / Path(url).name)

    saved = 0
    with zipfile.ZipFile(zip_path) as zf:
        members = [m for m in zf.namelist()
                   if Path(m).suffix.lower() in _IMG_EXT and not m.startswith("__")]
        members.sort()
        print(f"[extract] {len(members)} images in archive; taking {n}")
        for m in members[:n]:
            target = out_dir / Path(m).name
            if target.exists():
                saved += 1
                continue
            with zf.open(m) as src, open(target, "wb") as dst:
                dst.write(src.read())
            saved += 1
    return saved


def fetch_from_hf(n: int, out_dir: Path, dataset: str, split: str) -> int:
    from datasets import load_dataset  # heavy import kept local

    ds = load_dataset(dataset, split=split, streaming=True)
    saved = 0
    for i, ex in enumerate(ds):
        if saved >= n:
            break
        img = ex.get("image")
        if img is None:
            continue
        try:
            img.convert("RGB").save(out_dir / f"fp_{i:06d}.jpg", quality=90)
            saved += 1
        except Exception:
            continue
    return saved


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch a Fashionpedia image subset.")
    ap.add_argument("--n", type=int, default=800, help="number of images to keep")
    ap.add_argument("--source", choices=["s3", "hf"], default="s3",
                    help="s3 = official Fashionpedia bucket (default)")
    ap.add_argument("--split", default="val_test",
                    help="s3: val_test (~1.2GB, recommended) | train (~20GB)")
    ap.add_argument("--dataset", default="detection-datasets/fashionpedia",
                    help="HuggingFace dataset id (only used with --source hf)")
    ap.add_argument("--with-annotations", action="store_true",
                    help="also fetch val annotations (eval ground truth only)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out_dir = resolve(args.out) if args.out else resolve("data/images")
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = resolve("data/_cache")

    if args.source == "s3":
        saved = fetch_from_s3(args.n, out_dir, cache_dir, args.split)
    else:
        saved = fetch_from_hf(args.n, out_dir, args.dataset,
                              "train" if args.split in ("train", "val_test") else args.split)

    if args.with_annotations:
        ann_dir = resolve("data/eval")
        ann_dir.mkdir(parents=True, exist_ok=True)
        _download(ANNOTATIONS_VAL, ann_dir / "instances_attributes_val2020.json")

    print(f"\nSaved {saved} images to {out_dir}")
    print("Next:  python -m indexer.build --data data/images")


if __name__ == "__main__":
    main()
