"""Evaluation harness: quantify that we beat vanilla CLIP.

Two evaluations, both self-contained (no manual labelling required):

1. COLOUR-SWAP BINDING TEST  (the headline result)
   For each indexed image that has an upper garment of colour A and a lower
   garment of colour B (A != B), we form:
       correct  = "a <A> <upper> and a <B> <lower>"
       swapped  = "a <B> <upper> and a <A> <lower>"
   A model with true attribute binding ranks the correct image ABOVE its
   colour-swapped description; a bag-of-concepts model (vanilla CLIP) is near
   chance. We report binding accuracy for global-only vs. the full pipeline.
   Ground truth here is the per-region dominant-colour tag, which is NOT used
   by the retrieval scoring path (that uses CLIP region embeddings), so the
   test is not circular.

2. ATTRIBUTE RETRIEVAL METRICS  (Recall@k / Precision@k / MRR)
   For single-attribute queries we auto-judge relevance from region metadata
   (e.g. "yellow raincoat" -> images with a yellow upper region) and compare
   global-only vs. full pipeline.

    python -m eval.evaluate
"""
from __future__ import annotations

from collections import defaultdict
from typing import Callable, Dict, List

import numpy as np

from indexer.store import VectorStore
from retriever.search import FashionRetriever


# --------------------------------------------------------------------------
# Load region metadata straight from the Chroma store (our ground-truth source)
# --------------------------------------------------------------------------
def load_region_meta(store: VectorStore) -> Dict[str, List[dict]]:
    """image_id -> list of {region_class, color_name}."""
    got = store.region_col.get(include=["metadatas"])
    by_img: Dict[str, List[dict]] = defaultdict(list)
    for meta in got["metadatas"]:
        by_img[meta["image_id"]].append(meta)
    return by_img


def has_region_color(regions: List[dict], region_class: str, color: str) -> bool:
    return any(r["region_class"] == region_class and r["color_name"] == color
               for r in regions)


# --------------------------------------------------------------------------
# Metric helpers
# --------------------------------------------------------------------------
def recall_at_k(ranked_ids: List[str], relevant: set, k: int) -> float:
    if not relevant:
        return float("nan")
    return len(set(ranked_ids[:k]) & relevant) / len(relevant)


def precision_at_k(ranked_ids: List[str], relevant: set, k: int) -> float:
    if k == 0:
        return 0.0
    return len(set(ranked_ids[:k]) & relevant) / k


def mrr(ranked_ids: List[str], relevant: set) -> float:
    for i, _id in enumerate(ranked_ids, 1):
        if _id in relevant:
            return 1.0 / i
    return 0.0


# --------------------------------------------------------------------------
# 1. Colour-swap binding test
# --------------------------------------------------------------------------
def binding_test(retriever: FashionRetriever, by_img: Dict[str, List[dict]],
                 max_pairs: int = 60) -> Dict[str, float]:
    encoder = retriever.encoder
    samples = []
    for img_id, regions in by_img.items():
        upper = next((r for r in regions if r["region_class"] == "upper"), None)
        lower = next((r for r in regions if r["region_class"] == "lower"), None)
        if upper and lower and upper["color_name"] != lower["color_name"]:
            samples.append((img_id, upper["color_name"], lower["color_name"]))
        if len(samples) >= max_pairs:
            break

    if not samples:
        return {"pairs": 0}

    def image_vec(img_id):
        hit = retriever.store.global_col.get(ids=[img_id], include=["embeddings"])
        return np.array(hit["embeddings"][0], dtype="float32")

    def score(text, vec):
        t = encoder.encode_texts(text)[0]
        return float(np.dot(t, vec))

    correct = 0
    for img_id, up_c, lo_c in samples:
        vec = image_vec(img_id)
        s_correct = score(f"a {up_c} top and {lo_c} pants", vec)
        s_swapped = score(f"a {lo_c} top and {up_c} pants", vec)
        if s_correct > s_swapped:
            correct += 1
    return {"pairs": len(samples),
            "binding_accuracy": round(correct / len(samples), 4)}


# --------------------------------------------------------------------------
# 2. Attribute retrieval metrics (global-only vs full pipeline)
# --------------------------------------------------------------------------
def attribute_eval(retriever: FashionRetriever, by_img: Dict[str, List[dict]],
                   k: int = 5) -> None:
    # (query, judge) pairs. Judge decides if an image is relevant from metadata.
    judges: List[tuple] = [
        ("a person in a bright yellow raincoat",
         lambda regs: has_region_color(regs, "upper", "yellow")),
        ("someone wearing a blue shirt",
         lambda regs: has_region_color(regs, "upper", "blue")),
        ("a white shirt",
         lambda regs: has_region_color(regs, "upper", "white")),
        ("red pants",
         lambda regs: has_region_color(regs, "lower", "red")),
    ]

    print(f"\n{'query':<38}{'mode':<10}{'R@k':>7}{'P@k':>7}{'MRR':>7}")
    print("-" * 69)
    for query, judge in judges:
        relevant = {img for img, regs in by_img.items() if judge(regs)}
        for mode, only_global in [("global", True), ("full", False)]:
            res = retriever.search(query, k=max(k, 20), only_global=only_global)
            ranked = [r.image_id for r in res]
            r = recall_at_k(ranked, relevant, k)
            p = precision_at_k(ranked, relevant, k)
            m = mrr(ranked, relevant)
            print(f"{query[:37]:<38}{mode:<10}{r:>7.3f}{p:>7.3f}{m:>7.3f}")
        print("-" * 69)


def main() -> None:
    store = VectorStore()
    retriever = FashionRetriever(store=store)
    by_img = load_region_meta(store)
    print(f"Loaded metadata for {len(by_img)} images.")

    print("\n=== 1. Colour-swap binding test (global embedding) ===")
    print(binding_test(retriever, by_img))

    print("\n=== 2. Attribute retrieval: global-only vs full pipeline ===")
    attribute_eval(retriever, by_img, k=5)


if __name__ == "__main__":
    main()
