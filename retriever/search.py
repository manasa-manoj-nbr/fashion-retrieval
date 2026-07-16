"""Multi-stage compositional retrieval + score fusion (core ML logic).

Pipeline for a query:
  Stage 1  Recall      : ANN over GLOBAL vectors -> candidate pool (fast, scalable).
  Stage 2  Composition : for each (colour, garment) pair, score the pair prompt
                         against REGION vectors of the matching garment class,
                         then combine pairs with a MIN (logical AND). This is the
                         step vanilla CLIP cannot do -- it forces *every* named
                         attribute to be present on the *right* garment, so
                         "red tie + white shirt" no longer matches "white tie +
                         red shirt".
  Stage 3  Context     : score a scene/style prompt against GLOBAL vectors
                         (captures office / park / formal "where & vibe").
  Fusion  : weighted sum of the (min-max normalised) active components.

Components that don't apply to a query (e.g. no colour+garment pairs) are
dropped and their weight is redistributed, so single-axis queries aren't
penalised.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from common.config import CONFIG
from indexer.encoder import FashionEncoder
from indexer.store import VectorStore
from retriever.query_parser import ParsedQuery, parse


@dataclass
class SearchResult:
    image_id: str
    path: str
    score: float
    breakdown: Dict[str, float] = field(default_factory=dict)


def _minmax(scores: Dict[str, float]) -> Dict[str, float]:
    if not scores:
        return {}
    vals = np.array(list(scores.values()), dtype="float32")
    lo, hi = float(vals.min()), float(vals.max())
    if hi - lo < 1e-9:
        return {k: 1.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


class FashionRetriever:
    def __init__(self, encoder: FashionEncoder | None = None,
                store: VectorStore | None = None):
        self.encoder = encoder or FashionEncoder()
        self.store = store or VectorStore()
        rc = CONFIG["retrieval"]
        self.pool = int(rc["candidate_pool"])
        self.k = int(rc["top_k"])
        self.w = dict(rc["weights"])

    # -- component scorers ---------------------------------------------------
    def _global_scores(self, prompt: str) -> Dict[str, float]:
        emb = self.encoder.encode_texts(prompt)[0]
        hits = self.store.query_global(emb, self.pool)
        return {h["meta"]["image_id"]: h["score"] for h in hits}

    def _pair_scores(self, pair) -> Dict[str, float]:
        """Best region-match score per image for one (colour, garment) pair."""
        emb = self.encoder.encode_texts(pair.as_prompt())[0]
        hits = self.store.query_regions(
            emb, self.pool, where={"region_class": pair.region_class}
        )
        best: Dict[str, float] = {}
        for h in hits:
            img = h["meta"]["image_id"]
            if h["score"] > best.get(img, -1.0):
                best[img] = h["score"]
        return best

    def _composition_scores(self, parsed: ParsedQuery,
                            candidates: List[str]) -> Optional[Dict[str, float]]:
        if not parsed.pairs:
            return None
        per_pair = [self._pair_scores(p) for p in parsed.pairs]
        # AND semantics: an image's composition score is the WORST of its pair
        # scores. Missing garment region -> floor 0.0 (strong penalty).
        comp: Dict[str, float] = {}
        for img in candidates:
            comp[img] = min(ps.get(img, 0.0) for ps in per_pair)
        return comp

    def _scene_prompt(self, parsed: ParsedQuery) -> Optional[str]:
        bits = []
        if parsed.scene:
            bits.append(f"a scene in a {parsed.scene}")
        if parsed.style:
            bits.append(f"a {parsed.style} outfit")
        return ", ".join(bits) if bits else None

    # -- main entry ----------------------------------------------------------
    def search(self, query: str, k: int | None = None,
               only_global: bool = False) -> List[SearchResult]:
        """Run retrieval. ``only_global=True`` disables composition + scene,
        emulating single-vector (vanilla-CLIP-style) retrieval -- used as the
        ablation baseline to demonstrate the lift from attribute binding."""
        k = k or self.k
        parsed = parse(query)

        # Stage 1: recall pool from the raw query over global vectors.
        emb = self.encoder.encode_texts(query)[0]
        hits = self.store.query_global(emb, self.pool)
        global_raw = {h["meta"]["image_id"]: h["score"] for h in hits}
        candidates = list(global_raw.keys())
        paths = {h["meta"]["image_id"]: h["meta"].get("path", "") for h in hits}

        # Stage 2: composition (attribute binding).
        comp = None if only_global else self._composition_scores(parsed, candidates)

        # Stage 3: scene/style context.
        scene = None
        if not only_global:
            scene_prompt = self._scene_prompt(parsed)
            scene = self._global_scores(scene_prompt) if scene_prompt else None

        # Assemble active components + redistribute weights.
        components = {"global": (global_raw, self.w["global"])}
        if comp is not None:
            components["composition"] = (comp, self.w["composition"])
        if scene is not None:
            components["scene"] = (scene, self.w["scene"])
        total_w = sum(w for _, w in components.values())

        # Fuse over the union of all scored images.
        norm = {name: _minmax(sc) for name, (sc, _) in components.items()}
        all_ids = set().union(*[set(sc.keys()) for sc, _ in components.values()])

        results: List[SearchResult] = []
        for img in all_ids:
            breakdown = {}
            fused = 0.0
            for name, (_, w) in components.items():
                val = norm[name].get(img, 0.0)
                breakdown[name] = round(val, 4)
                fused += (w / total_w) * val
            results.append(SearchResult(
                image_id=img, path=paths.get(img, ""),
                score=round(fused, 4), breakdown=breakdown,
            ))

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:k]
