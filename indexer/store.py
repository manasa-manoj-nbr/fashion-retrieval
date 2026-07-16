"""Vector storage on top of ChromaDB.

Per the brief we pick the *easiest convenient* vector DB rather than building
our own. Chroma is zero-config, persistent, and uses HNSW (approximate nearest
neighbour) under the hood -- that ANN index is the reason the retrieval logic
scales to ~1M vectors instead of doing a brute-force matrix product.

Multi-vector design: TWO collections in one store.
  * global_emb  -> one vector per image (whole-scene / vibe / location).
  * region_emb  -> one vector per detected garment (enables attribute binding).
Region rows carry ``image_id`` + ``region_class`` metadata so the retriever can
score, say, "red" against only the "upper" regions of a candidate set.
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence

import chromadb

from common.config import CONFIG, resolve


class VectorStore:
    def __init__(self, index_dir: str | None = None):
        path = str(resolve(index_dir or CONFIG["paths"]["index_dir"]))
        self.client = chromadb.PersistentClient(path=path)
        space = CONFIG["index"]["distance"]
        self.global_col = self.client.get_or_create_collection(
            name=CONFIG["index"]["global_collection"],
            metadata={"hnsw:space": space},
        )
        self.region_col = self.client.get_or_create_collection(
            name=CONFIG["index"]["region_collection"],
            metadata={"hnsw:space": space},
        )

    # ---- writing -----------------------------------------------------------
    # NOTE: both writers no-op on an empty batch (a trailing flush after an
    # exact-multiple batch count would otherwise hand Chroma an empty list,
    # which it rejects), and use ``upsert`` so re-running the build overwrites
    # existing ids instead of raising on duplicates -- i.e. indexing is
    # idempotent and safely resumable.
    def _write(self, collection, ids, embeddings, metadatas) -> None:
        if not ids:
            return
        collection.upsert(ids=list(ids),
                          embeddings=[e.tolist() for e in embeddings],
                          metadatas=list(metadatas))

    def add_global(self, ids, embeddings, metadatas):
        self._write(self.global_col, ids, embeddings, metadatas)

    def add_regions(self, ids, embeddings, metadatas):
        self._write(self.region_col, ids, embeddings, metadatas)

    # ---- reading -----------------------------------------------------------
    def query_global(self, embedding, n: int) -> List[Dict[str, Any]]:
        """ANN search over whole-image vectors -> ranked candidates."""
        res = self.global_col.query(
            query_embeddings=[embedding.tolist()],
            n_results=n,
            include=["metadatas", "distances"],
        )
        return self._flatten(res)

    def query_regions(self, embedding, n: int,
                      where: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
        """ANN search over garment-region vectors, optionally metadata-filtered."""
        res = self.region_col.query(
            query_embeddings=[embedding.tolist()],
            n_results=n,
            where=where,
            include=["metadatas", "distances"],
        )
        return self._flatten(res)

    @staticmethod
    def _flatten(res: Dict[str, Any]) -> List[Dict[str, Any]]:
        ids = res.get("ids", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        out = []
        for i, _id in enumerate(ids):
            # cosine distance -> similarity in [-1, 1]
            out.append({"id": _id, "meta": metas[i], "score": 1.0 - float(dists[i])})
        return out

    def count(self) -> Dict[str, int]:
        return {"global": self.global_col.count(), "region": self.region_col.count()}
