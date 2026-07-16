"""CLI entry point for Part B: run a natural-language query.

    python -m retriever.query "a red tie and a white shirt in a formal setting" --k 5
    python -m retriever.query "a person in a bright yellow raincoat" --rerank

Prints the top-k image paths with the fused score and per-component breakdown.
"""
from __future__ import annotations

import argparse
import json

from retriever.query_parser import parse
from retriever.search import FashionRetriever


def main() -> None:
    ap = argparse.ArgumentParser(description="Query the fashion retrieval index.")
    ap.add_argument("query", help="natural-language description")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--rerank", action="store_true",
                    help="enable local VQA attribute verification (slower, precise)")
    ap.add_argument("--json", action="store_true", help="emit JSON")
    args = ap.parse_args()

    retriever = FashionRetriever()
    # Retrieve a slightly larger set when re-ranking so the VQA stage has room.
    fetch_k = max(args.k, args.k * 4) if args.rerank else args.k
    results = retriever.search(args.query, k=fetch_k)

    if args.rerank:
        from retriever.rerank import VQAReranker  # lazy import (heavy model)
        parsed = parse(args.query)
        results = VQAReranker().rerank(parsed, results)[: args.k]
    else:
        results = results[: args.k]

    if args.json:
        print(json.dumps([r.__dict__ for r in results], indent=2))
        return

    print(f'\nQuery: "{args.query}"')
    print(f"Parsed: {parse(args.query)}\n")
    for rank, r in enumerate(results, 1):
        print(f"{rank:>2}. {r.score:.3f}  {r.path}")
        print(f"       breakdown={r.breakdown}")


if __name__ == "__main__":
    main()
