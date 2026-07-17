"""Curated query battery for the report.

Runs a diverse, categorised set of queries so the write-up can demonstrate
coverage across the assignment's three axes (environment, clothing type,
colour) plus the compositional-binding differentiator and zero-shot behaviour.

    python -m eval.showcase                 # text output, all categories
    python -m eval.showcase --k 3
    python -m eval.showcase --only binding  # one category

For the compositional-binding category we print GLOBAL vs FULL side by side so
the attribute-binding lift is visible per query. Image-grid versions (better for
screenshots) are in the notebook cells shared alongside this file.
"""
from __future__ import annotations

import argparse
from typing import Dict, List

from retriever.search import FashionRetriever

# --------------------------------------------------------------------------
# Query sets, grouped by what each one demonstrates for the report.
# --------------------------------------------------------------------------
QUERY_SETS: Dict[str, List[str]] = {
    # The 5 mandatory evaluation prompts from the assignment.
    "official": [
        "A person in a bright yellow raincoat.",
        "Professional business attire inside a modern office.",
        "Someone wearing a blue shirt sitting on a park bench.",
        "Casual weekend outfit for a city walk.",
        "A red tie and a white shirt in a formal setting.",
    ],
    # THE differentiator: colour<->garment binding. Each line is a swap pair;
    # a bag-of-concepts model returns near-identical sets for the two, ours
    # should not. Printed global-vs-full.
    "binding": [
        "a red top and blue pants",
        "a blue top and red pants",
        "a white shirt and black pants",
        "a black shirt and white pants",
        "a green top and white pants",
    ],
    # Colour-theory axis: precise colour across a wide palette.
    "color": [
        "a bright red dress",
        "a navy blue blazer",
        "a pink hoodie",
        "a beige trench coat",
        "a purple skirt",
        "a mustard yellow sweater",
    ],
    # Clothing-type axis + style inference (zero-shot styles, not label words).
    "garment_style": [
        "a formal business suit",
        "a casual streetwear hoodie",
        "an elegant evening gown",
        "a denim jacket",
        "a knitted winter sweater",
        "a leather biker jacket",
    ],
    # Context / environment axis. NOTE: Fashionpedia is catalog/runway heavy,
    # so scene signal is weaker -- reported honestly in the write-up.
    "context": [
        "professional office attire",
        "relaxed outfit at home",
        "street style in the city",
        "summer outfit for the beach",
    ],
    # Zero-shot: descriptions unlikely to appear as any training label.
    "zeroshot": [
        "a floral summer dress",
        "a polka dot blouse",
        "a monochrome black outfit",
        "an oversized pastel coat",
    ],
}


def _fmt(results, k):
    return "  ".join(f"{r.image_id[:10]}({r.score:.2f})" for r in results[:k])


def run(only: str | None, k: int) -> None:
    retriever = FashionRetriever()
    sets = {only: QUERY_SETS[only]} if only else QUERY_SETS

    for name, queries in sets.items():
        print("\n" + "=" * 78)
        print(f"CATEGORY: {name.upper()}")
        print("=" * 78)
        for q in queries:
            if name == "binding":
                # side-by-side ablation to expose the binding lift
                g = retriever.search(q, k=k, only_global=True)
                f = retriever.search(q, k=k, only_global=False)
                print(f'\n"{q}"')
                print(f"   global : {_fmt(g, k)}")
                print(f"   full   : {_fmt(f, k)}")
            else:
                f = retriever.search(q, k=k, only_global=False)
                print(f'\n"{q}"')
                print(f"   {_fmt(f, k)}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the report query battery.")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--only", choices=list(QUERY_SETS.keys()), default=None)
    args = ap.parse_args()
    run(args.only, args.k)


if __name__ == "__main__":
    main()
