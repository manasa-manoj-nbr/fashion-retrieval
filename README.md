# Multimodal Fashion & Context Retrieval

Text-to-image search over a fashion photo database. Given a natural-language
query — *"a red tie and a white shirt in a formal setting"* — it returns the
top-k matching images, with a specific focus on **compositional / attribute
binding** queries where vanilla CLIP fails.

## Why this is better than vanilla CLIP

Vanilla CLIP encodes an image into a **single** vector, so it treats a photo as
a *bag of concepts*: it knows "red", "blue", "tie", "shirt" are present but
cannot reliably bind **which colour goes with which garment**. It scores
*"red tie + white shirt"* and *"white tie + red shirt"* almost identically.

This system fixes that with a **decompose → bind → verify** pipeline:

| Stage | What it does | Why it helps |
|-------|--------------|--------------|
| **FashionCLIP** | fashion-fine-tuned encoder | knows garment/fabric/style vocabulary |
| **Garment segmentation** | split image into per-garment regions, embed each | enables attribute→garment binding |
| **Query decomposition** | parse query into (colour, garment) pairs + scene + style | structured intent, not a bag of words |
| **Compositional AND-scoring** | each attribute must match its garment region (min-fusion) | kills the colour-swap failure mode |
| **VQA re-rank** | local BLIP re-checks colours it *can* isolate on the top candidates | sharpens precision where a garment is separable; conservatively skips attributes the segmenter can't isolate (e.g. tie vs shirt) rather than guessing |

## Architecture

```
INDEXING (Part A)                          RETRIEVAL (Part B)
 image ─ FashionCLIP ─► global vector       query ─► parse ─► {pairs, scene, style}
   │                                           │
   └─ segmenter ─► crops ─► region vectors     ├─ Stage1 global ANN  (recall)
             │                                 ├─ Stage2 region AND-score (bind)
             ▼                                 ├─ Stage3 scene score (context)
        Chroma (HNSW ANN)  ◄───────────────────┴─ fuse ─► VQA re-rank ─► top-k
```

## Repo layout (logic separated from data)

```
indexer/    Part A — encoder, segmenter, colours, vector store, build CLI
retriever/  Part B — query parser, multi-stage search, VQA re-rank, query CLI
eval/       data fetcher + evaluation harness (binding test, ablation)
common/     shared config loader
config.yaml all models / weights / paths (no magic numbers in code)
notebooks/  demo.ipynb (Colab)
```

## Quickstart (Colab T4 recommended)

```bash
pip install -r requirements.txt

# 1. get ~800 Fashionpedia images (official S3 val/test split, 236 MB download)
python -m eval.fetch_data --n 800

# 2. build the index (Part A)
python -m indexer.build --data data/images

# 3. query (Part B)
python -m retriever.query "a red tie and a white shirt in a formal setting" --k 5 --rerank

# 4. evaluate (binding test + global-vs-full ablation)
python -m eval.evaluate
```

Or just open `notebooks/demo.ipynb` in Colab and run top to bottom.

## Scalability to 1M images

- **ANN, not brute force.** Chroma uses HNSW, giving sub-linear search — the
  reason we use a real vector DB instead of a NumPy cosine matrix.
- **Two-stage funnel bounds cost.** Cheap global-ANN recall narrows 1M → ~200,
  then the expensive region/VQA scoring runs only on that pool. Cost scales
  with pool size, not corpus size.
- **At larger scale:** batch GPU embedding at index time, quantise vectors
  (PQ/int8), shard, and swap Chroma (local) for Qdrant/Milvus (distributed).

## Data

Images are **downloaded at runtime by `eval/fetch_data.py`** — nothing is
committed to git (`data/` is gitignored). Run it inside Colab, not locally.

Source: the official Fashionpedia S3 bucket (cvdfoundation). Verified endpoints:

| File | Size | Used for |
|------|------|----------|
| `images/val_test2020.zip` | **236 MB** | indexing (default) |
| `images/train2020.zip` | 3.3 GB | not needed |
| `annotations/instances_attributes_val2020.json` | 15 MB | eval ground truth only (optional) |

The val/test split alone covers the assignment's 500–1,000 image requirement,
so there is no reason to pull the train split.

## Zero-shot & dataset-agnostic

The indexer and retriever **never** read Fashionpedia labels — they run on raw
pixels only. Dataset annotations are used *solely* by `eval/` to build ground
truth. So the system works on arbitrary unlabeled images and handles
descriptions unseen at training time.

## Known limitations (and how the design handles them)

- The human-parsing segmenter is **coarse** (one "upper" class — a tie isn't
  separated from a shirt). Coarse regions still fix upper-vs-lower binding; the
  **VQA re-rank** stage recovers the fine-grained tie-vs-shirt cases.
- Scene understanding relies on CLIP's global vector; a dedicated
  scene/place classifier (see *future work*) would sharpen contextual queries.

## Future work

- **Locations & weather:** add a scene/place classifier + weather tagger,
  store their outputs as extra metadata vectors + filterable facets, and fuse
  into the context score.
- **Precision:** hard-negative fine-tuning of the encoder, learned (not fixed)
  fusion weights, a larger VLM re-ranker, and attribute-level query expansion.
