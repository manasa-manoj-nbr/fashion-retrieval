"""Render the system architecture diagram to a high-res PNG for the report."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

# palette
NAVY = "#1F3864"
BLUE = "#2E5496"
LBLUE = "#DCE6F5"
GREEN = "#2E7D5B"
LGREEN = "#D8ECE1"
AMBER = "#B9770E"
LAMBER = "#FBEBD1"
GREY = "#555555"
LGREY = "#EFF1F4"

fig, ax = plt.subplots(figsize=(12, 6.6), dpi=200)
ax.set_xlim(0, 120)
ax.set_ylim(0, 66)
ax.axis("off")


def box(x, y, w, h, text, fc, ec, tc="#1a1a1a", fs=10, bold=False, rounded=0.4):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.02,rounding_size={rounded}",
                       linewidth=1.4, edgecolor=ec, facecolor=fc, zorder=2)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=tc, fontweight="bold" if bold else "normal", zorder=3)


def arrow(x1, y1, x2, y2, color=GREY, style="-|>", lw=1.6, ls="-"):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=14,
                        linewidth=lw, color=color, linestyle=ls, zorder=1,
                        shrinkA=2, shrinkB=2)
    ax.add_patch(a)


def band(x, y, w, h, label, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.6",
                 linewidth=1.2, edgecolor=color, facecolor="none", linestyle=(0, (5, 3)), zorder=0))
    ax.text(x + 1.4, y + h - 2.2, label, ha="left", va="center", fontsize=10.5,
            color=color, fontweight="bold")


# ---- INDEXING band (Part A) ------------------------------------------------
band(2, 34, 66, 29, "PART A  ·  INDEXER  (offline, once)", NAVY)
box(4, 50, 14, 7, "Fashion\nimages", LGREY, GREY, fs=10)
box(4, 39.5, 14, 7, "Garment\nsegmenter", LBLUE, BLUE, bold=True, fs=9.5)
box(21, 39.5, 12, 7, "per-garment\ncrops", LGREY, GREY, fs=9)
box(36, 44.5, 15, 8, "FashionCLIP\nencoder", LBLUE, BLUE, bold=True, fs=10)
box(54, 45, 9.5, 7, "global +\nregion\nvectors", LGREEN, GREEN, fs=8.5)

# indexing arrows: image -> segmenter -> crops; image + crops -> encoder -> vectors
arrow(11, 50, 11, 46.5)                          # images -> segmenter (down)
arrow(18, 43, 21, 43)                            # segmenter -> crops
arrow(18, 53, 36, 50)                            # whole image -> encoder
arrow(33, 43, 36, 47)                            # crops -> encoder
arrow(51, 48.5, 54, 48.5, color=BLUE)            # encoder -> vectors

ax.text(11, 38.0, "splits image into\nupper / lower / dress …",
        ha="center", va="top", fontsize=7.5, color=GREY, style="italic")

# ---- STORE (shared) --------------------------------------------------------
box(70, 42, 20, 12, "ChromaDB\n(HNSW ANN index)\n\nglobal_emb  ·  region_emb",
    LAMBER, AMBER, bold=True, fs=9.5)
arrow(63.5, 48.5, 70, 48, color=GREEN, lw=2)     # vectors -> store

# ---- RETRIEVAL band (Part B) ----------------------------------------------
band(2, 2, 116, 28, "PART B  ·  RETRIEVER  (per query)", GREEN)

box(5, 18, 20, 8, 'Query:\n"a red tie and a\nwhite shirt…"', LGREY, GREY, fs=9)
box(29, 18, 17, 8, "Query\ndecomposition", LBLUE, BLUE, bold=True, fs=10)
ax.text(37.5, 16.4, "(colour, garment) pairs\n+ scene + style", ha="center", va="top",
        fontsize=8, color=GREY, style="italic")

arrow(25, 22, 29, 22)

# three scoring stages
box(52, 24, 24, 6.2, "Stage 1 — global recall (ANN)", "#EAF0FA", BLUE, fs=9.5)
box(52, 15.5, 24, 6.2, "Stage 2 — region AND-scoring", "#E6F3EC", GREEN, bold=True, fs=9.5)
box(52, 7, 24, 6.2, "Stage 3 — scene / style score", "#EAF0FA", BLUE, fs=9.5)

# parser -> stages
arrow(46, 22, 52, 27, color=GREY)
arrow(46, 22, 52, 18.6, color=GREY)
arrow(46, 22, 52, 10.1, color=GREY)

# store <-> stages (queries)
arrow(80, 42, 74, 30.5, color=AMBER, ls=(0, (4, 3)))
ax.text(82, 36, "ANN\nlookups", ha="left", va="center", fontsize=8, color=AMBER, style="italic")

# stages -> fuse
box(82, 15.5, 12, 6.2, "Weighted\nfusion", LGREEN, GREEN, bold=True, fs=9.5)
arrow(76, 27, 82, 20, color=GREY)
arrow(76, 18.6, 82, 18.6, color=GREEN, lw=2)
arrow(76, 10.1, 82, 17, color=GREY)

# fuse -> rerank -> results
box(99, 15.5, 15, 6.2, "VQA re-rank\n(optional)", "#EAF0FA", BLUE, fs=9)
arrow(94, 18.6, 99, 18.6)
box(99, 6.5, 15, 6.2, "Top-k\nimages", LAMBER, AMBER, bold=True, fs=10)
arrow(106.5, 15.5, 106.5, 12.7)

# ---- title -----------------------------------------------------------------
ax.text(60, 64.6, "Decompose → Bind → Verify:  a compositional fashion-retrieval pipeline",
        ha="center", va="center", fontsize=13, fontweight="bold", color="#1a1a1a")

# ---- legend ----------------------------------------------------------------
legend = [
    Line2D([0], [0], marker="s", color="w", markerfacecolor=LBLUE, markeredgecolor=BLUE,
           markersize=11, label="neural model"),
    Line2D([0], [0], marker="s", color="w", markerfacecolor=LGREEN, markeredgecolor=GREEN,
           markersize=11, label="our contribution / binding"),
    Line2D([0], [0], marker="s", color="w", markerfacecolor=LAMBER, markeredgecolor=AMBER,
           markersize=11, label="storage / output"),
]
ax.legend(handles=legend, loc="lower left", bbox_to_anchor=(0.015, -0.02),
          ncol=3, frameon=False, fontsize=8.5)

plt.tight_layout()
fig.savefig("docs/architecture.png", dpi=200, bbox_inches="tight",
            facecolor="white", pad_inches=0.15)
print("wrote docs/architecture.png")
