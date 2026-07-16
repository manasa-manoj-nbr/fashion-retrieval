"""Load the central YAML config once and expose it project-wide.

Keeping every tunable in ``config.yaml`` (models, weights, paths, garment map)
means the retrieval/indexing *logic* never hard-codes data-specific values.
"""
from __future__ import annotations

import functools
from pathlib import Path
from typing import Any, Dict

import yaml

# Repo root = parent of the ``common`` package directory.
ROOT = Path(__file__).resolve().parent.parent


@functools.lru_cache(maxsize=1)
def load_config(path: str | None = None) -> Dict[str, Any]:
    """Return the parsed config dict (cached after first read)."""
    cfg_path = Path(path) if path else ROOT / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    return cfg


def resolve(path_str: str) -> Path:
    """Resolve a config-relative path against the repo root."""
    p = Path(path_str)
    return p if p.is_absolute() else ROOT / p


CONFIG = load_config()
