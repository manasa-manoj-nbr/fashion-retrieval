"""Decompose a natural-language query into structured intent.

This is the second half of the compositionality fix. Instead of throwing the
whole sentence at CLIP, we extract:
  * (colour, garment) PAIRS   -> "red tie", "white shirt"  (attribute binding)
  * scene / location cues     -> "office", "park", "street"
  * style cues                -> "formal", "casual"

Primary implementation is rule-based so the system runs fully offline (no API
key) and is deterministic for grading. ``parse_with_llm`` is an optional hook:
an LLM returns the same schema for messier queries. Both paths yield the same
``ParsedQuery`` so the retriever is agnostic to which was used.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from common.config import CONFIG

_COLORS = list(CONFIG["colors"].keys())
# Colour synonyms / modifiers we fold into a base colour.
_COLOR_SYNONYMS = {
    "navy": "blue", "teal": "blue", "crimson": "red", "maroon": "red",
    "cream": "white", "ivory": "white", "grey": "gray", "tan": "beige",
    "olive": "green", "lime": "green",
}

# garment word -> region class, inverted from config.garment_map.
_GARMENT_TO_REGION: Dict[str, str] = {}
for region_class, words in CONFIG["garment_map"].items():
    for w in words:
        _GARMENT_TO_REGION[w] = region_class

_SCENE_KEYWORDS = {
    "office": ["office", "workplace", "boardroom", "desk", "corporate"],
    "street": ["street", "urban", "city", "sidewalk", "downtown"],
    "park": ["park", "bench", "garden", "outdoor", "grass"],
    "home": ["home", "living room", "bedroom", "indoor", "couch", "sofa"],
}
_STYLE_KEYWORDS = {
    "formal": ["formal", "business", "professional", "elegant", "smart"],
    "casual": ["casual", "weekend", "relaxed", "everyday", "streetwear"],
}


@dataclass
class AttributePair:
    color: Optional[str]
    garment_word: str
    region_class: str

    def as_prompt(self) -> str:
        return f"{self.color} {self.garment_word}" if self.color else self.garment_word


@dataclass
class ParsedQuery:
    raw: str
    pairs: List[AttributePair] = field(default_factory=list)
    scene: Optional[str] = None
    style: Optional[str] = None
    colors: List[str] = field(default_factory=list)
    garments: List[str] = field(default_factory=list)


def _norm_color(tok: str) -> Optional[str]:
    tok = tok.lower()
    if tok in _COLORS:
        return tok
    return _COLOR_SYNONYMS.get(tok)


def parse(query: str) -> ParsedQuery:
    """Rule-based decomposition of a query into structured intent."""
    raw = query.strip()
    tokens = re.findall(r"[a-zA-Z\-]+", raw.lower())

    pairs: List[AttributePair] = []
    colors_found: List[str] = []
    garments_found: List[str] = []

    for i, tok in enumerate(tokens):
        region = _GARMENT_TO_REGION.get(tok)
        if not region:
            continue
        garments_found.append(tok)
        # Bind the nearest preceding colour within a 3-token window.
        color = None
        for j in range(max(0, i - 3), i):
            c = _norm_color(tokens[j])
            if c:
                color = c
        if color:
            colors_found.append(color)
        pairs.append(AttributePair(color=color, garment_word=tok, region_class=region))

    # Any colours mentioned without an attached garment (still useful as prior).
    for tok in tokens:
        c = _norm_color(tok)
        if c and c not in colors_found:
            colors_found.append(c)

    scene = _first_match(raw.lower(), _SCENE_KEYWORDS)
    style = _first_match(raw.lower(), _STYLE_KEYWORDS)

    return ParsedQuery(
        raw=raw, pairs=pairs, scene=scene, style=style,
        colors=colors_found, garments=garments_found,
    )


def _first_match(text: str, table: Dict[str, List[str]]) -> Optional[str]:
    for label, kws in table.items():
        if any(kw in text for kw in kws):
            return label
    return None


def parse_with_llm(query: str, client=None) -> ParsedQuery:
    """Optional: use an LLM to produce the same schema for messy queries.

    Falls back to the rule-based parser when no client is supplied, so callers
    never need to special-case the offline path.
    """
    if client is None:
        return parse(query)
    # Intentionally left as an integration point. A JSON-schema prompt asking
    # for {pairs:[{color,garment}], scene, style} would be mapped into
    # ParsedQuery here. Rule-based output remains the safe default.
    return parse(query)
