"""Attribute-verification re-ranking with a local VQA model (BLIP).

The coarse segmenter cannot isolate a tie from a shirt, so fine-grained
compositional queries ("red tie AND white shirt") can still slip through. This
stage takes the top-N fused candidates and *verifies* each named attribute by
asking a Visual-Question-Answering model a grounded yes/no question per pair,
e.g. "Is the person wearing a red tie?". Candidates that satisfy more of the
query's constraints float to the top.

Uses ``Salesforce/blip-vqa-base`` which runs locally on the Colab T4 -> no API
key, fully self-contained. The whole stage is optional (toggled by the caller);
retrieval works without it.
"""
from __future__ import annotations

from typing import List

import torch
from PIL import Image
from transformers import BlipForQuestionAnswering, BlipProcessor

from common.config import CONFIG
from retriever.query_parser import ParsedQuery
from retriever.search import SearchResult

# A VQA model may name a colour with a near-synonym; accept those as matches
# rather than punishing the candidate for vocabulary choice.
_COLOR_ALIASES = {
    "blue": ("blue", "navy", "teal", "denim"),
    "red": ("red", "crimson", "maroon", "burgundy"),
    "white": ("white", "cream", "ivory"),
    "black": ("black", "dark"),
    "gray": ("gray", "grey", "silver"),
    "green": ("green", "olive", "lime"),
    "yellow": ("yellow", "gold", "mustard"),
    "brown": ("brown", "tan", "khaki"),
    "beige": ("beige", "tan", "cream"),
    "pink": ("pink", "rose"),
    "purple": ("purple", "violet"),
    "orange": ("orange",),
}
_SCENE_ALIASES = {
    "office": "indoor", "home": "indoor",
    "park": "outdoor", "street": "outdoor",
}


def _color_matches(want: str, answer: str) -> bool:
    """True if a VQA colour answer names the requested colour (or a synonym)."""
    answer = answer.lower()
    return any(a in answer for a in _COLOR_ALIASES.get(want, (want,)))


class VQAReranker:
    def __init__(self, model_name: str | None = None, device: str | None = None):
        self.model_name = model_name or CONFIG["models"]["vqa"]
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = BlipProcessor.from_pretrained(self.model_name)
        self.model = (
            BlipForQuestionAnswering.from_pretrained(self.model_name)
            .to(self.device)
            .eval()
        )

    @torch.no_grad()
    def _ask(self, image: Image.Image, question: str) -> str:
        inputs = self.processor(image, question, return_tensors="pt").to(self.device)
        out = self.model.generate(**inputs, max_new_tokens=8)
        return self.processor.decode(out[0], skip_special_tokens=True).strip().lower()

    def _verify_fraction(self, image: Image.Image, parsed: ParsedQuery) -> float:
        """Fraction of query constraints the image satisfies (0..1).

        IMPORTANT -- we ask OPEN-ENDED questions ("what color is the shirt?"),
        never leading yes/no ones ("is the shirt red?"). VQA models including
        BLIP have a strong yes-bias: they answer "yes" to almost any plausible
        leading question, which made this stage return 1.0 for every candidate
        and therefore reorder nothing. Forcing the model to *name* the colour
        makes the check discriminative -- a wrong colour cannot be papered over.
        """
        checks = 0
        satisfied = 0
        for pair in parsed.pairs:
            checks += 1
            if pair.color:
                ans = self._ask(image, f"what color is the {pair.garment_word}?")
                if _color_matches(pair.color, ans):
                    satisfied += 1
            else:
                # Presence-only: open-ended "what is the person wearing?" is
                # unreliable, so a yes/no is acceptable here (no colour to bind).
                if self._ask(image, f"is the person wearing a {pair.garment_word}?").startswith("y"):
                    satisfied += 1
        # Optional scene check, also open-ended.
        if parsed.scene:
            checks += 1
            ans = self._ask(image, "where was this photo taken?")
            if parsed.scene in ans or _SCENE_ALIASES.get(parsed.scene, "") in ans:
                satisfied += 1
        return satisfied / checks if checks else 0.0

    def rerank(self, parsed: ParsedQuery, results: List[SearchResult],
               top_n: int | None = None, alpha: float = 0.5) -> List[SearchResult]:
        """Blend fused score with VQA verification and re-sort.

        final = (1 - alpha) * fused_score + alpha * verified_fraction
        """
        if not parsed.pairs and not parsed.scene:
            return results
        top_n = top_n or CONFIG["retrieval"]["rerank_top_n"]
        head = results[:top_n]
        for r in head:
            try:
                img = Image.open(r.path).convert("RGB")
            except Exception:
                r.breakdown["vqa"] = 0.0
                continue
            frac = self._verify_fraction(img, parsed)
            r.breakdown["vqa"] = round(frac, 4)
            r.score = round((1 - alpha) * r.score + alpha * frac, 4)
        head.sort(key=lambda r: r.score, reverse=True)
        return head + results[top_n:]
