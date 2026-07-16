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
        """Fraction of query constraints the image satisfies (0..1)."""
        checks = 0
        satisfied = 0
        for pair in parsed.pairs:
            checks += 1
            if pair.color:
                q = f"is the person wearing a {pair.color} {pair.garment_word}?"
                ans = self._ask(image, q)
                if ans.startswith("y"):
                    satisfied += 1
                else:
                    # Back off to a direct colour question for robustness.
                    c = self._ask(image, f"what color is the {pair.garment_word}?")
                    if pair.color in c:
                        satisfied += 1
            else:
                q = f"is the person wearing a {pair.garment_word}?"
                if self._ask(image, q).startswith("y"):
                    satisfied += 1
        # Optional scene check.
        if parsed.scene:
            checks += 1
            if self._ask(image, f"is this in a {parsed.scene}?").startswith("y"):
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
