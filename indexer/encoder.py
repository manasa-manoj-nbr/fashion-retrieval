"""FashionCLIP encoder wrapper.

Why FashionCLIP and not vanilla CLIP: it is CLIP fine-tuned on ~800k fashion
(image, text) pairs, so its text tower already understands garment, fabric and
style vocabulary ("button-down", "raincoat", "blazer"). This is a free accuracy
win over the ``openai/clip-vit-base-patch32`` baseline and the first step of the
"be better than vanilla CLIP" requirement.

All embeddings are L2-normalised so that a dot product == cosine similarity.
"""
from __future__ import annotations

from typing import List, Sequence, Union

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from common.config import CONFIG


class FashionEncoder:
    """Thin wrapper around a (Fashion)CLIP model producing unit-norm vectors."""

    def __init__(self, model_name: str | None = None, device: str | None = None):
        self.model_name = model_name or CONFIG["models"]["clip"]
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(self.model_name).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(self.model_name)

    @property
    def dim(self) -> int:
        return int(self.model.config.projection_dim)

    @staticmethod
    def _normalize(x: torch.Tensor) -> torch.Tensor:
        return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    @staticmethod
    def _as_features(out, projection) -> torch.Tensor:
        """Coerce a get_*_features() return value into projected features.

        transformers changed this API's return type across versions:
          * older: returns the already-projected tensor  -> pass through.
          * newer: wraps it in an output object          -> unwrap it.

        We disambiguate by DIMENSION rather than by version sniffing, since the
        wrapped ``pooler_output`` may be either already-projected features
        (last dim == projection.out_features) or a raw encoder pooled state
        (last dim == projection.in_features). For CLIP ViT-B/32 the vision
        tower is 768-d and the shared embedding space is 512-d, so the two
        cases are distinguishable. We prefer the already-projected reading when
        both match, which is what current transformers returns.
        """
        if isinstance(out, torch.Tensor):
            feats = out
        else:
            feats = getattr(out, "pooler_output", None)
            if feats is None and isinstance(out, (tuple, list)):
                feats = out[1] if len(out) > 1 else out[0]
            if feats is None:
                last = getattr(out, "last_hidden_state", None)
                if last is None:
                    raise TypeError(f"Cannot extract features from {type(out)}")
                feats = last[:, 0]

        dim = feats.shape[-1]
        if dim == projection.out_features:
            return feats                    # already in the shared space
        if dim == projection.in_features:
            return projection(feats)        # raw pooled state -> project
        raise TypeError(
            f"Unexpected feature dim {dim}; projection expects "
            f"in={projection.in_features} or out={projection.out_features}"
        )

    @torch.no_grad()
    def encode_images(self, images: Sequence[Image.Image], batch_size: int = 32) -> np.ndarray:
        """Encode a list of PIL images -> (N, dim) float32 unit vectors."""
        out: List[np.ndarray] = []
        for i in range(0, len(images), batch_size):
            batch = [im.convert("RGB") for im in images[i : i + batch_size]]
            inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
            feats = self.model.get_image_features(**inputs)
            feats = self._as_features(feats, self.model.visual_projection)
            feats = self._normalize(feats)
            out.append(feats.cpu().numpy().astype("float32"))
        return np.concatenate(out, axis=0) if out else np.zeros((0, self.dim), "float32")

    @torch.no_grad()
    def encode_texts(self, texts: Union[str, Sequence[str]], batch_size: int = 64) -> np.ndarray:
        """Encode text prompt(s) -> (N, dim) float32 unit vectors."""
        if isinstance(texts, str):
            texts = [texts]
        out: List[np.ndarray] = []
        for i in range(0, len(texts), batch_size):
            batch = list(texts[i : i + batch_size])
            inputs = self.processor(
                text=batch, return_tensors="pt", padding=True, truncation=True
            ).to(self.device)
            feats = self.model.get_text_features(**inputs)
            feats = self._as_features(feats, self.model.text_projection)
            feats = self._normalize(feats)
            out.append(feats.cpu().numpy().astype("float32"))
        return np.concatenate(out, axis=0) if out else np.zeros((0, self.dim), "float32")
