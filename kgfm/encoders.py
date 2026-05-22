"""Pluggable text encoders.

All encoders implement the same contract:

    forward(texts: Sequence[str]) -> Tensor of shape [B, embedding_dim]

Two implementations are provided:

- HashedNgramEncoder: trainable char-n-gram hashing (FastText-style). No deps.
- TransformerEncoder: HuggingFace AutoModel/AutoTokenizer wrapper (BERT etc.).
  Requires `transformers` to be installed. Pass `freeze=True` to use it as a
  fixed feature extractor; otherwise it fine-tunes end-to-end.
"""

from __future__ import annotations

import zlib
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Hashed n-gram encoder
# ---------------------------------------------------------------------------


def extract_ngrams(text: str, n_min: int = 3, n_max: int = 5, max_ngrams: int = 96) -> List[str]:
    s = "^" + text + "$"
    out: List[str] = []
    L = len(s)
    for n in range(n_min, n_max + 1):
        if L < n:
            continue
        for i in range(L - n + 1):
            out.append(s[i : i + n])
            if len(out) >= max_ngrams:
                return out
    if not out:
        out.append(s)
    return out


def hash_text_to_buckets(
    text: str, vocab_size: int, n_min: int = 3, n_max: int = 5, max_ngrams: int = 96
) -> List[int]:
    grams = extract_ngrams(text, n_min, n_max, max_ngrams)
    return [zlib.crc32(g.encode("utf-8")) % vocab_size for g in grams]


def encode_batch_ngram(
    texts: Sequence[str],
    vocab_size: int,
    n_min: int = 3,
    n_max: int = 5,
    max_ngrams: int = 96,
) -> Tuple[torch.Tensor, torch.Tensor]:
    flat: List[int] = []
    offsets: List[int] = []
    cur = 0
    for s in texts:
        offsets.append(cur)
        ids = hash_text_to_buckets(s, vocab_size, n_min, n_max, max_ngrams)
        flat.extend(ids)
        cur += len(ids)
    return (
        torch.tensor(flat, dtype=torch.long),
        torch.tensor(offsets, dtype=torch.long),
    )


class HashedNgramEncoder(nn.Module):
    name = "ngram"

    def __init__(
        self,
        vocab_size: int = 1 << 20,
        embedding_dim: int = 256,
        n_min: int = 3,
        n_max: int = 5,
        max_ngrams: int = 96,
        sparse: bool = False,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.n_min = n_min
        self.n_max = n_max
        self.max_ngrams = max_ngrams
        self.bag = nn.EmbeddingBag(
            vocab_size, embedding_dim, mode="mean", sparse=sparse
        )
        nn.init.normal_(self.bag.weight, mean=0.0, std=0.1 / (embedding_dim ** 0.5))

    def forward(self, texts: Sequence[str]) -> torch.Tensor:
        flat, offsets = encode_batch_ngram(
            texts, self.vocab_size, self.n_min, self.n_max, self.max_ngrams
        )
        device = self.bag.weight.device
        return self.bag(flat.to(device), offsets.to(device))


# ---------------------------------------------------------------------------
# HuggingFace transformer encoder
# ---------------------------------------------------------------------------


class TransformerEncoder(nn.Module):
    """Wraps a HuggingFace AutoModel as a sentence encoder.

    Pooling: "mean" (attention-masked mean over token states) or "cls".
    Set `freeze=True` to disable gradients through the LM (use as a feature
    extractor + a learnable projection head trained downstream).
    """

    name = "transformer"

    def __init__(
        self,
        model_name: str = "bert-base-multilingual-cased",
        max_length: int = 128,
        pooling: str = "mean",
        freeze: bool = False,
    ):
        super().__init__()
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "TransformerEncoder requires the `transformers` package.\n"
                "Install it with:  pip install transformers"
            ) from e
        self.model_name = model_name
        self.max_length = max_length
        self.pooling = pooling
        self.freeze = freeze

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        except (ValueError, OSError):
            # Some older checkpoints ship only a slow tokenizer.
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        self.model = AutoModel.from_pretrained(model_name)
        self.embedding_dim: int = int(self.model.config.hidden_size)

        if freeze:
            for p in self.model.parameters():
                p.requires_grad_(False)
            self.model.eval()

    def train(self, mode: bool = True):  # type: ignore[override]
        super().train(mode)
        if self.freeze:
            self.model.eval()
        return self

    def _device(self) -> torch.device:
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def forward(self, texts: Sequence[str]) -> torch.Tensor:
        # Replace empty strings to avoid tokenizer edge cases.
        clean = [t if t else " " for t in texts]
        tok = self.tokenizer(
            clean,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        device = self._device()
        tok = {k: v.to(device) for k, v in tok.items()}

        ctx = torch.no_grad() if self.freeze else torch.enable_grad()
        with ctx:
            out = self.model(**tok)
            h = out.last_hidden_state  # [B, L, D]

            if self.pooling == "cls":
                pooled = h[:, 0]
            else:
                mask = tok["attention_mask"].unsqueeze(-1).to(h.dtype)
                pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

        # If frozen, detach so downstream autograd sees a leaf-like tensor.
        if self.freeze:
            pooled = pooled.detach()
        return pooled


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def make_encoder(
    name: str,
    *,
    # ngram args
    vocab_size: int = 1 << 20,
    embedding_dim: int = 256,
    n_min: int = 3,
    n_max: int = 5,
    max_ngrams: int = 96,
    # transformer args
    transformer_model: str = "bert-base-multilingual-cased",
    transformer_max_length: int = 128,
    transformer_pooling: str = "mean",
    freeze_encoder: bool = False,
) -> nn.Module:
    name = name.lower()
    if name in ("ngram", "hash", "hashed-ngram", "hashngram"):
        return HashedNgramEncoder(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            n_min=n_min,
            n_max=n_max,
            max_ngrams=max_ngrams,
        )
    if name in ("transformer", "bert", "hf"):
        return TransformerEncoder(
            model_name=transformer_model,
            max_length=transformer_max_length,
            pooling=transformer_pooling,
            freeze=freeze_encoder,
        )
    raise ValueError(f"Unknown encoder: {name!r} (use 'ngram' or 'transformer')")
