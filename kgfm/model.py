"""DistMult-style scorer over an arbitrary text encoder.

The scorer is encoder-agnostic: any module that maps `Sequence[str] -> [B, D]`
will work. See `kgfm.encoders` for ngram and transformer implementations.

An optional `proj_dim` adds a shared `Linear` head used for h, r, and t — useful
when the encoder is frozen (e.g., a frozen BERT) or when you want to score in a
smaller dimension than the encoder's hidden size.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn

# Re-export for backward compatibility with earlier code.
from .encoders import HashedNgramEncoder, TransformerEncoder, make_encoder  # noqa: F401


class DistMultScorer(nn.Module):
    """score(h, r, t) = sum(h_d * r_d * t_d).

    h and t are L2-normalized when `normalize=True`; r is left unnormalized
    (its magnitude carries information about relation strength).
    """

    def __init__(
        self,
        encoder: nn.Module,
        proj_dim: Optional[int] = None,
        normalize: bool = True,
    ):
        super().__init__()
        self.encoder = encoder
        self.normalize = normalize
        in_dim = int(getattr(encoder, "embedding_dim"))
        if proj_dim is None or proj_dim == in_dim:
            self.proj: nn.Module = nn.Identity()
            self.dim = in_dim
        else:
            self.proj = nn.Linear(in_dim, proj_dim)
            self.dim = proj_dim

    @staticmethod
    def _maybe_norm(x: torch.Tensor, do: bool) -> torch.Tensor:
        if not do:
            return x
        return x / (x.norm(dim=-1, keepdim=True).clamp_min(1e-6))

    def encode(self, texts: Sequence[str]) -> torch.Tensor:
        return self.proj(self.encoder(texts))

    def encode_triple(
        self, h_text: Sequence[str], r_text: Sequence[str], t_text: Sequence[str]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Single batched call to the encoder for the entire (h, r, t) bundle.
        # This is a 3x speedup for transformer encoders versus three forwards.
        B = len(h_text)
        all_text = list(h_text) + list(r_text) + list(t_text)
        emb = self.encode(all_text)
        h = emb[:B]
        r = emb[B : 2 * B]
        t = emb[2 * B : 3 * B]
        h = self._maybe_norm(h, self.normalize)
        t = self._maybe_norm(t, self.normalize)
        return h, r, t

    def score(self, h: torch.Tensor, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return (h * r * t).sum(dim=-1)

    def forward(
        self, h_text: Sequence[str], r_text: Sequence[str], t_text: Sequence[str]
    ) -> torch.Tensor:
        h, r, t = self.encode_triple(h_text, r_text, t_text)
        return self.score(h, r, t)
