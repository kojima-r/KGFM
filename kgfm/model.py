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
from .heads import DEFAULT_HEAD, head_out_dim, make_head


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
        head_dropout: float = 0.0,
        head: str = DEFAULT_HEAD,
    ):
        super().__init__()
        self.encoder = encoder
        self.normalize = normalize
        # Regularizes the *coupling* between encoder and score, deliberately a
        # separate knob from the encoder's own dropout: with a frozen encoder
        # this head is the only thing that trains, and with a fine-tuned one
        # the two halves overfit at different rates.
        self.head_dropout = float(head_dropout)
        self.drop: nn.Module = (
            nn.Dropout(self.head_dropout) if self.head_dropout > 0 else nn.Identity()
        )
        in_dim = int(getattr(encoder, "embedding_dim"))
        self.head = head
        # `auto` reproduces the original behaviour (Identity when the width
        # already matches, Linear otherwise); see kgfm/heads.py.
        self.proj: nn.Module = make_head(
            head, in_dim, proj_dim, dropout=self.head_dropout
        )
        self.dim = head_out_dim(head, in_dim, proj_dim)

    def head_parameters(self):
        """Everything that is not the encoder — the projection head.

        Kept next to the module that owns it so the optimizer's parameter
        groups stay correct if the head ever grows past a single Linear.
        """
        encoder_ids = {id(p) for p in self.encoder.parameters()}
        return [p for p in self.parameters() if id(p) not in encoder_ids]

    @staticmethod
    def _maybe_norm(x: torch.Tensor, do: bool) -> torch.Tensor:
        if not do:
            return x
        return x / (x.norm(dim=-1, keepdim=True).clamp_min(1e-6))

    def encode(self, texts: Sequence[str]) -> torch.Tensor:
        # Dropout sits between the two regularized halves: on the encoder's
        # output, before the head consumes it.
        return self.proj(self.drop(self.encoder(texts)))

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
        self,
        h_text: Sequence[str],
        r_text: Sequence[str],
        t_text: Sequence[str],
        return_embeddings: bool = False,
    ):
        """Score the triples, or return the (h, r, t) embeddings behind them.

        `return_embeddings` exists for the training loss, which needs all three
        embeddings to build its [B, B] in-batch score matrix. It must be reached
        **through this forward** rather than by calling `encode_triple` on the
        module directly: under DDP, `DistributedDataParallel.forward` is what
        calls `reducer.prepare_for_backward()`, and without that call no
        gradient all-reduce happens at all and each rank silently trains its own
        model. See `train.in_batch_negative_loss`.
        """
        h, r, t = self.encode_triple(h_text, r_text, t_text)
        if return_embeddings:
            return h, r, t
        return self.score(h, r, t)
