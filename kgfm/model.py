"""Triple scorer: text encoder -> projection head(s) -> scoring function.

Encoder-agnostic (anything mapping `Sequence[str] -> [B, D]`; see
`kgfm.encoders`), head-agnostic (`kgfm.heads`) and now scorer-agnostic
(`kgfm.scorers`). `proj_dim` sets the width the score is computed in, which
matters most with a frozen encoder — there the head is the only thing that
trains.

Three axes meet here and they are deliberately independent:

* **head** — what the projection *is* (linear, mlp, ...).
* **head_mode** — whether h, r and t share one head or get one each.
* **scorer** — what the three vectors are then combined by (DistMult,
  ComplEx, TransE, RotatE).

The class is still called `DistMultScorer` because checkpoints name it and
`state_dict` keys are load-bearing; with the defaults (`shared`, `distmult`)
it is bit-for-bit the module it always was, `proj.*` keys included.
`TripleScorer` is the honest alias for new code.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn

# Re-export for backward compatibility with earlier code.
from .encoders import HashedNgramEncoder, TransformerEncoder, make_encoder  # noqa: F401
from .heads import (DEFAULT_HEAD, DEFAULT_HEAD_MODE, HEAD_MODES, ROLES,
                    head_out_dim, make_head)
from .scorers import DEFAULT_SCORER, make_scorer


class TripleScorer(nn.Module):
    """Encode (h, r, t) as text, project, and score.

    h and t are L2-normalized when `normalize=True`; r is left unnormalized
    (its magnitude carries information about relation strength). That
    asymmetry predates the scorer registry and is kept for every scorer, so a
    scorer comparison is not also a normalization comparison.
    """

    def __init__(
        self,
        encoder: nn.Module,
        proj_dim: Optional[int] = None,
        normalize: bool = True,
        head_dropout: float = 0.0,
        head: str = DEFAULT_HEAD,
        head_mode: str = DEFAULT_HEAD_MODE,
        scorer: str = DEFAULT_SCORER,
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
        self.head_mode = (head_mode or DEFAULT_HEAD_MODE).lower()
        if self.head_mode not in HEAD_MODES:
            raise SystemExit(
                f"Unknown head_mode {head_mode!r}. "
                f"Choose from: {', '.join(HEAD_MODES)}"
            )
        self.scorer_name = (scorer or DEFAULT_SCORER).lower()
        self.scorer = make_scorer(self.scorer_name)
        self.dim = head_out_dim(head, in_dim, proj_dim)
        if self.scorer.needs_even_dim and self.dim % 2:
            raise SystemExit(
                f"scorer={self.scorer_name} reads the vector as {self.dim}/2 "
                f"complex numbers, so the scoring width must be even (got "
                f"{self.dim}). Pass an even --proj-dim."
            )
        # `auto` reproduces the original behaviour (Identity when the width
        # already matches, Linear otherwise); see kgfm/heads.py.
        #
        # In `shared` mode the head is stored as `self.proj`, which is the key
        # every existing checkpoint uses — so old checkpoints keep loading.
        # `separate` builds three and is a new set of keys by construction.
        if self.head_mode == "shared":
            self.proj: nn.Module = make_head(
                head, in_dim, proj_dim, dropout=self.head_dropout
            )
        else:
            self.proj_by_role = nn.ModuleDict({
                role: make_head(head, in_dim, proj_dim,
                                dropout=self.head_dropout)
                for role in ROLES
            })

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

    def _head(self, role: str) -> nn.Module:
        return self.proj if self.head_mode == "shared" \
            else self.proj_by_role[role]

    def encode(self, texts: Sequence[str], role: str = "t") -> torch.Tensor:
        """Encode and project a batch of strings **in a given role**.

        `role` defaults to "t" because every bulk caller outside training is
        building a bank of candidate *tails* (`eval.build_candidate_pool`,
        `build_filter_index`). Under `head_mode=shared` the argument does
        nothing; under `separate` it decides which head runs, and getting it
        wrong would score a query against tails projected by the wrong matrix —
        silently, with plausible-looking numbers.
        """
        # Dropout sits between the two regularized halves: on the encoder's
        # output, before the head consumes it.
        return self._head(role)(self.drop(self.encoder(texts)))

    def encode_triple(
        self, h_text: Sequence[str], r_text: Sequence[str], t_text: Sequence[str]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Single batched call to the encoder for the entire (h, r, t) bundle.
        # This is a 3x speedup for transformer encoders versus three forwards.
        B = len(h_text)
        all_text = list(h_text) + list(r_text) + list(t_text)
        # One encoder forward for the whole bundle; the heads are applied
        # afterwards, per slice. Separate heads therefore cost head parameters
        # and three small matmuls, not a third encoder pass.
        emb = self.drop(self.encoder(all_text))
        if self.head_mode == "shared":
            emb = self.proj(emb)
            h, r, t = emb[:B], emb[B:2 * B], emb[2 * B:3 * B]
        else:
            h = self.proj_by_role["h"](emb[:B])
            r = self.proj_by_role["r"](emb[B:2 * B])
            t = self.proj_by_role["t"](emb[2 * B:3 * B])
        h = self._maybe_norm(h, self.normalize)
        t = self._maybe_norm(t, self.normalize)
        return h, r, t

    def score(self, h: torch.Tensor, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.scorer.score(h, r, t)

    def query(self, h: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        """The half of the score that does not involve t — see kgfm/scorers.py."""
        return self.scorer.query(h, r)

    def score_against(self, q: torch.Tensor, tails: torch.Tensor) -> torch.Tensor:
        """``[B, N]`` scores of queries against a bank of already-projected tails."""
        return self.scorer.score_against(q, self.scorer.tail(tails))

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


# The class was called DistMultScorer for this repo's whole history and the
# name is written into every checkpoint's reconstruction path. Keep it.
DistMultScorer = TripleScorer
