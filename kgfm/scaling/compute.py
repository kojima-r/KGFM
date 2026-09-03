"""Turning a kgfm training run into FLOPs, and FLOPs into PF-days.

A scaling-law plot needs an x-axis of *compute*, and compute has to be derived
rather than measured — no counter on the GPU tells you how many useful FLOPs a
step cost. The standard accounting (Kaplan et al. 2020, appendix F) is

    forward   ~ 2 * N * T          FLOPs
    backward  ~ 4 * N * T          FLOPs   (two of the three matmul products)
    total     ~ 6 * N * T

for a dense transformer with ``N`` parameters processing ``T`` tokens. The
factor of 2 is multiply-and-add; backward is twice forward because it computes
gradients with respect to both activations and weights.

What that means *here*, where the model is a text encoder under a DistMult
scorer rather than a language model:

* **T is measured, not assumed.** ``encode_triple`` pushes ``3B`` sequences per
  step (h, r and t together) and the tokenizer pads to the longest in the
  batch, so tokens-per-step is data-dependent. The trainer counts the real
  padded token total and logs it as ``tokens=``; this module just reads it.
  Deriving T as ``3 * B * max_length`` instead would overcount by ~3.5x on
  ChEMBL, where sequences pad to ~36 of the 128 allowed.

* **A frozen encoder costs 2NT, not 6NT.** ``requires_grad=False`` means no
  backward through the LM at all — only the head gets gradients. Treating a
  frozen 7B encoder as if it cost 6NT would overstate its compute threefold and
  put it in the wrong place on every plot.

* **The projection head is counted separately** because it consumes *vectors*
  (3B of them per step), not tokens. It is small enough to be a rounding error
  for a 335M encoder and the entire cost for a frozen one, so it is not
  optional.

* **ngram is not on this axis.** ``HashedNgramEncoder`` is an EmbeddingBag: its
  268M parameters are a lookup table, and a gather does no multiply-accumulate
  proportional to N. The 6ND model would score it as the most expensive model
  in the sweep when it is by far the cheapest. `flops_per_step` returns the
  head cost only for it, and `is_compute_comparable` says so, so the report can
  mark it rather than quietly plotting a wrong point.

PF-days is the unit Kaplan et al. use: ``1 PF-day = 1e15 * 86400 = 8.64e19``
FLOPs, i.e. a petaflop-per-second machine running for a day.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

PF_DAY = 1e15 * 86400.0          # 8.64e19 FLOPs

# Multiply-accumulate is 2 FLOPs; backward costs twice the forward pass.
FLOPS_FWD_PER_PARAM_TOKEN = 2.0
FLOPS_BWD_PER_PARAM_TOKEN = 4.0


@dataclass(frozen=True)
class CellCompute:
    """The compute profile of one training cell."""

    encoder: str
    encoder_params: int          # parameters in the text encoder
    head_params: int             # parameters in the projection head
    frozen: bool                 # encoder had requires_grad=False throughout
    is_transformer: bool         # False for ngram (see module docstring)

    @property
    def trainable_params(self) -> int:
        return self.head_params if self.frozen else self.encoder_params + self.head_params

    @property
    def comparable(self) -> bool:
        """Whether a FLOPs axis means anything for this cell."""
        return self.is_transformer

    def flops(self, tokens: int, vectors: int) -> float:
        """FLOPs for ``tokens`` encoder tokens and ``vectors`` head inputs.

        ``tokens`` and ``vectors`` are cumulative totals across all ranks.
        """
        total = 0.0
        if self.is_transformer:
            per = (FLOPS_FWD_PER_PARAM_TOKEN if self.frozen
                   else FLOPS_FWD_PER_PARAM_TOKEN + FLOPS_BWD_PER_PARAM_TOKEN)
            total += per * self.encoder_params * tokens
        # The head always trains, so it always pays the backward pass.
        total += (FLOPS_FWD_PER_PARAM_TOKEN + FLOPS_BWD_PER_PARAM_TOKEN) * \
            self.head_params * vectors
        return total

    def pf_days(self, tokens: int, vectors: int) -> float:
        return self.flops(tokens, vectors) / PF_DAY


def head_params(params_total: int, params_trainable: int, frozen: bool) -> int:
    """Split the logged parameter counts into encoder vs head.

    `train()` logs ``params total=`` and ``trainable=``. For a frozen encoder
    everything trainable *is* the head, which is the only case where the split
    is recoverable from the log alone; otherwise the head is a rounding error
    on the total and is reported as 0 rather than guessed.
    """
    if frozen:
        return max(0, int(params_trainable))
    return 0


def encoder_params(params_total: int, head: int) -> int:
    return max(0, int(params_total) - int(head))


def vectors_per_step(global_batch_size: int) -> int:
    """Head inputs per optimizer step: h, r and t for every row in the batch."""
    return 3 * int(global_batch_size)


def format_pf_days(v: float) -> str:
    if v <= 0:
        return "0"
    if v < 1e-3:
        return f"{v:.2e}"
    if v < 1:
        return f"{v:.4f}"
    return f"{v:,.2f}"
