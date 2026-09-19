"""Projection heads: the layer between the text encoder and the DistMult score.

The head is the only trainable part of a frozen-encoder run and a small
fraction of a fine-tuned one, so it is worth being able to swap. All heads map
``[B, in_dim] -> [B, out_dim]``.

Two ways to wire them to the triple, chosen with ``head_mode``: ``shared``
(one head for h, r and t, which is what ``encode`` did for this repo's whole
history) or ``separate`` (one head per role). See ``HEAD_MODES`` below for why
neither is obviously right.

Choosing between them
---------------------
``auto`` is the default and reproduces exactly what every result before this
module was trained with. ``identity`` is only meaningful when the encoder's own
dimension is already the scoring dimension; with a frozen encoder it leaves
**zero trainable parameters**, which `train()` detects and warns about. ``mlp``
and
``residual_mlp`` add capacity where it is cheap — over a frozen 7B encoder the
head is the entire model, and a single Linear is a hard ceiling on what the
comparison can show.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

HEADS = ("auto", "identity", "linear", "mlp", "residual_mlp")

# Whether h, r and t go through the same head or one each.
#
# `shared` is what this repo did for its entire history, and it is not an
# arbitrary default: the encoder forward is already shared (encode_triple
# bundles h+r+t into one 3B-length batch for a 3x speedup), so sharing the head
# too keeps the three roles in one representation space. That is what makes
# `hr @ pool.t()` meaningful — the pool is encoded once, as tails, and the
# query lives in the same space.
#
# `separate` gives each role its own parameters. The argument for it is that
# the three roles are genuinely different: h and t are entities, r is a
# relation, and DistMult's h/t symmetry means a shared head cannot even learn
# to treat head-position and tail-position differently. The argument against is
# 3x the head parameters and, for a frozen encoder, 3x the only thing that
# trains — so a difference between the modes is not purely architectural
# unless the comparison holds the parameter budget in mind. The benchmark in
# benchmark_scorer/ reports head parameter counts next to the metrics for
# exactly that reason.
HEAD_MODES = ("shared", "separate")
DEFAULT_HEAD_MODE = "shared"
ROLES = ("h", "r", "t")
# "auto" is what DistMultScorer always did: a Linear when proj_dim actually
# changes the width, nn.Identity when it does not. It stays the default so
# every result produced before this module is reproducible unchanged — note
# that `auto` + a frozen encoder + no proj_dim is the zero-trainable-parameter
# trap, which is exactly why the explicit names exist.
DEFAULT_HEAD = "auto"


class MLPHead(nn.Module):
    """Linear -> GELU -> Dropout -> Linear.

    The hidden width follows the *input* dimension rather than the output: the
    point of a projection is usually to compress (1024 -> 256), and sizing the
    hidden layer off the narrow end would throw the information away before the
    non-linearity ever sees it.
    """

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0,
                 hidden_dim: Optional[int] = None):
        super().__init__()
        hidden = hidden_dim or in_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualMLPHead(nn.Module):
    """Pre-norm residual block, then project.

    ``LayerNorm -> Linear -> GELU -> Dropout -> Linear`` added back to the
    input, followed by the output projection. The residual path means the head
    starts close to the encoder's own representation instead of a random
    rotation of it, which matters most when the encoder is frozen and the head
    is all there is to train.
    """

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0,
                 hidden_dim: Optional[int] = None):
        super().__init__()
        hidden = hidden_dim or in_dim
        self.norm = nn.LayerNorm(in_dim)
        self.block = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden, in_dim),
        )
        self.out = nn.Linear(in_dim, out_dim) if out_dim != in_dim else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(x + self.block(self.norm(x)))


def make_head(
    name: str, in_dim: int, out_dim: Optional[int] = None, *, dropout: float = 0.0
) -> nn.Module:
    """Build a head. ``out_dim=None`` (or == in_dim) means "keep the width".

    Returns the module and leaves it to the caller to record the resulting
    dimension — read it off ``head_out_dim`` rather than assuming.
    """
    name = (name or DEFAULT_HEAD).lower()
    if name not in HEADS:
        raise SystemExit(
            f"Unknown head {name!r}. Choose from: {', '.join(HEADS)}"
        )
    target = in_dim if out_dim is None else int(out_dim)

    if name == "auto":
        return nn.Identity() if target == in_dim else nn.Linear(in_dim, target)
    if name == "identity":
        if target != in_dim:
            raise SystemExit(
                f"head=identity cannot change the dimension "
                f"({in_dim} -> {target}). Use --head linear, or drop --proj-dim."
            )
        return nn.Identity()
    if name == "linear":
        # A Linear that neither compresses nor expands is still a learnable
        # rotation, so it is NOT folded away into Identity here — that
        # collapse is what left frozen-encoder runs with no parameters.
        return nn.Linear(in_dim, target)
    if name == "mlp":
        return MLPHead(in_dim, target, dropout=dropout)
    return ResidualMLPHead(in_dim, target, dropout=dropout)


def head_out_dim(name: str, in_dim: int, out_dim: Optional[int]) -> int:
    """The scoring dimension a head produces."""
    if (name or DEFAULT_HEAD).lower() == "identity" or out_dim is None:
        return in_dim
    return int(out_dim)


def is_trainable(name: str, in_dim: int, out_dim: Optional[int]) -> bool:
    """Whether this head has parameters — the frozen-encoder sanity check."""
    name = (name or DEFAULT_HEAD).lower()
    if name == "identity":
        return False
    if name == "auto":
        return out_dim is not None and int(out_dim) != in_dim
    return True
