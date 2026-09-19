"""Scoring functions over (h, r, t) embeddings.

The scorer decides what "this triple is plausible" means once the encoder and
head have turned three strings into three vectors. DistMult was the only one
for most of this repo's life; this module makes it a choice.

THE INTERFACE IS SHAPED BY HOW EVALUATION HAS TO WORK, NOT BY THE MATH
Ranking a batch against a candidate pool means scoring ``[B]`` queries against
``[P]`` tails, and P is 6,000-2,000,000. Materialising ``[B, P, D]`` to apply an
arbitrary ``f(h, r, t)`` elementwise is not affordable, so every scorer here
must expose the two-stage form

    q = query(h, r)                # [B, Dq] — everything not involving t
    S = score_against(q, T)        # [B, P]  — one fused op against the bank

which is what makes ``hr @ pool.t()`` possible. A scorer that cannot be written
this way (a full bilinear ``h^T W_r t`` with a per-relation matrix, say) does
not belong here — it would silently turn a 2M-tail evaluation into an OOM.

Two families satisfy it:

* **inner-product** — ``S = q @ T.t()``. DistMult and ComplEx are both of this
  form; ComplEx only reorders the halves of the vector before the matmul.
* **distance** — ``S = -||q - T||``. TransE and RotatE. `torch.cdist` computes
  the whole matrix without expanding, so the cost is the same order as a matmul.

`score_diag` exists separately because the true tail is scored per-row, and
computing it as the diagonal of the full matrix would be quadratic waste. Note
the two paths can disagree in the last bits — `hr @ T.t()` and `(hr * t).sum()`
accumulate in different orders — which is exactly the ~1e-7 systematic offset
`eval._ranks_with_ties` had to grow a tolerance for.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

SCORERS = ("distmult", "complex", "transe", "rotate")
DEFAULT_SCORER = "distmult"


def _l2(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-6)


def _halves(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split the last dim into (real, imaginary) halves."""
    d = x.size(-1) // 2
    return x[..., :d], x[..., d:]


class Scorer(nn.Module):
    """Base class. Subclasses implement `query` and, if not inner-product,
    `score_against`.

    Stateless by design — every scorer here is a fixed formula over the
    embeddings, so all the parameters live in the encoder and head. That keeps
    a scorer swap from changing the parameter count, which is what lets the
    benchmark attribute a difference to the scoring function rather than to
    capacity.
    """

    kind: str = "inner"           # "inner" or "distance"
    needs_even_dim: bool = False  # complex-valued scorers split D in half

    def query(self, h: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def tail(self, t: torch.Tensor) -> torch.Tensor:
        """The tail representation the query is scored against.

        Usually ``t`` unchanged; ComplEx reorders it so that a plain matmul
        computes the real part of the Hermitian product.
        """
        return t

    def score_against(self, q: torch.Tensor, tails: torch.Tensor) -> torch.Tensor:
        """``[B, N]`` scores of each query against every tail in ``tails``."""
        return q @ tails.t()

    def score_diag(self, q: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """``[B]`` score of query i against tail i."""
        return (q * t).sum(dim=-1)

    # -- convenience ------------------------------------------------------
    def score(self, h: torch.Tensor, r: torch.Tensor,
              t: torch.Tensor) -> torch.Tensor:
        return self.score_diag(self.query(h, r), self.tail(t))

    def logits(self, h: torch.Tensor, r: torch.Tensor, t: torch.Tensor, *,
               normalize: bool = False) -> torch.Tensor:
        """``[B, B]`` in-batch matrix: row i is (h_i, r_i) against every tail.

        ``normalize`` L2-normalizes query and tails first, turning an
        inner-product score into a cosine. It is a **per-row positive
        rescaling only for inner-product scorers**, so it leaves their ranking
        untouched; for a distance scorer it changes the geometry, which is why
        `contrastive` is documented as an inner-product loss.
        """
        q, T = self.query(h, r), self.tail(t)
        if normalize:
            q, T = _l2(q), _l2(T)
        return self.score_against(q, T)


class DistMult(Scorer):
    """``score = sum_d h_d * r_d * t_d`` (Yang et al., 2015).

    The original and still the default. Symmetric in h and t, which is a real
    modelling limitation — it cannot represent an antisymmetric relation — and
    the reason ComplEx exists.
    """

    kind = "inner"

    def query(self, h: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        return h * r


class ComplEx(Scorer):
    """``score = Re(<h, r, conj(t)>)`` (Trouillon et al., 2016).

    The vector's two halves are read as the real and imaginary parts of a
    D/2-dimensional complex vector. Writing out the real part of the Hermitian
    product,

        Re<h,r,conj(t)> = <h_re*r_re - h_im*r_im, t_re>
                        + <h_re*r_im + h_im*r_re, t_im>

    which is a plain inner product between a query built from (h, r) and the
    *unmodified* concatenation of t's halves — so the efficient `q @ T.t()`
    path still applies and no reordering of `t` is needed after all.

    Unlike DistMult it is not symmetric in h and t, so it can express
    antisymmetric relations ("A inhibits B" without "B inhibits A"), at the
    cost of scoring in D/2 effective complex dimensions.
    """

    kind = "inner"
    needs_even_dim = True

    def query(self, h: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        h_re, h_im = _halves(h)
        r_re, r_im = _halves(r)
        return torch.cat([h_re * r_re - h_im * r_im,
                          h_re * r_im + h_im * r_re], dim=-1)


class TransE(Scorer):
    """``score = -||h + r - t||_2`` (Bordes et al., 2013).

    Translation in the embedding space: the relation is a displacement rather
    than a rescaling. A *distance*, so higher-is-better requires the minus, and
    the scale is unbounded below — which matters for the losses that read raw
    scores (`margin`, `self_adversarial`) and is why their `--margin` has to be
    tuned to the scorer, not just to the data.

    The full ``[B, P]`` matrix comes from `torch.cdist`, which does not expand
    to ``[B, P, D]``.
    """

    kind = "distance"

    def query(self, h: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        return h + r

    def score_against(self, q: torch.Tensor, tails: torch.Tensor) -> torch.Tensor:
        return -torch.cdist(q, tails, p=2)

    def score_diag(self, q: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return -(q - t).norm(dim=-1)


class RotatE(Scorer):
    """``score = -||h ∘ r - t||`` with ``|r_d| = 1`` (Sun et al., 2019).

    Rotation in complex space instead of translation. The unit-modulus
    constraint on the relation is what makes it a rotation rather than an
    arbitrary complex scaling, and it is imposed here by normalizing each
    complex component of r — not by a penalty, so it holds exactly rather than
    approximately.

    Like TransE this is a distance, and like ComplEx it reads the vector as
    D/2 complex numbers.
    """

    kind = "distance"
    needs_even_dim = True

    def query(self, h: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        h_re, h_im = _halves(h)
        r_re, r_im = _halves(r)
        # Per-component unit modulus: r/|r| elementwise over the complex pairs.
        mod = (r_re ** 2 + r_im ** 2).sqrt().clamp_min(1e-6)
        r_re, r_im = r_re / mod, r_im / mod
        return torch.cat([h_re * r_re - h_im * r_im,
                          h_re * r_im + h_im * r_re], dim=-1)

    def score_against(self, q: torch.Tensor, tails: torch.Tensor) -> torch.Tensor:
        return -torch.cdist(q, tails, p=2)

    def score_diag(self, q: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return -(q - t).norm(dim=-1)


_REGISTRY: Dict[str, type] = {
    "distmult": DistMult,
    "complex": ComplEx,
    "transe": TransE,
    "rotate": RotatE,
}


def make_scorer(name: Optional[str] = None) -> Scorer:
    key = (name or DEFAULT_SCORER).lower()
    if key not in _REGISTRY:
        raise SystemExit(
            f"Unknown scorer {name!r}. Choose from: {', '.join(SCORERS)}"
        )
    return _REGISTRY[key]()


def is_distance(name: Optional[str] = None) -> bool:
    """Whether the scorer returns a (negated) distance rather than a product.

    Callers that reason about score *scale* need this: a distance is unbounded
    below and its magnitude grows with the embedding norm, so a margin tuned
    for DistMult means something different here.
    """
    return make_scorer(name).kind == "distance"


def needs_even_dim(name: Optional[str] = None) -> bool:
    """Whether this scorer reads the vector as D/2 complex numbers."""
    return make_scorer(name).needs_even_dim
