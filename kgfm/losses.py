"""Link-prediction training objectives over in-batch negatives.

Every loss here scores a batch of triples against itself: for row *i* the
positive is ``t_i`` and the negatives are the other ``B-1`` tails in the batch.
So the batch size *is* the negative count — see CLAUDE.md.

Choosing between them
---------------------
``contrastive`` (the default) is InfoNCE / NT-Xent: cosine similarities divided
by an explicit temperature. It exists because the alternative,
``softmax_ce``, has no scale control at all. ``DistMultScorer`` normalizes h
and t but deliberately leaves r unnormalized, so the logit scale is whatever
``‖h*r‖`` happens to grow to — measured at 22.3 logit std after 60k steps on a
fine-tuned transformer, which drove validation cross-entropy to 13.2 while the
ranking was fine (median rank 12 of 512). Dividing those same logits by a
constant brought the loss to 5.3 without changing a single rank.

The fix is to normalize the query ``h*r`` per row and set the scale explicitly.
**Normalizing per row cannot change any ranking**: it divides every candidate
score for that row by the same positive number, so MRR / Hit@k are identical
and only the loss geometry changes.

The others are provided for comparison, and are *not* drop-in — ``margin`` and
``self_adversarial`` operate on raw, unnormalized scores and need their
``margin`` tuned to that scale.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

LOSSES = ("contrastive", "softmax_ce", "bce", "margin", "self_adversarial")
DEFAULT_LOSS = "contrastive"
# Measured, on a 1500-step ngram probe at B=512 (valid loss / valid MRR):
#   tau=0.05 -> 6.39 / 0.816      tau=0.1 -> 5.64 / 0.816
# Equal ranking, but 0.1 keeps the loss clearly below the ln(B)=6.24 that a
# random model scores, so the number stays interpretable. Related work spans
# this range (SimCSE 0.05, CLIP 0.07 then learned).
DEFAULT_TEMPERATURE = 0.1


def _l2(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-6)


def in_batch_logits(
    h: torch.Tensor, r: torch.Tensor, t: torch.Tensor, *, normalize: bool = False
) -> torch.Tensor:
    """``[B, B]`` score matrix: row i is (h_i, r_i) scored against every tail.

    With ``normalize`` the query and the tails are L2-normalized first, making
    every entry a cosine similarity in [-1, 1]. That is a per-row positive
    rescaling, so it leaves the induced ranking untouched.
    """
    q = h * r
    if normalize:
        return _l2(q) @ _l2(t).t()
    return q @ t.t()


def _targets(logits: torch.Tensor) -> torch.Tensor:
    return torch.arange(logits.size(0), device=logits.device)


def _offdiag_mask(logits: torch.Tensor) -> torch.Tensor:
    return torch.eye(logits.size(0), device=logits.device, dtype=torch.bool)


def duplicate_tail_mask(
    t_text: Sequence[str], device: Optional[torch.device] = None
) -> torch.Tensor:
    """``[B, B]`` mask of *false* negatives: cell (i, j) where j != i but the
    two rows share the same tail string.

    In-batch corruption assumes every other tail is wrong, but a batch drawn
    from a real KG repeats tails constantly — measured on ChEMBL at B=512,
    48% of train rows and 56% of valid rows share their tail with another row
    in the same batch. Those "negatives" encode to the *identical* vector as
    the positive, so no model can separate them and the loss inherits a floor
    of ``E[log multiplicity]`` — 1.14 nats on train, 1.52 on valid. Masking
    them removes that floor and makes the loss comparable to the pooled eval
    protocol, which deduplicates its candidate pool for the same reason.

    The diagonal is always False: row i's own tail is the positive, not a
    false negative.
    """
    lookup: dict = {}
    ids = [lookup.setdefault(s, len(lookup)) for s in t_text]
    idx = torch.as_tensor(ids, device=device)
    same = idx.unsqueeze(0) == idx.unsqueeze(1)
    return same & ~torch.eye(len(ids), device=idx.device, dtype=torch.bool)


def _apply_mask(logits: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    """Drop false negatives out of the softmax by sending them to -inf."""
    if mask is None:
        return logits
    return logits.masked_fill(mask.to(logits.device), float("-inf"))


def contrastive(
    h: torch.Tensor, r: torch.Tensor, t: torch.Tensor, *,
    temperature: float = DEFAULT_TEMPERATURE, label_smoothing: float = 0.0,
    false_negative_mask: Optional[torch.Tensor] = None, **_: object,
) -> torch.Tensor:
    """InfoNCE / NT-Xent over in-batch negatives (the default).

    Softmax cross-entropy over *cosine* similarities scaled by 1/temperature,
    so the sharpness of the distribution is a hyperparameter instead of an
    emergent property of ‖r‖.
    """
    logits = _apply_mask(in_batch_logits(h, r, t, normalize=True) / temperature,
                         false_negative_mask)
    return F.cross_entropy(logits, _targets(logits), label_smoothing=label_smoothing)


def softmax_ce(
    h: torch.Tensor, r: torch.Tensor, t: torch.Tensor, *,
    label_smoothing: float = 0.0,
    false_negative_mask: Optional[torch.Tensor] = None, **_: object,
) -> torch.Tensor:
    """Softmax cross-entropy on raw scores — no normalization, no temperature.

    Kept because it is what every result before this module was trained with,
    and because the contrast makes the calibration problem visible.
    """
    logits = _apply_mask(in_batch_logits(h, r, t), false_negative_mask)
    return F.cross_entropy(logits, _targets(logits), label_smoothing=label_smoothing)


def bce(
    h: torch.Tensor, r: torch.Tensor, t: torch.Tensor, *,
    label_smoothing: float = 0.0,
    false_negative_mask: Optional[torch.Tensor] = None, **_: object,
) -> torch.Tensor:
    """Binary cross-entropy per candidate — the "1-N scoring" of ConvE.

    Each of the B^2 cells is an independent yes/no decision, so the positives
    are outnumbered B-1 to 1; label smoothing (ConvE uses 0.1) is the usual
    counterweight.

    False negatives cannot be sent to -inf here — every cell is its own term,
    not a softmax competitor — so they are dropped from the mean instead.
    """
    logits = in_batch_logits(h, r, t)
    target = torch.eye(logits.size(0), device=logits.device, dtype=logits.dtype)
    if label_smoothing:
        target = target * (1.0 - label_smoothing) + label_smoothing / logits.size(0)
    per_cell = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    if false_negative_mask is None:
        return per_cell.mean()
    keep = ~false_negative_mask.to(logits.device)
    return (per_cell * keep).sum() / keep.sum().clamp_min(1)


def margin_ranking(
    h: torch.Tensor, r: torch.Tensor, t: torch.Tensor, *,
    margin: float = 1.0,
    false_negative_mask: Optional[torch.Tensor] = None, **_: object,
) -> torch.Tensor:
    """Max-margin hinge: every negative must sit ``margin`` below the positive.

    The mean is over the off-diagonal pairs only — including the always-zero
    diagonal would dilute the loss by a factor of B/(B-1). False negatives are
    excluded from both the numerator and that pair count.
    """
    logits = in_batch_logits(h, r, t)
    pos = logits.diagonal().unsqueeze(1)
    drop = _offdiag_mask(logits)
    if false_negative_mask is not None:
        drop = drop | false_negative_mask.to(logits.device)
    per_pair = F.relu(margin - pos + logits).masked_fill(drop, 0.0)
    n_pairs = int((~drop).sum())
    return per_pair.sum() / max(n_pairs, 1)


def self_adversarial(
    h: torch.Tensor, r: torch.Tensor, t: torch.Tensor, *,
    margin: float = 1.0, adversarial_temperature: float = 1.0,
    false_negative_mask: Optional[torch.Tensor] = None, **_: object,
) -> torch.Tensor:
    """RotatE's self-adversarial negative sampling.

    Negatives are weighted by how hard they are — the weights are a softmax
    over the negatives' own scores, and are detached because they are sample
    weights, not part of the objective (Sun et al., 2019).
    """
    logits = in_batch_logits(h, r, t)
    mask = _offdiag_mask(logits)
    if false_negative_mask is not None:
        # Excluded from the softmax *and* from the summed negative term: a
        # false negative is the hardest cell in its row, so leaving it in
        # would hand it nearly all the adversarial weight.
        mask = mask | false_negative_mask.to(logits.device)
    pos = logits.diagonal()
    neg = logits.masked_fill(mask, float("-inf"))
    weight = F.softmax(neg * adversarial_temperature, dim=1).detach()
    # A row whose every negative was masked away softmaxes to NaN; it has no
    # negative term to contribute, so zero it out.
    weight = torch.nan_to_num(weight, nan=0.0)
    # Positive should score above -margin, negatives below it.
    pos_term = F.logsigmoid(margin + pos)
    neg_term = (weight * F.logsigmoid(-margin - neg.masked_fill(mask, 0.0))).sum(1)
    return -(pos_term + neg_term).mean()


_REGISTRY = {
    "contrastive": contrastive,
    "softmax_ce": softmax_ce,
    "bce": bce,
    "margin": margin_ranking,
    "self_adversarial": self_adversarial,
}


def compute_loss(name: str, h: torch.Tensor, r: torch.Tensor, t: torch.Tensor,
                 **kwargs: object) -> torch.Tensor:
    """Dispatch to one of LOSSES. Extra kwargs are ignored by losses that
    don't take them, so callers can pass the whole hyperparameter set."""
    try:
        fn = _REGISTRY[name]
    except KeyError:
        raise SystemExit(
            f"Unknown loss {name!r}. Choose from: {', '.join(LOSSES)}"
        ) from None
    return fn(h, r, t, **kwargs)
