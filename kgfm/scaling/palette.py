"""Viridis, without requiring matplotlib.

The scaling plots colour one line per model size, and the colour has to *mean*
size — a categorical palette cycling through six hues encodes rank at best and
nothing at all once it wraps. Viridis is perceptually uniform and monotone in
lightness, so "darker = smaller" survives greyscale printing and the common
forms of colour blindness.

`kgfm/charts.py` has three backends and only matplotlib could supply a
colormap, so the anchors are inlined and interpolated here instead. These are
the standard viridis control points at 0, 0.1, ... 1.0.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

_VIRIDIS: Tuple[Tuple[int, int, int], ...] = (
    (68, 1, 84), (72, 40, 120), (62, 74, 137), (49, 104, 142),
    (38, 130, 142), (31, 158, 137), (53, 183, 121), (109, 205, 89),
    (180, 222, 44), (253, 231, 37),
)


def viridis(t: float) -> str:
    """Hex colour at position ``t`` in [0, 1], linearly interpolated."""
    t = 0.0 if t < 0 else 1.0 if t > 1 else float(t)
    pos = t * (len(_VIRIDIS) - 1)
    i = min(int(pos), len(_VIRIDIS) - 2)
    frac = pos - i
    lo, hi = _VIRIDIS[i], _VIRIDIS[i + 1]
    rgb = tuple(round(a + (b - a) * frac) for a, b in zip(lo, hi))
    return "#%02x%02x%02x" % rgb


def by_size(values: Sequence[float]) -> List[str]:
    """A viridis colour per value, scaled on ``log10`` of the value.

    Log rather than linear because model sizes span decades: on a linear ramp
    a 4M and a 41M model would be almost the same colour while everything
    below 100M crowded into the dark end. With one distinct value the mid
    colour is used rather than an arbitrary end.
    """
    import math

    usable = [v for v in values if v and v > 0]
    if not usable:
        return [viridis(0.5)] * len(values)
    lo, hi = math.log10(min(usable)), math.log10(max(usable))
    if hi <= lo:
        return [viridis(0.5)] * len(values)
    out = []
    for v in values:
        if not v or v <= 0:
            out.append(viridis(0.5))
        else:
            out.append(viridis((math.log10(v) - lo) / (hi - lo)))
    return out
