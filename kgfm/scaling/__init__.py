"""Scaling-law analysis over kgfm benchmark runs.

`kgfm bench run` does the training; this package only re-reads its logs and
re-expresses them in (compute, loss) coordinates. Keeping it separate means a
scaling study needs no new training code — just a config that sweeps model
sizes and a report that knows what a FLOP is.
"""

from .compute import PF_DAY, CellCompute
from .points import ScalingPoint, ScalingSeries, collect, fit_power_law, frontier

__all__ = [
    "PF_DAY", "CellCompute", "ScalingPoint", "ScalingSeries",
    "collect", "fit_power_law", "frontier",
]
