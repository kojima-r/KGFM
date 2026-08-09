"""kgfm's own ChEMBL benchmark. One module per step:

    prep       raw kgfm TSVs -> entity-ID KG the baselines can load
    sweep      kgfm cells (encoders x freezes x protocols)
    cell       one kgfm cell: train, evaluate, write JSON
    pipeline   run the above into one timestamped run directory

Parameters live in `config`; scale settings come from `benchmarks/*.yaml`. Run directories, logging
(`kgfm.runs`) and conda-env resolution (`kgfm.envs`) are shared with the
baseline commands, so they sit outside this package.

Deliberately *not* here: ULTRA / MOTIF (separate methods, separate commands
`kgfm-ultra` / `kgfm-motif`) and the comparison table (`kgfm report`). This
package only knows how to produce kgfm's own numbers.
"""

from .config import BenchConfig, available_configs, build_config, load_config_file

__all__ = [
    "BenchConfig",
    "available_configs",
    "build_config",
    "load_config_file",
]
