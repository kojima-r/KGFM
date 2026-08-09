"""ULTRA / MOTIF — external KG foundation models compared against kgfm.

These are separate methods with their own commands (`kgfm-ultra`,
`kgfm-motif`), deliberately outside `kgfm bench`: they share nothing with
kgfm's model or training path. What they do share is the *run directory* —
they read the entity-ID KG that `kgfm bench prep` builds and drop a JSON
record that `kgfm report` collects into one table.
"""

from .common import Baseline

__all__ = ["Baseline"]
