"""kgfm — Knowledge-graph foundation model with streaming text-relation prediction.

Public API:
    DistMultScorer          — DistMult-style scorer over a text encoder.
    HashedNgramEncoder      — trainable char-n-gram hashed encoder.
    TransformerEncoder      — HuggingFace AutoModel wrapper.
    make_encoder            — encoder factory (``"ngram"`` or ``"transformer"``).
    StreamingTripleDataset  — streaming TSV iterable dataset.
    collate_triples         — DataLoader collate function.
    evaluate                — tail-prediction MRR / Hit@k / nDCG.
    TrainConfig, train      — training entrypoint.
"""

from .encoders import HashedNgramEncoder, TransformerEncoder, make_encoder
from .model import DistMultScorer
from .data import StreamingTripleDataset, collate_triples
from .eval import evaluate
from .train import TrainConfig, train

__version__ = "0.1.0"

__all__ = [
    "DistMultScorer",
    "HashedNgramEncoder",
    "TransformerEncoder",
    "make_encoder",
    "StreamingTripleDataset",
    "collate_triples",
    "evaluate",
    "TrainConfig",
    "train",
    "__version__",
]
