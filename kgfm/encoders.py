"""Pluggable text encoders.

All encoders implement the same contract:

    forward(texts: Sequence[str]) -> Tensor of shape [B, embedding_dim]

Two implementations are provided:

- HashedNgramEncoder: trainable char-n-gram hashing (FastText-style). No deps.
- TransformerEncoder: HuggingFace AutoModel/AutoTokenizer wrapper (BERT etc.).
  Requires `transformers` to be installed. Pass `freeze=True` to use it as a
  fixed feature extractor; otherwise it fine-tunes end-to-end.
"""

from __future__ import annotations

import zlib
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Hashed n-gram encoder
# ---------------------------------------------------------------------------


def extract_ngrams(text: str, n_min: int = 3, n_max: int = 5, max_ngrams: int = 96) -> List[str]:
    s = "^" + text + "$"
    out: List[str] = []
    L = len(s)
    for n in range(n_min, n_max + 1):
        if L < n:
            continue
        for i in range(L - n + 1):
            out.append(s[i : i + n])
            if len(out) >= max_ngrams:
                return out
    if not out:
        out.append(s)
    return out


def hash_text_to_buckets(
    text: str, vocab_size: int, n_min: int = 3, n_max: int = 5, max_ngrams: int = 96
) -> List[int]:
    grams = extract_ngrams(text, n_min, n_max, max_ngrams)
    return [zlib.crc32(g.encode("utf-8")) % vocab_size for g in grams]


def encode_batch_ngram(
    texts: Sequence[str],
    vocab_size: int,
    n_min: int = 3,
    n_max: int = 5,
    max_ngrams: int = 96,
) -> Tuple[torch.Tensor, torch.Tensor]:
    flat: List[int] = []
    offsets: List[int] = []
    cur = 0
    for s in texts:
        offsets.append(cur)
        ids = hash_text_to_buckets(s, vocab_size, n_min, n_max, max_ngrams)
        flat.extend(ids)
        cur += len(ids)
    return (
        torch.tensor(flat, dtype=torch.long),
        torch.tensor(offsets, dtype=torch.long),
    )


class HashedNgramEncoder(nn.Module):
    name = "ngram"

    def __init__(
        self,
        vocab_size: int = 1 << 20,
        embedding_dim: int = 256,
        n_min: int = 3,
        n_max: int = 5,
        max_ngrams: int = 96,
        sparse: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.n_min = n_min
        self.n_max = n_max
        self.max_ngrams = max_ngrams
        self.dropout = float(dropout)
        self.bag = nn.EmbeddingBag(
            vocab_size, embedding_dim, mode="mean", sparse=sparse
        )
        nn.init.normal_(self.bag.weight, mean=0.0, std=0.1 / (embedding_dim ** 0.5))
        self.drop: nn.Module = (
            nn.Dropout(self.dropout) if self.dropout > 0 else nn.Identity()
        )

    def forward(self, texts: Sequence[str]) -> torch.Tensor:
        flat, offsets = encode_batch_ngram(
            texts, self.vocab_size, self.n_min, self.n_max, self.max_ngrams
        )
        device = self.bag.weight.device
        return self.drop(self.bag(flat.to(device), offsets.to(device)))


# ---------------------------------------------------------------------------
# HuggingFace transformer encoder
# ---------------------------------------------------------------------------


class TransformerEncoder(nn.Module):
    """Wraps a HuggingFace AutoModel as a sentence encoder.

    Pooling: "mean" (attention-masked mean over token states) or "cls".
    Set `freeze=True` to disable gradients through the LM (use as a feature
    extractor + a learnable projection head trained downstream).
    """

    name = "transformer"

    def __init__(
        self,
        model_name: str = "bert-base-multilingual-cased",
        max_length: int = 128,
        pooling: str = "mean",
        freeze: bool = False,
        dropout: Optional[float] = None,
        trust_remote_code: bool = False,
        load_dtype: Optional[str] = None,
    ):
        super().__init__()
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "TransformerEncoder requires the `transformers` package.\n"
                "Install it with:  pip install transformers"
            ) from e
        self.model_name = model_name
        self.max_length = max_length
        self.pooling = pooling
        self.freeze = freeze
        self.dropout = dropout
        self.trust_remote_code = trust_remote_code
        self.load_dtype = load_dtype
        kw = {"trust_remote_code": True} if trust_remote_code else {}
        model_kw = dict(kw)
        if load_dtype is not None:
            # Only ever set for frozen_only presets: casting weights that are
            # about to be updated would change the optimization, but weights
            # that are never updated lose nothing and cost half the memory.
            # `dtype` since transformers 4.5x; `torch_dtype` is deprecated.
            model_kw["dtype"] = getattr(torch, load_dtype)

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, **kw)
        except (ValueError, OSError):
            # Some older checkpoints ship only a slow tokenizer.
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, use_fast=False, **kw
            )
        if self.tokenizer.pad_token is None:
            # Decoder-derived embedding models (e5-mistral, gte-Qwen2) ship no
            # PAD token, and batching is impossible without one. EOS is the
            # conventional stand-in; the attention mask keeps it out of the
            # mean pooling either way.
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Dropout has to be set at construction: it lives in the HF config and
        # the layers read it when they are built. None keeps the pretrained
        # model's own value (0.1 for BERT), which is why this is Optional
        # rather than defaulting to 0.
        if dropout is None:
            self.model = AutoModel.from_pretrained(model_name, **model_kw)
        else:
            from transformers import AutoConfig

            # Every architecture spells these differently (BERT/RoBERTa use
            # hidden_dropout_prob, DistilBERT uses dropout, ...), so set
            # whichever the loaded config actually has rather than passing
            # BERT's names as kwargs and crashing on anything else.
            hf_cfg = AutoConfig.from_pretrained(model_name, **kw)
            known = ("hidden_dropout_prob", "attention_probs_dropout_prob",
                     "dropout", "attention_dropout")
            for attr in known:
                if hasattr(hf_cfg, attr):
                    setattr(hf_cfg, attr, dropout)
            self.model = AutoModel.from_pretrained(
                model_name, config=hf_cfg, **model_kw
            )
        self.embedding_dim: int = int(self.model.config.hidden_size)

        if freeze:
            for p in self.model.parameters():
                p.requires_grad_(False)
            self.model.eval()

    def train(self, mode: bool = True):  # type: ignore[override]
        super().train(mode)
        if self.freeze:
            self.model.eval()
        return self

    def _device(self) -> torch.device:
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def forward(self, texts: Sequence[str]) -> torch.Tensor:
        # Replace empty strings to avoid tokenizer edge cases.
        clean = [t if t else " " for t in texts]
        tok = self.tokenizer(
            clean,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        device = self._device()
        tok = {k: v.to(device) for k, v in tok.items()}

        ctx = torch.no_grad() if self.freeze else torch.enable_grad()
        with ctx:
            out = self.model(**tok)
            h = out.last_hidden_state  # [B, L, D]

            if self.pooling == "cls":
                pooled = h[:, 0]
            else:
                mask = tok["attention_mask"].unsqueeze(-1).to(h.dtype)
                pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

        # If frozen, detach so downstream autograd sees a leaf-like tensor.
        if self.freeze:
            pooled = pooled.detach()
        if self.load_dtype is not None:
            # An encoder loaded in bf16 emits bf16, but the projection head and
            # the scorer are fp32, and evaluation runs *outside* autocast — so
            # without this the head's Linear raises "mat1 and mat2 must have
            # the same dtype" as soon as the first eval pass starts. Every
            # encoder returns fp32; reduced precision is an internal detail of
            # how the weights happen to be stored.
            pooled = pooled.float()
        return pooled


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


# Named pretrained encoders, so a benchmark can sweep `encoders: [bge-large,
# e5-large, ...]` and get one cell per model with a readable tag. The value is
# the HuggingFace id; `trust_remote` marks models whose architecture only
# exists in their own repo (a real supply-chain decision, so it is explicit
# rather than blanket-enabled).
#
# `dim` is the encoder's hidden size, recorded here only so a config can be
# sanity-checked without downloading gigabytes. It is never used to build the
# model — that always comes from the loaded config.
# Every entry below was loaded and run a forward pass in this environment
# before being listed. Two models were tried and deliberately left out,
# because a preset that always crashes is worse than no preset:
#   microsoft/deberta-v3-large   its sentencepiece tokenizer cannot be built by
#                                the installed transformers (AttributeError on a
#                                None vocab_file) on the fast *or* slow path.
#   Alibaba-NLP/gte-Qwen2-7B-instruct
#                                its bundled modeling_qwen.py calls
#                                past_key_values.get_usable_length(), which
#                                DynamicCache no longer has.
# Both are upstream/version incompatibilities, not configuration mistakes.
ENCODER_PRESETS: dict = {
    # --- baselines / small ---
    "bert-multilingual": {"model": "bert-base-multilingual-cased", "dim": 768},
    "mpnet":             {"model": "sentence-transformers/all-mpnet-base-v2", "dim": 768},
    # --- strong mid-size retrieval encoders (fine-tunable on one H200) ---
    "bge-large":         {"model": "BAAI/bge-large-en-v1.5", "dim": 1024},
    "e5-large":          {"model": "intfloat/e5-large-v2", "dim": 1024},
    "gte-large":         {"model": "thenlper/gte-large", "dim": 1024},
    "xlm-roberta-large": {"model": "xlm-roberta-large", "dim": 1024},
    # --- 7B-class embedding models, marked `frozen_only`. encode_triple pushes
    # 3B sequences through the encoder per step, so fine-tuning one at any
    # useful batch size does not fit on a single H200 — and these are trained
    # to be used as-is anyway. `load_dtype` halves the 28 GB an fp32 load would
    # take; safe precisely because the weights are never updated. ---
    "e5-mistral-7b":     {"model": "intfloat/e5-mistral-7b-instruct", "dim": 4096,
                          "frozen_only": True, "load_dtype": "bfloat16"},
}


def preset_info(name: str) -> dict:
    """The preset dict for an encoder name, or {} if it is not a preset."""
    return ENCODER_PRESETS.get(name.lower(), {})


def is_frozen_only(name: str) -> bool:
    """Encoders too large to fine-tune — the sweep skips their freeze=off cell."""
    return bool(preset_info(name).get("frozen_only", False))


def resolve_encoder(name: str) -> Tuple[str, Optional[str], bool, Optional[str]]:
    """(kind, hf_model_id, trust_remote_code, load_dtype) for an encoder name.

    ``kind`` is "ngram" or "transformer". A preset name resolves to a
    transformer with that model id; "transformer" itself keeps whatever
    --transformer-model says.
    """
    key = name.lower()
    if key in ("ngram", "hash", "hashed-ngram", "hashngram"):
        return "ngram", None, False, None
    if key in ENCODER_PRESETS:
        preset = ENCODER_PRESETS[key]
        return ("transformer", preset["model"],
                bool(preset.get("trust_remote", False)),
                preset.get("load_dtype"))
    if key in ("transformer", "bert", "hf"):
        return "transformer", None, False, None
    raise SystemExit(
        f"Unknown encoder {name!r}.\n"
        f"Use 'ngram', 'transformer' (with --transformer-model), or a preset: "
        f"{', '.join(sorted(ENCODER_PRESETS))}"
    )


def is_transformer(name: str) -> bool:
    """Whether this encoder name is a pretrained LM.

    Used for the fine-tuning learning rate and for deciding which cells the
    freeze ablation applies to — both of which would silently do the wrong
    thing if a preset name were treated as an unknown/ngram encoder.
    """
    try:
        return resolve_encoder(name)[0] == "transformer"
    except SystemExit:
        return False


def make_encoder(
    name: str,
    *,
    # ngram args
    vocab_size: int = 1 << 20,
    embedding_dim: int = 256,
    n_min: int = 3,
    n_max: int = 5,
    max_ngrams: int = 96,
    # transformer args
    transformer_model: str = "bert-base-multilingual-cased",
    transformer_max_length: int = 128,
    transformer_pooling: str = "mean",
    freeze_encoder: bool = False,
    # Regularization inside the encoder. None = keep the pretrained model's own
    # value; the head's dropout is a separate knob on DistMultScorer.
    encoder_dropout: Optional[float] = None,
) -> nn.Module:
    kind, preset_model, trust_remote, load_dtype = resolve_encoder(name)
    if kind == "ngram":
        return HashedNgramEncoder(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            n_min=n_min,
            n_max=n_max,
            max_ngrams=max_ngrams,
            dropout=encoder_dropout or 0.0,
        )
    return TransformerEncoder(
        # A preset wins over --transformer-model: the preset *is* the choice
        # the cell tag names, so honouring a stale flag would silently
        # benchmark a different model than the one being reported.
        model_name=preset_model or transformer_model,
        max_length=transformer_max_length,
        pooling=transformer_pooling,
        freeze=freeze_encoder,
        dropout=encoder_dropout,
        trust_remote_code=trust_remote,
        load_dtype=load_dtype,
    )
