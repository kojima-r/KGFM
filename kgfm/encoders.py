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

import contextlib
import zlib
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
        # Cumulative units fed to the encoder, for the FLOPs accounting in
        # kgfm/scaling. For the ngram encoder a "token" is one n-gram lookup;
        # it is a gather, not a matmul, so it is NOT comparable to a
        # transformer token — see scaling/compute.py.
        self.tokens_seen: int = 0
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
        self.tokens_seen += int(flat.numel())
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
        random_init: bool = False,
        config_overrides: Optional[Dict[str, Any]] = None,
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
        self.random_init = random_init
        self.config_overrides = dict(config_overrides or {})
        # Cumulative *padded* tokens through the LM. Measured rather than
        # estimated because padding is what the GPU actually pays for, and
        # `padding=True` makes it batch-dependent.
        self.tokens_seen: int = 0
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
        if random_init:
            from transformers import AutoConfig

            # Architecture from the config, weights from scratch. The
            # *tokenizer* still comes from the repo: the vocabulary is a fixed
            # design choice, not learned capacity, and re-deriving one per
            # size would vary two things at once.
            hf_cfg = AutoConfig.from_pretrained(model_name, **kw)
            # Architecture overrides let the size axis carry points that no
            # published checkpoint happens to sit on. Only meaningful with
            # random_init: there are no weights to be inconsistent with, so
            # changing depth or width is free. Unknown keys are rejected
            # rather than silently ignored, because a typo here would quietly
            # train the wrong size and label it as the intended one.
            for key, value in (config_overrides or {}).items():
                if not hasattr(hf_cfg, key):
                    raise ValueError(
                        f"config_overrides: {model_name} has no config field "
                        f"{key!r}"
                    )
                setattr(hf_cfg, key, value)
            if dropout is not None:
                for attr in ("hidden_dropout_prob", "attention_probs_dropout_prob",
                             "dropout", "attention_dropout"):
                    if hasattr(hf_cfg, attr):
                        setattr(hf_cfg, attr, dropout)
            self.model = AutoModel.from_config(hf_cfg)
        elif dropout is None:
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

        # BERT-family checkpoints carry a `pooler` (a Linear over the [CLS]
        # state) that this encoder never calls — pooling here is done over
        # `last_hidden_state`. Its parameters therefore never receive a
        # gradient, and under DDP that is fatal rather than merely wasteful:
        # `find_unused_parameters=False` (the default) raises "Expected to have
        # finished reduction in the prior iteration" on the second step.
        # Freezing is the precise fix — a parameter that cannot receive a
        # gradient cannot learn — and unlike `add_pooling_layer=False` it keeps
        # the weights in the state dict, so checkpoints stay loadable.
        pooler = getattr(self.model, "pooler", None)
        if pooler is not None:
            for p in pooler.parameters():
                p.requires_grad_(False)

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
        self.tokens_seen += int(tok["input_ids"].numel())
        tok = {k: v.to(device) for k, v in tok.items()}

        # A frozen encoder never needs a graph, so force it off. An unfrozen
        # one must *inherit* the ambient mode, not turn grad on: under
        # `torch.enable_grad()` this forward re-enabled recording inside every
        # `@torch.no_grad()` evaluation and built a full training-sized graph
        # for 3B sequences, which is the memory the eval path was actually
        # paying. Measured at B=256 on the eval-side encode_triple, peak
        # allocated: scratch-base 5.299 -> 0.293 GiB, scratch-xl 9.170 ->
        # 0.387, scratch-large 13.684 -> 0.387; scores bit-identical
        # (`torch.equal`-level, diff 0.0). Training is unaffected — it never
        # runs under an ambient no_grad, so nullcontext is the same thing
        # there. This is why `scratch-large` OOMed in its first in-loop
        # evaluation while training at the same batch size fit.
        ctx = torch.no_grad() if self.freeze else contextlib.nullcontext()
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
    # --- randomly initialised, for a genuine capacity axis. `random_init`
    # builds the architecture from its config and skips the pretrained
    # weights, so N means "how much capacity this model has" instead of "what
    # its pretraining gave it". That distinction is why the pretrained sweep
    # produced a power law with an exponent of only -0.018: across the
    # pretrained BERT family the loss level is set by transfer, not capacity,
    # and scaling laws describe from-scratch training. The configs are
    # borrowed from the same family so N lines up with the pretrained cells
    # and the two can be compared point for point. ---
    "scratch-tiny":      {"model": "prajjwal1/bert-tiny", "dim": 128,
                          "random_init": True},
    "scratch-mini":      {"model": "prajjwal1/bert-mini", "dim": 256,
                          "random_init": True},
    "scratch-small":     {"model": "prajjwal1/bert-small", "dim": 512,
                          "random_init": True},
    "scratch-medium":    {"model": "prajjwal1/bert-medium", "dim": 512,
                          "random_init": True},
    "scratch-base":      {"model": "bert-base-uncased", "dim": 768,
                          "random_init": True},
    # --- above BERT-base. An intermediate shape, added when bert-large's own
    # 24x1024 appeared not to fit; it does fit (see `scratch-large` below),
    # so this is now a genuine interior point of the size axis rather than a
    # substitute for one. `encode_triple` pushes 3B sequences per step, so
    # activation memory is ~3x what a plain LM of the same shape would need,
    # and B is the in-batch negative count so it cannot be reduced without
    # changing the task. Depth and width are raised together, because the
    # 2026-08-26 run showed the 4-layer cells were nearly degenerate in loss
    # while doubling depth moved it.
    "scratch-xl":        {"model": "bert-large-uncased", "dim": 1024,
                          "random_init": True,
                          "config_overrides": {"num_hidden_layers": 16}},
    # 24x1024 / 335M, and the real top of the axis on one H200. It used to
    # OOM in the first in-loop *evaluation* while training fine at the same
    # B=256, which read as an eval-memory limit; it was not. The eval forward
    # was building a training-sized autograd graph because
    # `TransformerEncoder.forward` forced `torch.enable_grad()` over the
    # ambient `no_grad`. With that fixed the whole cell — training, both
    # in-loop evals and the final test — runs end to end, verified.
    # Measured at B=256: train_peak 101.81 GiB, eval_peak 7.98 GiB of 139.8.
    # So it is TRAINING that binds now, and the next size up would need a
    # smaller B, which is not available (B is the in-batch negative count).
    "scratch-large":     {"model": "bert-large-uncased", "dim": 1024,
                          "random_init": True},
    # --- tiny, for the left end of a scaling study. Pretrained BERTs from the
    # same family (Turc et al. 2019), so size varies while the recipe does
    # not — which is what makes them a size *axis* rather than five unrelated
    # models. 4.4M to 41M fills the two decades below bert-base, where the
    # loss has not yet saturated and a scaling slope is still measurable. ---
    "bert-tiny":         {"model": "prajjwal1/bert-tiny", "dim": 128},
    "bert-mini":         {"model": "prajjwal1/bert-mini", "dim": 256},
    "bert-small":        {"model": "prajjwal1/bert-small", "dim": 512},
    "bert-medium":       {"model": "prajjwal1/bert-medium", "dim": 512},
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


def config_overrides(name: str) -> Dict[str, Any]:
    """Architecture overrides a preset applies to its HF config."""
    return dict(preset_info(name).get("config_overrides", {}) or {})


def is_random_init(name: str) -> bool:
    """Whether this preset builds from config instead of pretrained weights."""
    return bool(preset_info(name).get("random_init", False))


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
    random_init = is_random_init(name)
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
        random_init=random_init,
        config_overrides=config_overrides(name),
    )
