#!/usr/bin/env bash
# bootstrap_chembl.sh — Run the full kgfm vs ULTRA vs MOTIF benchmark on
# ChEMBL end-to-end and record everything under
#   benchmarks/results/chembl/<UTC timestamp>/
#
# kgfm is swept across all combinations of protocols × encoders so the
# comparison table includes every variant side-by-side. Each kgfm
# encoder is trained once; evaluation is then repeated per protocol on
# the same checkpoint (the protocol only affects scoring).
#
# A `latest` symlink in the same directory always points at the most
# recent run. Each run dir contains:
#   meta.json                          run parameters + host/git/torch info
#   chembl_kg_stats.json               copy of prepared KG stats
#   kgfm_<protocol>_<encoder>[_frozen].json
#                                      one file per kgfm sweep cell; the
#                                      "_frozen" suffix is added when the
#                                      cell was run with --freeze-encoder.
#   ultra.json                         ULTRA metrics record
#   motif.json                         MOTIF metrics record
#   table.md                           aggregated comparison table
#   run.log                            combined log
#   <step>.log                         per-step logs
#   kgfm_ckpts_<encoder>[_frozen]/     kgfm checkpoints (one tree per cell)
#
# Setup steps (clone, conda env) are idempotent and re-run cheaply, so
# the only skip flags are for the per-method work and the data prep.
#
# Usage:
#   bash benchmarks/bootstrap_chembl.sh [flags]
#
# Common flags:
#   --max-train N          chembl train triple cap   (default 50000)
#   --max-valid N                                    (default 2000)
#   --max-test  N                                    (default 2000)
#   --max-steps N          kgfm training steps       (default 200)
#   --batch-size N         kgfm per-device micro-batch size (default 256).
#                          Backwards-compatible alias for
#                          --per-device-train-batch-size; under DDP this
#                          is per GPU, not global.
#   --per-device-train-batch-size N
#                          HF-style per-GPU micro-batch size. When set,
#                          overrides --batch-size. Effective global batch
#                          size = per_device_train * kgfm_nproc *
#                          gradient_accumulation_steps.
#   --per-device-eval-batch-size N
#                          Per-GPU eval batch size, used by the in-loop
#                          validation and final test pass. Defaults to
#                          max(64, per_device_train // 2).
#   --gradient-accumulation-steps N
#                          Number of micro-batches accumulated per
#                          optimizer step (default 1).
#   --kgfm-nproc N         How many GPUs (= processes) to launch per kgfm
#                          cell via torchrun. Default 1 (no DDP). When
#                          >1 the script invokes
#                          `torchrun --nproc-per-node=N run_kgfm.py ...`.
#   --kgfm-master-port P   torchrun --master-port. Default 29500.
#   --transformer-batch-size N
#                          Override --batch-size only for transformer-encoder
#                          cells (BERT-base full fine-tune cannot fit B=1024
#                          on a single GPU; default = same as --batch-size).
#   --proj-dim N           Add a learnable Linear projection of this size
#                          before scoring. Required when --kgfm-freezes
#                          contains "on" (with proj_dim=None and a frozen
#                          encoder, the optimizer has zero trainable params).
#                          A no-op for ngram when N equals its embedding_dim.
#   --kgfm-protocols LIST  Comma-separated kgfm final-eval protocols
#                          (default "pooled,filtered"). Each value must
#                          be one of pooled|filtered.
#   --kgfm-encoders  LIST  Comma-separated kgfm encoders
#                          (default "ngram,transformer"). Each value is
#                          forwarded to run_kgfm.py --encoder.
#   --kgfm-freezes   LIST  Comma-separated freeze modes (default "off").
#                          Each value is one of off|on; "on" forwards
#                          --freeze-encoder to run_kgfm.py. Only meaningful
#                          for transformer encoders — for ngram the "on"
#                          variant is silently skipped.
#   --max-filter-tails N   filtered protocol vocab cap    (default 50000)
#   --max-filter-rows  N   filtered protocol row cap      (default 1000000)
#   --ultra-gpus  "..."    GPU JSON list for ULTRA   (default "null" — CPU,
#                                                    works around the
#                                                    sm_90 rspmm bug)
#   --motif-gpus  "..."    GPU JSON list for MOTIF   (default "[0]")
#   --ultra-ckpt PATH      ULTRA checkpoint path
#   --motif-ckpt PATH      MOTIF checkpoint path
#   --skip-prep            Reuse the existing benchmarks/chembl_kg/
#   --skip-kgfm            Skip every kgfm sweep cell
#   --skip-ultra           Skip run_ultra.py
#   --skip-motif           Skip run_motif.py
#   --skip-aggregate       Skip aggregate.py
#   --resume [TS|PATH]     Resume an interrupted run instead of starting a
#                          fresh one. Reuses the existing results directory
#                          (and its meta.json / checkpoints) and skips any
#                          step whose output file is already present.
#                          - With no argument: resumes the run pointed to by
#                            benchmarks/results/chembl/latest.
#                          - With a bare timestamp (e.g. 20260507T155834Z):
#                            resumes benchmarks/results/chembl/<TS>.
#                          - With an absolute or relative path: resumes that
#                            directory directly.
#                          Per-step resume rules:
#                            prep    skipped if chembl_kg_stats.json exists
#                            kgfm    a cell's protocol is skipped if its
#                                    kgfm_*.json exists; otherwise the
#                                    first protocol that runs for the cell
#                                    is launched with --resume so train.py
#                                    auto-loads the latest checkpoint
#                                    (final.pt > last.pt > best.pt) and
#                                    continues training up to --max-steps.
#                                    Already-finished training (saved step
#                                    >= max-steps) skips the loop and just
#                                    re-runs eval. Resume applies to kgfm
#                                    only — ULTRA / MOTIF have no training
#                                    state and are still pure file-skip.
#                            ultra   skipped if ultra.json exists
#                            motif   skipped if motif.json exists
#                            aggregate always re-runs (cheap, picks up new files)
#   -h, --help             Show this header.
set -eo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # benchmarks/
ROOT="$(cd "$HERE/.." && pwd)"
cd "$ROOT"

# -------------------------------- defaults --------------------------------
MAX_TRAIN=50000
MAX_VALID=2000
MAX_TEST=2000
MAX_STEPS=200
BATCH_SIZE=256
PER_DEVICE_TRAIN_BS=""      # empty means: use $BATCH_SIZE
PER_DEVICE_EVAL_BS=""       # empty means: max(64, per_device_train_bs // 2)
GRAD_ACCUM_STEPS=1
KGFM_NPROC=1                # 1 = single-process; >1 = launch via torchrun
KGFM_MASTER_PORT=29500
TRANSFORMER_BATCH_SIZE=""   # empty means: use $BATCH_SIZE for transformer cells too
PROJ_DIM=""                 # empty means: don't pass --proj-dim (run_kgfm.py default: None)
KGFM_PROTOCOLS="pooled,filtered"
KGFM_ENCODERS="ngram,transformer"
KGFM_FREEZES="off"
MAX_FILTER_TAILS=50000
MAX_FILTER_ROWS=1000000
ULTRA_GPUS=null
MOTIF_GPUS="[0]"
ULTRA_CKPT="$HERE/ULTRA/ckpts/ultra_50g.pth"
MOTIF_CKPT="$HERE/MOTIF/ckpts/motif_3g.pth"
SKIP_PREP=0
SKIP_KGFM=0
SKIP_ULTRA=0
SKIP_MOTIF=0
SKIP_AGGREGATE=0
RESUME=0
RESUME_TARGET=""

show_help() { sed -n '2,/^set -eo pipefail/p' "$0" | sed 's/^# //; s/^#//' | head -n -1; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --max-train)         MAX_TRAIN=$2; shift 2 ;;
        --max-valid)         MAX_VALID=$2; shift 2 ;;
        --max-test)          MAX_TEST=$2; shift 2 ;;
        --max-steps)         MAX_STEPS=$2; shift 2 ;;
        --batch-size)        BATCH_SIZE=$2; shift 2 ;;
        --per-device-train-batch-size) PER_DEVICE_TRAIN_BS=$2; shift 2 ;;
        --per-device-eval-batch-size)  PER_DEVICE_EVAL_BS=$2; shift 2 ;;
        --gradient-accumulation-steps) GRAD_ACCUM_STEPS=$2; shift 2 ;;
        --kgfm-nproc)        KGFM_NPROC=$2; shift 2 ;;
        --kgfm-master-port)  KGFM_MASTER_PORT=$2; shift 2 ;;
        --transformer-batch-size) TRANSFORMER_BATCH_SIZE=$2; shift 2 ;;
        --proj-dim)          PROJ_DIM=$2; shift 2 ;;
        --kgfm-protocols)    KGFM_PROTOCOLS=$2; shift 2 ;;
        --kgfm-encoders)     KGFM_ENCODERS=$2; shift 2 ;;
        --kgfm-freezes)      KGFM_FREEZES=$2; shift 2 ;;
        --max-filter-tails)  MAX_FILTER_TAILS=$2; shift 2 ;;
        --max-filter-rows)   MAX_FILTER_ROWS=$2; shift 2 ;;
        --ultra-gpus)        ULTRA_GPUS=$2; shift 2 ;;
        --motif-gpus)        MOTIF_GPUS=$2; shift 2 ;;
        --ultra-ckpt)        ULTRA_CKPT=$2; shift 2 ;;
        --motif-ckpt)        MOTIF_CKPT=$2; shift 2 ;;
        --skip-prep)         SKIP_PREP=1; shift ;;
        --skip-kgfm)         SKIP_KGFM=1; shift ;;
        --skip-ultra)        SKIP_ULTRA=1; shift ;;
        --skip-motif)        SKIP_MOTIF=1; shift ;;
        --skip-aggregate)    SKIP_AGGREGATE=1; shift ;;
        --resume)
            RESUME=1
            # Accept an optional argument. Anything that doesn't start with "--"
            # is treated as the resume target (timestamp or path); otherwise
            # we default to "latest".
            if [[ $# -ge 2 && "$2" != --* ]]; then
                RESUME_TARGET=$2; shift 2
            else
                RESUME_TARGET="latest"; shift
            fi
            ;;
        -h|--help)           show_help; exit 0 ;;
        *) echo "Unknown flag: $1" >&2; echo "Try --help" >&2; exit 1 ;;
    esac
done

# ------------------------------- run dir ---------------------------------
if [[ $RESUME -eq 1 ]]; then
    # Resolve RESUME_TARGET into an absolute OUT_DIR. Accepted forms:
    #   "latest"                              -> follow the latest symlink
    #   "20260507T155834Z"                    -> $HERE/results/chembl/<ts>
    #   "/abs/path" or "rel/path"             -> taken verbatim
    if [[ "$RESUME_TARGET" == /* || "$RESUME_TARGET" == */* ]]; then
        OUT_DIR="$RESUME_TARGET"
    else
        OUT_DIR="$HERE/results/chembl/$RESUME_TARGET"
    fi
    if [[ ! -d "$OUT_DIR" ]]; then
        echo "Resume target not found: $OUT_DIR" >&2
        exit 1
    fi
    # Follow symlinks so $TS reflects the underlying directory name.
    OUT_DIR=$(cd "$OUT_DIR" && pwd -P)
    TS=$(basename "$OUT_DIR")
else
    TS=$(date -u +%Y%m%dT%H%M%SZ)
    OUT_DIR="$HERE/results/chembl/$TS"
    mkdir -p "$OUT_DIR"
    ln -sfn "$TS" "$HERE/results/chembl/latest"
fi

LOG="$OUT_DIR/run.log"
log()       { printf '[%(%H:%M:%S)T] %s\n' -1 "$*" | tee -a "$LOG" ; }
log_quiet() { printf '[%(%H:%M:%S)T] %s\n' -1 "$*" >> "$LOG" ; }

# ------------------------------ metadata ---------------------------------
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    GIT_REV=$(git rev-parse HEAD)
    GIT_DIRTY=$(git status --porcelain | wc -l | tr -d '[:space:]')
else
    GIT_REV="n/a"
    GIT_DIRTY=0
fi
HOSTNAME=$(hostname)
GPU_LINE=$(nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null \
    | head -1 || echo "n/a")
PY_VERSION=$(python --version 2>&1)
TORCH_VERSION=$(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "n/a")

IFS=',' read -ra KGFM_PROTOCOL_ARR <<< "$KGFM_PROTOCOLS"
IFS=',' read -ra KGFM_ENCODER_ARR  <<< "$KGFM_ENCODERS"
IFS=',' read -ra KGFM_FREEZE_ARR   <<< "$KGFM_FREEZES"

for f in "${KGFM_FREEZE_ARR[@]}"; do
    case "$f" in
        off|on) ;;
        *) echo "Invalid --kgfm-freezes value: '$f' (use off|on)" >&2; exit 1 ;;
    esac
done

# Render the bash arrays as JSON arrays for meta.json.
json_array() {
    local first=1
    printf '['
    for v in "$@"; do
        [[ $first -eq 0 ]] && printf ', '
        printf '"%s"' "$v"
        first=0
    done
    printf ']'
}
KGFM_PROTOCOLS_JSON=$(json_array "${KGFM_PROTOCOL_ARR[@]}")
KGFM_ENCODERS_JSON=$(json_array "${KGFM_ENCODER_ARR[@]}")
KGFM_FREEZES_JSON=$(json_array "${KGFM_FREEZE_ARR[@]}")

if [[ $RESUME -eq 0 || ! -f "$OUT_DIR/meta.json" ]]; then
    cat > "$OUT_DIR/meta.json" <<EOF
{
  "benchmark": "chembl",
  "timestamp_utc": "$TS",
  "host": "$HOSTNAME",
  "git_rev": "$GIT_REV",
  "git_dirty_files": $GIT_DIRTY,
  "python": "$PY_VERSION",
  "torch": "$TORCH_VERSION",
  "gpu": "$GPU_LINE",
  "params": {
    "max_train": $MAX_TRAIN,
    "max_valid": $MAX_VALID,
    "max_test": $MAX_TEST,
    "max_steps": $MAX_STEPS,
    "batch_size": $BATCH_SIZE,
    "per_device_train_batch_size": "${PER_DEVICE_TRAIN_BS:-null}",
    "per_device_eval_batch_size": "${PER_DEVICE_EVAL_BS:-null}",
    "gradient_accumulation_steps": $GRAD_ACCUM_STEPS,
    "kgfm_nproc": $KGFM_NPROC,
    "transformer_batch_size": "${TRANSFORMER_BATCH_SIZE:-$BATCH_SIZE}",
    "proj_dim": "${PROJ_DIM:-null}",
    "kgfm_protocols": $KGFM_PROTOCOLS_JSON,
    "kgfm_encoders": $KGFM_ENCODERS_JSON,
    "kgfm_freezes": $KGFM_FREEZES_JSON,
    "max_filter_tails": $MAX_FILTER_TAILS,
    "max_filter_rows": $MAX_FILTER_ROWS,
    "ultra_gpus": "$ULTRA_GPUS",
    "motif_gpus": "$MOTIF_GPUS",
    "ultra_ckpt": "$ULTRA_CKPT",
    "motif_ckpt": "$MOTIF_CKPT"
  }
}
EOF
fi

if [[ $RESUME -eq 1 ]]; then
    log "==> bootstrap_chembl.sh (RESUME mode)"
else
    log "==> bootstrap_chembl.sh"
fi
log "    OUT_DIR    = $OUT_DIR"
log "    GPU        = $GPU_LINE"
log "    torch      = $TORCH_VERSION  python = $PY_VERSION"
log "    git_rev    = $GIT_REV (dirty=$GIT_DIRTY)"
log "    params     = max_train=$MAX_TRAIN max_steps=$MAX_STEPS batch_size=$BATCH_SIZE"
log "    transformer_batch_size=${TRANSFORMER_BATCH_SIZE:-$BATCH_SIZE} proj_dim=${PROJ_DIM:-<unset>}"
log "    per_device_train_batch_size=${PER_DEVICE_TRAIN_BS:-<unset>} per_device_eval_batch_size=${PER_DEVICE_EVAL_BS:-<unset>}"
log "    gradient_accumulation_steps=$GRAD_ACCUM_STEPS kgfm_nproc=$KGFM_NPROC master_port=$KGFM_MASTER_PORT"
log "    kgfm sweep = protocols=[${KGFM_PROTOCOLS}] encoders=[${KGFM_ENCODERS}] freezes=[${KGFM_FREEZES}]"
log ""

# ------------------------------- helpers ---------------------------------
# Wraps a step so its stdout/stderr lands in both run.log and a per-step log
# while still streaming to the terminal. Returns the step's exit status.
run_step() {
    local name="$1"; shift
    local step_log="$OUT_DIR/${name}.log"
    log "==> $name"
    set +e
    "$@" 2>&1 | tee "$step_log" | tee -a "$LOG"
    local rc=${PIPESTATUS[0]}
    set -e
    if [[ $rc -ne 0 ]]; then
        log "    !! $name exited $rc — continuing"
    fi
    return $rc
}

# ------------------------ 1) clone ULTRA + MOTIF -------------------------
run_step setup bash "$HERE/setup.sh" || true

# ------------------------ 2) ULTRA conda env -----------------------------
if [[ $SKIP_ULTRA -eq 0 ]]; then
    run_step setup_ultra_env bash "$HERE/setup_ultra_env.sh" || true
else
    log "skip setup_ultra_env (--skip-ultra)"
fi

# ----------------------- 3) prepare ChEMBL KG ----------------------------
if [[ $RESUME -eq 1 && -f "$OUT_DIR/chembl_kg_stats.json" ]]; then
    log "skip prepare_chembl_kg (resume: $OUT_DIR/chembl_kg_stats.json exists)"
elif [[ $SKIP_PREP -eq 0 ]]; then
    rm -rf "$HERE/chembl_kg/processed" "$HERE/chembl_kg/processed_motif" 2>/dev/null || true
    run_step prepare_chembl_kg python "$HERE/prepare_chembl_kg.py" \
        --max-train "$MAX_TRAIN" \
        --max-valid "$MAX_VALID" \
        --max-test "$MAX_TEST" \
        --out-dir "$HERE/chembl_kg" || true
    cp -f "$HERE/chembl_kg/stats.json" "$OUT_DIR/chembl_kg_stats.json" 2>/dev/null || true
else
    log "skip prepare_chembl_kg (--skip-prep) — using existing $HERE/chembl_kg"
    cp -f "$HERE/chembl_kg/stats.json" "$OUT_DIR/chembl_kg_stats.json" 2>/dev/null || true
fi

# --------------------------- 4) kgfm sweep -------------------------------
# For each encoder we train once, then re-evaluate the resulting checkpoint
# under each requested protocol (the protocol only affects scoring, not
# training). Output filenames follow kgfm_<protocol>_<encoder>.json so the
# aggregate step can render one row per cell.
KGFM_FILES=()
if [[ $SKIP_KGFM -eq 0 ]]; then
    for encoder in "${KGFM_ENCODER_ARR[@]}"; do
        for freeze in "${KGFM_FREEZE_ARR[@]}"; do
            # Freezing the ngram bag has no effect (no LM to freeze) and would
            # produce a duplicate cell, so we silently skip it.
            if [[ "$encoder" == "ngram" && "$freeze" == "on" ]]; then
                log "skip kgfm cell encoder=ngram freeze=on (no LM to freeze)"
                continue
            fi
            tag="$encoder"
            [[ "$freeze" == "on" ]] && tag="${encoder}_frozen"
            ckpt_dir="$OUT_DIR/kgfm_ckpts_${tag}"
            # Per-encoder batch-size override (BERT full FT cannot fit large B).
            cell_batch_size=$BATCH_SIZE
            if [[ "$encoder" != "ngram" && -n "$TRANSFORMER_BATCH_SIZE" ]]; then
                cell_batch_size=$TRANSFORMER_BATCH_SIZE
            fi
            # Within one (encoder, freeze) cell every protocol shares the
            # same checkpoint — only the first protocol that actually runs
            # triggers training (or resumes it under --resume); subsequent
            # protocols re-evaluate the produced ckpt with --skip-train.
            # `cell_trained_yet` tracks whether the per-cell training pass
            # has already been launched in this script invocation.
            cell_trained_yet=0
            for protocol in "${KGFM_PROTOCOL_ARR[@]}"; do
                out_name="kgfm_${protocol}_${tag}.json"
                step_name="run_kgfm_${protocol}_${tag}"
                if [[ $RESUME -eq 1 && -f "$OUT_DIR/$out_name" ]]; then
                    log "skip $step_name (resume: $out_name exists)"
                    KGFM_FILES+=("$out_name")
                    continue
                fi
                extra_flags=()
                if [[ "$freeze" == "on" ]]; then
                    extra_flags+=("--freeze-encoder")
                fi
                if [[ -n "$PROJ_DIM" ]]; then
                    extra_flags+=("--proj-dim" "$PROJ_DIM")
                fi
                if [[ -n "$PER_DEVICE_TRAIN_BS" ]]; then
                    extra_flags+=("--per-device-train-batch-size" "$PER_DEVICE_TRAIN_BS")
                fi
                if [[ -n "$PER_DEVICE_EVAL_BS" ]]; then
                    extra_flags+=("--per-device-eval-batch-size" "$PER_DEVICE_EVAL_BS")
                fi
                if [[ "$GRAD_ACCUM_STEPS" != "1" ]]; then
                    extra_flags+=("--gradient-accumulation-steps" "$GRAD_ACCUM_STEPS")
                fi
                launcher=(python)
                if [[ $cell_trained_yet -eq 1 ]]; then
                    # An earlier protocol already trained this cell.
                    # Just re-score on the resulting ckpt.
                    extra_flags+=("--skip-train")
                else
                    # First protocol to run for this cell — owns the
                    # training pass. In resume mode, ask run_kgfm.py to
                    # pick up any existing ckpt and continue from its
                    # saved step up to --max-steps; if no ckpt is there
                    # train.py just trains from scratch as usual.
                    if [[ $RESUME -eq 1 ]]; then
                        extra_flags+=("--resume")
                    fi
                    if [[ $KGFM_NPROC -gt 1 ]]; then
                        launcher=(torchrun
                                  --standalone
                                  --nproc-per-node="$KGFM_NPROC"
                                  --master-port="$KGFM_MASTER_PORT")
                    fi
                    cell_trained_yet=1
                fi
                run_step "$step_name" "${launcher[@]}" "$HERE/run_kgfm.py" \
                    --encoder "$encoder" \
                    --max-steps "$MAX_STEPS" \
                    --batch-size "$cell_batch_size" \
                    --protocol "$protocol" \
                    --max-filter-tails "$MAX_FILTER_TAILS" \
                    --max-filter-rows "$MAX_FILTER_ROWS" \
                    --ckpt-dir "$ckpt_dir" \
                    --out "$OUT_DIR/$out_name" \
                    "${extra_flags[@]}" || true
                if [[ -f "$OUT_DIR/$out_name" ]]; then
                    KGFM_FILES+=("$out_name")
                fi
            done
        done
    done
else
    log "skip run_kgfm sweep (--skip-kgfm)"
fi

# --------------------------- 5) ULTRA ------------------------------------
if [[ $RESUME -eq 1 && -f "$OUT_DIR/ultra.json" ]]; then
    log "skip run_ultra (resume: ultra.json exists)"
elif [[ $SKIP_ULTRA -eq 0 ]]; then
    run_step run_ultra python "$HERE/run_ultra.py" \
        --gpus "$ULTRA_GPUS" \
        --ckpt "$ULTRA_CKPT" \
        --out "$OUT_DIR/ultra.json" || true
else
    log "skip run_ultra (--skip-ultra)"
fi

# --------------------------- 6) MOTIF ------------------------------------
if [[ $RESUME -eq 1 && -f "$OUT_DIR/motif.json" ]]; then
    log "skip run_motif (resume: motif.json exists)"
elif [[ $SKIP_MOTIF -eq 0 ]]; then
    run_step run_motif python "$HERE/run_motif.py" \
        --gpus "$MOTIF_GPUS" \
        --ckpt "$MOTIF_CKPT" \
        --out "$OUT_DIR/motif.json" || true
else
    log "skip run_motif (--skip-motif)"
fi

# --------------------------- 7) aggregate --------------------------------
if [[ $SKIP_AGGREGATE -eq 0 ]]; then
    AGG_FILES=("${KGFM_FILES[@]}")
    [[ -f "$OUT_DIR/ultra.json" ]] && AGG_FILES+=("ultra.json")
    [[ -f "$OUT_DIR/motif.json" ]] && AGG_FILES+=("motif.json")
    if [[ ${#AGG_FILES[@]} -eq 0 ]]; then
        log "skip aggregate (no result files produced)"
    else
        run_step aggregate python "$HERE/aggregate.py" \
            --results-dir "$OUT_DIR" \
            --files "${AGG_FILES[@]}" \
            --out "$OUT_DIR/table.md" || true
    fi
else
    log "skip aggregate (--skip-aggregate)"
fi

log ""
log "==> done. Results in $OUT_DIR"
log "    Latest pointer: $HERE/results/chembl/latest -> $TS"
echo
if [[ -f "$OUT_DIR/table.md" ]]; then
    echo "Comparison table:"
    cat "$OUT_DIR/table.md"
fi
