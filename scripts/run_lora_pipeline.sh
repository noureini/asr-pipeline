#!/usr/bin/env bash
# End-to-end Bengali ASR LoRA pipeline for remote compute.
#
# Usage on the remote box (after `git pull`):
#   export HF_TOKEN="hf_xxxxxxxxx"   # if using gated repos
#   bash scripts/run_lora_pipeline.sh [--quick | --full]
#
# --quick : Bengali_AI_Speech + banspeech + FLEURS only  (~1.5h extract)
# --full  : + SKNahin (default 24h budget)               (~25h total)
#
# Resumable at any stage. Re-run after Ctrl+C and it picks up.

set -euo pipefail

cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"

DATA_DIR="${REPO_ROOT}/lora_data"
MODEL_DIR="${REPO_ROOT}/lora_models/aya8b_v1"

MODE="${1:---full}"

case "$MODE" in
  --quick)
    SOURCES="fleurs bengali_ai_speech banspeech"
    MAX_HOURS=2
    ;;
  --full)
    SOURCES="fleurs bengali_ai_speech banspeech sknahin"
    MAX_HOURS=24
    ;;
  *)
    echo "Usage: $0 [--quick | --full]"
    exit 1
    ;;
esac

# ── Setup environment ──────────────────────────────────────────────────
echo "==[1/4]== Setting up Python environment"
if ! command -v uv &>/dev/null; then
  echo "uv not found, falling back to pip"
  PIP="pip install -q"
else
  PIP="uv pip install -q"
fi

# Pinned versions to avoid the script-loading issues we hit
$PIP "datasets==2.21.0" "huggingface_hub==0.24.7" "fsspec<=2024.6.1"
$PIP onnxruntime-gpu librosa soundfile torch torchaudio
$PIP unsloth trl

# ── Stage 1: Build dataset ─────────────────────────────────────────────
echo ""
echo "==[2/4]== Building dataset → $DATA_DIR (mode=$MODE)"
python scripts/build_lora_dataset_remote.py \
  --output-dir "$DATA_DIR" \
  --sources $SOURCES \
  --max-hours "$MAX_HOURS"

DATASET="$DATA_DIR/lora_dataset_full.jsonl"
N_PAIRS=$(wc -l < "$DATASET")
echo "Dataset ready: $N_PAIRS pairs at $DATASET"

# ── Stage 2: Train LoRA ────────────────────────────────────────────────
echo ""
echo "==[3/4]== Training LoRA → $MODEL_DIR"
python scripts/train_lora_remote.py \
  --dataset "$DATASET" \
  --output-dir "$MODEL_DIR" \
  --model "unsloth/aya-expanse-8b-bnb-4bit" \
  --lora-r 16 \
  --epochs 3 \
  --batch-size 4 \
  --grad-accum 2 \
  --lr 1e-4 \
  --export-gguf q4_k_m

# ── Stage 3: Done ──────────────────────────────────────────────────────
echo ""
echo "==[4/4]== Pipeline complete"
echo "  Adapter:  $MODEL_DIR/final/"
echo "  GGUF:     $MODEL_DIR/gguf/"
echo ""
echo "Use the GGUF in your inference pipeline:"
echo "  from llama_cpp import Llama"
echo "  m = Llama(model_path='$MODEL_DIR/gguf/<file>.gguf', n_ctx=2048)"
