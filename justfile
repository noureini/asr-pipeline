# ASR Pipeline — task runner (https://github.com/casey/just)
#
# Runs on Linux (Ubuntu 22.04/24.04, native or WSL2). Several dependencies
# (NeMo, the CUDA llama-cpp-python wheel, fairseq2) ship Linux-only builds.
#
#   just                       # list recipes
#   just setup                 # deps + pinned Bengali models (one-time)
#   just quality ./interviews  # Step 0: audio quality report (no GPU)
#   just fusion interview.m4a  # Steps 1-4: transcribe one interview

set shell := ["bash", "-cu"]

config := "fusion_production.yaml"

# List available recipes
default:
    @just --list

# ─── Setup ────────────────────────────────────────────────────────────────

# One-time setup: Python deps + pinned Bengali fusion models
setup: sync setup-models

# Install the locked Python environment (uv.lock)
sync: _check-ffmpeg
    uv sync --frozen

# Download titu, regional Whisper and Gemma at the pinned commits; build CT2 Whisper
setup-models:
    uv run python scripts/download_models.py -c {{config}}

# Optional: models for the general multilingual path (Whisper large-v3, Omnilingual, pyannote)
setup-multilingual: _hint-hf-token
    uv run asr-pipeline setup --skip-ollama

# ─── Step 0 · Audio quality (no GPU) ──────────────────────────────────────

# Per-file quality report → quality_report.xlsx in the folder
quality folder *flags:
    uv run python scripts/audio_quality_report.py "{{folder}}" {{flags}}

# Per-file + per-speaker report (needs HF_TOKEN + pyannote terms accepted)
quality-speakers folder *flags: _hint-hf-token
    uv run python scripts/audio_quality_report.py "{{folder}}" --with-speakers {{flags}}

# ─── Steps 1-4 · Bengali two-witness fusion (production path) ─────────────

# Transcribe ONE Bengali interview. Usage: just fusion in.m4a -o ./out --log-file run.log
fusion file *flags='':
    HF_HUB_OFFLINE=1 uv run asr-pipeline transcribe "{{file}}" -l ben -c {{config}} {{flags}}

# Transcribe a folder of interviews (Survey Solutions layout, resume-safe)
fusion-folder folder *flags='':
    HF_HUB_OFFLINE=1 uv run asr-pipeline transcribe-folder "{{folder}}" -l ben -c {{config}} {{flags}}

# Calibrate the GER snap threshold from a run's .reasoning.txt
calibrate-ger reasoning_file:
    uv run python scripts/ger_calibrate.py "{{reasoning_file}}"

# ─── General multilingual path ────────────────────────────────────────────

# Transcribe any language (ISO 639-3). Usage: just transcribe in.m4a swa
transcribe file lang *flags='':
    uv run asr-pipeline transcribe "{{file}}" -l {{lang}} {{flags}}

# Transcribe a folder in any language
transcribe-folder folder lang *flags='':
    uv run asr-pipeline transcribe-folder "{{folder}}" -l {{lang}} {{flags}}

# ─── Development ──────────────────────────────────────────────────────────

# Run the unit tests (no GPU, no models)
test *flags='':
    uv run --frozen --extra dev pytest {{flags}}

# Show CUDA / GPU status
gpu:
    @uv run python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"

# Check that every runtime dependency imports
check:
    uv run asr-pipeline check-deps

# ─── Internal helpers ─────────────────────────────────────────────────────

_check-ffmpeg:
    @ffmpeg -version > /dev/null 2>&1 && echo "[OK] ffmpeg found" || (echo "[!] ffmpeg not found — sudo apt install ffmpeg" && exit 1)

_hint-hf-token:
    @test -f .env && grep -q 'HF_TOKEN' .env && echo "[OK] HF_TOKEN found in .env" || echo "[!] No HF_TOKEN in .env — needed for pyannote. Get one at https://huggingface.co/settings/tokens"
