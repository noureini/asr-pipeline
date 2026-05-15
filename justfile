# ASR Pipeline — task runner
#
# Install just: https://github.com/casey/just
#   Windows : winget install Casey.Just  (or scoop install just)
#   macOS   : brew install just
#   Linux   : cargo install just  (or apt/dnf)
#
# Then:  just                  # show all recipes
#        just setup            # install everything (one-shot)
#        just quality FOLDER   # audio quality report
#
# Recipes prefixed `setup-` install just the deps needed for one workflow.
# `just setup` runs all of them.

set positional-arguments

# This justfile is written for WSL/bash (the user's primary environment).
# On native Windows PowerShell, the _check-* helpers may need tweaks.

# Default: list available recipes
default:
    @just --list

# ─── Setup ────────────────────────────────────────────────────────────────

# Install everything (audio quality + ASR + translation + LLM)
setup: setup-quality setup-transcribe setup-llm
    @echo ""
    @echo "[OK] Full setup complete."
    @echo "Try:  just quality test_data/bangladesh_mic_test"

# Sync core Python deps (run once before any other recipe)
sync:
    uv sync

# Audio quality assessment deps (librosa, openpyxl, pyannote, ffmpeg)
setup-quality: sync _check-ffmpeg _hint-hf-token
    @echo ""
    @echo "[OK] Audio quality setup ready."
    @echo "  Basic:           just quality FOLDER"
    @echo "  With speakers:   just quality-speakers FOLDER"
    @echo ""
    @echo "Per-speaker mode also needs you to accept pyannote terms (once):"
    @echo "  https://huggingface.co/pyannote/speaker-diarization-3.1"
    @echo "  https://huggingface.co/pyannote/segmentation-3.0"

# ASR + alignment deps (whisper, omnilingual, mms-fa)
setup-transcribe: sync _check-ffmpeg
    uv run asr-pipeline setup --skip-ollama
    @echo ""
    @echo "[OK] Transcription models downloaded."

# LLM correction / refinement via Ollama
setup-llm: _check-ollama
    ollama pull qwen2.5:7b
    @echo ""
    @echo "[OK] Ollama + qwen2.5:7b ready."

# Run the full ASR setup (models + ollama)
setup-full: sync
    uv run asr-pipeline setup

# ─── Audio quality ────────────────────────────────────────────────────────

# Per-file quality report → quality_report.xlsx in the folder
quality folder *flags:
    uv run python scripts/audio_quality_report.py "{{folder}}" {{flags}}

# Per-file + per-speaker (ENUMERATOR/RESPONDENT) report
quality-speakers folder *flags:
    uv run python scripts/audio_quality_report.py "{{folder}}" --with-speakers {{flags}}

# ─── ASR pipeline ─────────────────────────────────────────────────────────

# Transcribe a single audio file. Usage: just transcribe path/to.wav bn
transcribe file lang="bn" *flags:
    uv run asr-pipeline transcribe "{{file}}" --language {{lang}} {{flags}}

# Transcribe + translate to English
translate file lang="bn" *flags:
    uv run asr-pipeline transcribe "{{file}}" --language {{lang}} --translate {{flags}}

# Batch-transcribe a folder of audio
transcribe-folder folder lang="bn" *flags:
    uv run asr-pipeline transcribe "{{folder}}" --language {{lang}} {{flags}}

# ─── Research experiments ─────────────────────────────────────────────────

# Step A: build the P2G TSV lexicons (Epitran + WikiPron + CMUDict)
phys-build-lexicon:
    uv run python scripts/build_p2g_dictionaries.py

# Step B: build the panphon-feature index over the TSVs (~1 min)
phys-build:
    uv run python scripts/test_phys_lattice_recall.py --build

# Verify panphon's diacritic grouping handles ZIPA-style tokens
phys-diagnose:
    uv run python scripts/test_phys_lattice_recall.py --diagnose

# Recall@K sanity check — clean IPA (should be ~100%)
phys-recall-clean n="500":
    uv run python scripts/test_phys_lattice_recall.py --mode clean --n {{n}}

# Recall@K with synthetic ZIPA-style noise — the decisive number
phys-recall n="500":
    uv run python scripts/test_phys_lattice_recall.py --mode noisy --n {{n}}

# Recall@K on your own (gold, ZIPA-output) JSONL pairs
phys-recall-jsonl path n="500":
    uv run python scripts/test_phys_lattice_recall.py --mode jsonl --test-jsonl "{{path}}" --n {{n}}

# ─── Maintenance ──────────────────────────────────────────────────────────

# Show CUDA / GPU status
gpu:
    @uv run python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"

# Clear cached models (asks for confirmation)
clean-models:
    @echo "About to delete ~/.asr-pipeline/models"
    @echo "Press Ctrl-C to cancel, Enter to proceed"
    @read _
    rm -rf ~/.asr-pipeline/models

# ─── Internal helpers ─────────────────────────────────────────────────────

_check-ffmpeg:
    @ffmpeg -version > /dev/null 2>&1 && echo "[OK] ffmpeg found" || (echo "[!] ffmpeg not found — install: sudo apt install ffmpeg (WSL) or winget install ffmpeg" && exit 1)

_check-ollama:
    @ollama --version > /dev/null 2>&1 && echo "[OK] ollama found" || (echo "[!] ollama not found — install from https://ollama.com" && exit 1)

_hint-hf-token:
    @test -f .env && grep -q -i 'HF_TOKEN\|HUGGING' .env && echo "[OK] HF token found in .env" || echo "[!] No HF_TOKEN in .env — needed for --with-speakers mode. Get one at https://huggingface.co/settings/tokens"
