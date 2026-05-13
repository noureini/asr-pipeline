# ASR Pipeline — Multilingual Speech Transcription

Production-ready multilingual ASR pipeline with speaker diarization, LLM post-processing, and English translation. Built for qualitative research in low-resource language contexts.

Supports **1,600+ languages** through a two-tier engine architecture that routes high-resource languages to Whisper Large-v3 and everything else to Meta's Omnilingual CTC 300M.

## Table of Contents

- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [LoRA Fine-Tuning Experiments](#lora-fine-tuning-experiments)
- [Testing](#testing)
- [Project Structure](#project-structure)
- [Supported Languages](#supported-languages)
- [Output Format](#output-format)
- [Hardware Requirements](#hardware-requirements)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## Architecture

```
Audio --> Preprocess --> Language Detect --> Route
                                             |
                          +------------------+------------------+
                          |                                     |
                    HIGH RESOURCE                         NON-HIGH RESOURCE
                    (Spanish, English,                   (Hindi, Bengali, Nepali,
                     French, German...)                   Swahili, Amharic...)
                          |                                     |
                     Whisper LV3                         Omnilingual ASR
                     + word timestamps                   CTC 300M (16GB GPU)
                          |                                     |
                          +------------------+------------------+
                                             |
                                        pyannote 3.1
                                        (speaker diarization)
                                             |
                                      Merge + Align
                                             |
                                  +----------+----------+
                                  |                     |
                            Stage 1: LLM           Stage 1: Conservative
                            full correction        cleanup only
                                  |                     |
                                  +----------+----------+
                                             |
                                      Stage 2: Translation
                                      TranslateGemma 4B (default)
                                      or NLLB-200 (legacy)
                                             |
                                      Stage 3: LLM
                                      (English refinement)
                                             |
                                      Standard transcript
                                      output (.txt / .json / .srt)
```

### Two-Tier Routing

| Tier | Engine | Languages | Expected Accuracy |
|------|--------|-----------|-------------------|
| **HIGH** | Whisper Large-v3 | English, Spanish, French, German, Portuguese, Russian, Chinese, Japanese, Korean, Italian, Dutch, Polish, Turkish, Czech, Swedish, Ukrainian, Romanian, Arabic | WER <10% |
| **NON-HIGH** | Omnilingual CTC 300M | Hindi, Bengali, Nepali, Swahili, Amharic, Oromo, Hausa, Yoruba, Igbo, Tagalog, Burmese, Khmer, + 1,600 more | CER <10% for 78% |

### Code-Switching

Omnilingual ASR handles code-switching natively through its multilingual encoder. When a Nepali speaker drops into Hindi mid-sentence, the model just transcribes without switching engines.

## Prerequisites

Before installing the pipeline, make sure you have:

### System Dependencies

| Dependency | Required | How to Install |
|------------|----------|----------------|
| **Python >= 3.10** | Yes | [python.org](https://www.python.org/downloads/) or your system package manager |
| **FFmpeg** | Yes | `sudo apt install ffmpeg` (Ubuntu/Debian) or `brew install ffmpeg` (macOS) |
| **CUDA GPU (16GB+ VRAM)** | Recommended | NVIDIA driver + CUDA toolkit. CPU mode works but is much slower |
| **uv** | Recommended | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |

### HuggingFace Token (Required for Speaker Diarization)

The pyannote speaker diarization model requires a free HuggingFace token:

1. Create a free account at [huggingface.co](https://huggingface.co/join)
2. Go to [Settings > Access Tokens](https://huggingface.co/settings/tokens) and create a token
3. Accept the model terms at:
   - [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
   - [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
4. Set the token in your `.env` file (see [Installation](#installation) step 3)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/asr-pipeline.git
cd asr-pipeline
```

### 2. Install Dependencies

Using **uv** (recommended):

```bash
uv sync
```

Using **pip**:

```bash
pip install -e .
```

For NeMo MSDD diarization backend (optional):

```bash
pip install -e ".[nemo]"
```

For development tools (pytest, ruff, mypy):

```bash
uv sync --extra dev
# or
pip install -e ".[dev]"
```

### 3. Set Up Environment Variables

```bash
cp .env.example .env
```

Edit `.env` and replace `hf_your_token_here` with your actual HuggingFace token:

```
HF_TOKEN=hf_abc123...
```

### 4. Set Up Translation Models

**Option A: TranslateGemma (default, recommended)**

No manual setup needed. The model auto-downloads from HuggingFace on first transcription run (~3 GB download).

```bash
# Verify your environment is ready
uv run asr-pipeline setup
```

**Option B: CT2 NLLB + Ollama (legacy)**

This uses CTranslate2 NLLB-200 for translation and Ollama for LLM refinement:

```bash
# Install and start Ollama
curl -fsSL https://ollama.ai/install.sh | sh
ollama serve  # in a separate terminal

# Run the setup command
uv run asr-pipeline setup --translation-backend ct2_nllb
```

This will:
1. Pull the Ollama LLM model (default: `qwen2.5:1.5b`)
2. Convert NLLB-200 to CTranslate2 format
3. Update the config with model paths

### 5. Verify Installation

```bash
uv run asr-pipeline check-deps
```

This prints a table showing the status of all required dependencies, GPU availability, and installed packages.

## Configuration

All pipeline settings live in `src/asr_pipeline/default.yaml`. Key sections:

| Section | What it controls |
|---------|-----------------|
| `pipeline` | Device (cuda/cpu), compute type, batch size |
| `preprocessing` | Sample rate, VAD threshold, chunking, noise reduction |
| `languages` | Language registry with tier assignments |
| `whisper` | Whisper model size, beam search params |
| `omnilingual` | Model variant, zero-shot config |
| `diarization` | pyannote model, speaker count limits |
| `postprocessing` | Translation backend, LLM correction, refinement |
| `output` | Format (txt/json/srt), timestamp style |
| `logging` | Log level, file output, progress bars |

### Overriding Configuration

You can override defaults three ways:

1. **CLI flags** (highest priority):
   ```bash
   uv run asr-pipeline transcribe audio.m4a -l spa --device cpu --format json
   ```

2. **Custom YAML file**:
   ```bash
   uv run asr-pipeline transcribe audio.m4a -l spa --config my_config.yaml
   ```

3. **Environment variables** (prefix with `ASR_PIPELINE_`):
   ```bash
   ASR_PIPELINE_DEVICE=cpu uv run asr-pipeline transcribe audio.m4a -l spa
   ```

### Adding New Languages

New languages always go through the Omnilingual engine. Add an entry to the `languages` section in `default.yaml`:

```yaml
languages:
  wol:
    name: "Wolof"
    tier: "non_high"
    bcp47: "wo"
    script: "Latn"
    nllb_code: "wol_Latn"
```

## Usage

### CLI Commands

#### Transcribe a Single File

```bash
# Basic transcription (Spanish)
uv run asr-pipeline transcribe recording.m4a --language spa

# Hindi focus group with max 5 speakers
uv run asr-pipeline transcribe focus_group.wav -l hin --max-speakers 5

# Amharic with custom output directory and project name
uv run asr-pipeline transcribe interview.mp3 -l amh -o ./transcripts -p "Ethiopia Field Study"

# All output formats (TXT + JSON + SRT)
uv run asr-pipeline transcribe meeting.wav -l eng -f all

# Run on CPU (no GPU required)
uv run asr-pipeline transcribe audio.m4a -l spa --device cpu

# Use NeMo MSDD diarization backend
uv run asr-pipeline transcribe audio.m4a -l eng --diarization-backend nemo_msdd

# Debug mode with log file
uv run asr-pipeline transcribe audio.m4a -l spa --log-level DEBUG --log-file run.log
```

#### Transcribe a Folder (Batch Processing)

Process an entire folder of Survey Solutions interviews at once. The command expects the following folder structure:

```
FOLDER/
  {interview_key}/
    AudioAudit/
      *.m4a
```

Each interview directory gets a consolidated transcript with absolute timestamps derived from the recording start times encoded in the audio filenames.

```bash
# Transcribe all interviews in a folder (Bengali)
uv run asr-pipeline transcribe-folder ./interviews --language ben

# With custom output directory and all formats
uv run asr-pipeline transcribe-folder ./data -l spa -o ./transcripts -f all

# With project name and speaker limits
uv run asr-pipeline transcribe-folder ./fieldwork -l amh -p "Ethiopia Study" --max-speakers 5
```

**Resume support:** If a run is interrupted, simply re-run the same command. Already-completed interviews (those with existing output directories) are automatically skipped. Only remaining interviews are processed.

```bash
# Safe to re-run — skips already-completed interviews
uv run asr-pipeline transcribe-folder ./interviews -l ben -o ./outputs/interviews_2026-03-12 -f all
```

**Key features:**
- Models are loaded once and reused across all files (no per-file startup cost)
- Multiple audio files per interview are merged into a single consolidated transcript
- Timestamps are adjusted to absolute time based on filename-encoded recording start times
- Progress is displayed per-interview and per-file

#### List Supported Languages

```bash
uv run asr-pipeline list-languages
```

#### Check Dependencies

```bash
uv run asr-pipeline check-deps
```

#### Setup Models

```bash
# Full setup (downloads all ASR models + TranslateGemma)
uv run asr-pipeline setup

# Skip ASR models (only set up translation)
uv run asr-pipeline setup --skip-asr-models

# Skip translation setup
uv run asr-pipeline setup --skip-translation

# Use legacy CT2 NLLB + Ollama backend
uv run asr-pipeline setup --translation-backend ct2_nllb
```

#### Audio Quality Assessment

##### Quick path — one command, one Excel (recommended)

Point at a folder of recordings, get a color-coded Excel report. No setup,
no mapping files, no language flag.

```bash
uv run python scripts/audio_quality_report.py /path/to/audio/folder
```

That's it. The script will:

1. Walk the folder recursively for audio files (`.m4a`, `.wav`, `.mp3`, `.flac`, etc.)
2. Compute acoustic metrics per file (SNR, RMS, clipping %, speech %, peak level, duration)
3. Classify each file: **GOOD** / **OK** / **LOW SPEECH** / **POOR** / **BAD** / **CLIPPED**
4. Write `quality_report.xlsx` to the folder, with:
   - One row per file
   - Color-coded `quality` column (green for GOOD, red for BAD, etc.)
   - Auto-filter on the header row
   - Summary footer (distribution + averages)

Sample output:

| file | folder | duration_s | snr_db | rms_dbfs | clipping_pct | speech_pct | quality | comment |
|---|---|---|---|---|---|---|---|---|
| recording_001.m4a | interview_42 | 1841.2 | 24.3 | -18.1 | 0.000 | 67.4 | **GOOD** | clean, speech-rich |
| recording_002.m4a | interview_43 | 122.0 | 11.0 | -42.1 | 0.002 | 18.0 | **POOR** | very noisy, expect high WER |
| recording_003.m4a | interview_44 | 1543.7 | 18.5 | -34.2 | 1.450 | 32.1 | **CLIPPED** | distortion: 1.5% clipped samples |

Common options:

```bash
# Custom output path
uv run python scripts/audio_quality_report.py /path/to/audio -o ./reports/q.xlsx

# Limit to specific extensions
uv run python scripts/audio_quality_report.py /path/to/audio --extensions .m4a .wav

# Cap at first N files (for quick spot-checks)
uv run python scripts/audio_quality_report.py /path/to/audio --max-files 50
```

##### Advanced — `test-mics` CLI (compare microphones, run with transcription)

For comparing multiple microphones across the same recordings, use the
`test-mics` CLI command which produces a comparative JSON report.

**Expected folder structure** (Survey Solutions–style, same as `transcribe-folder`):

```
FOLDER/
  README.txt                        <-- maps interview keys to mic names
  {interview_key}/
    AudioAudit/
      *.m4a
```

**`README.txt` format** — one `# Mic Name` heading per microphone followed by
the interview keys recorded with that mic:

```
# DJI Mic 3
35-36-03-87
84-00-39-86

# Hollyland Lark M2
53-78-11-24
```

If every folder uses the same mic (e.g. a single-device test), put every key
under one heading.

**Run it:**

```bash
# Acoustic metrics only (fast — no ASR)
uv run asr-pipeline test-mics ./mic-test-data -l ben --skip-transcription

# Full run including transcription / WER comparison
uv run asr-pipeline test-mics ./mic-test-data -l ben

# Custom output directory
uv run asr-pipeline test-mics ./mic-test-data -l ben -o ./outputs/mic-test
```

**Per-file metrics computed:**
- **SNR (dB)** — VAD-based signal-to-noise ratio (higher = better)
- **Clipping %** — fraction of samples saturating to ±1.0 (lower = better)
- **Plosive spikes** — count of detected plosive bursts
- **Spectral rolloff (Hz)** — frequency below which 85 % of energy lies
- **Effective bandwidth (Hz)** — span of significant spectral energy
- **Crosstalk** — non-speech / speech energy ratio (lower = better)
- **RMS (dBFS)** — recording loudness
- **Speech %** — fraction of audio classified as speech

The CLI prints two Rich tables (per-file and per-mic averages) plus a textual
recommendation, and writes the full report to
`<output_dir>/mic-test-report.json`.

**Per-folder summary table (CSV + PNG + XLSX):** for the common case where each
folder is a separate respondent / interview, a helper script aggregates the
per-file metrics into one row per folder, classifies each folder with a
quality flag (`GOOD` / `OK` / `LOW SPEECH` / `POOR` / `BAD`), and exports the
result in three formats:

```bash
uv run python scripts/mic_test_per_folder_report.py \
    outputs/mic-test/mic-test-report.json
```

This writes the following files next to the JSON:

- `per-folder-report.csv` — sortable spreadsheet
- `per-file-report.csv` — same metrics broken down per individual audio file
- `per-folder-report.png` — rendered table image; best/worst SNR rows highlighted, Quality column colour-coded, MEAN footer row
- `per-folder-report.xlsx` — formatted Excel workbook with two tabs (`Per folder`, `Per file`), frozen header, auto-filter, colour-coded Quality column, and a live `=AVERAGE(...)` MEAN footer

**Per-speaker analysis (optional, requires diarization):** to assess audio quality
*separately for the enumerator and the respondent* (very useful for phone-mic
interviews where the respondent's voice comes through the phone earpiece and is
intrinsically lower quality), run:

```bash
# Heavy: re-runs speaker diarization on every file (~30-60 min on GPU for 100 files)
uv run python scripts/mic_test_per_speaker.py \
    ./mic-test-data -l ben -o outputs/mic-test
```

This writes:

- `per-speaker-report.csv` — one row per (file × speaker) with their own SNR, bandwidth, talk-time, etc.
- `per-speaker-report.json` — same data; automatically picked up by `mic_test_per_folder_report.py` to add a third **`Per speaker`** tab to the XLSX

Speakers are labelled **ENUMERATOR** or **RESPONDENT** using simple heuristics
(who speaks first, total turns, average turn length). Re-run the per-folder
report afterwards to refresh the XLSX with the new tab.

### Quick Test with Included Audio Files

The repository includes two test audio files you can use to verify the pipeline:

```bash
# English test audio
uv run asr-pipeline transcribe test_audio.m4a -l eng

# Swahili test audio
uv run asr-pipeline transcribe dmi_swa.mp3 -l swa
```

### Python API

```python
from asr_pipeline.config import load_config
from asr_pipeline.pipeline import ASRPipeline

config = load_config()
pipeline = ASRPipeline(config)

result = pipeline.transcribe(
    audio_path="focus_group.m4a",
    language="swa",
    project_name="Kenya SACCO Study",
)

# Access segments programmatically
for seg in result.segments:
    print(f"[{seg.speaker_id}] {seg.corrected_text}")
    print(f"  -> {seg.refined_translation}")
```

## LoRA Fine-Tuning Experiments

Beyond the core ASR pipeline, the repo contains scripts for fine-tuning LLMs
to correct noisy phoneme-based transcriptions. Use these when the lattice +
post-processing baseline isn't accurate enough for your target language.

### Pipeline overview

```
Audio → ZIPA (universal phoneme model) → IPA tokens → FST → noisy Bengali
                                                   ↘
                                       (alternative: skip FST,
                                        feed IPA directly to LLM)
                                                       ↓
                                         LoRA-fine-tuned LLM
                                                       ↓
                                                 clean Bengali
```

### Workflow scripts

| Stage | Script | Purpose |
|---|---|---|
| **Build dataset** | `scripts/extract_ipa_local.py` | Stream FLEURS/Bengali_AI_Speech/banspeech/SKNahin, run ZIPA, save IPA + Bengali (+English) pairs as JSONL. Stratified per source. Resume-safe. |
| | `scripts/build_lora_dataset_remote.py` | Same as above but with `--output-format` flag for FST vs IPA vs both. For remote compute / GitHub-clone workflow. |
| | `scripts/split_ipa_dataset.py` | Stratified train/val split by (source, subsource) — eval set spans all distributions equally. |
| **Train** | `scripts/train_lora_ipa_local.py` | LoRA fine-tune a small LLM (Qwen2.5-1.5B/3B by default) locally on RTX 3060-class GPU. Outputs adapter + GGUF. |
| | `scripts/train_lora_remote.py` | Same training but configurable for 16-24GB GPU boxes (rented A10/A100). |
| | `scripts/train_lora_corrector.py` | Original FST-input variant (Unsloth-canonical, uses chat template). |
| **Evaluate** | `scripts/test_lora_local.py` | Run a trained GGUF on held-out eval samples, report CER per source. |
| | `scripts/compare_lora_vs_baseline.py` | Apples-to-apples: compare LoRA output vs your prior lattice/Gemma baselines on the same FLEURS test samples. |
| | `scripts/compare_lora_chain_strategies.py` | Test 3 LoRA input strategies (raw FST / lattice top-1 / Gemma-corrected) to find the optimal pipeline. |
| | `scripts/test_lora_translation.py` | Test BN→EN translation capability of the trained model (two-pass vs combined prompt). |
| | `scripts/test_banglallama_zeroshot.py` | Zero-shot baseline using BanglaLlama 3.1 8B for comparison. |
| | `scripts/eval_lora_corrector.py` | Streaming eval on FLEURS test (CER, WER, per-source breakdown). |
| **Orchestration** | `scripts/run_lora_pipeline.sh` | End-to-end on a remote box: dataset build → training → GGUF export. Single command. |

### Quick start — local laptop (RTX 3060+)

```bash
# 1. Extract dataset with IPA tokens (~3-5h with cached audio)
export HF_TOKEN=hf_xxxx   # or place in .env file at repo root
uv run python scripts/extract_ipa_local.py \
    --output-dir ./lora_data_ipa \
    --max-hours-sknahin 4

# 2. Stratified train/val split (5 sec)
# (already done by extract_ipa_local.py)

# 3. Train LoRA (30-60 min on RTX 3060)
uv run python scripts/train_lora_ipa_local.py \
    --train ./lora_data_ipa/lora_dataset_full_ipa_train.jsonl \
    --val   ./lora_data_ipa/lora_dataset_full_ipa_val.jsonl

# 4. Evaluate
uv run python scripts/compare_lora_vs_baseline.py \
    --gguf models/qwen_ipa_lora/gguf/*.gguf \
    --baseline-json results/baseline_merged.json
```

### Documentation

See `scripts/README_lora_remote.md` for the remote-compute workflow and detailed
explanations of dataset diversity strategies, quality filters, and hyperparameter
choices.

## Testing

### Running All Tests

```bash
uv run pytest
```

### Running with Coverage

```bash
uv run pytest --cov=asr_pipeline --cov-report=term-missing
```

### Running Specific Test Classes

```bash
# Config tests only
uv run pytest tests/test_pipeline.py::TestConfig -v

# Data model tests
uv run pytest tests/test_pipeline.py::TestDataModels -v

# Alignment tests
uv run pytest tests/test_pipeline.py::TestAlignment -v

# Formatter tests
uv run pytest tests/test_pipeline.py::TestFormatter -v

# Language registry tests
uv run pytest tests/test_pipeline.py::TestLanguageRegistry -v

# VAD chunking tests
uv run pytest tests/test_pipeline.py::TestVADChunking -v
```

### What the Tests Cover

| Test Class | What it tests |
|------------|---------------|
| `TestConfig` | Configuration loading from YAML, engine routing, default values |
| `TestDataModels` | Pydantic data model creation and validation (segments, transcripts, etc.) |
| `TestAlignment` | Speaker-to-segment alignment, overlap computation, segment merging |
| `TestAlignmentWords` | Word-level timestamp preservation through alignment |
| `TestFormatter` | TXT and SRT output formatting, timestamp formatting |
| `TestLanguageRegistry` | Language lookup, tier routing, Whisper code mapping |
| `TestVADChunking` | Non-speech region extraction, gap thresholds |
| `TestNonSpeechModels` | Non-speech data models, audio quality metrics |
| `TestFormatterNonSpeech` | TXT output with non-speech placeholders |

> **Note**: The tests are unit tests that do not require a GPU or any downloaded models. They test configuration, data models, alignment logic, and formatting. End-to-end pipeline tests would require GPU + models.

### Linting and Type Checking

```bash
# Lint with ruff
uv run ruff check src/ tests/

# Type check with mypy
uv run mypy src/asr_pipeline/
```

## Project Structure

```
asr-pipeline/
|-- pyproject.toml              # Project metadata, dependencies, build config
|-- README.md                   # This file
|-- .env.example                # Template for environment variables
|-- .gitignore                  # Git ignore rules
|-- test_audio.m4a              # Sample English audio for testing
|-- dmi_swa.mp3                 # Sample Swahili audio for testing
|
|-- src/asr_pipeline/
|   |-- __init__.py             # Package init, version
|   |-- __main__.py             # python -m asr_pipeline entry point
|   |-- cli.py                  # Click CLI (transcribe, transcribe-folder, setup, etc.)
|   |-- config.py               # Pydantic config loading from YAML
|   |-- default.yaml            # Default configuration file
|   |-- pipeline.py             # Main ASR orchestration pipeline
|   |-- batch.py                # Folder batch processing (interview discovery, merging)
|   |-- preprocessor.py         # Audio preprocessing (VAD, noise reduction, chunking)
|   |-- alignment.py            # Speaker-segment alignment and merging
|   |-- forced_aligner.py       # wav2vec2 MMS forced alignment
|   |-- diarization.py          # pyannote speaker diarization backend
|   |-- nemo_diarization.py     # NeMo MSDD diarization backend (optional)
|   |-- postprocessor.py        # LLM correction, translation, refinement
|   |-- formatter.py            # Output formatting (TXT, JSON, SRT)
|   |-- language.py             # Language registry and routing
|   |-- logging_config.py       # Logging setup (Rich console)
|   |-- models.py               # Pydantic data models (segments, transcripts)
|   |-- finetune.py             # Model fine-tuning utilities
|   |-- engines/
|       |-- __init__.py         # Engine exports
|       |-- whisper_engine.py   # Whisper Large-v3 ASR engine
|       |-- omnilingual_engine.py # Omnilingual CTC 300M ASR engine
|
|-- tests/
|   |-- test_pipeline.py        # Unit tests (config, models, alignment, formatting)
|
|-- test_data/                  # Sample interview folders for batch testing
|
|-- presentation/
|   |-- ASR_Pipeline_Feb26.pdf  # Project presentation slides
|   |-- asr_pipeline_presentation.tex  # LaTeX source
|
|-- outputs/                    # Generated transcripts (git-ignored)
```

## Supported Languages

### Pre-configured (33 languages)

**High-resource (Whisper):** English, Spanish, French, German, Portuguese, Russian, Chinese, Japanese, Korean, Italian, Dutch, Polish, Turkish, Czech, Swedish, Ukrainian, Romanian, Arabic

**Non-high-resource (Omnilingual):** Hindi, Bengali, Nepali, Swahili, Amharic, Afaan Oromo, Hausa, Yoruba, Igbo, Tagalog, Burmese, Khmer, Kinyarwanda, Somali, Tigrinya

### Adding More

Any of the 1,600+ languages supported by Omnilingual ASR can be added via the config file. See [Adding New Languages](#adding-new-languages).

## Output Format

Standard qualitative research transcript:

```
========================================================================
TRANSCRIPT
========================================================================
Project:        Ethiopia Field Study
Date:           2026-02-07
Duration:       00:45:23
Audio File:     focus_group.m4a
Languages:      Amharic
Speakers:       3 identified
Transcription:  Intelligent Verbatim
ASR Engines:    Omnilingual omniASR_CTC_300M_v2
Post-processed: LLM Correction, NLLB-200 Translation, LLM Refinement
========================================================================

[00:00:12] SPEAKER_00:
[am] <original text in source language>
[en] <English translation>

[00:00:25] SPEAKER_01:
[am] <original text in source language>
[en] <English translation>

========================================================================
END OF TRANSCRIPT
========================================================================
```

Three output formats are available:
- **TXT**: Human-readable transcript (default)
- **JSON**: Structured data with all metadata and segments
- **SRT**: Subtitle format for video editors

## Hardware Requirements

| Component | 16GB GPU | 24GB GPU | CPU Only |
|-----------|----------|----------|----------|
| Whisper Large-v3 | ~10GB | ~10GB | Slow but works |
| Omnilingual CTC 300M | ~2GB | ~2GB | Slow but works |
| Omnilingual CTC 1B | N/A | ~6GB | N/A |
| pyannote 3.1 | ~2GB | ~2GB | Slow but works |
| TranslateGemma 4B (4-bit) | ~3GB | ~3GB | Slow but works |
| NLLB-200 1.3B (CT2) | ~3GB | ~3GB | Works |

A **16GB NVIDIA GPU** (e.g., RTX 4080, A4000, T4) is sufficient for the full pipeline. Models are loaded sequentially, not all at once.

## Troubleshooting

### Common Issues

**`CUDA out of memory`**

Lower the batch size in `default.yaml`:
```yaml
pipeline:
  batch_size: 4   # default is 8
whisper:
  batch_size: 4   # default is 8
```

Or switch to CPU mode:
```bash
uv run asr-pipeline transcribe audio.m4a -l spa --device cpu
```

**`pyannote` authentication error**

Make sure your `.env` file has a valid `HF_TOKEN` and you've accepted both model terms:
- [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
- [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)

**CT2 NLLB translation model corrupt**

If using the legacy CT2 backend and the model fails to load:
```bash
rm -rf ~/.asr-pipeline/models/ct2-nllb
uv run asr-pipeline setup --translation-backend ct2_nllb
```

**FFmpeg not found**

Install FFmpeg for your platform:
```bash
# Ubuntu/Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg

# Windows (with Chocolatey)
choco install ffmpeg
```

**Tests fail with import errors**

Make sure dev dependencies are installed:
```bash
uv sync --extra dev
```

## License

MIT
