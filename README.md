# ASR Pipeline — Multilingual Speech Transcription

Production-ready multilingual ASR pipeline with speaker diarization, LLM post-processing, and English translation. Built for qualitative research in low-resource language contexts.

Supports **1,600+ languages** through a two-tier engine architecture that routes high-resource languages to Whisper Large-v3 and everything else to Meta's Omnilingual CTC 300M.

## Table of Contents

- [Pipeline Walkthrough](#pipeline-walkthrough) ← **start here**
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage) — full CLI reference
- [LoRA Fine-Tuning Experiments](#lora-fine-tuning-experiments)
- [Testing](#testing)
- [Project Structure](#project-structure)
- [Supported Languages](#supported-languages)
- [Output Format](#output-format)
- [Hardware Requirements](#hardware-requirements)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## Pipeline Walkthrough

A complete survey transcription workflow runs in 4 sequential steps. Each step has
its own command and produces its own output. You can run them independently or
chain them together.

```
┌─────────────────┐     ┌──────────────────┐     ┌──────────────────┐     ┌────────────────┐
│  Step 0         │     │  Step 1          │     │  Step 2          │     │  Step 3        │
│  Audio quality  │ ──▶ │  Transcribe      │ ──▶ │  Diarize         │ ──▶ │  Output JSON / │
│  → Excel report │     │  + LLM correct   │     │  speakers        │     │  Excel / SRT   │
└─────────────────┘     └──────────────────┘     └──────────────────┘     └────────────────┘
       ↓ (optional)            ↓ (optional)                                     ↓
   Filter out               Use custom-trained                          Plug into your
   POOR/BAD audio           LoRA model for                              survey research
                            Bengali (or other                           pipeline
                            low-resource langs)
```

### Step 0 — Audio Quality Check

Before transcribing anything, run a quality assessment so you know what you're
working with. **One command, one Excel.**

```bash
just quality /path/to/audio/folder
# or, without just:
uv run python scripts/audio_quality_report.py /path/to/audio/folder
```

This recursively scans the folder, computes per-file metrics (SNR, clipping,
speech %, RMS, peak), classifies each recording (`EXCELLENT`/`GOOD`/`FAIR`/
`LOW SPEECH`/`EMPTY`/`POOR`/`BAD`/`CLIPPED`/`MUTED`/`BROKEN`), and writes
`quality_report.xlsx` to the folder.

Open the Excel — color-coded `quality` column and a `score` (0-100) per file.
Filter the spreadsheet by `quality` to find broken recordings to flag for
re-recording, or pre-filter your transcription queue to skip the BAD ones.

[Detailed audio quality docs →](#audio-quality-assessment)

### Step 1 — Transcribe Audio

Once you've filtered out unusable recordings, transcribe the rest:

```bash
# Single file
uv run asr-pipeline transcribe interview.m4a -l ben

# Whole folder (Survey Solutions style)
uv run asr-pipeline transcribe-folder /path/to/audio --language ben
```

Outputs JSON with full transcript, per-segment text, speaker IDs, and timestamps.
Also produces SRT subtitles and Excel summary.

The pipeline automatically:
- Preprocesses audio (resample to 16kHz, normalize loudness, denoise)
- Detects language (or uses your `--language` flag)
- Routes to the right engine (Whisper for high-resource, Omnilingual CTC for the rest)
- Runs speaker diarization (pyannote 3.1)
- LLM post-processing for correction and English translation

[Full CLI reference →](#cli-commands)

### Step 2 — (Optional) Custom LoRA for Bengali / Low-Resource Languages

If the default LLM correction isn't accurate enough for your target language,
fine-tune a custom LoRA on phoneme-based correction data. Comprehensive workflow
included for Bengali (extends to any low-resource language with the same
infrastructure).

```bash
# Build dataset from public corpora (FLEURS, Bengali_AI_Speech, banspeech, SKNahin)
uv run python scripts/extract_ipa_local.py --output-dir ./lora_data_ipa

# Train LoRA on RTX 3060+ (30-60 min)
uv run python scripts/train_lora_ipa_local.py \
    --train ./lora_data_ipa/lora_dataset_full_ipa_train.jsonl \
    --val   ./lora_data_ipa/lora_dataset_full_ipa_val.jsonl

# Compare against your prior baselines
uv run python scripts/compare_lora_vs_baseline.py \
    --gguf models/qwen_ipa_lora/gguf/*.gguf \
    --baseline-json results/baseline_merged.json
```

[Full LoRA workflow →](#lora-fine-tuning-experiments)

### Step 3 — Use Outputs

The pipeline emits structured JSON with per-segment text, speaker IDs, timestamps,
and refined translations. Plug into your downstream survey analysis workflow
(R / Python / Excel / Stata / etc.).

```python
import json
with open("output.json") as f:
    result = json.load(f)
for seg in result["segments"]:
    print(f"[{seg['speaker_id']}] {seg['corrected_text']}")
    print(f"  → {seg['refined_translation']}")
```

[Full output format reference →](#output-format)

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

| Required | What | Notes |
|---|---|---|
| Yes | Python ≥ 3.10 | |
| Yes | FFmpeg | for audio decoding (m4a, mp3, …) |
| Yes (for diarization) | HuggingFace token | free, set up in [Installation step 3](#3-set-up-environment-variables-huggingface-token) |
| Recommended | `uv` | Python env manager |
| Recommended | `just` | task runner for the one-line workflows below |
| Recommended | NVIDIA GPU (≥ 16 GB VRAM) | CPU works but is slow |

All exact install commands are in [Installation step 1](#1-install-system-tools-uv-just-ffmpeg).

## Installation

Follow these steps top-to-bottom on a fresh machine. After step 4 you can run
the audio quality check; transcription needs steps 5–6 as well.

### 1. Install system tools (uv, just, ffmpeg)

```bash
# uv — Python package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# just — task runner (powers all `just <recipe>` commands below)
curl --proto '=https' --tlsv1.2 -sSf https://just.systems/install.sh \
    | bash -s -- --to ~/.local/bin
export PATH="$HOME/.local/bin:$PATH"   # add to ~/.bashrc to make permanent

# ffmpeg — audio decoding (m4a, mp3, etc.)
sudo apt install ffmpeg          # Ubuntu / WSL / Debian
# brew install ffmpeg            # macOS
# winget install ffmpeg          # Windows
```

Verify:

```bash
uv --version && just --version && ffmpeg -version | head -1
```

### 2. Clone the repository

```bash
git clone https://github.com/<your-username>/asr-pipeline.git
cd asr-pipeline
```

### 3. Set up environment variables (HuggingFace token)

Needed for speaker diarization (Step 0 `--with-speakers`, and Step 1 transcription).
The basic per-file audio quality check works **without** a token.

```bash
cp .env.example .env
# edit .env and set HF_TOKEN=hf_abc123...
```

Get a free token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens),
then accept the model terms at
[pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
and [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0).

### 4. Install Python dependencies for audio quality

```bash
just setup-quality
# equivalent to: uv sync, then verify ffmpeg + HF token
```

You can now run the audio quality check:

```bash
just quality /path/to/audio/folder              # per-file
just quality-speakers /path/to/audio/folder     # + per-speaker (ENUMERATOR/RESPONDENT)
```

### 5. (Optional) Download transcription models

Only needed if you want to run the ASR / translation pipeline (Step 1 onward).

```bash
just setup-transcribe
# equivalent to: uv run asr-pipeline setup --skip-ollama
```

This downloads Whisper Large-v3, Omnilingual CTC, MMS-FA alignment, and NLLB-200
translation weights into `~/.asr-pipeline/models` (~15 GB).

### 6. (Optional) LLM correction / refinement via Ollama

```bash
just setup-llm
# equivalent to: pulls qwen2.5:7b via ollama (~4.7 GB)
```

Requires [Ollama](https://ollama.com) installed and running.

### One-shot install (everything)

If you want all three paths at once:

```bash
just setup
```

### Extras

| Extra | Install | When you need it |
|-------|---------|------------------|
| NeMo MSDD diarization (alternative to pyannote) | `uv sync --extra nemo` | Only if pyannote doesn't fit your hardware constraints |
| Dev tools (pytest, ruff, mypy) | `uv sync --extra dev` | Contributing code |

### Without `just` (raw commands)

Everything `just` does is just a wrapper. If you don't want to install `just`:

| Recipe | Raw equivalent |
|--------|----------------|
| `just setup-quality` | `uv sync` |
| `just quality FOLDER` | `uv run python scripts/audio_quality_report.py FOLDER` |
| `just quality-speakers FOLDER` | `uv run python scripts/audio_quality_report.py FOLDER --with-speakers` |
| `just setup-transcribe` | `uv run asr-pipeline setup --skip-ollama` |
| `just transcribe FILE bn` | `uv run asr-pipeline transcribe FILE --language bn` |

### Verify everything is wired up

```bash
just gpu                          # shows CUDA availability
uv run asr-pipeline check-deps    # full dependency / model status table
```

### Translation backends

The default backend (TranslateGemma 4B) auto-downloads on first transcription run.
For the legacy CT2 NLLB + Ollama backend instead, use:

```bash
uv run asr-pipeline setup --translation-backend ct2_nllb
```

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

The basic command is shown in [Step 0 of the walkthrough](#step-0--audio-quality-check).
The sections below document the **quality flag taxonomy** and the **advanced
`test-mics` CLI** for comparing microphones.

##### Quality flags

| Flag | Score | Meaning | Use? |
|---|---|---|---|
| 🟢 **EXCELLENT** | 90-100 | Studio-quality (SNR ≥ 25, ≥ 50% speech, no clipping) | Yes — premium training data |
| 🟢 **GOOD** | 80-89 | Clean, speech-rich (SNR ≥ 20, ≥ 35% speech) | Yes — standard training/inference |
| 🟡 **FAIR** | 50-69 | Acceptable (SNR ≥ 15, ≥ 25% speech) | Yes — expect minor errors |
| 🟠 **LOW SPEECH** | ~55 | Clean audio but mostly silent (mic far / muted speaker) | Maybe — speaker may be inaudible |
| 🟠 **EMPTY** | ~30 | Almost no speech (<10%) — possibly background recording | Investigate |
| 🟠 **POOR** | 25-40 | Noisy (SNR 10-13 dB) — high WER expected | Last resort only |
| 🔴 **BAD** | 10-15 | Barely audible (SNR < 10 dB) | No — discard |
| 🟣 **CLIPPED** | 10-60 | Distortion from mic gain too high | Reduce gain, re-record |
| ⚫ **MUTED** | 0 | No signal (RMS < -55 dBFS) — mic off | Fix recording setup |
| ⚫ **BROKEN** | 0 | Too short (< 1s) — incomplete recording | Discard |

Sample output:

| file | folder | duration_s | snr_db | rms_dbfs | clipping_pct | speech_pct | quality | score | comment |
|---|---|---|---|---|---|---|---|---|---|
| recording_001.m4a | interview_42 | 1841.2 | 27.3 | -18.1 | 0.000 | 67.4 | **EXCELLENT** | 95 | studio-quality (SNR 27 dB, 67% speech) — ideal for ASR |
| recording_002.m4a | interview_43 | 122.0 | 11.0 | -42.1 | 0.002 | 18.0 | **POOR** | 25 | very noisy (SNR 11 dB) — ASR will struggle, high WER expected |
| recording_003.m4a | interview_44 | 1543.7 | 18.5 | -34.2 | 1.450 | 32.1 | **CLIPPED** | 30 | heavy distortion: 1.45% samples clipped — reduce mic gain |

Common options:

```bash
# Custom output path
uv run python scripts/audio_quality_report.py /path/to/audio -o ./reports/q.xlsx

# Limit to specific extensions
uv run python scripts/audio_quality_report.py /path/to/audio --extensions .m4a .wav

# Cap at first N files (for quick spot-checks)
uv run python scripts/audio_quality_report.py /path/to/audio --max-files 50
```

**Customizing thresholds:** quality flags are determined by the
`classify_quality()` function in `scripts/audio_quality_report.py`. Edit the
function directly to match your project's tolerance for noise / silence /
distortion. The thresholds shipped here are tuned for phone-mic field
interview audio.

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
