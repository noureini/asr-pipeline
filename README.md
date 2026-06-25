# ASR Pipeline — Multilingual Speech Transcription

Multilingual ASR pipeline with speaker diarization, LLM post-processing, and English translation, built for **qualitative research in low-resource language contexts**. It produces speaker-attributed transcripts with the source language and a faithful English translation, in a schema designed for downstream survey analysis.

Supports **1,600+ languages** through a two-tier engine architecture that routes high-resource languages to Whisper and everything else to a low-resource engine (Omnilingual CTC, or a language-specialized model).

> **Flagship path — Bengali two-witness fusion.** The production-hardened use case is **Bengali climate-disaster household surveys** in coastal Bangladesh: a **fully-local, 8 GB-GPU, hallucination-resistant** pipeline that reconciles **two independent ASR witnesses** with a local LLM and **deterministic guards that refuse to fabricate numbers**. See [Bengali Two-Witness Fusion](#bengali-two-witness-fusion-production-path). English and the other languages continue to run through the general multilingual path.

## Table of Contents

- [Pipeline Walkthrough](#pipeline-walkthrough) ← **start here**
- [Architecture](#architecture)
- [Bengali Two-Witness Fusion](#bengali-two-witness-fusion-production-path) ← **the production path**
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

A complete survey transcription workflow runs in sequential steps. Each step has
its own command and produces its own output. You can run them independently or
chain them together.

```
┌─────────────────┐     ┌──────────────────┐     ┌──────────────────┐     ┌────────────────┐
│  Step 0         │     │  Step 1          │     │  Step 2          │     │  Step 3        │
│  Audio quality  │ ──▶ │  Transcribe      │ ──▶ │  Diarize         │ ──▶ │  Output JSON / │
│  → Excel report │     │  + reconcile/    │     │  speakers        │     │  TXT / SRT     │
└─────────────────┘     │    translate     │     └──────────────────┘     └────────────────┘
       ↓ (optional)     └──────────────────┘                                     ↓
   Filter out               ↓ (Bengali)                                   Plug into your
   POOR/BAD audio       two-witness fusion                                survey research
                        (titu ‖ Whisper → Gemma)                          pipeline
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
# General multilingual path (e.g. Swahili)
uv run asr-pipeline transcribe interview.m4a -l swa

# Bengali production path — two-witness fusion (recommended for Bengali)
just fusion interview.m4a
# (equivalent to: uv run asr-pipeline transcribe interview.m4a -l ben -c fusion_production.yaml)

# Whole folder (Survey Solutions style)
just fusion-folder /path/to/audio
```

Outputs a transcript with full text, per-segment source + translation, speaker
IDs, and timestamps. JSON / SRT / Excel are available via `-f`.

The pipeline automatically:
- Preprocesses audio (resample, normalize loudness, VAD-guided chunking)
- Detects language (or uses your `--language` flag)
- Routes to the right engine (Whisper for high-resource; a low-resource engine otherwise)
- Runs speaker diarization (pyannote)
- LLM post-processing for correction and English translation (backend-dependent — see [Translation backends](#translation-backends))

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

The pipeline emits structured output with per-segment text, speaker IDs, timestamps,
and translations. Plug into your downstream survey analysis workflow
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
                     Whisper                              Omnilingual CTC / titu /
                     + word timestamps                    Qwen / IndicConformer
                          |                                     |
                          +------------------+------------------+
                                             |
                                        pyannote
                                        (speaker diarization)
                                             |
                                      Merge + Align
                                             |
                                   Post-processing (backend-dependent)
                                             |
            +--------------------------------+--------------------------------+
            |                  |                      |                        |
       fusion (Bengali)   qwen (default)        translategemma           ct2_nllb
       two-witness +      local GGUF            TranslateGemma 4B         CT2 NLLB +
       Gemma reconcile    correct+translate                              Ollama refine
            |                  |                      |                        |
            +--------------------------------+--------------------------------+
                                             |
                                  Transcript output (.txt / .json / .srt)
```

### Two-Tier Routing

| Tier | Engine | Languages | Expected Accuracy |
|------|--------|-----------|-------------------|
| **HIGH** | Whisper | English, Spanish, French, German, Portuguese, Russian, Chinese, Japanese, Korean, Italian, Dutch, Polish, Turkish, Czech, Swedish, Ukrainian, Romanian, Arabic | WER <10% |
| **NON-HIGH** | Omnilingual CTC / titu / Qwen / IndicConformer | Hindi, Bengali, Nepali, Swahili, Amharic, Oromo, Hausa, Yoruba, Igbo, Tagalog, Burmese, Khmer, + 1,600 more | CER <10% for 78% |

Engines available: `whisper`, `omnilingual`, `titu`, `qwen`, `indic_conformer`.
The engine can be pinned per run via `pipeline.force_engine` (the Bengali
production config forces `titu` as one of the two fusion witnesses).

### Code-Switching

The low-resource engines handle code-switching natively through their multilingual encoders. When a Nepali speaker drops into Hindi mid-sentence, the model just transcribes without switching engines.

## Bengali Two-Witness Fusion (production path)

This is the production-hardened path and the project's main goal: **fully-local, open-source Bengali transcription for sensitive coastal-Bangladesh climate-survey interviews, on an 8 GB GPU, engineered to resist hallucination — especially on numbers.** Select it with `-c fusion_production.yaml`.

### Why this design

| Constraint | Consequence |
|---|---|
| **Sensitive data, no cloud** | Everything is local + open-source — ASR, diarization, and the fusion LLM. No API calls. |
| **8 GB VRAM ceiling** (RTX 2000 Ada) | Models load **sequentially**; the fusion LLM is a 4-bit GGUF with Q8 KV cache. |
| **Survey integrity — numbers must be exact** | A deterministic **GER guard** abstains rather than guess a digit; the LLM decodes **greedy (temp=0)** to resist confabulation. |
| **Degraded 8 kHz dialectal phone audio** | A rate-adaptive VAD spine and **two complementary witnesses** are reconciled rather than trusted individually. |

### How it works

Both witnesses transcribe the **same** VAD-defined spans (a "coupled shared spine"), so the fusion LLM compares like with like:

- **Witness #1 — `titu` FastConformer-CTC** (`hishab/titu_stt_bn_fastconformer`): acoustically literal, preserves how unusual/dialect words actually *sounded*; native CTC word timestamps.
- **Witness #2 — `tugstugi` Whisper-medium** (`bengaliAI/tugstugi_bengaliai-regional-asr_whisper-medium`): fluent seq2seq, well-formed grammar; run as **CT2 batched N-best=3** (auto-falls-back to HF transformers if the CT2 dir is absent).
- **Fusion — Gemma-3-12B** (`unsloth/gemma-3-12b-it-GGUF`, Q4_K_XL, in-process llama.cpp): reconciles the two into a clean `[bn]` (Whisper's structure, titu's word choices where they differ, every content word grounded in at least one witness), then translates to `[en]` — **one call**.
- **Diarization — pyannote community-1**; roles resolved to **ENUMERATOR / RESPONDENT** by a one-shot cluster→role classifier.
- **Forced alignment is intentionally disabled** — titu's native CTC timestamps are the source of truth (an external aligner drifted ~2.15 s and scrambled attribution).

### Hallucination guards (deterministic, post-generation)

This is what makes the output trustworthy for survey use — these hold regardless of what the LLM does:

- **Number-abstain** — a digit in `[bn]` unsupported by either witness is blanked to `[অস্পষ্ট]`/`[unclear]`. Fabricated numbers never ship.
- **Foreign-script abstain** — a token with characters outside Bengali + Latin (a Cyrillic «хозяйства» or Devanagari «लेके» hallucination) is blanked deterministically.
- **Witness-disagree / deviated-from-consensus** — output diverging sharply from both witnesses is flagged for human QA.
- **Greedy decoding (temp=0)** — chosen on evidence: on near-empty backchannel segments, temperature sampling confabulated plausible (sometimes coincidentally true) survey turns; greedy stays grounded and loses no accuracy. `repeat_penalty=1.0` is load-bearing (the llama-cpp default 1.1 drops legitimate dialectal repetition).

Every flagged segment lands on a **review worklist** so a human reviews only the handful that need it.

### Output artifacts (per audio)

| File | Contents |
|---|---|
| `<name>.txt` | The deliverable transcript (`[bn-raw]`/`[bn]`/`[en]`, ENUMERATOR/RESPONDENT) |
| `<name>.review.txt` | Human-QA worklist — only flagged segments, with reasons |
| `<name>.reasoning.txt` | Per-segment `<think>` trace + both witnesses + GER metrics |
| `<name>.raw.txt` | Pre-LLM titu + diarization transcript |

### Run it: a worked example

Transcribe one Bengali interview with the production config. The easiest way is
the `just` recipe:

```bash
# One-liner (wraps the fusion config):
just fusion interview.m4a

# …with options (output dir, project name, log file):
just fusion interview.m4a -o ./outputs -p "Coastal Survey 2026" --log-file ./outputs/run.log

# A whole folder of interviews (Survey Solutions layout, resume-safe):
just fusion-folder ./interviews
```

The recipe just expands to the full CLI command — equivalent to:

```bash
uv run asr-pipeline transcribe interview.m4a -l ben -c fusion_production.yaml
```

The run proceeds through the stages printed in the pipeline plan:

```
Stage 1  Preprocessing      → VAD finds speech spans, chunks onto the shared spine
Stage 2  Language Detection → routed to titu (force_engine=titu)
Stage 3+4 Transcription +   → titu transcribes; pyannote diarizes (2 speakers) — in parallel
          Diarization
Stage 3b Word Timestamps    → titu native CTC word times (forced alignment skipped)
Stage 5  Alignment          → speaker-pure segments
Stage 6a Second Witness     → tugstugi Whisper (CT2), N-best=3 per segment
Stage 6b Fusion+Translation → Gemma-3-12B reconciles both witnesses → [bn] → [en]  (greedy)
Stage 6c Role Attribution   → cluster→role: SPEAKER_01 → ENUMERATOR, other → RESPONDENT
Stage 7  Output             → transcript + review worklist + reasoning trace + raw dump
```

**How the model reconciles one segment.** Take a span where the two witnesses
disagree on a survey-critical word. `titu` (acoustic) preserves the colloquial
word but garbles the run; Whisper (fluent) smooths the grammar but swaps in a
plausible-but-wrong word — here খানা (household) → থানা (police station):

```
LITERAL (titu)   : খানায় ছয়জন সদস্য আছে আমরার ঘরে থাকে সবাই
FLUENT (whisper) : (a) থানায় ছয় জন সদস্য আছে। সবাই আমাদের ঘরে থাকে।
                   (b) ...   (c) ...
```

Gemma runs **once** on both. It is told to take Whisper's sentence structure but
titu's word choices where they differ, and to keep every content word grounded
in at least one witness. Its `<think>` trace (saved to `.reasoning.txt`) reasons
explicitly about the conflict, then emits the two output lines:

```
think: ...LITERAL's "খানা" vs FLUENT's "থানা" — in a household-roster survey
       খানা (household) is intended, not থানা (police station); take LITERAL's
       word, keep FLUENT's two-sentence structure; the number ছয়/6 agrees in
       both witnesses, so copy it exactly...
[bn]   খানায় ছয়জন সদস্য আছে। সবাই আমরার ঘরে থাকে।
[en]   There are six members in the household. They all live in our house.
```

*(Illustrative example — invented, not real interview data.)*

**Then the guards run** (deterministically, on the produced lines):
- the fused `[bn]` is compared against the witness pool — if it contained a
  **number** or **foreign-script token** unsupported by either witness, that
  token is blanked to `[অস্পষ্ট]` and the segment is flagged;
- large divergence from both witnesses raises a `witness-disagree` flag.

Anything flagged is written to `interview.review.txt` so a reviewer checks only
those segments. The clean result lands in `interview.txt`:

```
[00:01:12] RESPONDENT:
[bn-raw] খানায় ছয়জন সদস্য আছে আমরার ঘরে থাকে সবাই
[bn] খানায় ছয়জন সদস্য আছে। সবাই আমরার ঘরে থাকে।
[en] There are six members in the household. They all live in our house.
```

That single segment shows the whole idea: **two imperfect witnesses, one local
LLM that reconciles rather than trusts either, and deterministic guards that
refuse to let a fabricated number or script through.**

### What you need to run it (runtime stack)

You do **not** compile anything or hand-download models. `uv sync` installs the
engines, and the models auto-download from HuggingFace on the first run:

| Component | Package / source | How it's obtained |
|---|---|---|
| Fusion LLM **engine** | `llama-cpp-python` (CUDA wheel) | `uv sync` **automatically downloads** the prebuilt **cu124 GPU wheel** from the configured index — no manual download, and no compiling llama.cpp |
| Fusion **model** | `gemma-3-12b-it-UD-Q4_K_XL.gguf` | auto-downloads from `unsloth/gemma-3-12b-it-GGUF` on first run (~7 GB) |
| Witness #1 — titu | `nemo-toolkit[asr]` | model auto-downloads from HF |
| Witness #2 — Whisper | `ctranslate2` + `faster-whisper` / `transformers` | model auto-downloads from HF |
| Diarization | `pyannote.audio` | **gated** model — needs your HF token |

**Requirements for the GPU fusion path:**
- **x86_64 Linux + NVIDIA GPU** — on this platform `uv` selects the CUDA
  `llama-cpp-python` wheel and the cu128 PyTorch build automatically (the cu124
  CUDA runtime is bundled inside the llama-cpp wheel and runs on a cu128 driver).
  On macOS / non-Linux, `uv` installs the CPU build (works, but slow).
- **~10 GB free disk** for the model weights (HuggingFace cache).
- **HF token** (for the gated pyannote diarization model).

> **About `vendor/llamacpp/`** — the `*.so` files and `llama-server` binary are
> used **only** by the optional server-mode fusion (`fusion_backend: server`,
> for parallel/continuous-batching throughput). The **default in-process path**
> uses the pip-installed `llama-cpp-python` package, so a normal run never
> touches those binaries — which is why they were safe to leave out of the
> commit. You do not need them to run the pipeline.

## Prerequisites

| Required | What | Notes |
|---|---|---|
| Yes | Python ≥ 3.10 | |
| Yes | FFmpeg | for audio decoding (m4a, mp3, …) |
| Yes (for diarization) | HuggingFace token | free, set up in [Installation step 3](#3-set-up-environment-variables-huggingface-token) |
| Recommended | `uv` | Python env manager |
| Recommended | `just` | task runner for the one-line workflows below |
| Recommended | NVIDIA GPU | ≥ 8 GB for the Bengali fusion path; ≥ 16 GB for the full multilingual stack. CPU works but is slow |

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
git clone https://github.com/noureini/asr-pipeline.git
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
[pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1)
(the Bengali fusion path) and/or
[pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
for the general path.

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

The Bengali fusion models (`titu`, `tugstugi` Whisper, Gemma-3-12B GGUF) also
auto-download from HuggingFace on first transcription run into the HF cache
(~10 GB). The CT2-converted Whisper at `models/tugstugi_ct2` is an optional
speed optimization (~1.9× on the second witness); without it the pipeline
auto-falls-back to HF transformers.

### 6. (Optional) LLM correction / refinement via Ollama

```bash
just setup-llm
# equivalent to: pulls a model via ollama
```

Requires [Ollama](https://ollama.com) installed and running. Only used by the
`ct2_nllb` and `ollama`-mode backends — the Bengali `fusion` path runs the LLM
**in-process** (llama.cpp) and needs no Ollama daemon.

### One-shot install (everything)

```bash
just setup
```

### Extras

| Extra | Install | When you need it |
|-------|---------|------------------|
| NeMo MSDD diarization (alternative to pyannote) | `uv sync --extra nemo` | Only if pyannote doesn't fit your hardware constraints |
| Dev tools (pytest, ruff, mypy) | `uv sync --extra dev` | Contributing code |

### Verify everything is wired up

```bash
just gpu                          # shows CUDA availability
uv run asr-pipeline check-deps    # full dependency / model status table
```

### Translation backends

The post-ASR backend is set by `postprocessing.translation_backend`:

| Backend | What it does | Select with |
|---|---|---|
| **`fusion`** | **Bengali two-witness fusion** — titu ‖ Whisper reconciled + translated by Gemma-3-12B, fully local, with GER guards. **Recommended for Bengali.** | `-c fusion_production.yaml` |
| `qwen` | Local GGUF: source-correct then translate (general default in `default.yaml`) | default |
| `translategemma` | TranslateGemma 4B translate + English cleanup | `--config` with `translation_backend: translategemma` |
| `ct2_nllb` | Legacy CT2 NLLB + Ollama joint refinement | `uv run asr-pipeline setup --translation-backend ct2_nllb` |

## Configuration

General defaults live in `src/asr_pipeline/default.yaml`; the **Bengali production
settings live in `fusion_production.yaml`**. Key sections:

| Section | What it controls |
|---------|-----------------|
| `pipeline` | Device (cuda/cpu), compute type, batch size, `force_engine`, seed |
| `preprocessing` | Sample rate, VAD threshold, chunking, resampler (`soxr`) |
| `languages` | Language registry with tier assignments |
| `whisper` / `omnilingual` | Model size / variant for the general engines |
| `diarization` | pyannote model (community-1 in the fusion config), speaker count limits |
| `postprocessing` | Translation backend + fusion hyperparameters (see below) |
| `output` | Format (txt/json/srt), timestamp style |
| `logging` | Log level, file output, progress bars |

### Bengali fusion hyperparameters (`fusion_production.yaml`)

| Setting | Value | Why |
|---|---|---|
| `temperature` | **0.0** | greedy — resists confabulation on near-empty segments (evidence-based) |
| `repeat_penalty` | **1.0** | **must stay 1.0** — 1.1 drops legitimately repeated dialectal entities (repeated river/place names, months) |
| `whisper_nbest` | 3 | second-witness hypotheses fed to fusion |
| `n_ctx` / `flash_attn` / `kv_quant` | 2048 / true / true | the 8 GB-fit knobs |
| `diarization.model` | `…community-1` | min 2 / max 4 speakers |
| `alignment.enabled` | false | use titu native CTC timestamps |

### Overriding Configuration

1. **CLI flags** (highest priority):
   ```bash
   uv run asr-pipeline transcribe audio.m4a -l spa --device cpu --format json
   ```
2. **Custom YAML file**:
   ```bash
   uv run asr-pipeline transcribe audio.m4a -l ben --config fusion_production.yaml
   ```
3. **Environment variables** (prefix with `ASR_PIPELINE_`):
   ```bash
   ASR_PIPELINE_DEVICE=cpu uv run asr-pipeline transcribe audio.m4a -l spa
   ```

### Adding New Languages

New languages go through a NON-HIGH engine. Add an entry to the `languages` section in `default.yaml`:

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
# Bengali — two-witness fusion production path
uv run asr-pipeline transcribe interview.m4a -l ben -c fusion_production.yaml

# General multilingual (Spanish)
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
uv run asr-pipeline transcribe audio.m4a -l ben -c fusion_production.yaml --log-level DEBUG --log-file run.log
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
# Transcribe all interviews in a folder (Bengali fusion)
uv run asr-pipeline transcribe-folder ./interviews --language ben -c fusion_production.yaml

# With custom output directory and all formats
uv run asr-pipeline transcribe-folder ./data -l spa -o ./transcripts -f all

# With project name and speaker limits
uv run asr-pipeline transcribe-folder ./fieldwork -l amh -p "Ethiopia Study" --max-speakers 5
```

**Resume support:** If a run is interrupted, simply re-run the same command. Already-completed interviews (those with existing output directories) are automatically skipped. Only remaining interviews are processed.

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
# Full setup
uv run asr-pipeline setup

# Skip ASR models (only set up translation)
uv run asr-pipeline setup --skip-asr-models

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

##### Advanced — `test-mics` CLI (compare microphones)

For comparing multiple microphones across the same recordings, use the
`test-mics` CLI command which produces a comparative JSON report. The expected
folder structure is Survey Solutions–style with a `README.txt` mapping interview
keys to mic names:

```bash
# Acoustic metrics only (fast — no ASR)
uv run asr-pipeline test-mics ./mic-test-data -l ben --skip-transcription

# Full run including transcription / WER comparison
uv run asr-pipeline test-mics ./mic-test-data -l ben
```

The CLI prints per-file and per-mic tables plus a recommendation, and writes the
full report to `<output_dir>/mic-test-report.json`. Helper scripts
(`scripts/mic_test_per_folder_report.py`, `scripts/mic_test_per_speaker.py`)
aggregate the metrics into per-folder and per-speaker CSV/PNG/XLSX reports.

### Quick Test with Included Audio Files

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

# General: load_config()  |  Bengali fusion: load_config("fusion_production.yaml")
config = load_config("fusion_production.yaml")
pipeline = ASRPipeline(config)

result = pipeline.transcribe(
    audio_path="interview.m4a",
    language="ben",
    project_name="Coastal Survey 2026",
)

for seg in result.segments:
    print(f"[{seg.speaker_id}]")
    print(f"  [bn-raw] {seg.raw_text}")             # what the ASR witness heard
    print(f"  [bn]     {seg.corrected_text}")       # cleaned / fused source
    print(f"  [en]     {seg.refined_translation}")  # English translation
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
| **Build dataset** | `scripts/extract_ipa_local.py` | Stream FLEURS/Bengali_AI_Speech/banspeech/SKNahin, run ZIPA, save IPA + Bengali (+English) pairs as JSONL. Resume-safe. |
| | `scripts/build_lora_dataset_remote.py` | Same, with `--output-format` flag for FST vs IPA vs both. |
| | `scripts/split_ipa_dataset.py` | Stratified train/val split by (source, subsource). |
| **Train** | `scripts/train_lora_ipa_local.py` | LoRA fine-tune a small LLM (Qwen2.5-1.5B/3B) locally on RTX 3060-class GPU. Outputs adapter + GGUF. |
| | `scripts/train_lora_remote.py` | Same training, configured for 16-24 GB GPU boxes. |
| **Evaluate** | `scripts/test_lora_local.py` | Run a trained GGUF on held-out eval samples, report CER per source. |
| | `scripts/compare_lora_vs_baseline.py` | Compare LoRA output vs prior lattice/Gemma baselines on the same FLEURS test samples. |
| | `scripts/eval_lora_corrector.py` | Streaming eval on FLEURS test (CER, WER, per-source breakdown). |
| **Orchestration** | `scripts/run_lora_pipeline.sh` | End-to-end on a remote box: dataset build → training → GGUF export. |

See `scripts/README_lora_remote.md` for the remote-compute workflow and detailed
explanations of dataset diversity strategies, quality filters, and hyperparameter
choices.

## Testing

```bash
uv run pytest -q                                       # all unit tests (no GPU / no models)
uv run pytest --cov=asr_pipeline --cov-report=term-missing
```

| Test area | What it covers |
|------------|---------------|
| Config | YAML loading, engine routing, default values |
| Data models | Pydantic model creation/validation (segments, transcripts) |
| Alignment | Speaker-segment alignment, overlap, merging, word timestamps |
| Formatter | TXT / SRT output, timestamp formatting, non-speech placeholders |
| Language registry | Language lookup, tier routing, Whisper code mapping |
| VAD chunking | Non-speech region extraction, gap thresholds |
| Fusion parsing | `<think>` handling, `[bn]`/`[en]` extraction, role-tagged turns, YAML↔in-code prompt drift guard, think-off ablation |

> The tests are unit tests that do not require a GPU or downloaded models. End-to-end pipeline runs require GPU + models and are validated manually on the reference clip.

```bash
uv run ruff check src/ tests/    # lint
uv run mypy src/asr_pipeline/    # type check
```

## Project Structure

```
asr-pipeline/
|-- pyproject.toml              # Project metadata, dependencies, build config
|-- README.md                   # This file
|-- fusion_production.yaml      # Bengali two-witness fusion production config
|-- .env.example                # Template for environment variables
|-- test_audio.m4a              # Sample English audio for testing
|-- dmi_swa.mp3                 # Sample Swahili audio for testing
|
|-- src/asr_pipeline/
|   |-- cli.py                  # Click CLI (transcribe, transcribe-folder, setup, …)
|   |-- config.py               # Pydantic config loading from YAML
|   |-- default.yaml            # Default configuration file
|   |-- pipeline.py             # Main ASR orchestration pipeline
|   |-- batch.py                # Folder batch processing (interview discovery, merging)
|   |-- preprocessor.py         # Rate-adaptive VAD + shared-spine chunking
|   |-- alignment.py            # Speaker-segment alignment and merging
|   |-- forced_aligner.py       # wav2vec2 MMS forced alignment (disabled in fusion path)
|   |-- diarization.py          # pyannote speaker diarization backend
|   |-- nemo_diarization.py     # NeMo MSDD diarization backend (optional)
|   |-- fusion.py               # ★ two-witness fusion + translation + role classifier
|   |-- ger_guard.py            # ★ number-abstain, foreign-script, witness-disagree guards
|   |-- postprocessor.py        # LLM correction/translation backends (GGUFProcessor)
|   |-- formatter.py            # Output formatting (TXT, JSON, SRT, review, reasoning)
|   |-- language.py             # Language registry and routing
|   |-- logging_config.py       # Logging setup (Rich console, pipeline plan)
|   |-- models.py               # Pydantic data models (segments, transcripts)
|   |-- engines/
|       |-- whisper_engine.py       # Whisper ASR engine (HIGH tier)
|       |-- omnilingual_engine.py   # Omnilingual CTC engine (NON-HIGH)
|       |-- titu_engine.py          # ★ titu FastConformer-CTC (Bengali witness #1)
|       |-- qwen_engine.py          # Qwen ASR engine
|       |-- indic_conformer_engine.py # IndicConformer engine
|
|-- config/prompts/bn_survey.yaml   # fusion + role prompts (operational config)
|-- tests/                      # Unit tests
|-- test_data/                  # Sample interview folders for batch testing
|-- scripts/                    # Audio quality, mic-test, LoRA fine-tuning tooling
|-- outputs/                    # Generated transcripts (git-ignored)
```

## Supported Languages

### Pre-configured (33 languages)

**High-resource (Whisper):** English, Spanish, French, German, Portuguese, Russian, Chinese, Japanese, Korean, Italian, Dutch, Polish, Turkish, Czech, Swedish, Ukrainian, Romanian, Arabic

**Non-high-resource:** Hindi, Bengali, Nepali, Swahili, Amharic, Afaan Oromo, Hausa, Yoruba, Igbo, Tagalog, Burmese, Khmer, Kinyarwanda, Somali, Tigrinya

### Adding More

Any of the 1,600+ languages supported by the Omnilingual engine can be added via the config file. See [Adding New Languages](#adding-new-languages).

## Output Format

Standard qualitative research transcript. For the Bengali fusion path, source
lines are split into raw ASR (`[bn-raw]`, audit trail) and the cleaned
reconstruction (`[bn]`), and speaker labels are resolved to roles:

```
========================================================================
TRANSCRIPT
========================================================================
Project:        Coastal Survey 2026
Date:           2026-06-25
Duration:       00:05:00
Audio File:     interview.m4a
Languages:      Bengali
Speakers:       2 identified
Transcription:  Intelligent Verbatim
ASR Engines:    titu (FastConformer-CTC Bengali)
Post-processed: Two-witness fusion + translation (titu ‖ Whisper → Gemma-3-12B)
========================================================================

[00:01:12] RESPONDENT:
[bn-raw] খানায় ছয়জন সদস্য আছে আমরার ঘরে থাকে সবাই
[bn] খানায় ছয়জন সদস্য আছে। সবাই আমরার ঘরে থাকে।
[en] There are six members in the household. They all live in our house.

========================================================================
END OF TRANSCRIPT
========================================================================
```

For the general multilingual path the source is a single `[lang]` line plus
`[en]`. Three output formats are available:
- **TXT**: Human-readable transcript (default). The fusion path additionally
  writes `.review.txt`, `.reasoning.txt`, and `.raw.txt` (see
  [Output artifacts](#output-artifacts-per-audio)).
- **JSON**: Structured data with all metadata and segments
- **SRT**: Subtitle format for video editors

## Hardware Requirements

| Component | 8 GB GPU | 16 GB GPU | CPU Only |
|-----------|----------|-----------|----------|
| **Bengali fusion stack** (titu + Whisper + Gemma-3-12B Q4_K_XL) | ✅ sequential load | ✅ | Slow but works |
| Whisper (general HIGH tier) | ~10 GB | ~10 GB | Slow but works |
| Omnilingual CTC 300M | ~2 GB | ~2 GB | Slow but works |
| pyannote diarization | ~2 GB | ~2 GB | Slow but works |
| TranslateGemma 4B (4-bit) | ~3 GB | ~3 GB | Slow but works |

The **Bengali fusion path is tuned to fit 8 GB** (RTX 2000 Ada) by loading models
sequentially: titu and Whisper run and unload, then Gemma-3-12B loads with
`n_ctx=2048`, flash-attention, and Q8 KV cache. A 5-minute clip takes ~6 minutes.
The full general multilingual stack is comfortable on a **16 GB** GPU.

## Troubleshooting

**`CUDA out of memory`** — the fusion path is tuned for 8 GB and loads models sequentially. If you still hit OOM, lower `n_ctx` in `fusion_production.yaml`, or lower the batch size in `default.yaml`:
```yaml
pipeline:
  batch_size: 4   # default is 8
whisper:
  batch_size: 4   # default is 8
```
Or switch to CPU mode: `uv run asr-pipeline transcribe audio.m4a -l spa --device cpu`

**`pyannote` authentication error** — ensure `.env` has a valid `HF_TOKEN` and you've accepted the model terms ([community-1](https://huggingface.co/pyannote/speaker-diarization-community-1) for the fusion path, or [3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) for the general path).

**Second witness (Whisper) is slow** — you're on the HF-transformers fallback. Provide a CT2-converted model at `models/tugstugi_ct2` for the ~1.9× batched path.

**A number shows as `[অস্পষ্ট]`** — that's the guard working: the witnesses didn't agree on the digit, so it abstained rather than guess. Check the segment in `*.review.txt` against the audio.

**FFmpeg not found** — `sudo apt install ffmpeg` (Linux/WSL) / `brew install ffmpeg` (macOS) / `choco install ffmpeg` (Windows).

**Tests fail with import errors** — `uv sync --extra dev`.

## License

MIT
