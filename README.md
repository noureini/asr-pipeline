# ASR Pipeline — Multilingual Speech Transcription

Production-ready multilingual ASR pipeline with speaker diarization, LLM post-processing, and English translation. Built for qualitative research in low-resource language contexts.

Supports **1,600+ languages** through a two-tier engine architecture that routes high-resource languages to Whisper Large-v3 and everything else to Meta's Omnilingual CTC 300M.

## Table of Contents

- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
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

#### Transcribe Audio

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

#### List Supported Languages

```bash
uv run asr-pipeline list-languages
```

#### Check Dependencies

```bash
uv run asr-pipeline check-deps
```

#### Setup Post-Processing Models

```bash
uv run asr-pipeline setup
uv run asr-pipeline setup --translation-backend ct2_nllb
```

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
|   |-- cli.py                  # Click CLI (transcribe, list-languages, check-deps, setup)
|   |-- config.py               # Pydantic config loading from YAML
|   |-- default.yaml            # Default configuration file
|   |-- pipeline.py             # Main ASR orchestration pipeline
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
|   |-- engines/
|       |-- __init__.py         # Engine exports
|       |-- whisper_engine.py   # Whisper Large-v3 ASR engine
|       |-- omnilingual_engine.py # Omnilingual CTC 300M ASR engine
|
|-- tests/
|   |-- test_pipeline.py        # Unit tests (config, models, alignment, formatting)
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
