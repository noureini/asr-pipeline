"""
ASR model benchmarking tool.

Compares multiple ASR models (Omnilingual variants, HuggingFace fine-tuned
Whisper models) on the same audio files side-by-side. Useful for finding the
best model for a specific language before committing to a full pipeline run.

Models are loaded one at a time and unloaded before the next to stay within
GPU memory limits.
"""

from __future__ import annotations

import gc
import json
import logging
import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger("asr_pipeline")

# Known Omnilingual model cards (prefix check)
_OMNILINGUAL_PREFIXES = ("omniASR_CTC_", "omniASR_LLM_")


def is_omnilingual_model(model_id: str) -> bool:
    """Check if a model ID refers to an Omnilingual model card."""
    return any(model_id.startswith(p) for p in _OMNILINGUAL_PREFIXES)


@dataclass
class BenchmarkResult:
    """Result from running a single model on a single audio file."""

    model_id: str
    audio_file: str
    transcript: str
    elapsed_s: float
    error: str | None = None
    translation: str | None = None
    translation_elapsed_s: float = 0.0


@dataclass
class BenchmarkReport:
    """Full benchmark report across all models and audio files."""

    language: str
    device: str
    results: list[BenchmarkResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "language": self.language,
            "device": self.device,
            "results": [
                {
                    "model_id": r.model_id,
                    "audio_file": r.audio_file,
                    "transcript": r.transcript,
                    "elapsed_s": round(r.elapsed_s, 2),
                    "error": r.error,
                    "translation": r.translation,
                    "translation_elapsed_s": round(r.translation_elapsed_s, 2),
                }
                for r in self.results
            ],
        }

    def save_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False))


# ─────────────────────────────────────────────────────────────────────
# Audio preprocessing
# ─────────────────────────────────────────────────────────────────────


def preprocess_audio(audio_path: Path, target_sr: int = 16_000) -> Path:
    """
    Convert audio to 16kHz mono WAV via FFmpeg (matches pipeline preprocessor).

    Returns path to the temporary WAV file.
    """
    import subprocess

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()

    subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(audio_path),
            "-ac", "1", "-ar", str(target_sr),
            "-acodec", "pcm_s16le",
            "-loglevel", "error",
            tmp.name,
        ],
        check=True,
        capture_output=True,
    )
    return Path(tmp.name)


# ─────────────────────────────────────────────────────────────────────
# Omnilingual model runner
# ─────────────────────────────────────────────────────────────────────


def run_omnilingual_model(
    model_card: str,
    audio_path: Path,
    language: str,
    device: str = "cuda",
) -> BenchmarkResult:
    """
    Run an Omnilingual ASR model on a single audio file.

    Args:
        model_card: Omnilingual model card (e.g., "omniASR_CTC_300M_v2").
        audio_path: Path to 16kHz mono WAV file.
        language: Language in {code}_{script} format (e.g., "ben_Beng").
        device: "cuda" or "cpu".

    Returns:
        BenchmarkResult with transcript and timing.
    """
    import torch

    try:
        # fairseq2 CONDA_PREFIX workaround
        conda_prefix = os.environ.pop("CONDA_PREFIX", None)
        from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

        if conda_prefix is not None:
            os.environ["CONDA_PREFIX"] = conda_prefix

        t0 = time.perf_counter()
        pipeline = ASRInferencePipeline(model_card=model_card)

        is_ctc = "CTC" in model_card.upper()
        kwargs: dict = {"batch_size": 1}
        if not is_ctc:
            kwargs["lang"] = [language]

        transcriptions = pipeline.transcribe([str(audio_path)], **kwargs)
        elapsed = time.perf_counter() - t0

        text = transcriptions[0].strip() if transcriptions else ""

        # Cleanup
        del pipeline
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return BenchmarkResult(
            model_id=model_card,
            audio_file=audio_path.name,
            transcript=text,
            elapsed_s=elapsed,
        )

    except Exception as e:
        return BenchmarkResult(
            model_id=model_card,
            audio_file=audio_path.name,
            transcript="",
            elapsed_s=0.0,
            error=str(e),
        )


# ─────────────────────────────────────────────────────────────────────
# HuggingFace model runner
# ─────────────────────────────────────────────────────────────────────


def run_hf_model(
    model_id: str,
    audio_path: Path,
    language: str | None = None,
    device: str = "cuda",
) -> BenchmarkResult:
    """
    Run a HuggingFace ASR model (Whisper fine-tune, wav2vec2, etc.)
    on a single audio file.

    Uses the transformers `automatic-speech-recognition` pipeline.

    Args:
        model_id: HuggingFace model ID (e.g., "bangla-speech-processing/BanglaASR").
        audio_path: Path to audio file (any format torchaudio can read).
        language: Optional language hint (used for Whisper models).
        device: "cuda" or "cpu".

    Returns:
        BenchmarkResult with transcript and timing.
    """
    import torch

    try:
        import transformers

        device_idx = 0 if device == "cuda" and torch.cuda.is_available() else -1
        torch_dtype = torch.float16 if device_idx >= 0 else torch.float32

        t0 = time.perf_counter()

        pipe = transformers.pipeline(
            "automatic-speech-recognition",
            model=model_id,
            torch_dtype=torch_dtype,
            device=device_idx,
            chunk_length_s=30,
            stride_length_s=(4, 2),
        )

        # Build generate kwargs for Whisper models
        generate_kwargs = {}
        if language and hasattr(pipe.model.config, "forced_decoder_ids"):
            generate_kwargs["language"] = language

        output = pipe(
            str(audio_path),
            generate_kwargs=generate_kwargs,
            return_timestamps=True,
        )

        elapsed = time.perf_counter() - t0
        text = output["text"].strip() if isinstance(output, dict) else str(output).strip()

        # Cleanup
        del pipe
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return BenchmarkResult(
            model_id=model_id,
            audio_file=audio_path.name,
            transcript=text,
            elapsed_s=elapsed,
        )

    except Exception as e:
        return BenchmarkResult(
            model_id=model_id,
            audio_file=audio_path.name,
            transcript="",
            elapsed_s=0.0,
            error=str(e),
        )


# ─────────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────────


def benchmark_models(
    audio_paths: list[Path],
    model_ids: list[str],
    language: str,
    language_script: str,
    device: str = "cuda",
    baseline_model: str = "omniASR_CTC_300M_v2",
) -> BenchmarkReport:
    """
    Run all models on all audio files and collect results.

    The baseline model (current pipeline engine) is always included.
    Models are loaded one at a time to fit in limited VRAM.

    Args:
        audio_paths: List of audio file paths.
        model_ids: List of model IDs to benchmark (HF or Omnilingual).
        language: ISO 639-3 language code (e.g., "ben").
        language_script: Language in {code}_{script} format (e.g., "ben_Beng").
        device: "cuda" or "cpu".
        baseline_model: Omnilingual model card for the baseline.

    Returns:
        BenchmarkReport with all results.
    """
    report = BenchmarkReport(language=language, device=device)

    # Build the full model list: baseline first, then user-specified
    all_models = [baseline_model]
    for m in model_ids:
        if m != baseline_model:
            all_models.append(m)

    # Preprocess audio files once
    wav_paths: list[tuple[Path, Path]] = []  # (original, preprocessed)
    for audio_path in audio_paths:
        wav_path = preprocess_audio(audio_path)
        wav_paths.append((audio_path, wav_path))

    try:
        for model_id in all_models:
            for original_path, wav_path in wav_paths:
                if is_omnilingual_model(model_id):
                    result = run_omnilingual_model(
                        model_card=model_id,
                        audio_path=wav_path,
                        language=language_script,
                        device=device,
                    )
                else:
                    result = run_hf_model(
                        model_id=model_id,
                        audio_path=wav_path,
                        language=language,
                        device=device,
                    )
                # Use the original filename for display
                result.audio_file = original_path.name
                report.results.append(result)
    finally:
        # Clean up temp files
        for _, wav_path in wav_paths:
            try:
                wav_path.unlink(missing_ok=True)
            except OSError:
                pass

    return report


# ─────────────────────────────────────────────────────────────────────
# Translation pass
# ─────────────────────────────────────────────────────────────────────


def translate_results(
    report: BenchmarkReport,
    language: str,
    device: str = "cuda",
) -> None:
    """
    Translate all benchmark transcripts using TranslateGemma.

    Modifies results in-place, adding translation and timing.

    Args:
        report: BenchmarkReport with transcripts to translate.
        language: ISO 639-3 language code (e.g., "ben").
        device: "cuda" or "cpu".
    """
    import torch

    from asr_pipeline.postprocessor import TranslateGemmaTranslator

    # Skip if language is English
    if language in ("eng", "en"):
        for r in report.results:
            r.translation = r.transcript
        return

    # Map ISO 639-3 to BCP-47 for TranslateGemma
    _ISO3_TO_BCP47 = {
        "ben": "bn", "hin": "hi", "spa": "es", "fra": "fr", "ara": "ar",
        "por": "pt", "deu": "de", "zho": "zh", "jpn": "ja", "kor": "ko",
        "rus": "ru", "tur": "tr", "ita": "it", "nld": "nl", "pol": "pl",
        "vie": "vi", "tha": "th", "ind": "id", "swa": "sw", "amh": "am",
        "urd": "ur", "tam": "ta", "tel": "te", "mar": "mr", "guj": "gu",
        "kan": "kn", "mal": "ml", "pan": "pa", "mya": "my", "khm": "km",
        "nep": "ne", "sin": "si", "ukr": "uk", "ces": "cs", "ron": "ro",
        "hun": "hu", "ell": "el", "heb": "he", "fas": "fa", "fil": "tl",
    }
    source_bcp47 = _ISO3_TO_BCP47.get(language, language)

    translator = TranslateGemmaTranslator(
        quantize="4bit",
        device=device,
    )
    translator.load()

    try:
        # Translate each result individually so we get per-model timing
        for r in report.results:
            if r.error or not r.transcript.strip():
                continue
            t0 = time.perf_counter()
            translations = translator.translate_batch(
                [r.transcript],
                source_bcp47=source_bcp47,
                target_bcp47="en",
            )
            r.translation_elapsed_s = time.perf_counter() - t0
            r.translation = translations[0] if translations else ""
    finally:
        translator.unload()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ─────────────────────────────────────────────────────────────────────
# Display
# ─────────────────────────────────────────────────────────────────────


def display_results(report: BenchmarkReport) -> None:
    """Display benchmark results as a Rich table."""
    from rich.console import Console
    from rich.table import Table
    from rich.text import Text

    console = Console()

    has_translations = any(r.translation is not None for r in report.results)

    # Group results by audio file
    audio_files = list(dict.fromkeys(r.audio_file for r in report.results))

    for audio_file in audio_files:
        file_results = [r for r in report.results if r.audio_file == audio_file]

        table = Table(
            title=f"Benchmark: {audio_file}",
            show_header=True,
            header_style="bold cyan",
            show_lines=True,
            width=160 if has_translations else 120,
        )
        table.add_column("Model", style="bold", width=35, no_wrap=True)
        table.add_column("Transcript", width=55)
        if has_translations:
            table.add_column("English Translation", width=45)
        table.add_column("Time", width=8, justify="right")
        table.add_column("Status", width=8, justify="center")

        for r in file_results:
            if r.error:
                status = Text("\u2717 FAIL", style="red")
                transcript = Text(r.error, style="dim red")
                row = [r.model_id, transcript]
                if has_translations:
                    row.append(Text("-", style="dim"))
            else:
                status = Text("\u2713 OK", style="green")
                transcript = Text(
                    r.transcript[:200] + ("..." if len(r.transcript) > 200 else ""),
                )
                row = [r.model_id, transcript]
                if has_translations:
                    if r.translation:
                        trans_text = r.translation[:180] + ("..." if len(r.translation) > 180 else "")
                        row.append(Text(trans_text, style="italic"))
                    else:
                        row.append(Text("-", style="dim"))

            time_str = f"{r.elapsed_s:.1f}s" if r.elapsed_s > 0 else "-"
            if has_translations and r.translation_elapsed_s > 0:
                time_str += f" +{r.translation_elapsed_s:.1f}s"

            row.extend([time_str, status])
            table.add_row(*row)

        console.print(table)
        console.print()


def collect_audio_files(path: Path) -> list[Path]:
    """Collect audio files from a path (file or directory)."""
    audio_extensions = {".m4a", ".wav", ".mp3", ".flac", ".ogg", ".opus", ".wma"}

    if path.is_file():
        return [path]

    if path.is_dir():
        files = []
        for ext in audio_extensions:
            files.extend(path.rglob(f"*{ext}"))
        return sorted(files)

    return []
