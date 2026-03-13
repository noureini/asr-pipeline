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

# ISO 639-3 → Whisper full language name (for HF Whisper fine-tunes)
_ISO3_TO_WHISPER_LANG: dict[str, str] = {
    "ben": "bengali", "hin": "hindi", "spa": "spanish", "fra": "french",
    "ara": "arabic", "por": "portuguese", "deu": "german", "zho": "chinese",
    "jpn": "japanese", "kor": "korean", "rus": "russian", "tur": "turkish",
    "ita": "italian", "nld": "dutch", "pol": "polish", "vie": "vietnamese",
    "tha": "thai", "ind": "indonesian", "swa": "swahili", "amh": "amharic",
    "urd": "urdu", "tam": "tamil", "tel": "telugu", "mar": "marathi",
    "guj": "gujarati", "kan": "kannada", "mal": "malayalam", "pan": "punjabi",
    "mya": "burmese", "khm": "khmer", "nep": "nepali", "sin": "sinhala",
    "ukr": "ukrainian", "ces": "czech", "ron": "romanian", "hun": "hungarian",
    "ell": "greek", "heb": "hebrew", "fas": "persian", "fil": "tagalog",
    "cat": "catalan", "swe": "swedish", "fin": "finnish", "dan": "danish",
    "nor": "norwegian", "hrv": "croatian", "bul": "bulgarian", "lit": "lithuanian",
    "slk": "slovak", "slv": "slovenian", "est": "estonian", "lav": "latvian",
    "mkd": "macedonian", "srp": "serbian", "bos": "bosnian", "sqi": "albanian",
    "aze": "azerbaijani", "kaz": "kazakh", "uzb": "uzbek", "mon": "mongolian",
    "hye": "armenian", "kat": "georgian", "bel": "belarusian", "tgk": "tajik",
    "asm": "assamese", "lao": "lao", "msa": "malay", "mri": "maori",
    "cym": "welsh", "gle": "irish", "eus": "basque", "glg": "galician",
    "oci": "occitan", "bre": "breton", "isl": "icelandic", "mlt": "maltese",
    "ltz": "luxembourgish", "yid": "yiddish", "hau": "hausa", "yor": "yoruba",
    "som": "somali", "afr": "afrikaans", "jav": "javanese", "sun": "sundanese",
    "hat": "haitian creole", "pus": "pashto", "tuk": "turkmen", "snd": "sindhi",
    "san": "sanskrit", "tib": "tibetan", "haw": "hawaiian", "lin": "lingala",
    "bak": "bashkir", "tat": "tatar", "mlg": "malagasy", "sho": "shona",
    "eng": "english",
}


def _flush_gpu():
    """Aggressively free GPU memory between model runs."""
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


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

    def save_txt(self, path: Path) -> None:
        """Save results as a readable text comparison file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        lines: list[str] = []
        lines.append("=" * 80)
        lines.append("ASR BENCHMARK COMPARISON")
        lines.append(f"Language: {self.language}  |  Device: {self.device}")
        lines.append("=" * 80)

        audio_files = list(dict.fromkeys(r.audio_file for r in self.results))
        for audio_file in audio_files:
            file_results = [r for r in self.results if r.audio_file == audio_file]
            lines.append("")
            lines.append(f"Audio: {audio_file}")
            lines.append("-" * 80)

            for r in file_results:
                status = "FAIL" if r.error else "OK"
                time_str = f"{r.elapsed_s:.1f}s" if r.elapsed_s > 0 else "-"
                lines.append(f"\n[{r.model_id}]  ({time_str}, {status})")

                if r.error:
                    lines.append(f"  ERROR: {r.error}")
                else:
                    lines.append(f"  TRANSCRIPT:")
                    lines.append(f"  {r.transcript}")

                    if r.translation:
                        lines.append(f"  TRANSLATION ({r.translation_elapsed_s:.1f}s):")
                        lines.append(f"  {r.translation}")

            lines.append("-" * 80)

        lines.append("")
        path.write_text("\n".join(lines), encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────
# Audio preprocessing
# ─────────────────────────────────────────────────────────────────────


@dataclass
class PreprocessedAudio:
    """Result of preprocessing a single audio file."""

    original_path: Path
    wav_path: Path
    chunks: list  # list[AudioChunk]
    work_dir: Path


def preprocess_audio(audio_path: Path) -> PreprocessedAudio:
    """
    Preprocess audio using the pipeline's AudioPreprocessor.

    Runs the full chain: FFmpeg conversion → loudness normalization →
    noise reduction → VAD → chunking.

    Returns PreprocessedAudio with the full WAV and individual chunks.
    """
    from asr_pipeline.config import load_config
    from asr_pipeline.preprocessor import AudioPreprocessor

    cfg = load_config()
    work_dir = Path(tempfile.mkdtemp(prefix="asr_bench_"))
    preprocessor = AudioPreprocessor(cfg.preprocessing, work_dir)

    wav_path, chunks, _non_speech = preprocessor.preprocess(audio_path)
    return PreprocessedAudio(
        original_path=audio_path,
        wav_path=wav_path,
        chunks=chunks,
        work_dir=work_dir,
    )


# ─────────────────────────────────────────────────────────────────────
# Omnilingual model runner
# ─────────────────────────────────────────────────────────────────────


def run_omnilingual_model(
    model_card: str,
    preprocessed: PreprocessedAudio,
    language: str,
    device: str = "cuda",
) -> BenchmarkResult:
    """
    Run an Omnilingual ASR model on preprocessed audio chunks.

    Args:
        model_card: Omnilingual model card (e.g., "omniASR_CTC_300M_v2").
        preprocessed: PreprocessedAudio with chunks from the pipeline preprocessor.
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

        # Feed all chunk WAV paths to the model
        chunk_paths = [str(c.waveform_path) for c in preprocessed.chunks]

        is_ctc = "CTC" in model_card.upper()
        kwargs: dict = {"batch_size": min(len(chunk_paths), 8)}
        if not is_ctc:
            kwargs["lang"] = [language] * len(chunk_paths)

        transcriptions = pipeline.transcribe(chunk_paths, **kwargs)
        elapsed = time.perf_counter() - t0

        # Join chunk transcripts
        text = " ".join(t.strip() for t in transcriptions if t.strip())

        # Cleanup
        del pipeline
        _flush_gpu()

        return BenchmarkResult(
            model_id=model_card,
            audio_file=preprocessed.original_path.name,
            transcript=text,
            elapsed_s=elapsed,
        )

    except Exception as e:
        _flush_gpu()
        return BenchmarkResult(
            model_id=model_card,
            audio_file=preprocessed.original_path.name,
            transcript="",
            elapsed_s=0.0,
            error=str(e),
        )


# ─────────────────────────────────────────────────────────────────────
# HuggingFace model runner
# ─────────────────────────────────────────────────────────────────────


def run_hf_model(
    model_id: str,
    preprocessed: PreprocessedAudio,
    language: str | None = None,
    device: str = "cuda",
) -> BenchmarkResult:
    """
    Run a HuggingFace ASR model (Whisper fine-tune, wav2vec2, etc.)
    on preprocessed audio chunks.

    Uses the transformers `automatic-speech-recognition` pipeline.

    Args:
        model_id: HuggingFace model ID (e.g., "bangla-speech-processing/BanglaASR").
        preprocessed: PreprocessedAudio with chunks from the pipeline preprocessor.
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
        )

        # Build generate kwargs for Whisper models
        # Map ISO 639-3 → Whisper's full language name
        generate_kwargs = {}
        if language and hasattr(pipe.model.config, "forced_decoder_ids"):
            whisper_lang = _ISO3_TO_WHISPER_LANG.get(language, language)
            generate_kwargs["language"] = whisper_lang

        # Transcribe each chunk separately
        texts = []
        for chunk in preprocessed.chunks:
            output = pipe(
                str(chunk.waveform_path),
                generate_kwargs=generate_kwargs,
                return_timestamps=True,
            )
            chunk_text = output["text"].strip() if isinstance(output, dict) else str(output).strip()
            if chunk_text:
                texts.append(chunk_text)

        elapsed = time.perf_counter() - t0
        text = " ".join(texts)

        # Cleanup
        del pipe
        _flush_gpu()

        return BenchmarkResult(
            model_id=model_id,
            audio_file=preprocessed.original_path.name,
            transcript=text,
            elapsed_s=elapsed,
        )

    except Exception as e:
        _flush_gpu()
        return BenchmarkResult(
            model_id=model_id,
            audio_file=preprocessed.original_path.name,
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

    # Preprocess audio files once (full pipeline: convert → normalize → denoise → VAD → chunk)
    preprocessed_files: list[PreprocessedAudio] = []
    for audio_path in audio_paths:
        preprocessed_files.append(preprocess_audio(audio_path))

    try:
        for model_idx, model_id in enumerate(all_models):
            # Flush GPU between models to avoid OOM
            if model_idx > 0:
                logger.info("Flushing GPU memory before loading %s", model_id)
                _flush_gpu()

            for preprocessed in preprocessed_files:
                if is_omnilingual_model(model_id):
                    result = run_omnilingual_model(
                        model_card=model_id,
                        preprocessed=preprocessed,
                        language=language_script,
                        device=device,
                    )
                else:
                    result = run_hf_model(
                        model_id=model_id,
                        preprocessed=preprocessed,
                        language=language,
                        device=device,
                    )
                report.results.append(result)
    finally:
        # Clean up temp directories created by preprocessor
        import shutil

        for p in preprocessed_files:
            try:
                shutil.rmtree(p.work_dir, ignore_errors=True)
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
