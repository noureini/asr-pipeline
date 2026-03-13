"""
Command-line interface for the ASR pipeline.

Provides a clean CLI using Click for transcribing audio files
with full control over language, output format, and pipeline
configuration.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

# Suppress torchaudio / pyannote / speechbrain deprecation warnings globally.
# These are triggered by torchaudio's upcoming torchcodec migration and are
# purely informational — they don't affect functionality.
warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote")
warnings.filterwarnings("ignore", category=UserWarning, module="speechbrain")

import click

from asr_pipeline import __version__


@click.group()
@click.version_option(version=__version__, prog_name="asr-pipeline")
def main() -> None:
    """
    Multilingual ASR Pipeline — Production-ready speech transcription.

    Transcribe audio files in 1,600+ languages with speaker diarization,
    LLM post-processing, and English translation.
    """
    pass


@main.command()
@click.argument("audio_file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--language", "-l",
    required=True,
    help="ISO 639-3 language code (e.g., eng, spa, hin, ben, swa, amh).",
)
@click.option(
    "--output-dir", "-o",
    type=click.Path(path_type=Path),
    default=None,
    help="Output directory for transcript files. Default: ./outputs",
)
@click.option(
    "--config", "-c",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to custom YAML config file.",
)
@click.option(
    "--format", "-f",
    "output_format",
    type=click.Choice(["txt", "json", "srt", "all"], case_sensitive=False),
    default="txt",
    help="Output format. Default: txt",
)
@click.option(
    "--project", "-p",
    default="",
    help="Project name for transcript header.",
)
@click.option(
    "--min-speakers",
    type=int,
    default=None,
    help="Minimum number of speakers (for diarization).",
)
@click.option(
    "--max-speakers",
    type=int,
    default=None,
    help="Maximum number of speakers (for diarization).",
)
@click.option(
    "--device",
    type=click.Choice(["cuda", "cpu"], case_sensitive=False),
    default=None,
    help="Compute device. Default: cuda (from config).",
)
@click.option(
    "--diarization-backend",
    type=click.Choice(["pyannote", "nemo_msdd"], case_sensitive=False),
    default=None,
    help="Diarization backend. Default: pyannote (from config).",
)
@click.option(
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
    default=None,
    help="Logging level. Default: INFO",
)
@click.option(
    "--log-file",
    type=click.Path(path_type=Path),
    default=None,
    help="Also write logs to this file.",
)
def transcribe(
    audio_file: Path,
    language: str,
    output_dir: Optional[Path],
    config: Optional[Path],
    output_format: str,
    project: str,
    min_speakers: Optional[int],
    max_speakers: Optional[int],
    device: Optional[str],
    diarization_backend: Optional[str],
    log_level: Optional[str],
    log_file: Optional[Path],
) -> None:
    """
    Transcribe an audio file with speaker diarization and translation.

    \b
    Examples:
        asr-pipeline transcribe recording.m4a --language spa
        asr-pipeline transcribe interview.wav -l hin -o ./transcripts
        asr-pipeline transcribe focus_group.mp3 -l amh --max-speakers 5
        asr-pipeline transcribe meeting.wav -l eng -f all
    """
    from dotenv import load_dotenv

    from asr_pipeline.config import load_config
    from asr_pipeline.logging_config import setup_logging
    from asr_pipeline.pipeline import ASRPipeline

    # Load .env file (for HF_TOKEN, etc.)
    load_dotenv()

    # Load configuration
    cfg = load_config(config)

    # Apply CLI overrides
    if device:
        cfg.pipeline.device = device
    if output_format:
        cfg.output.format = output_format
    if diarization_backend:
        cfg.diarization.backend = diarization_backend
    if log_level:
        cfg.logging.level = log_level

    # Setup logging
    setup_logging(
        level=cfg.logging.level,
        log_file=str(log_file) if log_file else cfg.logging.file,
        fmt=cfg.logging.format,
    )

    # Run pipeline
    pipeline = ASRPipeline(cfg)
    pipeline.transcribe(
        audio_path=audio_file,
        language=language,
        output_dir=output_dir,
        project_name=project,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
    )


@main.command("transcribe-folder")
@click.argument(
    "folder",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.option(
    "--language", "-l",
    required=True,
    help="ISO 639-3 language code (e.g., spa, eng, hin).",
)
@click.option(
    "--output-dir", "-o",
    type=click.Path(path_type=Path),
    default=None,
    help="Output directory. Default: ./outputs",
)
@click.option(
    "--config", "-c",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Custom YAML config file.",
)
@click.option(
    "--format", "-f", "output_format",
    type=click.Choice(["txt", "json", "srt", "all"], case_sensitive=False),
    default="all",
    show_default=True,
    help="Output format.",
)
@click.option("--project", "-p", default="", help="Project name for transcript header.")
@click.option("--min-speakers", type=int, default=None, help="Minimum speakers per file.")
@click.option("--max-speakers", type=int, default=None, help="Maximum speakers per file.")
@click.option(
    "--device",
    type=click.Choice(["cuda", "cpu"], case_sensitive=False),
    default=None,
    help="Override device.",
)
@click.option(
    "--diarization-backend",
    type=click.Choice(["pyannote", "nemo_msdd"], case_sensitive=False),
    default=None,
    help="Override diarization backend.",
)
@click.option(
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
    default=None,
)
@click.option("--log-file", type=click.Path(path_type=Path), default=None)
def transcribe_folder(
    folder: Path,
    language: str,
    output_dir: Optional[Path],
    config: Optional[Path],
    output_format: str,
    project: str,
    min_speakers: Optional[int],
    max_speakers: Optional[int],
    device: Optional[str],
    diarization_backend: Optional[str],
    log_level: Optional[str],
    log_file: Optional[Path],
) -> None:
    """
    Transcribe a folder of Survey Solutions interviews.

    \b
    Expects the folder structure:
        FOLDER/{interview_key}/AudioAudit/*.m4a

    Produces one consolidated transcript per interview with absolute
    timestamps derived from the recording start time encoded in each
    audio filename.

    \b
    Examples:
        asr-pipeline transcribe-folder ./interviews -l spa
        asr-pipeline transcribe-folder ./data -l spa -o ./transcripts -f all
    """
    import time

    from dotenv import load_dotenv

    from asr_pipeline.batch import discover_interviews, merge_interview_results
    from asr_pipeline.config import load_config
    from asr_pipeline.formatter import write_transcript
    from asr_pipeline.logging_config import console, setup_logging
    from asr_pipeline.pipeline import ASRPipeline

    load_dotenv()

    cfg = load_config(config)

    # Apply CLI overrides
    if device:
        cfg.pipeline.device = device
    if output_format:
        cfg.output.format = output_format
    if diarization_backend:
        cfg.diarization.backend = diarization_backend
    if log_level:
        cfg.logging.level = log_level

    setup_logging(
        level=cfg.logging.level,
        log_file=str(log_file) if log_file else cfg.logging.file,
        fmt=cfg.logging.format,
    )

    output_dir = Path(output_dir or cfg.pipeline.output_dir)

    # Discover interviews
    console.print()
    console.print("[bold cyan]ASR Pipeline — Batch Folder Processing[/bold cyan]")
    console.print()
    console.print(f"  Scanning [bold]{folder}[/bold] ...")

    interviews = discover_interviews(folder)
    if not interviews:
        console.print("[red]No interviews found.[/red] Check folder structure.")
        console.print("  Expected: FOLDER/{interview_key}/AudioAudit/*.m4a")
        return

    total_files = sum(len(files) for files in interviews.values())
    console.print(
        f"  Found [bold]{len(interviews)}[/bold] interview(s), "
        f"[bold]{total_files}[/bold] audio file(s)"
    )
    console.print()

    # Filter out already-completed interviews (resume support)
    skipped = 0
    remaining: dict[str, list] = {}
    for interview_key, files in sorted(interviews.items()):
        interview_output_dir = output_dir / interview_key
        if interview_output_dir.exists() and any(interview_output_dir.iterdir()):
            skipped += 1
            continue
        remaining[interview_key] = files

    if skipped:
        console.print(
            f"  [dim]Skipping {skipped} already-completed interview(s)[/dim]"
        )

    if not remaining:
        console.print("[green]All interviews already processed![/green]")
        return

    remaining_files = sum(len(files) for files in remaining.values())
    console.print(
        f"  Processing [bold]{len(remaining)}[/bold] remaining interview(s), "
        f"[bold]{remaining_files}[/bold] audio file(s)"
    )
    console.print()

    # Create pipeline (models loaded once, reused across all files)
    pipeline = ASRPipeline(cfg)
    batch_start = time.perf_counter()

    for interview_idx, (interview_key, files) in enumerate(
        sorted(remaining.items()), start=1
    ):
        console.print(
            f"[bold]Interview {interview_idx}/{len(remaining)}: "
            f"{interview_key}[/bold] ({len(files)} file(s))"
        )

        file_results = []
        for file_idx, file_info in enumerate(files, start=1):
            console.print(
                f"  [{file_idx}/{len(files)}] {file_info.filename}"
            )
            try:
                result = pipeline.transcribe(
                    audio_path=file_info.file_path,
                    language=language,
                    output_dir=output_dir,
                    project_name=project,
                    min_speakers=min_speakers,
                    max_speakers=max_speakers,
                    write_output=False,
                )
                file_results.append((file_info, result))
            except Exception as e:
                console.print(f"    [red]\u2717 Failed: {e}[/red]")
                continue

        if not file_results:
            console.print(f"  [yellow]No files processed for {interview_key}[/yellow]")
            continue

        # Merge into consolidated transcript
        merged = merge_interview_results(
            file_results,
            interview_key=interview_key,
            project_name=project,
        )

        # Write output
        interview_output_dir = output_dir / interview_key
        written_files = write_transcript(
            merged,
            interview_output_dir,
            "transcript",
            output_format=output_format,
            timestamp_fmt="HH:MM:SS.mmm",
            include_raw=cfg.output.include_raw_text,
            include_translation=cfg.output.include_translation,
        )

        for f in written_files:
            console.print(f"  [green]\u2713[/green] {f}")
        console.print()

    # Release all models at end of batch
    pipeline.cleanup_all()

    batch_elapsed = time.perf_counter() - batch_start
    console.print(
        f"[green]Batch complete![/green] "
        f"{len(remaining)} interview(s) processed in "
        f"{batch_elapsed:.1f}s"
        f"{f' ({skipped} skipped)' if skipped else ''}"
    )
    console.print(f"  Output: [bold]{output_dir}[/bold]")


@main.command()
@click.argument("audio", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--language", "-l",
    required=True,
    help="ISO 639-3 language code (e.g., ben, hin, swa).",
)
@click.option(
    "--model", "-m",
    "model_ids",
    multiple=True,
    required=True,
    help=(
        "Model to benchmark. Repeat for multiple models. "
        "Use Omnilingual model cards (e.g., omniASR_LLM_300M_v2) "
        "or HuggingFace model IDs (e.g., bangla-speech-processing/BanglaASR)."
    ),
)
@click.option(
    "--device",
    type=click.Choice(["cuda", "cpu"], case_sensitive=False),
    default=None,
    help="Compute device. Default: cuda.",
)
@click.option(
    "--save", "-s",
    type=click.Path(path_type=Path),
    default=None,
    help="Save results to a JSON file.",
)
@click.option(
    "--baseline",
    default="omniASR_CTC_300M_v2",
    show_default=True,
    help="Baseline Omnilingual model card (always included).",
)
def benchmark(
    audio: Path,
    language: str,
    model_ids: tuple[str, ...],
    device: Optional[str],
    save: Optional[Path],
    baseline: str,
) -> None:
    """
    Benchmark ASR models side-by-side on the same audio.

    Compares multiple ASR models (Omnilingual variants, HuggingFace
    fine-tuned models) on the same audio files. Models are loaded one
    at a time to stay within GPU memory limits.

    The current pipeline baseline model is always included for comparison.

    \b
    Examples:
        asr-pipeline benchmark audio.m4a -l ben -m omniASR_LLM_300M_v2
        asr-pipeline benchmark audio.m4a -l ben -m omniASR_LLM_300M_v2 -m bangla-speech-processing/BanglaASR
        asr-pipeline benchmark ./folder/ -l ben -m omniASR_LLM_300M_v2 --save results.json
    """
    from asr_pipeline.benchmark import (
        benchmark_models,
        collect_audio_files,
        display_results,
    )
    from asr_pipeline.config import load_config
    from asr_pipeline.logging_config import console

    console.print()
    console.print("[bold cyan]ASR Pipeline — Model Benchmark[/bold cyan]")
    console.print()

    # Resolve device
    if device is None:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Resolve language script from config
    cfg = load_config()
    lang_cfg = cfg.languages.get(language)
    if lang_cfg:
        language_script = f"{language}_{lang_cfg.script}"
    else:
        language_script = f"{language}_Latn"

    # Collect audio files
    audio_files = collect_audio_files(audio)
    if not audio_files:
        console.print(f"[red]No audio files found at {audio}[/red]")
        return

    console.print(f"  Language:  [bold]{language}[/bold] ({language_script})")
    console.print(f"  Device:    [bold]{device}[/bold]")
    console.print(f"  Baseline:  [bold]{baseline}[/bold]")
    console.print(f"  Models:    {', '.join(model_ids)}")
    console.print(f"  Audio:     {len(audio_files)} file(s)")
    console.print()

    report = benchmark_models(
        audio_paths=audio_files,
        model_ids=list(model_ids),
        language=language,
        language_script=language_script,
        device=device,
        baseline_model=baseline,
    )

    display_results(report)

    if save:
        report.save_json(save)
        console.print(f"[green]Results saved to {save}[/green]")


@main.command()
def list_languages() -> None:
    """List all configured languages and their tier assignments."""
    from rich.table import Table

    from asr_pipeline.config import load_config
    from asr_pipeline.logging_config import console

    cfg = load_config()

    table = Table(
        title="Configured Languages",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Code", style="bold", width=6)
    table.add_column("Name", width=20)
    table.add_column("Tier", width=10)
    table.add_column("Engine", width=22)
    table.add_column("NLLB Code", width=12)
    table.add_column("Script", width=8)

    for code, lang in sorted(cfg.languages.items()):
        tier_style = "green" if lang.tier.value == "high" else "yellow"
        engine = "Whisper Large-v3" if lang.tier.value == "high" else "Omnilingual CTC 300M"
        table.add_row(
            code,
            lang.name,
            f"[{tier_style}]{lang.tier.value}[/{tier_style}]",
            engine,
            lang.nllb_code,
            lang.script,
        )

    console.print(table)
    console.print(
        f"\n[dim]Total: {len(cfg.languages)} languages "
        f"({sum(1 for l in cfg.languages.values() if l.tier.value == 'high')} high, "
        f"{sum(1 for l in cfg.languages.values() if l.tier.value == 'non_high')} non-high)[/dim]"
    )


@main.command()
def check_deps() -> None:
    """Check that all required dependencies are installed and available."""
    import shutil
    import importlib

    from asr_pipeline.logging_config import console

    checks: list[tuple[str, str, bool]] = []

    # FFmpeg
    ffmpeg_ok = shutil.which("ffmpeg") is not None
    checks.append(("FFmpeg", "Audio conversion", ffmpeg_ok))

    # Python packages
    pkgs = [
        ("faster_whisper", "Whisper ASR engine"),
        ("omnilingual_asr", "Omnilingual ASR engine"),
        ("pyannote.audio", "Speaker diarization"),
        ("transformers", "NLLB-200 tokenizer"),
        ("ctranslate2", "CTranslate2 translation"),
        ("ollama", "LLM post-processing (Ollama client)"),
        ("torch", "PyTorch (GPU compute)"),
        ("torchaudio", "Audio I/O"),
        ("soundfile", "WAV file handling"),
        ("rich", "Console output"),
        ("click", "CLI framework"),
        ("pydantic", "Data validation"),
        ("yaml", "Configuration"),
    ]

    for pkg_name, description in pkgs:
        try:
            importlib.import_module(pkg_name)
            checks.append((pkg_name, description, True))
        except ImportError:
            checks.append((pkg_name, description, False))

    # GPU check
    try:
        import torch
        gpu_ok = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if gpu_ok else "N/A"
        gpu_mem = (
            f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
            if gpu_ok
            else "N/A"
        )
        checks.append((f"CUDA GPU ({gpu_name}, {gpu_mem})", "GPU acceleration", gpu_ok))
    except Exception:
        checks.append(("CUDA GPU", "GPU acceleration", False))

    from rich.table import Table

    table = Table(title="Dependency Check", show_header=True, header_style="bold cyan")
    table.add_column("Component", width=40)
    table.add_column("Purpose", width=28)
    table.add_column("Status", width=10)

    for name, desc, ok in checks:
        status = "[green]✓ OK[/green]" if ok else "[red]✗ Missing[/red]"
        table.add_row(name, desc, status)

    console.print(table)

    missing = [name for name, _, ok in checks if not ok]
    if missing:
        console.print(
            f"\n[yellow]⚠ {len(missing)} dependencies missing. "
            f"Install with: uv sync[/yellow]"
        )
    else:
        console.print("\n[green]✓ All dependencies satisfied![/green]")


@main.command()
@click.option(
    "--translation-backend",
    type=click.Choice(["translategemma", "ct2_nllb"], case_sensitive=False),
    default="translategemma",
    show_default=True,
    help="Translation backend to set up.",
)
@click.option(
    "--ollama-model",
    default="qwen2.5:1.5b",
    show_default=True,
    help="Ollama model for LLM refinement (only with ct2_nllb backend).",
)
@click.option(
    "--nllb-model",
    default="facebook/nllb-200-distilled-1.3B",
    show_default=True,
    help="HuggingFace NLLB model to convert (only with ct2_nllb backend).",
)
@click.option(
    "--models-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Directory to store converted models. Default: ~/.asr-pipeline/models",
)
@click.option(
    "--quantization",
    type=click.Choice(["int8", "float16", "int8_float16"], case_sensitive=False),
    default="int8",
    show_default=True,
    help="Quantization for CTranslate2 NLLB conversion.",
)
@click.option(
    "--skip-asr-models",
    is_flag=True,
    default=False,
    help="Skip downloading ASR models (Whisper, Omnilingual, etc.).",
)
@click.option(
    "--skip-translation",
    is_flag=True,
    default=False,
    help="Skip translation backend setup.",
)
def setup(
    translation_backend: str,
    ollama_model: str,
    nllb_model: str,
    models_dir: Optional[Path],
    quantization: str,
    skip_asr_models: bool,
    skip_translation: bool,
) -> None:
    """
    Download all models and set up the ASR pipeline.

    \b
    Downloads core ASR models:
      1. Silero VAD (~40 MB)
      2. Whisper large-v3 (~3 GB)
      3. Omnilingual CTC 300M (~600 MB)
      4. Pyannote diarization 3.1 (~80 MB) — needs HF_TOKEN
      5. MMS forced alignment (~600 MB)

    \b
    Then sets up translation:
      Default: TranslateGemma 4B (auto-downloads on first run)
      Legacy:  --translation-backend ct2_nllb (CT2 NLLB + Ollama)

    \b
    Examples:
        asr-pipeline setup
        asr-pipeline setup --skip-translation
        asr-pipeline setup --skip-asr-models
        asr-pipeline setup --translation-backend ct2_nllb
    """
    import gc
    import os
    import shutil
    import subprocess

    from dotenv import load_dotenv

    from asr_pipeline.logging_config import console

    # Load .env file (for HF_TOKEN, etc.)
    load_dotenv()

    if models_dir is None:
        models_dir = Path.home() / ".asr-pipeline" / "models"

    models_dir.mkdir(parents=True, exist_ok=True)

    console.print()
    console.print(
        "[bold cyan]ASR Pipeline — Full Setup[/bold cyan]"
    )
    console.print()

    # Track results for final summary
    results: dict[str, str] = {}  # model_name -> "ok" | "skip" | error message

    # ── Phase 1: Core ASR models ──────────────────────────────────
    if not skip_asr_models:
        console.print("[bold]Phase 1: Core ASR Models[/bold]")
        console.print()

        # Step 1: Silero VAD
        console.print("  [bold]1/5[/bold] Silero VAD (~40 MB)")
        try:
            import torch

            torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                trust_repo=True,
            )
            console.print("    [green]\u2713[/green] Silero VAD ready")
            results["Silero VAD"] = "ok"
        except Exception as e:
            console.print(f"    [red]\u2717[/red] Silero VAD failed: {e}")
            results["Silero VAD"] = str(e)

        # Step 2: Whisper large-v3
        console.print("  [bold]2/5[/bold] Whisper large-v3 (~3 GB)")
        try:
            from faster_whisper import WhisperModel

            _model = WhisperModel("large-v3", device="cpu", compute_type="int8")
            del _model
            gc.collect()
            console.print("    [green]\u2713[/green] Whisper large-v3 ready")
            results["Whisper large-v3"] = "ok"
        except Exception as e:
            console.print(f"    [red]\u2717[/red] Whisper large-v3 failed: {e}")
            results["Whisper large-v3"] = str(e)

        # Step 3: Omnilingual CTC 300M
        console.print("  [bold]3/5[/bold] Omnilingual CTC 300M (~600 MB)")
        try:
            # fairseq2 skips system library lookup when CONDA_PREFIX is set
            conda_prefix = os.environ.pop("CONDA_PREFIX", None)
            from omnilingual_asr.models.inference.pipeline import (
                ASRInferencePipeline,
            )

            if conda_prefix is not None:
                os.environ["CONDA_PREFIX"] = conda_prefix

            _pipeline = ASRInferencePipeline(model_card="omniASR_CTC_300M_v2")
            del _pipeline
            gc.collect()
            console.print("    [green]\u2713[/green] Omnilingual CTC 300M ready")
            results["Omnilingual CTC 300M"] = "ok"
        except Exception as e:
            console.print(f"    [red]\u2717[/red] Omnilingual CTC 300M failed: {e}")
            results["Omnilingual CTC 300M"] = str(e)

        # Step 4: Pyannote diarization
        console.print("  [bold]4/5[/bold] Pyannote diarization 3.1 (~80 MB)")
        try:
            import torch
            from pyannote.audio import Pipeline as PyannotePipeline

            hf_token = os.environ.get("HF_TOKEN")
            if not hf_token:
                console.print(
                    "    [yellow]\u26a0[/yellow] HF_TOKEN not set — "
                    "pyannote requires a HuggingFace token.\n"
                    "    Set it with: [bold]export HF_TOKEN=hf_...[/bold]"
                )
                results["Pyannote 3.1"] = "skip (no HF_TOKEN)"
            else:
                _original_torch_load = torch.load
                torch.load = lambda *args, **kwargs: _original_torch_load(
                    *args, **{**kwargs, "weights_only": False}
                )
                try:
                    PyannotePipeline.from_pretrained(
                        "pyannote/speaker-diarization-3.1",
                        use_auth_token=hf_token,
                    )
                finally:
                    torch.load = _original_torch_load
                console.print("    [green]\u2713[/green] Pyannote 3.1 ready")
                results["Pyannote 3.1"] = "ok"
        except Exception as e:
            console.print(f"    [red]\u2717[/red] Pyannote 3.1 failed: {e}")
            results["Pyannote 3.1"] = str(e)

        # Step 5: MMS forced alignment
        console.print("  [bold]5/5[/bold] MMS forced alignment (~600 MB)")
        try:
            import torchaudio

            bundle = torchaudio.pipelines.MMS_FA
            _fa_model = bundle.get_model()
            bundle.get_dict()
            del _fa_model
            gc.collect()
            console.print("    [green]\u2713[/green] MMS forced alignment ready")
            results["MMS FA"] = "ok"
        except Exception as e:
            console.print(f"    [red]\u2717[/red] MMS forced alignment failed: {e}")
            results["MMS FA"] = str(e)

        console.print()
    else:
        console.print("[dim]Skipping ASR model downloads (--skip-asr-models)[/dim]")
        console.print()

    # ── Phase 2: Translation setup ────────────────────────────────
    if skip_translation:
        console.print("[dim]Skipping translation setup (--skip-translation)[/dim]")
        console.print()
    elif translation_backend == "translategemma":
        console.print("[bold]Phase 2: Translation Setup[/bold]")
        console.print()
        # ── TranslateGemma (default) ────────────────────────────────
        console.print(
            "[bold]Translation backend:[/bold] TranslateGemma 4B (HuggingFace)"
        )
        console.print()
        console.print(
            "  TranslateGemma auto-downloads from HuggingFace on first run.\n"
            "  Model: [bold]google/translategemma-4b-it[/bold] (~3 GB)\n"
            "  No manual setup required!"
        )
        console.print()

        # Verify GPU availability
        try:
            import torch

            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
                console.print(
                    f"  [green]\u2713[/green] GPU: {gpu_name} ({gpu_mem:.1f} GB)"
                )
            else:
                console.print(
                    "  [yellow]\u26a0[/yellow] No CUDA GPU detected. "
                    "TranslateGemma will run on CPU (slower)."
                )
        except ImportError:
            console.print(
                "  [yellow]\u26a0[/yellow] PyTorch not found. Install with: uv sync"
            )

        # Verify transformers is available
        try:
            import transformers  # noqa: F401

            console.print(
                "  [green]\u2713[/green] HuggingFace Transformers installed"
            )
        except ImportError:
            console.print(
                "  [red]\u2717[/red] HuggingFace Transformers not installed. "
                "Run: uv sync"
            )

        results["TranslateGemma"] = "ok"

    else:
        # ── CT2 NLLB + Ollama (legacy) ──────────────────────────────
        console.print(
            "[bold]Translation backend:[/bold] CT2 NLLB + Ollama (legacy)"
        )
        console.print()

        config_updated = False
        config_updates: dict[str, str] = {}
        ct2_output_dir = models_dir / "ct2-nllb"

        # Step 1: Ollama
        console.print("[bold]Step 1:[/bold] Ollama LLM setup")
        ollama_bin = shutil.which("ollama")
        if ollama_bin is None:
            console.print(
                "  [red]✗ Ollama not found.[/red]\n"
                "  Install: [bold]curl -fsSL https://ollama.ai/install.sh | sh[/bold]\n"
                "  Then: [bold]ollama serve[/bold] + re-run setup."
            )
        else:
            console.print(f"  [green]✓[/green] Ollama found at {ollama_bin}")
            try:
                import ollama as ollama_client

                client = ollama_client.Client(host="http://localhost:11434")
                client.list()
                console.print("  [green]✓[/green] Ollama server is running")

                console.print(
                    f"  Pulling [bold]{ollama_model}[/bold]..."
                )
                result = subprocess.run(
                    ["ollama", "pull", ollama_model],
                    capture_output=False, text=True,
                )
                if result.returncode == 0:
                    console.print(
                        f"  [green]✓[/green] Model [bold]{ollama_model}[/bold] ready"
                    )
                    config_updates["ollama_model"] = ollama_model
                    config_updated = True
            except Exception as e:
                console.print(
                    f"  [yellow]⚠ Ollama not reachable: {e}[/yellow]"
                )

        console.print()

        # Step 2: CT2 NLLB
        console.print("[bold]Step 2:[/bold] CTranslate2 NLLB translation model")
        if ct2_output_dir.exists() and any(ct2_output_dir.iterdir()):
            console.print(
                f"  [green]✓[/green] CT2 model exists at [file]{ct2_output_dir}[/file]"
            )
            config_updates["ct2_model_path"] = str(ct2_output_dir)
            config_updated = True
        else:
            ct2_converter = shutil.which("ct2-transformers-converter")
            if ct2_converter is None:
                console.print(
                    "  [red]✗[/red] ct2-transformers-converter not found. "
                    "Try: [bold]uv sync[/bold]"
                )
            else:
                console.print(
                    f"  Converting [bold]{nllb_model}[/bold] → CT2 ({quantization})..."
                )
                try:
                    result = subprocess.run(
                        [
                            "ct2-transformers-converter",
                            "--model", nllb_model,
                            "--output_dir", str(ct2_output_dir),
                            "--quantization", quantization,
                        ],
                        capture_output=False, text=True,
                    )
                    if result.returncode == 0:
                        console.print(
                            f"  [green]✓[/green] NLLB converted ({quantization})"
                        )
                        config_updates["ct2_model_path"] = str(ct2_output_dir)
                        config_updated = True
                    else:
                        console.print("  [red]✗[/red] Conversion failed.")
                except Exception as e:
                    console.print(f"  [red]✗[/red] Conversion failed: {e}")

        console.print()

        # Step 3: Update config
        if config_updated:
            console.print("[bold]Step 3:[/bold] Updating default configuration")
            from asr_pipeline.config import _DEFAULT_CONFIG

            try:
                config_path = _DEFAULT_CONFIG
                config_text = config_path.read_text(encoding="utf-8")

                # Switch backend to ct2_nllb
                config_text = config_text.replace(
                    'translation_backend: "translategemma"',
                    'translation_backend: "ct2_nllb"',
                )

                # Enable CT2 translation
                config_text = config_text.replace(
                    "enabled: false                  "
                    "# Disabled by default (TranslateGemma handles translation)",
                    "enabled: true                   "
                    "# Enabled for ct2_nllb backend",
                )

                if "ct2_model_path" in config_updates:
                    ct2_path = config_updates["ct2_model_path"]
                    config_text = config_text.replace(
                        "model_path: null                "
                        "# Set via: asr-pipeline setup --translation-backend ct2_nllb",
                        f'model_path: "{ct2_path}"  # CTranslate2 model dir',
                    )
                    console.print(
                        f"  [green]✓[/green] translation.model_path = "
                        f"[file]{ct2_path}[/file]"
                    )

                if "ollama_model" in config_updates:
                    console.print(
                        f"  [green]✓[/green] correction.model = "
                        f"[bold]{config_updates['ollama_model']}[/bold]"
                    )

                config_path.write_text(config_text, encoding="utf-8")
                console.print(
                    f"  [green]✓[/green] Config saved: [file]{config_path}[/file]"
                )

            except Exception as e:
                console.print(
                    f"  [yellow]⚠ Could not update config: {e}[/yellow]"
                )

        console.print()
        console.print("[bold cyan]Setup Summary:[/bold cyan]")
        if "ollama_model" in config_updates:
            console.print(
                f"  [green]✓[/green] Ollama: [bold]{config_updates['ollama_model']}[/bold]"
            )
        if "ct2_model_path" in config_updates:
            console.print(
                f"  [green]✓[/green] CT2 NLLB: [file]{config_updates['ct2_model_path']}[/file]"
            )

        console.print()
        if config_updated:
            results["CT2 NLLB + Ollama"] = "ok"
        else:
            results["CT2 NLLB + Ollama"] = "incomplete"

    # ── Final summary ─────────────────────────────────────────────
    if results:
        console.print()
        console.print("[bold cyan]Setup Summary:[/bold cyan]")
        ok_count = 0
        fail_count = 0
        for name, status in results.items():
            if status == "ok":
                console.print(f"  [green]\u2713[/green] {name}")
                ok_count += 1
            elif status.startswith("skip"):
                console.print(f"  [yellow]\u26a0[/yellow] {name} — {status}")
            else:
                console.print(f"  [red]\u2717[/red] {name} — {status}")
                fail_count += 1

        console.print()
        if fail_count == 0:
            console.print(
                "[green]Setup complete![/green] Run:\n"
                "  [bold]asr-pipeline transcribe audio.m4a --language spa[/bold]"
            )
        else:
            console.print(
                f"[yellow]{fail_count} model(s) failed.[/yellow] "
                "Fix issues above and re-run: [bold]asr-pipeline setup[/bold]"
            )


if __name__ == "__main__":
    main()
