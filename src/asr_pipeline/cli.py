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
            f"{torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB"
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
def setup(
    translation_backend: str,
    ollama_model: str,
    nllb_model: str,
    models_dir: Optional[Path],
    quantization: str,
) -> None:
    """
    Set up post-processing models for translation and refinement.

    \b
    Default (TranslateGemma):
      TranslateGemma 4B auto-downloads from HuggingFace on first run.
      No manual setup needed — just run this to verify your environment.

    \b
    Legacy (CT2 NLLB + Ollama):
      Use --translation-backend ct2_nllb to set up the legacy pipeline:
        1. Pull Ollama LLM model for refinement
        2. Convert NLLB-200 to CTranslate2 format
        3. Update config with model paths

    \b
    Examples:
        asr-pipeline setup
        asr-pipeline setup --translation-backend ct2_nllb
        asr-pipeline setup --translation-backend ct2_nllb --ollama-model mistral:7b
    """
    import shutil
    import subprocess

    from asr_pipeline.logging_config import console

    if models_dir is None:
        models_dir = Path.home() / ".asr-pipeline" / "models"

    models_dir.mkdir(parents=True, exist_ok=True)

    console.print()
    console.print(
        "[bold cyan]ASR Pipeline — Post-Processing Setup[/bold cyan]"
    )
    console.print()

    if translation_backend == "translategemma":
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
                gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1e9
                console.print(
                    f"  [green]✓[/green] GPU: {gpu_name} ({gpu_mem:.1f} GB)"
                )
            else:
                console.print(
                    "  [yellow]⚠[/yellow] No CUDA GPU detected. "
                    "TranslateGemma will run on CPU (slower)."
                )
        except ImportError:
            console.print(
                "  [yellow]⚠[/yellow] PyTorch not found. Install with: uv sync"
            )

        # Verify transformers is available
        try:
            import transformers  # noqa: F401

            console.print(
                "  [green]✓[/green] HuggingFace Transformers installed"
            )
        except ImportError:
            console.print(
                "  [red]✗[/red] HuggingFace Transformers not installed. "
                "Run: uv sync"
            )

        console.print()
        console.print(
            "[green]Setup complete![/green] "
            "TranslateGemma will download on first transcription.\n"
            "  [bold]asr-pipeline transcribe audio.m4a --language spa[/bold]"
        )

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
            console.print(
                "[green]Setup complete![/green] Run:\n"
                "  [bold]asr-pipeline transcribe audio.m4a --language spa[/bold]"
            )
        else:
            console.print(
                "[yellow]Setup incomplete.[/yellow] "
                "Fix issues above and re-run: [bold]asr-pipeline setup[/bold]"
            )


if __name__ == "__main__":
    main()
