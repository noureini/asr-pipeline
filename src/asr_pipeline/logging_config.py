"""
Logging configuration for the ASR pipeline.

Provides structured, stage-aware logging using Rich for beautiful console
output. Each pipeline stage is clearly labelled so users always know
what is happening and at which stage they are.
"""

from __future__ import annotations

import logging
import sys
import time
from contextlib import contextmanager
from typing import Generator, Optional

from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.theme import Theme

# =============================================================================
# Custom theme
# =============================================================================

_THEME = Theme(
    {
        "stage": "bold cyan",
        "substage": "dim cyan",
        "success": "bold green",
        "warning": "bold yellow",
        "error": "bold red",
        "info": "white",
        "metric": "bold magenta",
        "file": "underline blue",
    }
)

console = Console(theme=_THEME)

# =============================================================================
# Pipeline stage definitions
# =============================================================================

PIPELINE_STAGES_BASE = [
    ("1", "Preprocessing", "Audio normalization, VAD-guided chunking"),
    ("2", "Language Detection", "Identifying languages and routing to engines"),
    ("3+4", "Transcription + Diarization", "ASR + speaker ID (parallel)"),
    ("3b", "Forced Alignment", "Wav2vec2 word-level timestamp refinement"),
    ("5", "Alignment", "Merging ASR segments with speaker labels"),
]

PIPELINE_STAGES_PP_TRANSLATEGEMMA = [
    ("6", "Translation + Cleanup", "TranslateGemma 4B: translate + English cleanup"),
]

PIPELINE_STAGES_PP_CT2_NLLB = [
    ("6a", "Translation", "CTranslate2 NLLB batch translation to English"),
    ("6b", "Joint Refinement", "Ollama LLM: source + translation cross-reference"),
]

PIPELINE_STAGES_OUTPUT = [
    ("7", "Output", "Formatting and writing transcript"),
]


def get_pipeline_stages(translation_backend: str = "translategemma") -> list[tuple[str, str, str]]:
    """Build the full pipeline stages list based on the active translation backend."""
    stages = list(PIPELINE_STAGES_BASE)
    if translation_backend == "translategemma":
        stages.extend(PIPELINE_STAGES_PP_TRANSLATEGEMMA)
    else:
        stages.extend(PIPELINE_STAGES_PP_CT2_NLLB)
    stages.extend(PIPELINE_STAGES_OUTPUT)
    return stages


# Default for backwards compatibility
PIPELINE_STAGES = get_pipeline_stages("translategemma")


# =============================================================================
# Logger setup
# =============================================================================


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    fmt: str = "rich",
) -> logging.Logger:
    """
    Configure the pipeline logger with Rich console handler.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR).
        log_file: Optional path to also write logs to a file.
        fmt: "rich" for console output, "plain" for file-friendly.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger("asr_pipeline")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()

    if fmt == "rich":
        handler = RichHandler(
            console=console,
            show_time=True,
            show_path=False,
            markup=True,
            rich_tracebacks=True,
            tracebacks_show_locals=True,
        )
        handler.setFormatter(logging.Formatter("%(message)s"))
    else:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )

    logger.addHandler(handler)

    # Optional file handler
    if log_file:
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        file_handler.setLevel(getattr(logging, level.upper(), logging.INFO))
        logger.addHandler(file_handler)

    return logger


def get_logger() -> logging.Logger:
    """Retrieve the pipeline logger (must be initialized first)."""
    return logging.getLogger("asr_pipeline")


# =============================================================================
# Stage context manager — logs entry/exit with timing
# =============================================================================


@contextmanager
def stage_log(
    stage_num: str,
    stage_name: str,
    description: str = "",
) -> Generator[None, None, None]:
    """
    Context manager that logs the start and end of a pipeline stage.

    Prints a clear banner when entering a stage and a completion
    message with elapsed time when exiting.

    Usage:
        with stage_log("1", "Preprocessing", "Normalizing audio"):
            do_preprocessing()
    """
    logger = get_logger()
    header = f"[stage]Stage {stage_num}: {stage_name}[/stage]"
    if description:
        header += f"  [substage]— {description}[/substage]"

    console.print()
    console.rule(f"[stage]Stage {stage_num}: {stage_name}[/stage]", style="cyan")
    if description:
        logger.info(f"  {description}")

    start = time.perf_counter()
    try:
        yield
    except Exception:
        elapsed = time.perf_counter() - start
        logger.error(
            f"[error]Stage {stage_num} FAILED[/error] after {elapsed:.1f}s"
        )
        raise
    else:
        elapsed = time.perf_counter() - start
        logger.info(
            f"[success]✓ Stage {stage_num} completed[/success] in {elapsed:.1f}s"
        )


# =============================================================================
# Progress bar factory
# =============================================================================


def create_progress(description: str = "Processing") -> Progress:
    """Create a Rich progress bar for batch operations."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    )


# =============================================================================
# Summary display helpers
# =============================================================================


def print_pipeline_plan(stages_active: list[tuple[str, str, str]]) -> None:
    """Display the pipeline execution plan as a table."""
    table = Table(title="Pipeline Execution Plan", show_header=True, header_style="bold cyan")
    table.add_column("Stage", style="stage", width=8)
    table.add_column("Name", style="info", width=22)
    table.add_column("Description", style="substage")

    for num, name, desc in stages_active:
        table.add_row(num, name, desc)

    console.print(table)


def print_audio_info(
    file_path: str,
    duration_s: float,
    sample_rate: int,
    channels: int,
    file_size_mb: float,
) -> None:
    """Display audio file metadata in a panel."""
    mins = int(duration_s // 60)
    secs = int(duration_s % 60)
    info = (
        f"[file]{file_path}[/file]\n"
        f"Duration: {mins}m {secs}s  |  "
        f"Sample rate: {sample_rate} Hz  |  "
        f"Channels: {channels}  |  "
        f"Size: {file_size_mb:.1f} MB"
    )
    console.print(Panel(info, title="[bold]Audio File[/bold]", border_style="blue"))


def print_routing_decision(language: str, tier: str, engine: str) -> None:
    """Log the routing decision for a detected language."""
    logger = get_logger()
    tier_label = "[success]HIGH[/success]" if tier == "high" else "[warning]NON-HIGH[/warning]"
    logger.info(
        f"  Language: [bold]{language}[/bold]  |  "
        f"Tier: {tier_label}  |  "
        f"Engine: [bold]{engine}[/bold]"
    )


def print_completion_summary(
    audio_file: str,
    duration_s: float,
    num_speakers: int,
    num_segments: int,
    languages: list[str],
    output_file: str,
    total_elapsed_s: float,
) -> None:
    """Display the final completion summary."""
    mins = int(duration_s // 60)
    secs = int(duration_s % 60)
    elapsed_mins = int(total_elapsed_s // 60)
    elapsed_secs = int(total_elapsed_s % 60)

    summary = (
        f"Audio: [file]{audio_file}[/file]\n"
        f"Duration: {mins}m {secs}s  |  "
        f"Speakers: {num_speakers}  |  "
        f"Segments: {num_segments}\n"
        f"Languages: {', '.join(languages)}\n"
        f"Output: [file]{output_file}[/file]\n"
        f"Total processing time: {elapsed_mins}m {elapsed_secs}s"
    )
    console.print()
    console.print(
        Panel(
            summary,
            title="[bold green]✓ Transcription Complete[/bold green]",
            border_style="green",
        )
    )
