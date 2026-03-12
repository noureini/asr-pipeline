"""
Fine-tuning utilities for the omnilingual ASR CTC model.

Provides dataset preparation (Parquet), config generation, and training
launcher for wav2vec2-based CTC fine-tuning via the omnilingual-asr recipe.
"""

from __future__ import annotations

import io
import logging
import random
import subprocess
from pathlib import Path
from typing import Optional

import pyarrow as pa
import pyarrow.parquet as pq
import torchaudio
import torch

logger = logging.getLogger("asr_pipeline")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TARGET_SAMPLE_RATE = 16_000


# ---------------------------------------------------------------------------
# 1. Dataset preparation
# ---------------------------------------------------------------------------


def prepare_parquet_dataset(
    audio_dir: Path,
    transcript_file: Path,
    language: str,
    output_dir: Path,
    corpus_name: str = "custom",
    dev_split: float = 0.1,
) -> Path:
    """Convert a directory of audio files + TSV transcript into Parquet shards.

    The TSV file is expected to be **tab-separated** with two columns
    (``filename<TAB>text``) and **no header row**.

    Audio is resampled to 16 kHz mono and encoded as FLAC bytes. The output
    Parquet dataset follows the Hive-partitioned layout used by the
    omnilingual-asr data loader::

        output_dir/
          version=0/
            corpus={name}/
              split=train/language={code}/part-00000.parquet
              split=dev/language={code}/part-00000.parquet

    Parameters
    ----------
    audio_dir:
        Directory containing the audio files referenced in *transcript_file*.
    transcript_file:
        Tab-separated file with ``filename\ttext`` rows (no header).
    language:
        Language tag in ``{code}_{script}`` format, e.g. ``hin_Deva``.
    output_dir:
        Root directory for the generated Parquet dataset.
    corpus_name:
        Corpus identifier written into the ``corpus`` column.
    dev_split:
        Fraction of samples held out for the dev split (default 0.1).

    Returns
    -------
    Path
        The *output_dir* root so callers can pass it to config generation.
    """
    audio_dir = Path(audio_dir)
    transcript_file = Path(transcript_file)
    output_dir = Path(output_dir)

    logger.info("Reading transcript file: %s", transcript_file)
    entries: list[tuple[str, str]] = []
    with open(transcript_file, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t", maxsplit=1)
            if len(parts) != 2:
                logger.warning(
                    "Skipping malformed line %d (expected 2 tab-separated columns)",
                    lineno,
                )
                continue
            entries.append((parts[0], parts[1]))

    if not entries:
        raise ValueError(f"No valid entries found in {transcript_file}")

    logger.info("Found %d utterances in transcript file", len(entries))

    # Deterministic shuffle + split
    rng = random.Random(42)
    indices = list(range(len(entries)))
    rng.shuffle(indices)
    n_dev = max(1, int(len(entries) * dev_split))
    dev_indices = set(indices[:n_dev])

    train_rows: list[dict] = []
    dev_rows: list[dict] = []

    for idx, (filename, text) in enumerate(entries):
        audio_path = audio_dir / filename
        if not audio_path.exists():
            logger.warning("Audio file not found, skipping: %s", audio_path)
            continue

        # Load and resample
        waveform, sr = torchaudio.load(str(audio_path))
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != _TARGET_SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sr, new_freq=_TARGET_SAMPLE_RATE
            )
            waveform = resampler(waveform)

        # Encode as FLAC bytes
        buf = io.BytesIO()
        torchaudio.save(buf, waveform, _TARGET_SAMPLE_RATE, format="flac")
        flac_bytes = buf.getvalue()

        audio_bytes_list = [int(b) for b in flac_bytes]
        audio_size = waveform.shape[-1]

        split_name = "dev" if idx in dev_indices else "train"
        row = {
            "text": text,
            "audio_bytes": audio_bytes_list,
            "audio_size": int(audio_size),
            "corpus": corpus_name,
            "split": split_name,
            "language": language,
        }
        if split_name == "dev":
            dev_rows.append(row)
        else:
            train_rows.append(row)

    logger.info(
        "Prepared %d train / %d dev samples", len(train_rows), len(dev_rows)
    )

    # Write Parquet shards
    for split_name, rows in [("train", train_rows), ("dev", dev_rows)]:
        if not rows:
            logger.warning("No rows for split=%s, skipping", split_name)
            continue
        shard_dir = (
            output_dir
            / "version=0"
            / f"corpus={corpus_name}"
            / f"split={split_name}"
            / f"language={language}"
        )
        shard_dir.mkdir(parents=True, exist_ok=True)
        shard_path = shard_dir / "part-00000.parquet"

        schema = pa.schema(
            [
                pa.field("text", pa.string()),
                pa.field("audio_bytes", pa.list_(pa.int8())),
                pa.field("audio_size", pa.int64()),
                pa.field("corpus", pa.string()),
                pa.field("split", pa.string()),
                pa.field("language", pa.string()),
            ]
        )
        table = pa.table(
            {
                "text": [r["text"] for r in rows],
                "audio_bytes": [r["audio_bytes"] for r in rows],
                "audio_size": [r["audio_size"] for r in rows],
                "corpus": [r["corpus"] for r in rows],
                "split": [r["split"] for r in rows],
                "language": [r["language"] for r in rows],
            },
            schema=schema,
        )
        pq.write_table(table, shard_path)
        logger.info("Wrote %d rows to %s", len(rows), shard_path)

    return output_dir


# ---------------------------------------------------------------------------
# 2. Config generation
# ---------------------------------------------------------------------------


def generate_finetune_config(
    dataset_dir: Path,
    output_dir: Path,
    model_card: str = "omniASR_CTC_300M_v2",
    model_arch: str = "300m",
    learning_rate: float = 1e-5,
    num_steps: int = 5_000,
    batch_accumulation: int = 4,
    max_audio_len: int = 640_000,
) -> Path:
    """Generate a YAML configuration for omnilingual-asr CTC fine-tuning.

    The generated config follows the omnilingual-asr recipe layout with
    top-level keys: ``model``, ``tokenizer``, ``dataset``, ``optimizer``,
    ``trainer``, and ``regime``.

    Parameters
    ----------
    dataset_dir:
        Root of the Hive-partitioned Parquet dataset (as produced by
        :func:`prepare_parquet_dataset`).
    output_dir:
        Directory where the config YAML will be written.
    model_card:
        Model card identifier for the base checkpoint.
    model_arch:
        Architecture size key (e.g. ``"300m"``).
    learning_rate:
        Peak learning rate for the optimizer.
    num_steps:
        Total number of training steps.
    batch_accumulation:
        Gradient accumulation steps.
    max_audio_len:
        Maximum audio length in samples (at 16 kHz).

    Returns
    -------
    Path
        Path to the generated config YAML file.
    """
    import yaml

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "finetune_config.yaml"

    config = {
        "model": {
            "name": model_card,
            "arch": model_arch,
            "pretrained": True,
        },
        "tokenizer": {
            "name": "omniASR_tokenizer_v1",
        },
        "dataset": {
            "path": str(Path(dataset_dir).resolve()),
            "max_audio_len": max_audio_len,
            "num_workers": 4,
            "batch_size": 8,
        },
        "optimizer": {
            "name": "adamw",
            "lr": learning_rate,
            "weight_decay": 0.01,
            "betas": [0.9, 0.98],
            "warmup_steps": min(500, num_steps // 10),
        },
        "trainer": {
            "dtype": "bfloat16",
            "max_steps": num_steps,
            "gradient_accumulation_steps": batch_accumulation,
            "log_interval": 50,
            "save_interval": 1000,
            "eval_interval": 500,
        },
        "regime": {
            "freeze_feature_encoder": True,
            "unfreeze_after": min(1000, num_steps // 5),
        },
    }

    with open(config_path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(config, fh, default_flow_style=False, sort_keys=False)

    logger.info("Wrote fine-tuning config to %s", config_path)
    return config_path


# ---------------------------------------------------------------------------
# 3. Training launcher
# ---------------------------------------------------------------------------


def run_finetune(config_path: Path, output_dir: Path) -> Path:
    """Launch omnilingual-asr CTC fine-tuning via the wav2vec2 recipe.

    Runs the training script as a subprocess::

        python -m workflows.recipes.wav2vec2.asr <output_dir> \\
               --config-file <config_path>

    Parameters
    ----------
    config_path:
        Path to the YAML config produced by :func:`generate_finetune_config`.
    output_dir:
        Directory for checkpoints and training logs.

    Returns
    -------
    Path
        Path to the checkpoints directory (``output_dir / "checkpoints"``).

    Raises
    ------
    RuntimeError
        If the training subprocess exits with a non-zero return code.
    """
    config_path = Path(config_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python",
        "-m",
        "workflows.recipes.wav2vec2.asr",
        str(output_dir.resolve()),
        "--config-file",
        str(config_path.resolve()),
    ]

    logger.info("Starting fine-tuning: %s", " ".join(cmd))

    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Fine-tuning failed with exit code {result.returncode}"
        )

    checkpoints_dir = output_dir / "checkpoints"
    logger.info("Fine-tuning complete. Checkpoints at: %s", checkpoints_dir)
    return checkpoints_dir
