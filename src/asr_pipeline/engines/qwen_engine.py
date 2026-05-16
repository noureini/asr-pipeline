"""
Qwen3-ASR engine — default ASR for non-high-resource languages.

Replaces Omnilingual as the production default. Bengali is not in
Qwen3-ASR's official supported set, but the model produces usable
Bengali (validated ~8.5% CER on FLEURS, contamination-free).

The pipeline's VAD chunking supplies segment boundaries, so this
engine only needs to produce *text per chunk* — no forced aligner,
no internal long-audio chunking. Each chunk's waveform_path (a temp
WAV written by the preprocessor) is fed straight to Qwen3-ASR.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from asr_pipeline.config import QwenConfig
from asr_pipeline.engines import BaseASREngine
from asr_pipeline.models import ASRSegment, AudioChunk

logger = logging.getLogger("asr_pipeline")


class QwenEngine(BaseASREngine):
    """Qwen3-ASR engine (Qwen/Qwen3-ASR-1.7B by default)."""

    def __init__(
        self,
        config: QwenConfig,
        device: str = "cuda",
    ) -> None:
        self._config = config
        # Pipeline passes a device string ("cuda"/"cpu"); honor a CPU
        # request, otherwise use the configured device_map.
        if device == "cpu":
            self._device_map = "cpu"
        else:
            self._device_map = config.device_map
        self._model: Optional[object] = None
        self._segment_counter: int = 0

    # ─────────────────────────────────────────────────────────────────
    # Lifecycle
    # ─────────────────────────────────────────────────────────────────

    def load(self, checkpoint_path: Optional[str] = None) -> None:
        """Load Qwen3-ASR into memory."""
        if self._model is not None:
            logger.debug("Qwen3-ASR already loaded, skipping")
            return
        try:
            import torch
            from qwen_asr import Qwen3ASRModel
        except ImportError as e:
            raise RuntimeError(
                f"qwen-asr not installed ({e}). Add 'qwen-asr' to the "
                f"environment (it is in pyproject.toml; run `uv sync`)."
            ) from e

        model_id = checkpoint_path or self._config.model
        dtype = getattr(torch, self._config.dtype, torch.bfloat16)
        logger.info(
            f"Loading {model_id} (dtype={self._config.dtype}, "
            f"device_map={self._device_map})..."
        )
        self._model = Qwen3ASRModel.from_pretrained(
            model_id,
            dtype=dtype,
            device_map=self._device_map,
        )
        logger.info("  ✓ Qwen3-ASR loaded")

    def unload(self) -> None:
        """Release the model from memory."""
        if self._model is not None:
            del self._model
            self._model = None
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
        logger.debug("Qwen3-ASR unloaded")

    @property
    def name(self) -> str:
        return f"Qwen3-ASR ({self._config.model})"

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    # ─────────────────────────────────────────────────────────────────
    # Validation
    # ─────────────────────────────────────────────────────────────────

    def _validate_chunk_duration(self, chunk: AudioChunk) -> bool:
        max_len = self._config.max_audio_length_s
        if chunk.duration_s > max_len:
            logger.warning(
                f"Chunk {chunk.chunk_id} duration ({chunk.duration_s:.1f}s) "
                f"exceeds Qwen3-ASR limit ({max_len}s), skipping"
            )
            return False
        return True

    # ─────────────────────────────────────────────────────────────────
    # Transcription
    # ─────────────────────────────────────────────────────────────────

    def _transcribe_audio(self, audio_path: str) -> tuple[str, str]:
        """Run the model on one audio file. Returns (text, detected_lang)."""
        if self._model is None:
            raise RuntimeError("Qwen3-ASR not loaded. Call load() first.")
        results = self._model.transcribe(  # type: ignore[union-attr]
            audio=audio_path,
            language=self._config.language,  # None = auto-detect
        )
        if not results:
            return "", ""
        r = results[0]
        text = (getattr(r, "text", "") or "").strip()
        det = getattr(r, "language", "") or ""
        return text, det

    def transcribe_chunk(
        self,
        chunk: AudioChunk,
        language: str,
    ) -> list[ASRSegment]:
        """Transcribe a single audio chunk with Qwen3-ASR.

        `language` is the pipeline's ISO code; Qwen's own language hint
        comes from config (None = auto-detect, the validated mode).
        """
        if self._model is None:
            raise RuntimeError("Qwen3-ASR not loaded. Call load() first.")
        if chunk.waveform_path is None:
            logger.warning(
                f"Chunk {chunk.chunk_id} has no waveform_path, skipping"
            )
            return []
        if not self._validate_chunk_duration(chunk):
            return []

        try:
            text, _det = self._transcribe_audio(str(chunk.waveform_path))
        except Exception as e:
            logger.warning(
                f"Qwen3-ASR transcription failed for chunk "
                f"{chunk.chunk_id}: {e}"
            )
            return []

        if not text:
            return []

        seg = ASRSegment(
            segment_id=self._segment_counter,
            start_s=chunk.start_s,
            end_s=chunk.end_s,
            text=text,
            language=language.split("_")[0],
            confidence=0.0,        # Qwen3-ASR does not expose confidence
            words=[],              # no word-level timestamps from this path
        )
        self._segment_counter += 1
        return [seg]

    def transcribe_batch(
        self,
        chunks: list[AudioChunk],
        language: str,
    ) -> list[ASRSegment]:
        """Transcribe a batch of chunks. Sequential — the pipeline's
        VAD chunking already bounds chunk size; per-chunk calls keep
        memory flat and behavior identical to transcribe_chunk."""
        all_segments: list[ASRSegment] = []
        for chunk in chunks:
            all_segments.extend(self.transcribe_chunk(chunk, language))
        return all_segments

    def detect_language(
        self,
        audio_path: Path,
    ) -> tuple[str, float]:
        """Detect language via Qwen3-ASR auto-detect.
        Returns (detected_language_string, confidence)."""
        if self._model is None:
            self.load()
        try:
            _text, det = self._transcribe_audio(str(audio_path))
        except Exception as e:
            logger.warning(f"Qwen3-ASR language detection failed: {e}")
            return "", 0.0
        return det, 1.0 if det else 0.0
