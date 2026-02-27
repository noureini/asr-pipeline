"""
Whisper ASR engine for high-resource languages.

Uses faster-whisper (CTranslate2 backend) for efficient inference
on high-resource languages where Whisper is reliably accurate.
Provides word-level timestamps via WhisperX alignment.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Optional

from asr_pipeline.config import WhisperConfig
from asr_pipeline.engines import BaseASREngine
from asr_pipeline.language import map_whisper_lang_to_iso639_3
from asr_pipeline.models import ASRSegment, AudioChunk, WordSegment

logger = logging.getLogger("asr_pipeline")


# Mapping from ISO 639-3 to Whisper's expected language codes
_ISO639_3_TO_WHISPER: dict[str, str] = {
    "eng": "en", "spa": "es", "fra": "fr", "deu": "de",
    "por": "pt", "rus": "ru", "zho": "zh", "jpn": "ja",
    "kor": "ko", "ita": "it", "nld": "nl", "pol": "pl",
    "tur": "tr", "ces": "cs", "swe": "sv", "ukr": "uk",
    "ron": "ro", "ara": "ar",
}


class WhisperEngine(BaseASREngine):
    """
    Whisper Large-v3 engine via faster-whisper.

    Handles high-resource languages with reliable accuracy and
    word-level timestamp alignment.
    """

    def __init__(
        self,
        config: WhisperConfig,
        device: str = "cuda",
        compute_type: str = "float16",
    ) -> None:
        self._config = config
        self._device = device
        self._compute_type = compute_type
        self._model: Optional[object] = None
        self._segment_counter: int = 0

    # ─────────────────────────────────────────────────────────────────
    # Lifecycle
    # ─────────────────────────────────────────────────────────────────

    def load(self) -> None:
        """Load the Whisper model into memory."""
        if self._model is not None:
            logger.debug("Whisper model already loaded, skipping")
            return

        from faster_whisper import WhisperModel

        logger.info(
            f"  Loading Whisper [bold]{self._config.model_size}[/bold] "
            f"on {self._device} ({self._compute_type})"
        )

        self._model = WhisperModel(
            self._config.model_size,
            device=self._device,
            compute_type=self._compute_type,
        )
        logger.info("  ✓ Whisper model loaded")

    def unload(self) -> None:
        """Release Whisper model from memory."""
        if self._model is not None:
            del self._model
            self._model = None
            logger.debug("Whisper model unloaded")

    @property
    def name(self) -> str:
        return f"Whisper {self._config.model_size}"

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    # ─────────────────────────────────────────────────────────────────
    # Transcription
    # ─────────────────────────────────────────────────────────────────

    def transcribe_chunk(
        self,
        chunk: AudioChunk,
        language: str,
    ) -> list[ASRSegment]:
        """Transcribe a single audio chunk with Whisper."""
        if self._model is None:
            raise RuntimeError("Whisper model not loaded. Call load() first.")

        whisper_lang = _ISO639_3_TO_WHISPER.get(language, language)

        segments_iter, info = self._model.transcribe(  # type: ignore[union-attr]
            str(chunk.waveform_path),
            language=whisper_lang,
            beam_size=self._config.beam_size,
            best_of=self._config.best_of,
            patience=self._config.patience,
            condition_on_previous_text=self._config.condition_on_previous_text,
            vad_filter=False,  # Preprocessing Silero VAD handles segmentation
            word_timestamps=self._config.word_timestamps,
        )

        results: list[ASRSegment] = []
        for seg in segments_iter:
            # Build word-level segments if available
            words: list[WordSegment] = []
            if seg.words:
                for w in seg.words:
                    words.append(
                        WordSegment(
                            word=w.word,
                            start_s=chunk.start_s + w.start,
                            end_s=chunk.start_s + w.end,
                            confidence=w.probability,
                        )
                    )

            results.append(
                ASRSegment(
                    segment_id=self._segment_counter,
                    start_s=chunk.start_s + seg.start,
                    end_s=chunk.start_s + seg.end,
                    text=seg.text.strip(),
                    language=language,
                    confidence=max(0.0, min(1.0, math.exp(seg.avg_logprob))) if seg.avg_logprob else 0.0,
                    words=words,
                )
            )
            self._segment_counter += 1

        return results

    def transcribe_batch(
        self,
        chunks: list[AudioChunk],
        language: str,
    ) -> list[ASRSegment]:
        """Transcribe a batch of chunks sequentially with Whisper."""
        all_segments: list[ASRSegment] = []
        for chunk in chunks:
            segments = self.transcribe_chunk(chunk, language)
            all_segments.extend(segments)
        return all_segments

    def transcribe_full_audio(
        self,
        audio_path: Path,
        language: str,
        clip_timestamps: Optional[list[dict[str, float]]] = None,
    ) -> list[ASRSegment]:
        """
        Transcribe a full audio file using BatchedInferencePipeline.

        Processes multiple chunks in parallel on the GPU for significantly
        faster throughput.

        Args:
            audio_path: Path to the full preprocessed WAV file.
            language: ISO 639-3 language code.
            clip_timestamps: List of dicts with ``"start"`` and ``"end"``
                keys (values in seconds) from the preprocessor's VAD-aligned
                chunks.  Example: ``[{"start": 0.0, "end": 28.5}, ...]``.
                Each dict becomes one independent Whisper inference chunk.
                When provided, Whisper skips its internal VAD.
                If None, falls back to Whisper's built-in Silero VAD.
        """
        if self._model is None:
            raise RuntimeError("Whisper model not loaded. Call load() first.")

        from faster_whisper import BatchedInferencePipeline

        whisper_lang = _ISO639_3_TO_WHISPER.get(language, language)

        batched_model = BatchedInferencePipeline(model=self._model)

        logger.info(
            f"  Batched inference: batch_size={self._config.batch_size}"
        )

        # BatchedInferencePipeline requires either vad_filter=True or
        # clip_timestamps.  Since our preprocessor already ran Silero VAD,
        # pass our speech boundaries as clip_timestamps to avoid re-running
        # VAD and to prevent the "No clip timestamps found" error.
        transcribe_kwargs: dict = dict(
            language=whisper_lang,
            batch_size=self._config.batch_size,
            beam_size=self._config.beam_size,
            best_of=self._config.best_of,
            patience=self._config.patience,
            condition_on_previous_text=self._config.condition_on_previous_text,
            word_timestamps=self._config.word_timestamps,
            without_timestamps=False,
        )

        if clip_timestamps:
            # Use preprocessor VAD boundaries
            transcribe_kwargs["clip_timestamps"] = clip_timestamps
            transcribe_kwargs["vad_filter"] = False
            logger.info(
                f"  Using {len(clip_timestamps)} VAD clip timestamps "
                f"from preprocessor"
            )
        else:
            # Fallback: let Whisper run its own Silero VAD
            transcribe_kwargs["vad_filter"] = True
            logger.info("  Using Whisper built-in VAD (no clip timestamps provided)")

        segments_iter, info = batched_model.transcribe(
            str(audio_path),
            **transcribe_kwargs,
        )

        results: list[ASRSegment] = []
        for seg in segments_iter:
            # Build word-level segments if available
            words: list[WordSegment] = []
            if seg.words:
                for w in seg.words:
                    words.append(
                        WordSegment(
                            word=w.word,
                            start_s=w.start,
                            end_s=w.end,
                            confidence=w.probability,
                        )
                    )

            results.append(
                ASRSegment(
                    segment_id=self._segment_counter,
                    start_s=seg.start,
                    end_s=seg.end,
                    text=seg.text.strip(),
                    language=language,
                    confidence=max(0.0, min(1.0, math.exp(seg.avg_logprob))) if seg.avg_logprob else 0.0,
                    words=words,
                )
            )
            self._segment_counter += 1

        return results

    # ─────────────────────────────────────────────────────────────────
    # Language detection
    # ─────────────────────────────────────────────────────────────────

    def detect_language(
        self,
        audio_path: Path,
    ) -> tuple[str, float]:
        """
        Detect language using Whisper's built-in detection.

        Returns (iso639_3_code, confidence).
        """
        if self._model is None:
            raise RuntimeError("Whisper model not loaded. Call load() first.")

        # faster-whisper language detection
        segments_iter, info = self._model.transcribe(  # type: ignore[union-attr]
            str(audio_path),
            beam_size=1,
            best_of=1,
            vad_filter=False,
            word_timestamps=False,
        )

        # We need to consume at least one segment to get language info
        # but info.language is set immediately
        detected = info.language
        probability = info.language_probability

        iso_code = map_whisper_lang_to_iso639_3(detected)
        logger.debug(
            f"Whisper detected language: {detected} → {iso_code} "
            f"(confidence: {probability:.2%})"
        )

        return (iso_code, probability)
