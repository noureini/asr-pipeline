"""
Omnilingual ASR engine for non-high-resource languages.

Uses Meta's Omnilingual ASR (CTC or LLM variants) for languages where
Whisper is unreliable. Handles code-switching natively through its
multilingual encoder. Supports zero-shot inference for entirely new
languages.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from asr_pipeline.config import OmnilingualConfig
from asr_pipeline.engines import BaseASREngine
from asr_pipeline.models import ASRSegment, AudioChunk

logger = logging.getLogger("asr_pipeline")


class OmnilingualEngine(BaseASREngine):
    """
    Omnilingual ASR engine (CTC 300M by default).

    Routes non-high-resource languages to Meta's Omnilingual ASR,
    which achieves <10% CER on 78% of 1,600+ languages.

    The CTC 300M variant fits comfortably on 16GB GPU while
    producing accuracy competitive with the 7B LLM variant
    when fine-tuned on specific languages.
    """

    def __init__(
        self,
        config: OmnilingualConfig,
        device: str = "cuda",
    ) -> None:
        self._config = config
        self._device = device
        self._pipeline: Optional[object] = None
        self._zero_shot_pipeline: Optional[object] = None
        self._segment_counter: int = 0

    # ─────────────────────────────────────────────────────────────────
    # Lifecycle
    # ─────────────────────────────────────────────────────────────────

    def load(self) -> None:
        """Load the Omnilingual ASR model into memory."""
        if self._pipeline is not None:
            logger.debug("Omnilingual model already loaded, skipping")
            return

        # fairseq2 skips system library lookup when CONDA_PREFIX is set,
        # causing libsndfile to not be found even if installed via apt.
        # Temporarily unset it so the system libsndfile is discovered.
        import os

        conda_prefix = os.environ.pop("CONDA_PREFIX", None)

        from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

        if conda_prefix is not None:
            os.environ["CONDA_PREFIX"] = conda_prefix

        logger.info(
            f"  Loading Omnilingual ASR [bold]{self._config.model_card}[/bold] "
            f"on {self._device}"
        )

        self._pipeline = ASRInferencePipeline(
            model_card=self._config.model_card,
        )
        logger.info("  ✓ Omnilingual ASR model loaded")

    def _load_zero_shot(self) -> None:
        """Load the zero-shot LLM variant for unseen languages."""
        if self._zero_shot_pipeline is not None:
            return

        from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

        logger.info(
            f"  Loading zero-shot model [bold]{self._config.zero_shot_model_card}[/bold]"
        )
        self._zero_shot_pipeline = ASRInferencePipeline(
            model_card=self._config.zero_shot_model_card,
        )
        logger.info("  ✓ Zero-shot model loaded")

    def unload(self) -> None:
        """Release models from memory."""
        if self._pipeline is not None:
            del self._pipeline
            self._pipeline = None
        if self._zero_shot_pipeline is not None:
            del self._zero_shot_pipeline
            self._zero_shot_pipeline = None
        logger.debug("Omnilingual models unloaded")

    @property
    def name(self) -> str:
        return f"Omnilingual ASR ({self._config.model_card})"

    @property
    def is_loaded(self) -> bool:
        return self._pipeline is not None

    # ─────────────────────────────────────────────────────────────────
    # Transcription
    # ─────────────────────────────────────────────────────────────────

    def transcribe_chunk(
        self,
        chunk: AudioChunk,
        language: str,
    ) -> list[ASRSegment]:
        """
        Transcribe a single audio chunk with Omnilingual ASR.

        The language code must be in Omnilingual's {code}_{script}
        format (e.g., "hin_Deva", "amh_Ethi").
        """
        if self._pipeline is None:
            raise RuntimeError("Omnilingual model not loaded. Call load() first.")

        # Omnilingual expects the {lang}_{script} format
        omni_lang = language if "_" in language else f"{language}_Latn"

        try:
            transcriptions = self._pipeline.transcribe(  # type: ignore[union-attr]
                [str(chunk.waveform_path)],
                lang=[omni_lang],
                batch_size=1,
            )
        except Exception as e:
            logger.warning(
                f"Omnilingual transcription failed for chunk {chunk.chunk_id}: {e}"
            )
            return []

        results: list[ASRSegment] = []
        for text in transcriptions:
            if not text or not text.strip():
                continue

            results.append(
                ASRSegment(
                    segment_id=self._segment_counter,
                    start_s=chunk.start_s,
                    end_s=chunk.end_s,
                    text=text.strip(),
                    language=language.split("_")[0],  # Extract base code
                    confidence=0.0,  # CTC models don't provide confidence
                    words=[],  # No word-level timestamps from CTC
                )
            )
            self._segment_counter += 1

        return results

    def transcribe_batch(
        self,
        chunks: list[AudioChunk],
        language: str,
    ) -> list[ASRSegment]:
        """
        Transcribe a batch of chunks with Omnilingual ASR.

        Uses native batch processing for efficiency.
        """
        if self._pipeline is None:
            raise RuntimeError("Omnilingual model not loaded. Call load() first.")

        omni_lang = language if "_" in language else f"{language}_Latn"

        audio_paths = [str(c.waveform_path) for c in chunks]
        lang_list = [omni_lang] * len(chunks)

        try:
            transcriptions = self._pipeline.transcribe(  # type: ignore[union-attr]
                audio_paths,
                lang=lang_list,
                batch_size=min(len(chunks), 8),
            )
        except Exception as e:
            logger.warning(f"Batch transcription failed: {e}, falling back to sequential")
            all_segments: list[ASRSegment] = []
            for chunk in chunks:
                segments = self.transcribe_chunk(chunk, language)
                all_segments.extend(segments)
            return all_segments

        results: list[ASRSegment] = []
        for chunk, text in zip(chunks, transcriptions):
            if not text or not text.strip():
                continue

            results.append(
                ASRSegment(
                    segment_id=self._segment_counter,
                    start_s=chunk.start_s,
                    end_s=chunk.end_s,
                    text=text.strip(),
                    language=language.split("_")[0],
                    confidence=0.0,
                    words=[],
                )
            )
            self._segment_counter += 1

        return results

    # ─────────────────────────────────────────────────────────────────
    # Zero-shot transcription for new languages
    # ─────────────────────────────────────────────────────────────────

    def transcribe_zero_shot(
        self,
        chunks: list[AudioChunk],
        language: str,
        context_examples: Optional[list[tuple[str, str]]] = None,
    ) -> list[ASRSegment]:
        """
        Transcribe using the zero-shot LLM variant for unseen languages.

        Args:
            chunks: Audio chunks to transcribe.
            language: Target language in {code}_{script} format.
            context_examples: Optional list of (audio_path, transcription)
                pairs for few-shot prompting.

        Returns:
            List of ASR segments.
        """
        self._load_zero_shot()

        if self._zero_shot_pipeline is None:
            raise RuntimeError("Zero-shot model failed to load.")

        audio_paths = [str(c.waveform_path) for c in chunks]
        lang_list = [language] * len(chunks)

        # The LLM-ASR variant supports in-context examples
        transcriptions = self._zero_shot_pipeline.transcribe(  # type: ignore[union-attr]
            audio_paths,
            lang=lang_list,
            batch_size=1,  # Zero-shot is slower, use batch_size=1
        )

        results: list[ASRSegment] = []
        for chunk, text in zip(chunks, transcriptions):
            if not text or not text.strip():
                continue
            results.append(
                ASRSegment(
                    segment_id=self._segment_counter,
                    start_s=chunk.start_s,
                    end_s=chunk.end_s,
                    text=text.strip(),
                    language=language.split("_")[0],
                    confidence=0.0,
                    words=[],
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
        Detect language using Omnilingual's encoder.

        For Omnilingual, language detection is less critical since
        the model handles code-switching natively. We primarily
        use this to confirm the user-specified language.
        """
        # Omnilingual doesn't have a separate language detection API
        # in the CTC variant. The user specifies the language.
        # Return a placeholder indicating manual specification needed.
        logger.debug("Omnilingual CTC does not have built-in language detection")
        return ("unknown", 0.0)
