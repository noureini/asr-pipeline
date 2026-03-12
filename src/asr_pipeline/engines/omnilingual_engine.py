"""
Omnilingual ASR engine for non-high-resource languages.

Uses Meta's Omnilingual ASR (CTC or LLM variants) for languages where
Whisper is unreliable. Handles code-switching natively through its
multilingual encoder. Supports zero-shot inference for entirely new
languages.

CTC models use a unified 9,812-symbol character vocabulary across all
1,600+ languages -- no per-language adapters or language conditioning.
The lang parameter is only meaningful for the LLM variants.
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


def _is_ctc_model(model_card: str) -> bool:
    """Check if a model card refers to a CTC (not LLM) variant."""
    return "CTC" in model_card.upper()


class OmnilingualEngine(BaseASREngine):
    """
    Omnilingual ASR engine (CTC 300M by default).

    Routes non-high-resource languages to Meta's Omnilingual ASR,
    which achieves <10% CER on 78% of 1,600+ languages.
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
        self._is_ctc = _is_ctc_model(config.model_card)

    # ─────────────────────────────────────────────────────────────────
    # Lifecycle
    # ─────────────────────────────────────────────────────────────────

    def load(self, checkpoint_path: Optional[str] = None) -> None:
        """
        Load the Omnilingual ASR model into memory.

        Args:
            checkpoint_path: Optional path to a fine-tuned checkpoint
                directory. If provided, loads the fine-tuned model
                instead of the base model card.
        """
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

        if checkpoint_path and Path(checkpoint_path).exists():
            logger.info(
                f"  Loading fine-tuned Omnilingual ASR from "
                f"[bold]{checkpoint_path}[/bold] on {self._device}"
            )
            self._pipeline = ASRInferencePipeline(
                model_card=self._config.model_card,
                checkpoint_dir=checkpoint_path,
            )
        else:
            if checkpoint_path:
                logger.warning(
                    f"  Fine-tuned checkpoint not found: {checkpoint_path}, "
                    f"falling back to base model"
                )
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

    def offload_to_cpu(self) -> None:
        """Move model to CPU to free GPU memory without destroying fairseq2 state.

        This avoids the fairseq2 thread-local gang context corruption that
        occurs when the model is fully destroyed and re-created.
        """
        if self._pipeline is not None and hasattr(self._pipeline, "model"):
            self._pipeline.model.cpu()
            import gc
            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.debug("Omnilingual model offloaded to CPU")

    def reload_to_gpu(self) -> None:
        """Move model back to GPU after offloading."""
        if self._pipeline is not None and hasattr(self._pipeline, "model"):
            import torch

            device = torch.device(self._device)
            self._pipeline.model.to(device)
            self._pipeline.device = device
            logger.debug(f"Omnilingual model reloaded to {self._device}")

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
    # Validation
    # ─────────────────────────────────────────────────────────────────

    def _validate_chunk_duration(self, chunk: AudioChunk) -> bool:
        """Check that chunk duration is within the model limit."""
        max_len = self._config.max_audio_length_s
        if chunk.duration_s > max_len:
            logger.warning(
                f"Chunk {chunk.chunk_id} duration ({chunk.duration_s:.1f}s) "
                f"exceeds Omnilingual limit ({max_len}s), skipping"
            )
            return False
        return True

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

        For CTC models, the language parameter is ignored (the unified
        vocabulary handles all languages). For LLM variants, the language
        code must be in {code}_{script} format (e.g., "hin_Deva").
        """
        if self._pipeline is None:
            raise RuntimeError("Omnilingual model not loaded. Call load() first.")

        if not self._validate_chunk_duration(chunk):
            return []

        omni_lang = language if "_" in language else f"{language}_Latn"

        # CTC models ignore lang -- omit it to avoid the library warning.
        # Only LLM variants use language conditioning.
        transcribe_kwargs: dict = {
            "batch_size": 1,
        }
        if not self._is_ctc:
            transcribe_kwargs["lang"] = [omni_lang]

        try:
            transcriptions = self._pipeline.transcribe(  # type: ignore[union-attr]
                [str(chunk.waveform_path)],
                **transcribe_kwargs,
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

        # Filter out chunks that exceed the max duration
        valid_chunks = [c for c in chunks if self._validate_chunk_duration(c)]
        if not valid_chunks:
            return []

        omni_lang = language if "_" in language else f"{language}_Latn"
        audio_paths = [str(c.waveform_path) for c in valid_chunks]

        transcribe_kwargs: dict = {
            "batch_size": min(len(valid_chunks), 8),
        }
        if not self._is_ctc:
            transcribe_kwargs["lang"] = [omni_lang] * len(valid_chunks)

        try:
            transcriptions = self._pipeline.transcribe(  # type: ignore[union-attr]
                audio_paths,
                **transcribe_kwargs,
            )
        except Exception as e:
            logger.warning(f"Batch transcription failed: {e}, falling back to sequential")
            all_segments: list[ASRSegment] = []
            for chunk in valid_chunks:
                segments = self.transcribe_chunk(chunk, language)
                all_segments.extend(segments)
            return all_segments

        results: list[ASRSegment] = []
        for chunk, text in zip(valid_chunks, transcriptions):
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

        # Filter out chunks that exceed the max duration
        valid_chunks = [c for c in chunks if self._validate_chunk_duration(c)]
        if not valid_chunks:
            return []

        audio_paths = [str(c.waveform_path) for c in valid_chunks]
        lang_list = [language] * len(valid_chunks)

        # Build kwargs for the LLM variant -- it supports in-context examples
        transcribe_kwargs: dict = {
            "lang": lang_list,
            "batch_size": 1,
        }
        if context_examples:
            transcribe_kwargs["context_examples"] = context_examples
            logger.info(
                f"  Using {len(context_examples)} context examples for few-shot"
            )

        transcriptions = self._zero_shot_pipeline.transcribe(  # type: ignore[union-attr]
            audio_paths,
            **transcribe_kwargs,
        )

        results: list[ASRSegment] = []
        for chunk, text in zip(valid_chunks, transcriptions):
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
        Detect language -- not supported for CTC variant.

        The CTC model uses a unified vocabulary and does not have
        a language identification head. The user must specify the
        target language explicitly.
        """
        # Omnilingual doesn't have a separate language detection API
        # in the CTC variant. The user specifies the language.
        # Return a placeholder indicating manual specification needed.
        logger.debug("Omnilingual CTC does not have built-in language detection")
        return ("unknown", 0.0)
