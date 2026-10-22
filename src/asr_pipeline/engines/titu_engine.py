"""titu ASR engine for Bengali (NeMo FastConformer-CTC).

Uses hishab/titu_stt_bn_fastconformer, an 18,000-hour-trained
Bengali FastConformer-CTC model. Phonetically faithful, no hallucination
loops (CTC guarantee), strong for coastal Bangladeshi dialect.

As a CTC model it emits WORD-LEVEL timestamps natively (NeMo
`transcribe(timestamps=True)` → `hyp.timestamp['word']`), so word timing
comes from the SAME model that produced the text — no separate aligner,
no romanization, and the timestamps can never disagree with the words.
This is what enables word-level speaker attribution downstream.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from asr_pipeline.engines import BaseASREngine
from asr_pipeline.models import ASRSegment, AudioChunk, WordSegment

logger = logging.getLogger("asr_pipeline")


class TituEngine(BaseASREngine):
    """
    titu ASR engine — Bengali FastConformer-CTC by Hishab.

    Designed for Bangladeshi Bengali (the model card calls out
    Standard Bengali, dialectal, code-switched, conversational speech).
    Loaded via NeMo. Emits native CTC word timestamps
    (transcribe(timestamps=True) → hyp.timestamp['word']), so no separate
    forced aligner is needed; MMS_FA is bypassed when these are present.
    """

    MODEL_ID = "hishab/titu_stt_bn_fastconformer"  # default (overridable)
    SUPPORTED_LANGUAGE = "ben"  # ISO 639-3

    def __init__(
        self,
        device: str = "cuda",
        batch_size: int = 8,
        model_id: Optional[str] = None,
        revision: Optional[str] = None,
    ) -> None:
        self._device = device
        self._revision = revision
        self._batch_size = batch_size
        # fastconformer by default; conformer_large is a config A/B (see
        # TituConfig). Both are NeMo EncDecCTC models.
        self._model_id = model_id or self.MODEL_ID
        self._model: Optional[object] = None
        self._segment_counter: int = 0

    # ─────────────────────────────────────────────────────────────────
    # Lifecycle
    # ─────────────────────────────────────────────────────────────────

    def load(self) -> None:
        if self._model is not None:
            return
        logger.info(
            f"  Loading titu [bold]{self._model_id}[/bold] "
            f"on {self._device}"
        )
        # nemo_toolkit is heavy; import lazily so the rest of the CLI
        # works without loading it.
        try:
            import nemo.collections.asr as nemo_asr
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "titu engine requires nemo_toolkit (a core dependency, "
                "~3 GB). Re-sync the environment: uv sync"
            ) from e
        if self._revision:
            # Pinned commit: fetch that exact .nemo checkpoint (cached,
            # offline-safe once downloaded) and restore it.
            from huggingface_hub import hf_hub_download
            nemo_path = hf_hub_download(
                repo_id=self._model_id,
                filename=f"{self._model_id.split('/')[-1]}.nemo",
                revision=self._revision,
            )
            self._model = nemo_asr.models.ASRModel.restore_from(nemo_path)
        else:
            self._model = nemo_asr.models.ASRModel.from_pretrained(
                self._model_id
            )
        if self._device == "cuda":
            import torch
            if torch.cuda.is_available():
                self._model = self._model.to(torch.device("cuda"))
        self._model.eval()
        logger.info("  ✓ titu loaded")

    def unload(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None

    @property
    def name(self) -> str:
        variant = self._model_id.split("titu_stt_bn_")[-1]
        return f"titu ({variant} CTC Bengali)"

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
        """Transcribe a single chunk and return one ASRSegment."""
        return self.transcribe_batch([chunk], language)

    def transcribe_batch(
        self,
        chunks: list[AudioChunk],
        language: str,
    ) -> list[ASRSegment]:
        """Transcribe a batch of chunks.

        titu's NeMo model.transcribe takes a list of WAV paths and
        returns either strings or hypothesis objects with .text.
        """
        if not chunks:
            return []
        if self._model is None:
            raise RuntimeError(
                "TituEngine not loaded — call load() first")
        if language != self.SUPPORTED_LANGUAGE:
            logger.warning(
                f"  titu engine called with language '{language}' "
                f"but only supports '{self.SUPPORTED_LANGUAGE}' "
                f"(Bengali). Transcribing anyway.")

        wav_paths = [str(c.waveform_path) for c in chunks
                     if c.waveform_path is not None]
        if not wav_paths:
            return []

        # timestamps=True → CTC word timestamps in hyp.timestamp['word'].
        # Fall back to plain transcription if the NeMo build rejects it.
        try:
            outs = self._model.transcribe(
                wav_paths, batch_size=self._batch_size, timestamps=True
            )
        except Exception as e:
            logger.warning("  titu word timestamps unavailable (%s) — "
                           "chunk-level text only", type(e).__name__)
            outs = self._model.transcribe(
                wav_paths, batch_size=self._batch_size
            )

        segments: list[ASRSegment] = []
        for chunk, out in zip(chunks, outs):
            if isinstance(out, str):
                text, words = out, []
            else:
                text = getattr(out, "text", str(out))
                words = self._extract_words(out, chunk.start_s)
            text = (text or "").strip()
            segments.append(ASRSegment(
                segment_id=self._segment_counter,
                start_s=chunk.start_s,
                end_s=chunk.end_s,
                text=text,
                language=self.SUPPORTED_LANGUAGE,
                confidence=0.0,  # CTC doesn't expose calibrated conf
                words=words,      # native CTC word timestamps (absolute)
            ))
            self._segment_counter += 1
        return segments

    @staticmethod
    def _extract_words(hyp, offset_s: float) -> list[WordSegment]:
        """Pull native CTC word timestamps from a NeMo Hypothesis.
        `hyp.timestamp['word']` = [{'word', 'start', 'end'(s), ...}]; times
        are chunk-relative, so add the chunk's absolute start (offset_s)."""
        ts = getattr(hyp, "timestamp", None)
        if not isinstance(ts, dict):
            return []
        out: list[WordSegment] = []
        for w in (ts.get("word") or []):
            word = str(w.get("word") or "").strip()
            s, e = w.get("start"), w.get("end")
            if word and s is not None and e is not None:
                out.append(WordSegment(
                    word=word,
                    start_s=offset_s + float(s),
                    end_s=offset_s + float(e),
                ))
        return out

    def detect_language(
        self,
        audio_path: Path,
    ) -> tuple[str, float]:
        """titu is a single-language Bengali model — always Bengali."""
        return (self.SUPPORTED_LANGUAGE, 1.0)
