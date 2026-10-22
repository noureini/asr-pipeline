"""CTC witness engine (witness 1 of the fusion pipeline).

An acoustically literal CTC model: weak language model, so it preserves how
names, numbers and dialect words actually sounded, and it cannot loop or
hallucinate (CTC guarantee). Two backends, chosen by the language profile:

  nemo  a NeMo `.nemo` checkpoint, e.g. Bengali titu
        (hishab/titu_stt_bn_fastconformer); word timestamps from
        `transcribe(timestamps=True)` → `hyp.timestamp['word']`.
  hf    a Hugging Face CTC model (wav2vec2 / XLS-R / MMS); word timestamps
        from the CTC frame offsets (`output_word_offsets=True`).

Either way the word timing comes from the SAME model that produced the text,
so no separate forced aligner is needed and timestamps never disagree with
the words.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from asr_pipeline.engines import BaseASREngine
from asr_pipeline.models import ASRSegment, AudioChunk, WordSegment

logger = logging.getLogger("asr_pipeline")

_SR = 16000  # preprocessed chunks are 16 kHz mono


class CTCEngine(BaseASREngine):
    """CTC witness: NeMo or Hugging Face backend, word timestamps."""

    def __init__(
        self,
        model_id: str,
        language: str,
        backend: str = "nemo",
        revision: Optional[str] = None,
        nemo_file: Optional[str] = None,
        target_lang: Optional[str] = None,
        device: str = "cuda",
        batch_size: int = 8,
    ) -> None:
        self._model_id = model_id
        self._language = language            # ISO 639-3, set on segments
        self._backend = backend
        self._revision = revision
        self._nemo_file = nemo_file
        self._target_lang = target_lang
        self._device = device
        self._batch_size = batch_size
        self._model: Optional[object] = None
        self._processor: Optional[object] = None   # hf backend only
        self._segment_counter: int = 0

    # ─────────────────────────────────────────────────────────────────
    # Lifecycle
    # ─────────────────────────────────────────────────────────────────

    def load(self) -> None:
        if self._model is not None:
            return
        logger.info(
            f"  Loading CTC witness [bold]{self._model_id}[/bold] "
            f"({self._backend}) on {self._device}"
        )
        if self._backend == "nemo":
            self._load_nemo()
        elif self._backend == "hf":
            self._load_hf()
        else:
            raise ValueError(f"Unknown CTC backend '{self._backend}' "
                             "(expected 'nemo' or 'hf')")
        logger.info("  ✓ CTC witness loaded")

    def _use_cuda(self) -> bool:
        if self._device != "cuda":
            return False
        import torch
        return torch.cuda.is_available()

    def _load_nemo(self) -> None:
        # nemo_toolkit is heavy; import lazily so the rest of the CLI
        # works without loading it.
        try:
            import nemo.collections.asr as nemo_asr
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "The nemo CTC backend requires nemo_toolkit (a core "
                "dependency, ~3 GB). Re-sync the environment: uv sync"
            ) from e
        if self._revision:
            # Pinned commit: fetch that exact .nemo checkpoint (cached,
            # offline-safe once downloaded) and restore it.
            from huggingface_hub import hf_hub_download
            nemo_path = hf_hub_download(
                repo_id=self._model_id,
                filename=self.nemo_filename,
                revision=self._revision,
            )
            self._model = nemo_asr.models.ASRModel.restore_from(nemo_path)
        else:
            self._model = nemo_asr.models.ASRModel.from_pretrained(
                self._model_id
            )
        if self._use_cuda():
            import torch
            self._model = self._model.to(torch.device("cuda"))
        self._model.eval()

    def _load_hf(self) -> None:
        from transformers import AutoModelForCTC, AutoProcessor
        kw: dict = {"revision": self._revision}
        if self._target_lang:
            # MMS-style multilingual CTC: per-language adapter + vocabulary.
            kw.update(target_lang=self._target_lang)
            self._processor = AutoProcessor.from_pretrained(
                self._model_id, **kw)
            self._model = AutoModelForCTC.from_pretrained(
                self._model_id, ignore_mismatched_sizes=True, **kw)
        else:
            self._processor = AutoProcessor.from_pretrained(
                self._model_id, **kw)
            self._model = AutoModelForCTC.from_pretrained(
                self._model_id, **kw)
        if self._use_cuda():
            self._model = self._model.to("cuda")
        self._model.eval()

    def unload(self) -> None:
        self._model = None
        self._processor = None

    @property
    def nemo_filename(self) -> str:
        return self._nemo_file or f"{self._model_id.split('/')[-1]}.nemo"

    @property
    def name(self) -> str:
        return f"{self._model_id} (CTC, {self._backend})"

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
        """Transcribe a batch of chunks into one ASRSegment each, with
        absolute word timestamps."""
        chunks = [c for c in chunks if c.waveform_path is not None]
        if not chunks:
            return []
        if self._model is None:
            raise RuntimeError("CTCEngine not loaded — call load() first")

        if self._backend == "nemo":
            results = self._transcribe_nemo(chunks)
        else:
            results = self._transcribe_hf(chunks)

        segments: list[ASRSegment] = []
        for chunk, (text, words) in zip(chunks, results):
            segments.append(ASRSegment(
                segment_id=self._segment_counter,
                start_s=chunk.start_s,
                end_s=chunk.end_s,
                text=(text or "").strip(),
                language=self._language,
                confidence=0.0,  # CTC doesn't expose calibrated conf
                words=words,      # native CTC word timestamps (absolute)
            ))
            self._segment_counter += 1
        return segments

    def _transcribe_nemo(
        self, chunks: list[AudioChunk],
    ) -> list[tuple[str, list[WordSegment]]]:
        wav_paths = [str(c.waveform_path) for c in chunks]
        # timestamps=True → CTC word timestamps in hyp.timestamp['word'].
        # Fall back to plain transcription if the NeMo build rejects it.
        try:
            outs = self._model.transcribe(
                wav_paths, batch_size=self._batch_size, timestamps=True
            )
        except Exception as e:
            logger.warning("  CTC word timestamps unavailable (%s) — "
                           "chunk-level text only", type(e).__name__)
            outs = self._model.transcribe(
                wav_paths, batch_size=self._batch_size
            )
        results = []
        for chunk, out in zip(chunks, outs):
            if isinstance(out, str):
                results.append((out, []))
            else:
                results.append((getattr(out, "text", str(out)),
                                self._nemo_words(out, chunk.start_s)))
        return results

    def _transcribe_hf(
        self, chunks: list[AudioChunk],
    ) -> list[tuple[str, list[WordSegment]]]:
        import soundfile as sf
        import torch

        arrays = []
        for c in chunks:
            audio, sr = sf.read(str(c.waveform_path), dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if sr != _SR:
                raise ValueError(f"expected {_SR} Hz chunks, got {sr} Hz")
            arrays.append(audio)

        # seconds per CTC output frame (e.g. 320 samples = 20 ms)
        frame_s = self._model.config.inputs_to_logits_ratio / _SR
        dev = next(self._model.parameters()).device
        results = []
        for b in range(0, len(arrays), self._batch_size):
            batch = arrays[b:b + self._batch_size]
            inputs = self._processor(batch, sampling_rate=_SR,
                                     return_tensors="pt", padding=True)
            with torch.no_grad():
                logits = self._model(
                    inputs.input_values.to(dev),
                    attention_mask=(inputs.attention_mask.to(dev)
                                    if "attention_mask" in inputs else None),
                ).logits
            ids = torch.argmax(logits, dim=-1).cpu()
            dec = self._processor.batch_decode(ids, output_word_offsets=True)
            for k, chunk in enumerate(chunks[b:b + self._batch_size]):
                results.append((
                    dec.text[k],
                    self.hf_words(dec.word_offsets[k], frame_s,
                                  chunk.start_s),
                ))
        return results

    @staticmethod
    def _nemo_words(hyp, offset_s: float) -> list[WordSegment]:
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

    @staticmethod
    def hf_words(offsets: list[dict], frame_s: float,
                 offset_s: float) -> list[WordSegment]:
        """CTC frame offsets ([{'word', 'start_offset', 'end_offset'}], in
        output frames) → absolute WordSegments."""
        out: list[WordSegment] = []
        for w in offsets or []:
            word = str(w.get("word") or "").strip()
            if word:
                out.append(WordSegment(
                    word=word,
                    start_s=offset_s + w["start_offset"] * frame_s,
                    end_s=offset_s + w["end_offset"] * frame_s,
                ))
        return out

    def detect_language(
        self,
        audio_path: Path,
    ) -> tuple[str, float]:
        """A CTC witness is single-language: the profile's language."""
        return (self._language, 1.0)
