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
        decoder: Optional[str] = None,
        language_head: Optional[str] = None,
        target_lang: Optional[str] = None,
        device: str = "cuda",
        batch_size: int = 8,
    ) -> None:
        self._model_id = model_id
        self._language = language            # ISO 639-3, set on segments
        self._backend = backend
        self._revision = revision
        self._nemo_file = nemo_file
        self._decoder = decoder               # hybrid nemo: "ctc" | "rnnt"
        # multi-softmax nemo (AI4Bharat fork): language head to decode
        self._language_head = language_head
        self._lang_tokenizer: Optional[object] = None
        self._lang_blank: int = 0
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
        if self._language_head:
            # AI4Bharat-fork multi-softmax checkpoint → compatibility patches
            from asr_pipeline.engines.indic_multisoftmax import (
                apply_multisoftmax_patches)
            apply_multisoftmax_patches()
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
        if self._language_head:
            self._setup_language_head()
        elif self._decoder:
            # Hybrid checkpoints (e.g. IndicConformer: CTC + RNN-T heads)
            # decode with RNN-T by default; the witness wants the CTC head.
            if not hasattr(self._model, "change_decoding_strategy") or \
                    not hasattr(self._model, "ctc_decoder"):
                raise ValueError(
                    f"{self._model_id} is not a hybrid CTC model; remove "
                    f"'decoder: {self._decoder}' from the profile")
            self._model.change_decoding_strategy(
                decoder_type=self._decoder, verbose=False)
            logger.info(f"  decoder: {self._decoder}")
        if self._use_cuda():
            import torch
            self._model = self._model.to(torch.device("cuda"))
        self._model.eval()

    def _setup_language_head(self) -> None:
        """Multi-softmax checkpoint: restrict the CTC head to one language's
        token block (+ blank), as AI4Bharat's fork does at inference."""
        tok = self._model.tokenizer
        lang = self._language_head
        if not hasattr(tok, "langs_by_token_id") or \
                lang not in getattr(tok, "tokenizers_dict", {}):
            raise ValueError(
                f"{self._model_id}: no '{lang}' head in an aggregate "
                f"(multi-language) tokenizer; check witness_ctc.language_head")
        masks = {}
        for name in tok.tokenizers_dict:
            mask = [tok.langs_by_token_id[i] == name
                    for i in range(len(tok.langs_by_token_id))]
            mask.append(True)                 # blank
            masks[name] = mask
        self._model.ctc_decoder.language_masks = masks
        # per-language blank sits right after that language's tokens
        self._lang_blank = (self._model.joint._vocab_size
                            // len(tok.tokenizers_dict))
        self._lang_tokenizer = tok.tokenizers_dict[lang]
        logger.info(f"  language head: {lang} (CTC)")

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

        if self._language_head:
            results = self._transcribe_language_head(chunks)
        elif self._backend == "nemo":
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

    def _transcribe_language_head(
        self, chunks: list[AudioChunk],
    ) -> list[tuple[str, list[WordSegment]]]:
        """Greedy CTC over one language head of a multi-softmax checkpoint,
        with word timestamps from the frames where each token is emitted."""
        import soundfile as sf
        import torch

        cfg = self._model.cfg
        frame_s = (cfg.preprocessor.window_stride
                   * self._model.encoder.subsampling_factor)
        dev = next(self._model.parameters()).device
        results = []
        for b in range(0, len(chunks), self._batch_size):
            batch = chunks[b:b + self._batch_size]
            wavs = []
            for c in batch:
                audio, sr = sf.read(str(c.waveform_path), dtype="float32")
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)
                if sr != _SR:
                    raise ValueError(f"expected {_SR} Hz chunks, got {sr} Hz")
                wavs.append(torch.from_numpy(audio))
            lengths = torch.tensor([len(w) for w in wavs], device=dev)
            padded = torch.nn.utils.rnn.pad_sequence(
                wavs, batch_first=True).to(dev)
            with torch.inference_mode():
                encoded, enc_len = self._model(
                    input_signal=padded, input_signal_length=lengths)
                log_probs = self._model.ctc_decoder(
                    encoder_output=encoded,
                    language_ids=[self._language_head] * len(batch))
            for k, chunk in enumerate(batch):
                preds = log_probs[k, :int(enc_len[k])].argmax(-1).tolist()
                results.append(self._ctc_words(preds, frame_s,
                                               chunk.start_s))
        return results

    def _ctc_words(self, preds: list[int], frame_s: float,
                   offset_s: float) -> tuple[str, list[WordSegment]]:
        """Collapse greedy CTC frames → text + words (a SentencePiece piece
        starting with ▁ opens a new word)."""
        tok = self._lang_tokenizer
        ids, frames, prev = [], [], None
        for t, p in enumerate(preds):
            if p != self._lang_blank and p != prev:
                ids.append(p)
                frames.append(t)
            prev = p
        if not ids:
            return "", []
        pieces = tok.ids_to_tokens(ids)
        groups: list[list[int]] = []
        for i, piece in enumerate(pieces):
            if not groups or piece.startswith("▁"):
                groups.append([])
            groups[-1].append(i)
        words = []
        for g in groups:
            word = tok.ids_to_text([ids[i] for i in g]).strip()
            if word:
                words.append(WordSegment(
                    word=word,
                    start_s=offset_s + frames[g[0]] * frame_s,
                    end_s=offset_s + (frames[g[-1]] + 1) * frame_s,
                ))
        return tok.ids_to_text(ids).strip(), words

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
