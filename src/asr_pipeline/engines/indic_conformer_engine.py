"""IndicConformer ASR engine (AI4Bharat, 22 Indic languages).

Uses ai4bharat/indic-conformer-600m-multilingual — a 600M-param
Conformer trained on the AI4Bharat data lineage (ULCA, KathBath,
Shrutilipi, MUCS + IndicVoices). Supports CTC and RNNT decoding;
RNNT is usually more accurate on conjuncts / word boundaries
(exactly titu's weak spots), CTC is faster.

Loads via stock `transformers` with trust_remote_code=True — NO
custom NeMo fork needed, so it coexists with the titu engine in the
same venv. Native output is chunk-level text without word
timestamps; the pipeline's MMS_FA ForcedAligner refines them
downstream, matching the WhisperX-style approach used elsewhere.

Model call signature (from the model card):
    model(waveform_tensor, lang_code, decoder)   # decoder: "ctc"|"rnnt"
where waveform is mono 16 kHz, lang_code is ISO-639-1 (Bengali = "bn").
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from asr_pipeline.engines import BaseASREngine
from asr_pipeline.models import ASRSegment, AudioChunk

logger = logging.getLogger("asr_pipeline")


# ISO 639-3 (pipeline-internal) → IndicConformer ISO 639-1 codes.
# Extend as needed for other Indic languages.
_LANG_3_TO_IC = {
    "ben": "bn",   # Bengali
    "hin": "hi",   # Hindi
    "tam": "ta",   # Tamil
    "tel": "te",   # Telugu
    "mar": "mr",   # Marathi
    "guj": "gu",   # Gujarati
    "kan": "kn",   # Kannada
    "mal": "ml",   # Malayalam
    "pan": "pa",   # Punjabi
    "ori": "or",   # Odia
    "asm": "as",   # Assamese
    "urd": "ur",   # Urdu
}


class IndicConformerEngine(BaseASREngine):
    """
    IndicConformer ASR — AI4Bharat 600M multilingual Conformer.

    Open-source, local, 8 GB-friendly. Drop-in alternative to titu for
    Bengali; also covers 21 other Indic languages. Native output is
    segment-level text only; word timestamps come from MMS_FA.
    """

    MODEL_ID = "ai4bharat/indic-conformer-600m-multilingual"
    SUPPORTED_LANGUAGE = "ben"  # ISO 639-3 default

    def __init__(
        self,
        device: str = "cuda",
        decoder: str = "rnnt",   # "rnnt" (more accurate) or "ctc" (faster)
    ) -> None:
        self._device = device
        self._decoder = decoder.lower()
        if self._decoder not in ("ctc", "rnnt"):
            raise ValueError(
                f"decoder must be 'ctc' or 'rnnt', got '{decoder}'")
        self._model: Optional[object] = None
        self._resamplers: dict = {}  # cache per (orig_sr) resampler
        self._segment_counter: int = 0

    # ─────────────────────────────────────────────────────────────────
    # Lifecycle
    # ─────────────────────────────────────────────────────────────────

    def load(self) -> None:
        if self._model is not None:
            return
        logger.info(
            f"  Loading IndicConformer [bold]{self.MODEL_ID}[/bold] "
            f"on {self._device} (decoder={self._decoder})"
        )
        try:
            from transformers import AutoModel
            import torch  # noqa: F401  (needed for device move)
        except ImportError as e:
            raise ImportError(
                "IndicConformer engine requires transformers + torch. "
                "Both should already be installed for the pipeline."
            ) from e

        # ── Disable the TorchScript GPU kernel fuser ──────────────────
        # IndicConformer's feature extractor computes an STFT, which
        # produces complex tensors. PyTorch's TorchScript fuser
        # (NVFuser / NNC tensorexpr) tries to JIT-compile a fused CUDA
        # kernel for ops on c10::complex<float> and emits invalid code
        # ("fabs((float) c10::complex)"), failing every chunk with an
        # NVRTC compile error. Disabling the fusers makes those ops run
        # as ordinary (unfused) kernels — correct, marginally slower.
        self._disable_jit_fusers()

        # trust_remote_code: the repo ships custom modeling code that
        # implements the model(wav, lang, decoder) call interface.
        self._model = AutoModel.from_pretrained(
            self.MODEL_ID, trust_remote_code=True
        )
        if self._device == "cuda":
            import torch
            if torch.cuda.is_available():
                self._model = self._model.to(torch.device("cuda"))
        # Many HF models expose .eval(); guard in case the custom class
        # doesn't subclass nn.Module in the usual way.
        if hasattr(self._model, "eval"):
            self._model.eval()
        logger.info("  ✓ IndicConformer loaded")

    @staticmethod
    def _disable_jit_fusers() -> None:
        """Turn off TorchScript profiling executor + CPU/GPU fusers.

        Prevents the NVRTC c10::complex compile failure on the model's
        STFT. All calls are wrapped defensively because these are
        private torch APIs whose availability varies across versions.
        """
        import torch

        for fn, args in [
            ("_jit_set_profiling_executor", (False,)),
            ("_jit_set_profiling_mode", (False,)),
            ("_jit_override_can_fuse_on_cpu", (False,)),
            ("_jit_override_can_fuse_on_gpu", (False,)),
            ("_jit_set_texpr_fuser_enabled", (False,)),
            ("_jit_set_nvfuser_enabled", (False,)),
        ]:
            try:
                getattr(torch._C, fn)(*args)
            except Exception:
                pass  # API absent in this torch build — fine
        try:
            torch.jit.enable_onednn_fusion(False)
        except Exception:
            pass

    def unload(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
            self._resamplers.clear()
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass

    @property
    def name(self) -> str:
        return f"IndicConformer-600M ({self._decoder.upper()})"

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    # ─────────────────────────────────────────────────────────────────
    # Transcription
    # ─────────────────────────────────────────────────────────────────

    def _ic_lang(self, language: str) -> str:
        """Map the pipeline's ISO-639-3 code to IndicConformer's 639-1."""
        code = _LANG_3_TO_IC.get(language)
        if code is None:
            logger.warning(
                f"  IndicConformer: no language mapping for '{language}', "
                f"defaulting to 'bn' (Bengali).")
            return "bn"
        return code

    def _load_wav_mono16k(self, wav_path: Path):
        """Load a WAV/FLAC, downmix to mono, resample to 16 kHz."""
        import torch
        import torchaudio
        wav, sr = torchaudio.load(str(wav_path))
        if wav.dim() == 2 and wav.size(0) > 1:
            wav = torch.mean(wav, dim=0, keepdim=True)
        elif wav.dim() == 1:
            wav = wav.unsqueeze(0)
        target = 16000
        if sr != target:
            rs = self._resamplers.get(sr)
            if rs is None:
                rs = torchaudio.transforms.Resample(
                    orig_freq=sr, new_freq=target)
                self._resamplers[sr] = rs
            wav = rs(wav)
        if self._device == "cuda":
            if torch.cuda.is_available():
                wav = wav.to(torch.device("cuda"))
        return wav

    def transcribe_chunk(
        self,
        chunk: AudioChunk,
        language: str,
    ) -> list[ASRSegment]:
        """Transcribe a single chunk → one ASRSegment."""
        if self._model is None:
            raise RuntimeError(
                "IndicConformerEngine not loaded — call load() first")
        if chunk.waveform_path is None:
            return []

        ic_lang = self._ic_lang(language)
        wav = self._load_wav_mono16k(chunk.waveform_path)

        try:
            import torch
            with torch.no_grad():
                out = self._model(wav, ic_lang, self._decoder)
        except Exception as e:
            logger.warning(
                f"  IndicConformer failed on chunk "
                f"[{chunk.start_s:.1f}-{chunk.end_s:.1f}s]: {e}")
            out = ""

        # The custom model may return a str, a list[str], or an object.
        if isinstance(out, str):
            text = out
        elif isinstance(out, (list, tuple)) and out:
            text = out[0] if isinstance(out[0], str) else str(out[0])
        else:
            text = getattr(out, "text", str(out)) if out else ""
        text = (text or "").strip()

        seg = ASRSegment(
            segment_id=self._segment_counter,
            start_s=chunk.start_s,
            end_s=chunk.end_s,
            text=text,
            language=language,
            confidence=0.0,   # CTC/RNNT don't expose a calibrated conf
            words=[],          # word timestamps via ForcedAligner
        )
        self._segment_counter += 1
        return [seg]

    def transcribe_batch(
        self,
        chunks: list[AudioChunk],
        language: str,
    ) -> list[ASRSegment]:
        """Transcribe a batch. The custom model call is per-utterance,
        so we loop; chunks are short (≤15 s) so this is fine."""
        segments: list[ASRSegment] = []
        for chunk in chunks:
            segments.extend(self.transcribe_chunk(chunk, language))
        return segments

    def detect_language(
        self,
        audio_path: Path,
    ) -> tuple[str, float]:
        """Single-config Bengali deployment — always Bengali here.
        (The model supports 22 languages but the pipeline routes by
        the user-specified --language.)"""
        return (self.SUPPORTED_LANGUAGE, 1.0)
