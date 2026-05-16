"""
Post-processing module: translation and refinement.

Two backends:

1. **TranslateGemma (default)** — single-pass translation+refinement via
   Google's TranslateGemma 4B model (HuggingFace Transformers).
   Handles all languages including English cleanup (en→en).
   True GPU batch inference for maximum throughput.

2. **CT2 NLLB + Ollama (legacy)** — two-stage pipeline:
   Stage 1: CTranslate2 NLLB-200 batch translation.
   Stage 2: Ollama LLM joint refinement (source + translation).
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from typing import Callable, Optional

from asr_pipeline.config import PostprocessingConfig
from asr_pipeline.language import LanguageRegistry
from asr_pipeline.logging_config import create_progress
from asr_pipeline.models import AlignedSegment, LanguageTier, ProcessedSegment

logger = logging.getLogger("asr_pipeline")


# =============================================================================
# Ollama LLM Processor (Stage 2 only — joint refinement)
# =============================================================================


class OllamaProcessor:
    """
    LLM-based text processor using the Ollama API.

    Used for Stage 2 (joint refinement of source + translation).
    Requires an external Ollama server (``ollama serve``) with
    the desired model already pulled (``ollama pull <model>``).
    """

    def __init__(
        self,
        model: Optional[str] = None,
        base_url: str = "http://localhost:11434",
    ) -> None:
        self._model_name = model
        self._base_url = base_url
        self._client: Optional[object] = None
        self._available = False

    def load(self) -> bool:
        """Verify Ollama connectivity and model availability.

        Returns False if model is None, server is unreachable,
        or model is not found.
        """
        if self._model_name is None:
            logger.info("  Ollama model not configured, skipping LLM stages")
            return False

        try:
            from ollama import Client

            client = Client(host=self._base_url)

            # Check server connectivity and list models
            models_response = client.list()
            available_models = [m.model for m in models_response.models]

            # Check if the requested model is available
            # Ollama model names may or may not include a tag
            found = any(
                self._model_name in name or name.startswith(self._model_name)
                for name in available_models
            )

            if not found:
                logger.warning(
                    f"  Ollama model '{self._model_name}' not found. "
                    f"Available: {available_models}. "
                    f"Pull it with: ollama pull {self._model_name}"
                )
                return False

            self._client = client
            self._available = True
            logger.info(
                f"  \u2713 Ollama connected — model "
                f"[bold]{self._model_name}[/bold] ready"
            )
            return True

        except ImportError:
            logger.warning("  ollama package not installed. pip install ollama")
            return False
        except Exception as e:
            logger.warning(
                f"  Failed to connect to Ollama at {self._base_url}: {e}"
            )
            return False

    def unload(self) -> None:
        """No-op: Ollama manages model lifecycle on the server."""
        self._available = False
        self._client = None

    @property
    def is_loaded(self) -> bool:
        return self._available

    def generate(
        self,
        prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> str:
        """Generate text from a prompt via Ollama."""
        if not self._available or self._client is None:
            return ""

        try:
            response = self._client.generate(  # type: ignore[union-attr]
                model=self._model_name,
                prompt=prompt,
                options={
                    "temperature": temperature,
                    "num_predict": max_tokens,
                    "stop": ["</output>", "\n\n\n"],
                },
                stream=False,
            )
            return response["response"].strip()  # type: ignore[index]

        except Exception as e:
            logger.warning(f"  Ollama generation failed: {e}")
            return ""


# =============================================================================
# Qwen3.5 Source-Text Corrector (runs BEFORE translation)
# =============================================================================


class QwenCorrector:
    """Local Qwen3.5-dense corrector for the source-language transcript.

    Fixes ASR recognition errors and restores code-switched English
    written phonetically in Bengali script — WITHOUT inventing content,
    translating, or (optionally) altering the speaker's register.

    Served via local Ollama (private; nothing leaves the machine).
    Composes OllamaProcessor for connectivity + generation.
    """

    # Validated constrained prompt (n=6 synthetic probe: 0 hallucinations,
    # correct brand/admin-term judgment). {register} swaps the dialect rule.
    _PROMPT = (
        "You are correcting a Bengali speech-to-text transcript. The "
        "recognizer makes phonetic errors and writes spoken English words "
        "in Bengali script.\n"
        "Rules:\n"
        "1. Fix obvious recognition errors (wrong/garbled characters).\n"
        "2. Restore code-switched English words to correct English "
        "spelling (e.g. ভেকসিন -> vaccine, বিকাশ brand -> bKash).\n"
        "3. Keep genuine Bengali words and Bengali administrative terms "
        "in Bengali (e.g. ইউনিয়ন পরিষদ stays Bengali).\n"
        "4. Do NOT add information. Do NOT translate. Do NOT change "
        "meaning. Do NOT invent words. If a span is unintelligible, "
        "leave it unchanged.\n"
        "{register}\n"
        "Output ONLY the corrected transcript on one line, nothing else.\n\n"
        "Transcript: {text}\n"
        "Corrected:"
    )
    _REGISTER_PRESERVE = (
        "5. Preserve the speaker's colloquial and regional forms exactly "
        "(e.g. keep পাঠাইছি, নাই, দিছি — do NOT standardize grammar)."
    )
    _REGISTER_STANDARD = (
        "5. You may normalize colloquial forms to standard Bengali."
    )

    def __init__(self, cfg) -> None:
        self._llm = _make_qwen_llm(cfg)
        self._temperature = cfg.temperature
        self._max_tokens = cfg.max_tokens
        self._register = (
            self._REGISTER_PRESERVE if cfg.preserve_register
            else self._REGISTER_STANDARD
        )
        self._backend = cfg.backend

    def load(self) -> bool:
        ok = self._llm.load()
        if ok:
            logger.info(
                f"  ✓ QwenCorrector ready ({self._backend}, "
                f"register={'preserve' if self._register is self._REGISTER_PRESERVE else 'standard'})"
            )
        else:
            logger.warning(
                "  QwenCorrector: model unavailable — source correction "
                "SKIPPED (raw text passes through unchanged)."
            )
        return ok

    def unload(self) -> None:
        self._llm.unload()

    @property
    def is_loaded(self) -> bool:
        return self._llm.is_loaded

    def correct(self, texts: list[str]) -> list[str]:
        """Correct each transcript. On any failure, fall back to the
        original text for that segment (never drop content)."""
        if not self._llm.is_loaded:
            return list(texts)
        out: list[str] = []
        for t in texts:
            src = (t or "").strip()
            if not src:
                out.append(t)
                continue
            prompt = self._PROMPT.format(register=self._register, text=src)
            try:
                resp = self._llm.generate(
                    prompt,
                    temperature=self._temperature,
                    max_tokens=self._max_tokens,
                ).strip()
            except Exception as e:
                logger.warning(f"  QwenCorrector failed on a segment: {e}")
                resp = ""
            # Guard against empty / refusal / runaway output: fall back.
            if not resp or len(resp) > max(40, len(src) * 4):
                out.append(t)
            else:
                out.append(resp)
        return out


class GGUFProcessor:
    """Local in-process LLM via llama-cpp-python (GGUF). No daemon.

    Drop-in for OllamaProcessor: same .load()/.is_loaded/.generate
    interface so QwenCorrector/QwenTranslator are backend-agnostic.
    """

    def __init__(
        self,
        gguf_path: Optional[str] = None,
        gguf_repo: Optional[str] = None,
        gguf_file: Optional[str] = None,
        n_gpu_layers: int = -1,
        n_ctx: int = 4096,
        n_threads: int = 8,
    ) -> None:
        self._path = gguf_path
        self._repo = gguf_repo
        self._file = gguf_file
        self._n_gpu_layers = n_gpu_layers
        self._n_ctx = n_ctx
        self._n_threads = n_threads
        self._llm: Optional[object] = None
        self._available = False

    def load(self) -> bool:
        try:
            from llama_cpp import Llama
        except ImportError:
            logger.warning(
                "  llama-cpp-python not installed — GGUF backend "
                "unavailable. uv pip install llama-cpp-python "
                "(or use backend: ollama)."
            )
            return False
        try:
            common = dict(
                n_ctx=self._n_ctx,
                n_threads=self._n_threads,
                n_gpu_layers=self._n_gpu_layers,
                verbose=False,
            )
            if self._path:
                self._llm = Llama(model_path=str(self._path), **common)
            elif self._repo and self._file:
                # One-time HF download into the local cache, then local.
                self._llm = Llama.from_pretrained(
                    repo_id=self._repo, filename=self._file, **common
                )
            else:
                logger.warning(
                    "  GGUF backend: set gguf_path OR "
                    "gguf_repo+gguf_file. Skipping."
                )
                return False
            self._available = True
            logger.info("  ✓ GGUF model loaded (llama.cpp, in-process)")
            return True
        except Exception as e:
            logger.warning(f"  Failed to load GGUF model: {e}")
            return False

    def unload(self) -> None:
        self._llm = None
        self._available = False

    @property
    def is_loaded(self) -> bool:
        return self._available

    def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str:
        if not self._available or self._llm is None:
            return ""
        try:
            out = self._llm.create_completion(  # type: ignore[union-attr]
                prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                stop=["</output>", "\n\n\n"],
            )
            return out["choices"][0]["text"].strip()
        except Exception as e:
            logger.warning(f"  GGUF generation failed: {e}")
            return ""


def _make_qwen_llm(cfg):
    """Build the right local LLM client for a Qwen post-processing
    stage from its config (backend: gguf | ollama). Both are local."""
    if getattr(cfg, "backend", "gguf") == "ollama":
        return OllamaProcessor(
            model=cfg.ollama_model, base_url=cfg.ollama_base_url
        )
    return GGUFProcessor(
        gguf_path=cfg.gguf_path,
        gguf_repo=cfg.gguf_repo,
        gguf_file=cfg.gguf_file,
        n_gpu_layers=cfg.n_gpu_layers,
        n_ctx=cfg.n_ctx,
        n_threads=cfg.n_threads,
    )


class QwenTranslator:
    """Local Qwen3.5-dense translator: (corrected) Bengali -> English.

    Faithful translation only. Composes OllamaProcessor (local/private).
    Per-segment fallback to '' on failure (never fabricates).
    """

    # User-specified lean prompt + 2 retained safeguards:
    #   - keep English/brands/numbers (code-switch fidelity = the point)
    #   - output-only (so no preamble pollutes the [en] line)
    _PROMPT = (
        "Translate this Bengali text to English. Translate faithfully — "
        "do not add interpretation, summarize, or omit. Keep English "
        "words, brand names, numbers, and proper nouns exactly as they "
        "are. Output only the English translation, nothing else.\n\n"
        "Bengali: {text}\n"
        "English:"
    )

    def __init__(self, cfg) -> None:
        self._llm = _make_qwen_llm(cfg)
        self._temperature = cfg.temperature
        self._max_tokens = cfg.max_tokens
        self._backend = cfg.backend

    def load(self) -> bool:
        ok = self._llm.load()
        if ok:
            logger.info(f"  ✓ QwenTranslator ready ({self._backend})")
        else:
            logger.warning(
                "  QwenTranslator: model unavailable — translation "
                "will be empty for affected segments."
            )
        return ok

    def unload(self) -> None:
        self._llm.unload()

    @property
    def is_loaded(self) -> bool:
        return self._llm.is_loaded

    def translate_batch(self, texts: list[str]) -> list[str]:
        if not self._llm.is_loaded:
            return [""] * len(texts)
        out: list[str] = []
        for t in texts:
            src = (t or "").strip()
            if not src:
                out.append("")
                continue
            try:
                resp = self._llm.generate(
                    self._PROMPT.format(text=src),
                    temperature=self._temperature,
                    max_tokens=self._max_tokens,
                ).strip()
            except Exception as e:
                logger.warning(f"  QwenTranslator failed on a segment: {e}")
                resp = ""
            out.append(resp)
        return out


# =============================================================================
# CTranslate2 NLLB Translator (Stage 1)
# =============================================================================


class CTranslate2Translator:
    """
    NLLB-200 translator using CTranslate2 for fast batched inference.

    Uses a pre-converted CTranslate2 model for inference and a
    HuggingFace tokenizer for text encoding/decoding. Supports
    translating multiple segments in a single batch call.

    Pre-conversion required::

        ct2-transformers-converter \\
            --model facebook/nllb-200-distilled-600M \\
            --output_dir /path/to/ct2-nllb \\
            --quantization int8
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        tokenizer_name: str = "facebook/nllb-200-distilled-1.3B",
        target_language: str = "eng_Latn",
        max_length: int = 512,
        beam_size: int = 4,
        device: str = "cuda",
    ) -> None:
        self._model_path = model_path
        self._tokenizer_name = tokenizer_name
        self._target_language = target_language
        self._max_length = max_length
        self._beam_size = beam_size
        self._device = device
        self._translator: Optional[object] = None
        self._tokenizer: Optional[object] = None

    def load(self) -> bool:
        """Load CTranslate2 translator and HuggingFace tokenizer."""
        if self._model_path is None:
            logger.info(
                "  CTranslate2 model path not configured, skipping translation"
            )
            return False

        try:
            import ctranslate2
            from transformers import AutoTokenizer

            logger.info(
                f"  Loading CTranslate2 NLLB from "
                f"[file]{self._model_path}[/file]"
            )

            # Determine device for CTranslate2
            ct2_device = "cuda" if self._device == "cuda" else "cpu"
            try:
                import torch

                if ct2_device == "cuda" and not torch.cuda.is_available():
                    ct2_device = "cpu"
            except ImportError:
                ct2_device = "cpu"

            self._translator = ctranslate2.Translator(
                self._model_path,
                device=ct2_device,
                device_index=0,
            )

            self._tokenizer = AutoTokenizer.from_pretrained(
                self._tokenizer_name
            )

            logger.info("  \u2713 CTranslate2 NLLB loaded")
            return True

        except Exception as e:
            logger.warning(f"  Failed to load CTranslate2 translator: {e}")
            return False

    def unload(self) -> None:
        if self._translator is not None:
            del self._translator
            self._translator = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None

    @property
    def is_loaded(self) -> bool:
        return self._translator is not None and self._tokenizer is not None

    def translate_batch(
        self,
        texts: list[str],
        source_language: str,
    ) -> list[str]:
        """
        Translate a batch of texts from source language to English.

        Args:
            texts: List of texts to translate.
            source_language: NLLB language code (e.g., ``"hin_Deva"``).

        Returns:
            List of translated English texts, same length as input.
            On per-item failure, returns the original text for that item.
        """
        if not self.is_loaded or not texts:
            return list(texts)

        try:
            # Set source language on tokenizer
            self._tokenizer.src_lang = source_language  # type: ignore[union-attr]

            # Tokenize all texts (no padding — CT2 handles variable lengths)
            encoded = self._tokenizer(  # type: ignore[misc]
                texts,
                padding=False,
                truncation=True,
                max_length=self._max_length,
                return_tensors=None,
            )
            source_tokens = [
                self._tokenizer.convert_ids_to_tokens(ids)  # type: ignore[union-attr]
                for ids in encoded["input_ids"]
            ]

            # Target prefix: the target language token for each item
            target_prefix = [[self._target_language]] * len(texts)

            # Batch translate
            results = self._translator.translate_batch(  # type: ignore[union-attr]
                source_tokens,
                target_prefix=target_prefix,
                beam_size=self._beam_size,
                max_decoding_length=self._max_length,
            )

            # Decode results
            translations: list[str] = []
            for i, result in enumerate(results):
                try:
                    output_tokens = result.hypotheses[0]
                    token_ids = self._tokenizer.convert_tokens_to_ids(  # type: ignore[union-attr]
                        output_tokens
                    )
                    translated = self._tokenizer.decode(  # type: ignore[union-attr]
                        token_ids, skip_special_tokens=True
                    )
                    translations.append(translated.strip())
                except Exception as e:
                    logger.warning(f"  Decoding failed for segment {i}: {e}")
                    translations.append(texts[i])

            return translations

        except Exception as e:
            logger.warning(f"  Batch translation failed: {e}")
            return list(texts)

    def translate(
        self,
        text: str,
        source_language: str,
    ) -> str:
        """Translate a single text. Convenience wrapper around translate_batch."""
        if not text.strip():
            return text
        results = self.translate_batch([text], source_language)
        return results[0]


# =============================================================================
# TranslateGemma Translator (default backend)
# =============================================================================


# Languages NOT supported by TranslateGemma (55 languages).
# These fall back to CT2 NLLB if configured.
_TRANSLATEGEMMA_UNSUPPORTED = {"mya", "khm"}


class TranslateGemmaTranslator:
    """
    Translation via Google's TranslateGemma 4B (HuggingFace Transformers).

    Single model handles both translation (any→en) and English cleanup (en→en).
    Uses true GPU batch inference via HuggingFace pipeline for fast throughput.
    Auto-downloads model from HuggingFace on first use (~3-8 GB).
    """

    def __init__(
        self,
        model_id: str = "google/translategemma-4b-it",
        batch_size: int = 8,
        max_new_tokens: int = 256,
        device: str = "cuda",
        quantize: Optional[str] = "4bit",
    ) -> None:
        self._model_id = model_id
        self._batch_size = batch_size
        self._max_new_tokens = max_new_tokens
        self._device = device
        self._quantize = quantize
        self._pipe: Optional[object] = None

    def load(self) -> bool:
        """Load TranslateGemma pipeline (downloads model on first use)."""
        try:
            import torch
            from transformers import pipeline as hf_pipeline

            device = self._device
            if device == "cuda" and not torch.cuda.is_available():
                device = "cpu"

            # Determine quantization config
            quantization_config = None
            use_quantization = self._quantize is not None and device == "cuda"

            if use_quantization:
                try:
                    from transformers import BitsAndBytesConfig

                    if self._quantize == "4bit":
                        quantization_config = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_quant_type="nf4",
                            bnb_4bit_use_double_quant=True,
                            bnb_4bit_compute_dtype=torch.bfloat16,
                        )
                    elif self._quantize == "8bit":
                        quantization_config = BitsAndBytesConfig(
                            load_in_8bit=True,
                        )
                except ImportError:
                    logger.warning(
                        "  bitsandbytes not installed — loading without "
                        "quantization. Install with: "
                        "pip install bitsandbytes>=0.41.0 accelerate>=0.20.0"
                    )
                    quantization_config = None
                    use_quantization = False

            if use_quantization and quantization_config is not None:
                # With quantization: use device_map="auto", do NOT pass device=
                logger.info(
                    f"  Loading TranslateGemma from "
                    f"[bold]{self._model_id}[/bold] "
                    f"({self._quantize} quantized, device_map=auto)"
                )
                self._pipe = hf_pipeline(
                    "image-text-to-text",
                    model=self._model_id,
                    model_kwargs={
                        "quantization_config": quantization_config,
                        "device_map": "auto",
                    },
                    torch_dtype=torch.bfloat16,
                )
            else:
                # Full precision: pass device directly
                dtype = torch.bfloat16 if device == "cuda" else torch.float32
                logger.info(
                    f"  Loading TranslateGemma from "
                    f"[bold]{self._model_id}[/bold] ({device}, {dtype})"
                )
                self._pipe = hf_pipeline(
                    "image-text-to-text",
                    model=self._model_id,
                    device=device,
                    torch_dtype=dtype,
                )

            logger.info("  \u2713 TranslateGemma loaded")
            return True

        except Exception as e:
            logger.warning(f"  Failed to load TranslateGemma: {e}")
            return False

    def unload(self) -> None:
        """Release TranslateGemma from memory."""
        if self._pipe is not None:
            del self._pipe
            self._pipe = None
            try:
                import gc
                import torch

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

    @property
    def is_loaded(self) -> bool:
        return self._pipe is not None

    def translate_batch(
        self,
        texts: list[str],
        source_bcp47: str,
        target_bcp47: str = "en",
        on_batch_done: Optional[Callable[[int], None]] = None,
    ) -> list[str]:
        """
        Translate a batch of texts using TranslateGemma.

        Uses TranslateGemma's chat template format with true GPU batching.

        Args:
            texts: List of texts to translate.
            source_bcp47: BCP-47 language code (e.g., "es", "hi", "en").
            target_bcp47: Target language code (default "en").
            on_batch_done: Optional callback called after each mini-batch with
                           the number of items completed in that batch.

        Returns:
            List of translated texts, same length as input.
        """
        if not self.is_loaded or not texts:
            return list(texts)

        try:
            # Build chat messages for each text
            all_messages = []
            for text in texts:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "source_lang_code": source_bcp47,
                                "target_lang_code": target_bcp47,
                                "text": text,
                            }
                        ],
                    }
                ]
                all_messages.append(messages)

            # Run batch inference through HF pipeline
            results: list[str] = []
            for batch_start in range(0, len(all_messages), self._batch_size):
                batch = all_messages[batch_start : batch_start + self._batch_size]
                outputs = self._pipe(
                    text=batch,
                    max_new_tokens=self._max_new_tokens,
                    batch_size=len(batch),
                )
                for output in outputs:
                    try:
                        translated = output[0]["generated_text"][-1]["content"]
                        results.append(translated.strip())
                    except (KeyError, IndexError, TypeError):
                        results.append("")
                if on_batch_done:
                    on_batch_done(len(batch))

            # Fill any missing with originals
            while len(results) < len(texts):
                results.append(texts[len(results)])

            return results

        except Exception as e:
            logger.warning(f"  TranslateGemma batch translation failed: {e}")
            return list(texts)


# =============================================================================
# Post-Processor Orchestrator
# =============================================================================

# ── Joint refinement prompt ─────────────────────────────────────────────────
# The LLM receives both the source transcript and translation together.
# This lets it cross-reference to catch ASR errors that propagated into
# the translation, fix translation artifacts, and correct obvious speaker
# attribution mistakes — all in a single call.

_JOINT_REFINEMENT_PROMPT = """\
You are a specialist in refining ASR transcripts that have been machine-translated to English.

You receive:
- The original {source_language} transcript (may contain ASR errors)
- The machine-translated English version
- Surrounding speaker context for diarization awareness

Your tasks (in order of priority):
1. TRANSLATION QUALITY: Fix awkward phrasing, unnatural word order, and mistranslations. \
Cross-reference the source text to verify meaning is preserved.
2. ASR ERROR RECOVERY: If the source text contains obvious ASR errors (repeated words, \
garbled phrases, hallucinated fragments), infer the correct meaning and fix the English.
3. SPEAKER CONTINUITY: If the context shows the same speaker was split across consecutive \
turns, or a mid-sentence speaker switch seems wrong, note this but still output only the \
refined English for THIS segment.
4. FLUENCY: Make the English natural and readable while preserving the speaker's register \
and intent (formal/informal, technical terms, etc.).

Rules:
- Output ONLY the refined English text, nothing else.
- Do NOT add information that isn't in the source.
- Do NOT transliterate — produce natural English.
- If the source and translation are consistent and fluent, return the translation unchanged.

<context>
{context}
</context>

<source language="{source_language}">
{source_text}
</source>

<translation language="en">
{translated_text}
</translation>

<output>"""

# For English segments: no translation step, just clean up ASR artifacts.
_ENGLISH_CLEANUP_PROMPT = """\
You are an ASR post-processor. Clean up the following English transcript segment.

Fix: repeated words/phrases from overlapping chunks, missing punctuation, \
obvious ASR hallucinations, garbled words.
Do NOT change the meaning, add new content, or rephrase unless the original is unintelligible.
Output ONLY the cleaned text.

<context>
{context}
</context>

<input>{text}</input>
<output>"""

# ── Batched refinement prompts ─────────────────────────────────────────────
# Process multiple segments per LLM call to drastically reduce latency.
# Each segment is numbered; the LLM returns numbered refined lines.

_BATCH_REFINEMENT_PROMPT = """\
You are a specialist in refining ASR transcripts that have been machine-translated to English.

For each numbered segment below, you receive the original {source_language} text and its \
machine translation. Refine each translation:
- Fix awkward phrasing and mistranslations (cross-reference the source).
- Fix obvious ASR errors (repeated words, garbled text, hallucinations).
- Make the English natural while preserving the speaker's register and intent.
- If the translation is already good, return it unchanged.

Output format: one line per segment, numbered to match. Output ONLY the refined English text \
for each line. Do NOT add explanations.

{segments_block}

<output>"""

_BATCH_ENGLISH_CLEANUP_PROMPT = """\
You are an ASR post-processor. For each numbered segment below, clean up the English text.

Fix: repeated words/phrases, missing punctuation, ASR hallucinations, garbled words.
Do NOT change meaning or add content. Output one refined line per segment, numbered to match.

{segments_block}

<output>"""

# Number of segments to group into a single LLM call.
_REFINEMENT_BATCH_SIZE = 10

# Maximum number of surrounding segments to include as speaker context.
_CONTEXT_WINDOW = 3


class PostProcessor:
    """
    Orchestrates post-processing: translation and optional refinement.

    Two backends:
    - **translategemma** (default): Single TranslateGemma 4B model for all
      languages. Handles translation (any→en) and English cleanup (en→en).
    - **ct2_nllb**: Legacy two-stage pipeline with CT2 NLLB translation
      + Ollama LLM joint refinement.
    """

    def __init__(
        self,
        config: PostprocessingConfig,
        language_registry: LanguageRegistry,
        device: str = "cuda",
    ) -> None:
        self._config = config
        self._registry = language_registry
        self._device = device

        # TranslateGemma translator (default backend)
        self._translategemma = TranslateGemmaTranslator(
            model_id=config.translategemma.model_id,
            batch_size=config.translategemma.batch_size,
            max_new_tokens=config.translategemma.max_new_tokens,
            device=device,
            quantize=config.translategemma.quantize,
        )

        # CT2 NLLB translator (legacy/fallback)
        self._translator = CTranslate2Translator(
            model_path=config.translation.model_path,
            tokenizer_name=config.translation.tokenizer_name,
            target_language=config.translation.target_language,
            max_length=config.translation.max_length,
            beam_size=config.translation.beam_size,
            device=device,
        )

        # Ollama LLM for joint refinement (only with ct2_nllb backend)
        self._llm = OllamaProcessor(
            model=config.correction.model,
            base_url=config.correction.base_url,
        )

        # Qwen translator (local; selected via translation_backend="qwen")
        self._qwen_translator: Optional[QwenTranslator] = None
        if config.translation_backend == "qwen":
            self._qwen_translator = QwenTranslator(config.qwen_translator)

        # Source-text corrector (runs before translation).
        self._qwen_corrector: Optional[QwenCorrector] = None
        if config.correction_backend == "qwen":
            self._qwen_corrector = QwenCorrector(config.qwen_corrector)

    def load(self) -> None:
        """Load post-processing models based on the active backend."""
        # Source corrector (independent of translation backend)
        if self._qwen_corrector is not None:
            self._qwen_corrector.load()

        backend = self._config.translation_backend

        if backend == "translategemma":
            self._translategemma.load()
            # Also load CT2 as fallback for unsupported languages
            if self._config.translation.enabled and self._config.translation.model_path:
                self._translator.load()
        elif backend == "qwen":
            if self._qwen_translator is not None:
                self._qwen_translator.load()
        else:
            # Legacy CT2 NLLB + Ollama path
            if self._config.translation.enabled:
                self._translator.load()
            if self._config.refinement.enabled:
                self._llm.load()

    def unload(self) -> None:
        """Release all post-processing models."""
        self._translategemma.unload()
        self._translator.unload()
        self._llm.unload()
        if self._qwen_corrector is not None:
            self._qwen_corrector.unload()
        if self._qwen_translator is not None:
            self._qwen_translator.unload()

    def process(
        self,
        segments: list[AlignedSegment],
    ) -> list[ProcessedSegment]:
        """
        Run post-processing pipeline on aligned segments.

        Backend determines flow:
        - translategemma: single-pass translation+refinement for all languages
        - ct2_nllb: CT2 translation → Ollama joint refinement (legacy)

        Returns list of fully processed segments.
        """
        if not segments:
            return []

        # Pre-compute language metadata for each segment
        lang_configs = []
        for seg in segments:
            lang_config = self._registry.get(seg.language)
            is_english = seg.language == "eng"
            is_high = lang_config.tier == LanguageTier.HIGH
            lang_configs.append((lang_config, is_english, is_high))

        # ── Source-text correction (before translation) ─────────────
        # Default off: corrected == raw (unchanged pipeline behavior).
        if (self._qwen_corrector is not None
                and self._qwen_corrector.is_loaded):
            corrected_texts = self._qwen_corrector.correct(
                [s.raw_text for s in segments]
            )
            # Translation should operate on the CORRECTED text. Use
            # shallow copies so the original raw_text is preserved for
            # the ProcessedSegment.raw_text field.
            tx_segments = [
                s.model_copy(update={"raw_text": ct})
                for s, ct in zip(segments, corrected_texts)
            ]
        else:
            corrected_texts = [s.raw_text for s in segments]
            tx_segments = segments

        backend = self._config.translation_backend

        if backend == "translategemma":
            # Single-pass: TranslateGemma handles translation + refinement
            translations = self._pass_translategemma(tx_segments, lang_configs)
            # For TranslateGemma, translation IS the refined output
            refined_translations = translations
        elif backend == "qwen":
            # Local Qwen3.5-dense: faithful translation of corrected text
            if (self._qwen_translator is not None
                    and self._qwen_translator.is_loaded):
                translations = self._qwen_translator.translate_batch(
                    [s.raw_text for s in tx_segments]
                )
            else:
                translations = [s.raw_text for s in tx_segments]
            refined_translations = translations
        else:
            # Legacy: CT2 translate → Ollama refine
            translations = self._pass_translation(tx_segments, lang_configs)
            refined_translations = self._pass_joint_refinement(
                tx_segments, translations, lang_configs
            )

        # ── Build results ────────────────────────────────────────────
        results: list[ProcessedSegment] = []
        for i, seg in enumerate(segments):
            results.append(
                ProcessedSegment(
                    segment_id=seg.segment_id,
                    start_s=seg.start_s,
                    end_s=seg.end_s,
                    speaker_id=seg.speaker_id,
                    language=seg.language,
                    raw_text=seg.raw_text,
                    corrected_text=corrected_texts[i],
                    english_translation=translations[i],
                    refined_translation=refined_translations[i],
                    confidence=seg.confidence,
                )
            )

        return results

    # ─────────────────────────────────────────────────────────────────
    # TranslateGemma backend
    # ─────────────────────────────────────────────────────────────────

    def _pass_translategemma(
        self,
        segments: list[AlignedSegment],
        lang_configs: list[tuple],
    ) -> list[str]:
        """
        Single-pass translation via TranslateGemma for all segments.

        Non-English: source_lang → en (translation)
        English: en → en (conservative ASR cleanup)
        Unsupported langs (Burmese, Khmer): fall back to CT2 if available.
        """
        n = len(segments)
        translations = [""] * n

        if not self._translategemma.is_loaded:
            # Fallback: pass through raw text
            return [seg.raw_text for seg in segments]

        # Group segments by BCP-47 language code
        lang_groups: dict[str, list[int]] = defaultdict(list)
        ct2_fallback: dict[str, list[int]] = defaultdict(list)

        for i, (lang_config, _is_english, _is_high) in enumerate(lang_configs):
            if not segments[i].raw_text.strip():
                translations[i] = ""
                continue

            if segments[i].language in _TRANSLATEGEMMA_UNSUPPORTED:
                # Route to CT2 fallback
                nllb_code = self._registry.get_nllb_code(segments[i].language)
                ct2_fallback[nllb_code].append(i)
            else:
                lang_groups[lang_config.bcp47].append(i)

        total = sum(len(idxs) for idxs in lang_groups.values())
        total += sum(len(idxs) for idxs in ct2_fallback.values())

        if total == 0:
            return translations

        logger.info(
            f"  Translating {total} segments via TranslateGemma "
            f"({len(lang_groups)} language group(s))"
        )

        progress = create_progress("Translation")
        with progress:
            task = progress.add_task(
                "TranslateGemma Translation", total=total,
            )

            # Process each language group
            for bcp47, indices in lang_groups.items():
                texts = [segments[i].raw_text for i in indices]
                results = self._translategemma.translate_batch(
                    texts,
                    source_bcp47=bcp47,
                    target_bcp47="en",
                    on_batch_done=lambda n: progress.advance(task, advance=n),
                )
                for idx, translated in zip(indices, results):
                    translations[idx] = translated if translated else segments[idx].raw_text

            # CT2 fallback for unsupported languages
            if ct2_fallback and self._translator.is_loaded:
                for nllb_code, indices in ct2_fallback.items():
                    texts = [segments[i].raw_text for i in indices]
                    results = self._translator.translate_batch(texts, nllb_code)
                    for idx, translated in zip(indices, results):
                        translations[idx] = translated
                    progress.advance(task, advance=len(indices))
            elif ct2_fallback:
                # No CT2 available — pass through raw text
                for indices in ct2_fallback.values():
                    for idx in indices:
                        translations[idx] = segments[idx].raw_text
                    progress.advance(task, advance=len(indices))

        return translations

    # ─────────────────────────────────────────────────────────────────
    # CT2 NLLB + Ollama backend (legacy)
    # ─────────────────────────────────────────────────────────────────

    def _pass_translation(
        self,
        segments: list[AlignedSegment],
        lang_configs: list[tuple],
    ) -> list[str]:
        """Stage 1: Batch-translate all non-English segments via CTranslate2."""
        n = len(segments)
        translations = [""] * n

        # For English segments, the "translation" is the raw text itself
        for i, (_lang_config, is_english, _is_high) in enumerate(lang_configs):
            if is_english:
                translations[i] = segments[i].raw_text

        if not (self._config.translation.enabled and self._translator.is_loaded):
            return translations

        # Group non-English segments by NLLB source language code
        lang_groups: dict[str, list[int]] = defaultdict(list)
        for i, (_lang_config, is_english, _is_high) in enumerate(lang_configs):
            if not is_english and segments[i].raw_text.strip():
                nllb_code = self._registry.get_nllb_code(segments[i].language)
                lang_groups[nllb_code].append(i)

        total_to_translate = sum(len(idxs) for idxs in lang_groups.values())
        if total_to_translate == 0:
            return translations

        logger.info(
            f"  Translating {total_to_translate} segments "
            f"in {len(lang_groups)} language group(s)"
        )

        progress = create_progress("Translation")
        with progress:
            task = progress.add_task(
                "CTranslate2 Batch Translation",
                total=total_to_translate,
            )

            for nllb_code, indices in lang_groups.items():
                # Collect texts for this language group
                batch_texts = [segments[i].raw_text for i in indices]

                # Batch translate
                batch_results = self._translator.translate_batch(
                    batch_texts,
                    source_language=nllb_code,
                )

                # Scatter results back
                for idx, translated in zip(indices, batch_results):
                    translations[idx] = translated

                progress.advance(task, advance=len(indices))

        return translations

    def _build_speaker_context(
        self,
        segments: list[AlignedSegment],
        raw_translations: list[str],
        current_idx: int,
    ) -> str:
        """
        Build a short context window of surrounding speaker turns.

        Gives the LLM awareness of the conversation flow so it can
        detect speaker attribution errors (e.g. same speaker split
        across consecutive turns, mid-sentence breaks).
        """
        context_lines: list[str] = []

        # Preceding turns
        start = max(0, current_idx - _CONTEXT_WINDOW)
        for i in range(start, current_idx):
            seg = segments[i]
            text = raw_translations[i] if raw_translations[i] else seg.raw_text
            context_lines.append(f"[{seg.speaker_id}] {text}")

        # Mark current segment
        context_lines.append(">>> CURRENT SEGMENT <<<")

        # Following turns
        end = min(len(segments), current_idx + _CONTEXT_WINDOW + 1)
        for i in range(current_idx + 1, end):
            seg = segments[i]
            text = raw_translations[i] if raw_translations[i] else seg.raw_text
            context_lines.append(f"[{seg.speaker_id}] {text}")

        return "\n".join(context_lines)

    def _pass_joint_refinement(
        self,
        segments: list[AlignedSegment],
        raw_translations: list[str],
        lang_configs: list[tuple],
    ) -> list[str]:
        """
        Stage 2: Batched joint refinement — multiple segments per LLM call.

        Groups segments into batches of ~10 and sends them to Ollama in a
        single prompt, reducing total LLM round-trips by ~10x. For a 248-
        segment file, this means ~25 calls instead of 248.

        For non-English segments: the LLM sees source text + translation.
        For English segments: lightweight ASR cleanup (batched separately).
        """
        refined: list[str] = list(raw_translations)  # start as copy

        if not (self._config.refinement.enabled and self._llm.is_loaded):
            return refined

        # Separate English vs non-English indices
        english_indices = []
        non_english_indices = []
        for i in range(len(segments)):
            if not raw_translations[i].strip():
                continue
            _lang_config, is_english, _is_high = lang_configs[i]
            if is_english:
                english_indices.append(i)
            else:
                non_english_indices.append(i)

        total = len(english_indices) + len(non_english_indices)
        if total == 0:
            return refined

        # Group non-English by source language for coherent batches
        lang_groups: dict[str, list[int]] = defaultdict(list)
        for i in non_english_indices:
            lang_config = lang_configs[i][0]
            lang_groups[lang_config.name].append(i)

        # Count total batches for progress
        total_batches = len(english_indices[::_REFINEMENT_BATCH_SIZE])
        for indices in lang_groups.values():
            total_batches += len(indices[::_REFINEMENT_BATCH_SIZE])

        progress = create_progress("Joint Refinement")
        with progress:
            task = progress.add_task(
                f"Ollama Batched Refinement ({total} segments, "
                f"{total_batches} batches)",
                total=total,
            )

            # Process non-English batches (grouped by language)
            for lang_name, indices in lang_groups.items():
                for batch_start in range(0, len(indices), _REFINEMENT_BATCH_SIZE):
                    batch_indices = indices[
                        batch_start : batch_start + _REFINEMENT_BATCH_SIZE
                    ]
                    self._refine_batch_non_english(
                        segments, raw_translations, refined,
                        batch_indices, lang_name,
                    )
                    progress.advance(task, advance=len(batch_indices))

            # Process English batches
            for batch_start in range(0, len(english_indices), _REFINEMENT_BATCH_SIZE):
                batch_indices = english_indices[
                    batch_start : batch_start + _REFINEMENT_BATCH_SIZE
                ]
                self._refine_batch_english(
                    segments, refined, batch_indices,
                )
                progress.advance(task, advance=len(batch_indices))

        return refined

    def _refine_batch_non_english(
        self,
        segments: list[AlignedSegment],
        raw_translations: list[str],
        refined: list[str],
        batch_indices: list[int],
        source_language: str,
    ) -> None:
        """Send a batch of non-English segments to Ollama for joint refinement."""
        # Build numbered segments block
        lines = []
        for seq, i in enumerate(batch_indices, 1):
            src = segments[i].raw_text.replace("\n", " ")
            tgt = raw_translations[i].replace("\n", " ")
            speaker = segments[i].speaker_id
            lines.append(
                f"{seq}. [{speaker}] Source: {src}\n"
                f"   Translation: {tgt}"
            )
        segments_block = "\n".join(lines)

        prompt = _BATCH_REFINEMENT_PROMPT.format(
            source_language=source_language,
            segments_block=segments_block,
        )

        result = self._llm.generate(
            prompt,
            temperature=self._config.refinement.temperature,
            max_tokens=self._config.refinement.max_tokens * min(len(batch_indices), 4),
        )

        if result:
            self._parse_batch_result(result, batch_indices, refined, raw_translations)

    def _refine_batch_english(
        self,
        segments: list[AlignedSegment],
        refined: list[str],
        batch_indices: list[int],
    ) -> None:
        """Send a batch of English segments to Ollama for ASR cleanup."""
        lines = []
        for seq, i in enumerate(batch_indices, 1):
            text = segments[i].raw_text.replace("\n", " ")
            speaker = segments[i].speaker_id
            lines.append(f"{seq}. [{speaker}] {text}")
        segments_block = "\n".join(lines)

        prompt = _BATCH_ENGLISH_CLEANUP_PROMPT.format(
            segments_block=segments_block,
        )

        result = self._llm.generate(
            prompt,
            temperature=self._config.refinement.temperature,
            max_tokens=self._config.refinement.max_tokens * min(len(batch_indices), 4),
        )

        if result:
            # For English, raw_translations == raw_text, use refined as fallback
            self._parse_batch_result(result, batch_indices, refined, refined)

    @staticmethod
    def _parse_batch_result(
        result: str,
        batch_indices: list[int],
        refined: list[str],
        fallback: list[str],
    ) -> None:
        """
        Parse numbered LLM output and scatter results back into refined[].

        Expected format from LLM:
            1. Refined text for first segment
            2. Refined text for second segment
            ...

        Falls back to the existing translation if a line can't be parsed.
        """
        # Parse lines like "1. text", "1) text", or just "1 text"
        parsed: dict[int, str] = {}
        for line in result.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            match = re.match(r"^(\d+)[.):\s]+\s*(.*)", line)
            if match:
                seq_num = int(match.group(1))
                text = match.group(2).strip()
                if text:
                    parsed[seq_num] = text

        # Scatter parsed results back
        for seq, idx in enumerate(batch_indices, 1):
            if seq in parsed:
                refined[idx] = parsed[seq]
            # else: keep existing translation (fallback)
