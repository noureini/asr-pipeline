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
# Survey wiki auto-loader (used by QwenCorrector / QwenTranslator)
# =============================================================================
# If data/survey_wiki/ is present (built once by
# scripts/build_survey_wiki.py on the archive/experiments branch from the MMI Protocol docx), inject
# the domain vocabulary into the corrector's system context so the
# LLM gravitates toward valid survey-domain terms when titu's
# output is ambiguous. Zero-config: drops in automatically when the
# wiki files exist.

def _load_survey_wiki_for_prompt() -> str:
    """Return a compact context string from the survey wiki, or ""
    if the wiki isn't present."""
    from pathlib import Path
    repo = Path(__file__).resolve().parents[2]
    wiki = repo / "data" / "survey_wiki"
    if not wiki.exists():
        return ""
    pieces: list[str] = []
    # Domain vocabulary — the main quality lever
    vfile = wiki / "domain_vocabulary.md"
    if vfile.exists():
        terms: list[str] = []
        for ln in vfile.read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            if ln.startswith("Extracted from"):
                continue
            terms.append(ln)
        if terms:
            pieces.append(
                "This is an MMI (Mental Model Interview) survey on "
                "climate risk in coastal Bangladesh (Satkhira / "
                "Sunamganj). Expected domain vocabulary:\n"
                + " | ".join(terms)[:1500]
            )
    # Section structure (helps the LLM understand which part of the
    # interview a chunk likely comes from)
    sfile = wiki / "sections.json"
    if sfile.exists():
        import json
        try:
            sections = json.loads(sfile.read_text(encoding="utf-8"))
            names = [s.get("title", "")
                     for s in sections if s.get("title")]
            if names:
                pieces.append(
                    "Interview sections: " + " → ".join(names))
        except Exception:
            pass
    return "\n\n".join(pieces)


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
        "{context}"
        "Rules:\n"
        "1. Fix ONLY clear recognition errors — garbled/non-word "
        "tokens and obvious phonetic slips.\n"
        "2. ONLY when you are CONFIDENT a token is a code-switched "
        "English word written phonetically in Bengali script, restore "
        "it to correct English spelling (e.g. ভেকসিন -> vaccine, "
        "বিকাশ brand -> bKash). If a token is garbled or you are not "
        "sure which English word was meant, leave it unchanged.\n"
        "3. Keep genuine Bengali words and Bengali administrative terms "
        "in Bengali (e.g. ইউনিয়ন পরিষদ stays Bengali).\n"
        "4. INTELLIGENT VERBATIM: do NOT add, translate, summarize, "
        "rephrase, normalize dialect, smooth grammar, or invent. Keep "
        "the speaker's words/disfluencies as spoken. If a span is "
        "unintelligible, leave it unchanged.\n"
        "{register}\n"
        "6. If the same word or short phrase repeats many times in a "
        "row, that is a recognizer loop (an artifact, NOT speech): "
        "keep ONE occurrence and replace the rest with [unclear].\n"
        "Output ONLY the corrected transcript on one line, nothing else.\n\n"
        "{dyn}"
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
        ctx_pieces = []
        # Explicit user-provided context
        user_ctx = (getattr(cfg, "context", "") or "").strip()
        if user_ctx:
            ctx_pieces.append(user_ctx)
        # Auto-load the survey wiki (built once via
        # scripts/build_survey_wiki.py on the archive/experiments branch) if present. Gives the
        # corrector the survey domain anchor — the LLM gravitates
        # toward valid domain terms when titu's output is ambiguous
        # (e.g. পৌরসা → পৌরসভা, আইলা preserved as the cyclone name).
        survey_ctx = _load_survey_wiki_for_prompt()
        if survey_ctx:
            ctx_pieces.append(survey_ctx)
            logger.info(
                "  QwenCorrector: loaded survey wiki vocabulary "
                f"({len(survey_ctx)} chars)")
        full_ctx = "\n\n".join(ctx_pieces)
        self._context = f"Context: {full_ctx}\n" if full_ctx else ""
        self._dyn = ""  # per-call rolling/entity context (caller-set)
        self._backend = cfg.backend

    def set_context(self, dyn: str) -> None:
        """Set per-call dynamic context (rolling prior + entity memory).
        Empty string disables it. Caller updates this before each
        correct() call; default '' keeps production behaviour."""
        self._dyn = dyn or ""

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
        progress = create_progress("Source correction (LLM)")
        with progress:
            task = progress.add_task(
                "QwenCorrector (per-segment)", total=len(texts))
            for t in texts:
                src = (t or "").strip()
                if not src:
                    out.append(t)
                    progress.advance(task)
                    continue
                prompt = self._PROMPT.format(
                    register=self._register, text=src,
                    context=self._context, dyn=self._dyn)
                try:
                    resp = self._llm.generate(
                        prompt,
                        temperature=self._temperature,
                        max_tokens=self._max_tokens,
                    ).strip()
                except Exception as e:
                    logger.warning(
                        f"  QwenCorrector failed on a segment: {e}")
                    resp = ""
                # Guard against empty / refusal / runaway output.
                if not resp or len(resp) > max(40, len(src) * 4):
                    out.append(t)
                else:
                    out.append(resp)
                progress.advance(task)
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
        gguf_revision: Optional[str] = None,
        n_gpu_layers: int = -1,
        n_ctx: int = 4096,
        n_threads: int = 8,
        prompt_lookup_tokens: int = 0,
        flash_attn: bool = False,
        kv_quant: bool = False,
    ) -> None:
        self._path = gguf_path
        self._repo = gguf_repo
        self._file = gguf_file
        self._revision = gguf_revision
        self._n_gpu_layers = n_gpu_layers
        self._n_ctx = n_ctx
        self._n_threads = n_threads
        # Flash attention shrinks the KV cache + compute buffers and
        # speeds decode — critical on 8GB where the 7GB model leaves
        # almost no room for KV (avoids paging to host RAM).
        self._flash_attn = flash_attn
        # Store the KV cache as q8_0 (half the memory) — frees VRAM so
        # the whole model+KV fits on-card instead of spilling.
        self._kv_quant = kv_quant
        # >0 enables prompt-lookup speculative decoding: the model
        # proposes this many tokens by copying n-grams from the prompt
        # and verifies them — lossless speedup on tasks where the output
        # echoes the input (translation, fusion). 0 = disabled.
        self._prompt_lookup_tokens = prompt_lookup_tokens
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
                flash_attn=self._flash_attn,
                verbose=False,
            )
            if self._kv_quant:
                try:
                    from llama_cpp import GGML_TYPE_Q8_0
                    common["type_k"] = GGML_TYPE_Q8_0
                    common["type_v"] = GGML_TYPE_Q8_0
                except Exception:
                    pass
            # Prompt-lookup speculative decoding (lossless): proposes
            # tokens by matching n-grams from the prompt and verifies
            # them with the full model. Big win when the output echoes
            # the input (translation, ASR fusion). Output is identical
            # to greedy — only faster.
            if self._prompt_lookup_tokens > 0:
                try:
                    # Lives in the llama_cpp.llama_speculative submodule,
                    # not the top-level package.
                    from llama_cpp.llama_speculative import (
                        LlamaPromptLookupDecoding)
                    common["draft_model"] = LlamaPromptLookupDecoding(
                        num_pred_tokens=self._prompt_lookup_tokens)
                    logger.info(
                        "  prompt-lookup speculative decoding ON "
                        f"(num_pred_tokens={self._prompt_lookup_tokens})")
                except Exception as e:
                    logger.warning("  prompt-lookup unavailable (%s) — "
                                   "running without it", e)
            logger.info(
                "  GGUF load opts: n_ctx=%s flash_attn=%s kv_quant=%s "
                "(type_k set=%s)", common.get("n_ctx"),
                common.get("flash_attn"), self._kv_quant,
                "type_k" in common)
            if self._path:
                self._llm = Llama(model_path=str(self._path), **common)
            elif self._repo and self._file:
                # Prefer the already-cached file via a pure local-cache
                # lookup (no HF API call) so this works fully offline
                # (HF_HUB_OFFLINE) and without a network. Download only if
                # the file is not cached yet.
                from huggingface_hub import hf_hub_download
                try:
                    model_path = hf_hub_download(
                        repo_id=self._repo, filename=self._file,
                        revision=self._revision, local_files_only=True)
                except Exception:
                    # Not cached — one-time download, then local.
                    model_path = hf_hub_download(
                        repo_id=self._repo, filename=self._file,
                        revision=self._revision)
                self._llm = Llama(model_path=model_path, **common)
            else:
                logger.warning(
                    "  GGUF backend: set gguf_path OR "
                    "gguf_repo+gguf_file. Skipping."
                )
                return False
            self._available = True
            # Sanity check: is llama-cpp-python actually GPU-offloaded?
            # If the wheel was built without CUDA, n_gpu_layers is
            # silently ignored → CPU inference → 5–10× slower.
            self._family = self._detect_family()
            self._chat_capable = self._has_chat_template()
            try:
                meta = getattr(self._llm, "metadata", {}) or {}
                arch = meta.get("general.architecture", "?")
            except Exception:
                arch = "?"
            logger.info(
                f"  ✓ GGUF model loaded (llama.cpp, in-process) "
                f"arch={arch} family={self._family} "
                f"chat_template={self._chat_capable} "
                f"n_gpu_layers={self._n_gpu_layers}"
            )
            return True
        except Exception as e:
            logger.warning(f"  Failed to load GGUF model: {e}")
            return False

    def _detect_family(self) -> str:
        """Identify the model family from filename / repo. Determines
        which prompt-formatting hacks (Qwen reasoning) or chat templates
        (Gemma) to use. TigerLLM is Gemma-2-it underneath."""
        name = ((self._file or "") + " " + (self._repo or "") + " "
                + (self._path or "")).lower()
        if "tiger" in name or "gemma" in name:
            return "gemma"
        if "qwen3" in name or "qwen2.5" in name or "qwen2" in name:
            return "qwen"
        if "llama" in name:
            return "llama"
        return "unknown"

    def _has_chat_template(self) -> bool:
        """Whether the GGUF carries a built-in chat template usable
        via create_chat_completion()."""
        try:
            meta = getattr(self._llm, "metadata", {}) or {}
            tpl = meta.get("tokenizer.chat_template")
            return bool(tpl)
        except Exception:
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
        top_p: float = 1.0,
        return_meta: bool = False,
        json_schema: dict | None = None,
        top_k: int | None = None,
        min_p: float | None = None,
        repeat_penalty: float | None = None,
        seed: int | None = None,
    ) -> str | tuple[str, dict]:
        """Generate a completion. With return_meta=True, returns
        (text, meta) where meta = {"finish_reason", "completion_tokens"} —
        finish_reason=="length" means the answer was truncated at the cap
        (the signal that distinguishes a genuine empty from a budget
        overrun). With json_schema, generation is GRAMMAR-CONSTRAINED to
        that JSON Schema (llama.cpp) so the output is always valid JSON of
        the requested shape — used for structured role attribution.
        Default returns just the text for back-compat."""
        def _ret(text: str, fr: str = "", ct: int | None = None):
            return (text, {"finish_reason": fr,
                           "completion_tokens": ct}) if return_meta else text

        if not self._available or self._llm is None:
            return _ret("")
        import re

        # JSON-schema-constrained output (llama.cpp converts the schema to a
        # GBNF grammar). Works through any chat-capable family.
        resp_fmt = ({"type": "json_object", "schema": json_schema}
                    if json_schema else None)

        # Optional sampling knobs (Gemma-team config: top_k=64, top_p=0.95,
        # min_p=0.0, repeat_penalty=1.0, + a fixed seed for reproducibility).
        # Only forwarded when set, so callers that omit them keep llama-cpp's
        # defaults / prior behaviour.
        sampling: dict = {}
        if top_k is not None:
            sampling["top_k"] = top_k
        if min_p is not None:
            sampling["min_p"] = min_p
        if repeat_penalty is not None:
            sampling["repeat_penalty"] = repeat_penalty
        if seed is not None:
            sampling["seed"] = seed

        family = getattr(self, "_family", "unknown")

        # ── Gemma / TigerLLM path ──
        # TigerLLM-9B-it is Gemma-2 instruction-tuned. It MUST be
        # called via its chat template; raw completion + a "Bengali:"
        # / "English:" prompt confuses the instruction-tuned model and
        # it generates to max_tokens of garbage every call (the source
        # of the 10-hour stall). create_chat_completion applies the
        # built-in <start_of_turn>user ... <end_of_turn> template and
        # stops at <end_of_turn>, so output length is bounded by the
        # actual answer.
        if family == "gemma" and getattr(self, "_chat_capable", False):
            try:
                out = self._llm.create_chat_completion(  # type: ignore[union-attr]
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    response_format=resp_fmt,
                    **sampling,
                )
                choice = out["choices"][0]
                msg = choice["message"]
                text = (msg.get("content") or "").strip()
                fr = choice.get("finish_reason") or ""
                ct = (out.get("usage") or {}).get("completion_tokens")
                return _ret(text, fr, ct)
            except Exception as e:
                logger.warning(
                    f"  GGUF chat generation failed (gemma): {e}")
                return _ret("")

        # ── Qwen reasoning-model path (original behaviour) ──
        # Qwen3/3.5 emit <think>...</think> before the answer; disable
        # via seeded empty think block + stop at </output>.
        if family == "qwen":
            try:
                out = self._llm.create_completion(  # type: ignore[union-attr]
                    prompt + "<think>\n\n</think>\n\n",
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    stop=["</output>"],
                )
                text = out["choices"][0]["text"]
                text = re.sub(r"<think>.*?</think>", "", text,
                              flags=re.DOTALL)
                text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL)
                fr = out["choices"][0].get("finish_reason") or ""
                ct = (out.get("usage") or {}).get("completion_tokens")
                return _ret(text.strip(), fr, ct)
            except Exception as e:
                logger.warning(f"  GGUF generation failed (qwen): {e}")
                return _ret("")

        # ── Generic fallback ──
        # If a chat template is present, prefer it (proper stops);
        # otherwise plain completion with newline-based stops.
        try:
            if getattr(self, "_chat_capable", False):
                out = self._llm.create_chat_completion(  # type: ignore[union-attr]
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                )
                choice = out["choices"][0]
                msg = choice["message"]
                fr = choice.get("finish_reason") or ""
                ct = (out.get("usage") or {}).get("completion_tokens")
                return _ret((msg.get("content") or "").strip(), fr, ct)
            out = self._llm.create_completion(  # type: ignore[union-attr]
                prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                stop=["\n\n\n"],
            )
            choice = out["choices"][0]
            fr = choice.get("finish_reason") or ""
            ct = (out.get("usage") or {}).get("completion_tokens")
            return _ret(choice["text"].strip(), fr, ct)
        except Exception as e:
            logger.warning(f"  GGUF generation failed: {e}")
            return _ret("")


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
        "{context}"
        "Translate this Bengali text to English. Translate faithfully — "
        "do not add interpretation, summarize, or omit. PRESERVE the "
        "speaker's hedging, uncertainty and register (e.g. 'maybe', "
        "'God willing', 'I can't say'); do NOT smooth into formal or "
        "essay-like prose. Keep English words, brand names, and "
        "proper nouns as they are.\n"
        "If part of the text is garbled or unintelligible, translate "
        "what is clear and mark the unclear span as [unclear] — do "
        "not guess.\n"
        "OUTPUT ENGLISH ONLY: never leave Bengali script in the "
        "output, and never substitute a Bengali word with its "
        "phonetic Roman spelling instead of translating it. (Proper "
        "nouns / place / person names ARE written in normal English "
        "romanization, e.g. Satkhira, Munshiganj, Amphan.)\n"
        "Render Bengali number words as digits (e.g. আঠারো -> 18, "
        "একশো আশি -> 180, পাঁচশ -> 500, পঞ্চাশ হাজার -> 50000).\n"
        "Output only the English translation, nothing else.\n\n"
        "{dyn}"
        "Bengali: {text}\n"
        "English:"
    )

    def __init__(self, cfg) -> None:
        self._llm = _make_qwen_llm(cfg)
        self._temperature = cfg.temperature
        self._max_tokens = cfg.max_tokens
        ctx_pieces = []
        user_ctx = (getattr(cfg, "context", "") or "").strip()
        if user_ctx:
            ctx_pieces.append(user_ctx)
        # Auto-load the survey wiki (same as QwenCorrector) so the
        # translator also knows the survey domain vocabulary —
        # helps it correctly handle code-switch English in input.
        survey_ctx = _load_survey_wiki_for_prompt()
        if survey_ctx:
            ctx_pieces.append(survey_ctx)
            logger.info(
                "  QwenTranslator: loaded survey wiki vocabulary "
                f"({len(survey_ctx)} chars)")
        full_ctx = "\n\n".join(ctx_pieces)
        self._context = f"Context: {full_ctx}\n\n" if full_ctx else ""
        self._dyn = ""  # per-call rolling/entity context (caller-set)
        self._backend = cfg.backend

    def set_context(self, dyn: str) -> None:
        """Set per-call dynamic context (rolling prior + entity
        memory). '' disables (default = production behaviour)."""
        self._dyn = dyn or ""

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
        progress = create_progress("Translation (LLM)")
        with progress:
            task = progress.add_task(
                "QwenTranslator (per-segment)", total=len(texts))
            for t in texts:
                src = (t or "").strip()
                if not src:
                    out.append("")
                    progress.advance(task)
                    continue
                try:
                    resp = self._llm.generate(
                        self._PROMPT.format(text=src,
                                            context=self._context,
                                            dyn=self._dyn),
                        temperature=self._temperature,
                        max_tokens=self._max_tokens,
                    ).strip()
                except Exception as e:
                    logger.warning(
                        f"  QwenTranslator failed on a segment: {e}")
                    resp = ""
                out.append(resp)
                progress.advance(task)
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
                # If corrector & translator point at the SAME local GGUF,
                # reuse the already-loaded llama.cpp instance instead of
                # loading the 9B weights a second time (which would OOM
                # the GPU and crash the translator stage).
                shared = self._maybe_share_gguf_llm()
                if shared is not None:
                    self._qwen_translator._llm = shared
                    logger.info(
                        "  QwenTranslator: reusing corrector's GGUF "
                        "instance (same model — avoids double VRAM load)"
                    )
                else:
                    self._qwen_translator.load()
        else:
            # Legacy CT2 NLLB + Ollama path
            if self._config.translation.enabled:
                self._translator.load()
            if self._config.refinement.enabled:
                self._llm.load()

    def _maybe_share_gguf_llm(self):
        """If the qwen corrector and qwen translator both use the GGUF
        backend pointing at the SAME local file (path OR repo+file),
        return the corrector's already-loaded inner GGUFProcessor so the
        translator can share it. Otherwise None.

        Loading a 9B GGUF a second time on the same GPU usually OOMs
        (the user hit exactly this: corrector loaded, translator's
        Llama() raised 'Failed to load model from file')."""
        c = self._qwen_corrector
        t = self._qwen_translator
        if c is None or t is None:
            return None
        c_cfg = self._config.qwen_corrector
        t_cfg = self._config.qwen_translator
        if getattr(c_cfg, "backend", "gguf") != "gguf":
            return None
        if getattr(t_cfg, "backend", "gguf") != "gguf":
            return None
        same_path = (
            c_cfg.gguf_path
            and t_cfg.gguf_path
            and c_cfg.gguf_path == t_cfg.gguf_path
        )
        same_repo_file = (
            c_cfg.gguf_repo
            and c_cfg.gguf_file
            and c_cfg.gguf_repo == t_cfg.gguf_repo
            and c_cfg.gguf_file == t_cfg.gguf_file
        )
        if not (same_path or same_repo_file):
            return None
        inner = getattr(c, "_llm", None)
        if inner is None or not getattr(inner, "is_loaded", False):
            return None
        return inner

    def unload(self) -> None:
        """Release all post-processing models."""
        self._translategemma.unload()
        self._translator.unload()
        self._llm.unload()
        if self._qwen_corrector is not None:
            self._qwen_corrector.unload()
        if self._qwen_translator is not None:
            # If translator shares corrector's inner GGUF, don't
            # double-unload (corrector already released it).
            shared = (
                self._qwen_corrector is not None
                and getattr(self._qwen_translator, "_llm", None)
                is getattr(self._qwen_corrector, "_llm", None)
            )
            if not shared:
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
