"""
Configuration management for the ASR pipeline.

Loads configuration from YAML files with environment variable overrides.
Uses Pydantic for validation and type coercion.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal, Optional, get_args

import yaml
from pydantic import BaseModel, Field

from asr_pipeline.models import LanguageConfig, LanguageTier

# =============================================================================
# Default config path
# =============================================================================

_DEFAULT_CONFIG = Path(__file__).parent / "default.yaml"


# =============================================================================
# Nested configuration models
# =============================================================================

# Canonical set of supported ASR engines. force_engine is validated against
# this at config-LOAD time (via the Literal below), so a YAML typo — e.g.
# "tito" instead of "titu" — fails LOUDLY at startup rather than silently
# mis-routing the survey to the wrong (or no) model. Keep in sync with the
# engine factory (engines/__init__.py create_engine).
ASREngine = Literal["whisper", "titu", "qwen", "omnilingual", "indic_conformer"]
VALID_ASR_ENGINES = frozenset(get_args(ASREngine))


class PipelineConfig(BaseModel):
    """Top-level pipeline settings."""

    device: str = "cuda"
    compute_type: str = "float16"
    batch_size: int = 8
    # Reproducibility: one knob seeds Python/NumPy/torch/CTranslate2 at the
    # start of every run (the sampling stages — CT2 Whisper, Gemma fusion — are
    # separately seeded from FusionConfig.seed; keep them aligned for a fully
    # deterministic deliverable).
    seed: int = 1234
    num_workers: int = 4
    output_dir: str = "./outputs"
    low_vram: bool = False  # Sequential execution: load/unload one model at a time
    # Force the ASR engine for non-high-resource languages, overriding
    # the default tier routing. Validated against ASREngine at load time —
    # an unknown value raises immediately instead of mis-routing silently.
    # null = use default routing (non-high -> omnilingual).
    force_engine: Optional[ASREngine] = None
    # IndicConformer decoder strategy when force_engine="indic_conformer".
    # "rnnt" = more accurate on conjuncts/boundaries; "ctc" = faster.
    indic_conformer_decoder: str = "rnnt"


class LoudnessConfig(BaseModel):
    # OFF by default: EBU R128 loudnorm measures the WHOLE file, so the same
    # audio gets a different gain depending on file length → VAD boundaries
    # drift → the same speech chunks (and transcribes) differently. The
    # standard ASR path (WhisperX/faster-whisper) runs VAD on the resampled
    # audio with no whole-file normalization. Enable only with per-chunk
    # normalization, never whole-file before VAD.
    enabled: bool = False
    target_lufs: float = -23.0


class VADConfig(BaseModel):
    # Model-neutral, recall-leaning VAD (A/B-validated on real 8 kHz telephony,
    # file 15_part1_mid5min): threshold 0.45 + 350 ms min-silence keeps
    # conversational/dialectal utterances WHOLE (the regional Whisper witness
    # lives on continuity) and avoids dropping quiet/breathy speech, while the
    # chunker enforces the per-model length budget. The old 0.50/100 ms
    # over-segmented (124 vs 63 speech regions on the same audio) and shredded
    # dialectal runs. Tuned for neither ASR model specifically — by design.
    enabled: bool = True
    threshold: float = 0.45
    min_speech_duration_ms: int = 250
    min_silence_duration_ms: int = 350


class ChunkingConfig(BaseModel):
    # 15s is the validated stable regime: 30s triggers seq2seq decoder
    # repetition-collapse on hard/dialectal audio (A/B-proven). Keep code
    # default == default.yaml so the pipeline is safe without a YAML override.
    max_duration_s: int = 15
    overlap_s: int = 2


class NoiseReductionConfig(BaseModel):
    # OFF by default: a global noise profile is another whole-file dependency
    # that shifts VAD boundaries, and spectral gating can add artifacts on
    # already-degraded 8 kHz/low-bitrate audio. Standard ASR pipelines don't
    # denoise (the models are trained on noisy speech).
    enabled: bool = False
    method: str = "spectral_gating"
    prop_decrease: float = 0.75


class PreprocessingConfig(BaseModel):
    target_sample_rate: int = 16000
    mono: bool = True
    # High-quality resampling via ffmpeg's soxr (polyphase sinc). Naive
    # resampling of degraded 8 kHz telephony adds artifacts; soxr is
    # artifact-free. NOTE: this does NOT restore the missing 4–8 kHz band
    # (that needs neural bandwidth extension, deliberately out of scope) —
    # it only avoids ADDING distortion when up/downsampling to 16 kHz.
    resampler: str = "soxr"
    loudness_normalization: LoudnessConfig = Field(default_factory=LoudnessConfig)
    vad: VADConfig = Field(default_factory=VADConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    noise_reduction: NoiseReductionConfig = Field(default_factory=NoiseReductionConfig)


class WhisperConfig(BaseModel):
    model_size: str = "large-v3"
    beam_size: int = 5
    best_of: int = 5
    patience: float = 1.0
    condition_on_previous_text: bool = True
    vad_filter: bool = True
    word_timestamps: bool = True
    batch_inference: bool = True
    batch_size: int = 8


class OmnilingualConfig(BaseModel):
    model_card: str = "omniASR_LLM_300M_v2"
    zero_shot_model_card: str = "omniASR_LLM_300M_v2"
    max_audio_length_s: int = 40


class TituConfig(BaseModel):
    """titu CTC grounding-witness selection.

    fastconformer (18,336 hr YouTube, 1024-token tokenizer) is the validated
    default: its standard orthography is COMPLEMENTARY to the dialectal Whisper
    witness in fusion (maximizes ensemble diversity) and is clean for the
    downstream Bengali number-word detection. conformer_large (4.4K hr
    non-telephony, 128-token near-character tokenizer) captures more dialect +
    content (incl. survey-critical flood months) but correlates with Whisper and
    emits phonetic non-standard spelling — kept as a one-line A/B switch.
    Compare with scripts/ab_titu_variants.py. Literal-validated so a typo fails
    at config load.
    """
    model_id: Literal[
        "hishab/titu_stt_bn_fastconformer",
        "hishab/titu_stt_bn_conformer_large",
    ] = "hishab/titu_stt_bn_fastconformer"
    batch_size: int = 8


class QwenConfig(BaseModel):
    """Qwen3-ASR engine — default ASR for non-high-resource languages.

    Bengali is not in Qwen3-ASR's official language list but the model
    produces usable Bengali (validated ~8.5% CER on FLEURS). `language`
    is the optional Qwen language hint ("Bengali", "English", ...);
    None = auto-detect (the validated mode).
    """

    model: str = "Qwen/Qwen3-ASR-1.7B"
    language: Optional[str] = None       # None = auto-detect
    dtype: str = "bfloat16"              # bfloat16 | float16 | float32
    device_map: str = "cuda:0"           # "cpu" for CPU-only boxes
    max_audio_length_s: int = 30


class NemoMsddConfig(BaseModel):
    """NeMo MSDD-specific diarization settings."""

    use_msdd: bool = False  # True = NeuralDiarizer (slower, overlap-aware)
    speaker_embeddings_model: str = "titanet_large"
    clustering_method: str = "agglomerative"
    max_rp_threshold: float = 0.25  # Clustering distance threshold
    sparse_search_volume: int = 30


class DiarizationConfig(BaseModel):
    # Voice diarization (speaker clusters). The Bengali fusion path turns it
    # OFF: the fusion LLM labels ENUMERATOR / RESPONDENT from what is said.
    enabled: bool = True
    backend: Literal["pyannote", "nemo_msdd"] = "pyannote"
    # community-1 (2025) is the actively-maintained pyannote pipeline: exclusive
    # frame-level diarization + stronger on noisy/multi-speaker audio. Ties 3.1
    # on clean 2-speaker clips (A/B 2026-06-25) but wins on harder full files.
    # Both are gated on HF — accept terms + set HF_TOKEN. 3.1 kept as fallback.
    model: str = "pyannote/speaker-diarization-community-1"  # pyannote backend only
    auth_token: Optional[str] = None
    min_speakers: Optional[int] = None
    max_speakers: Optional[int] = None
    min_segment_duration: float = 0.5
    nemo: NemoMsddConfig = Field(default_factory=NemoMsddConfig)


class AlignmentConfig(BaseModel):
    """Wav2vec2 forced phoneme alignment for precise word timestamps."""

    enabled: bool = True


class CorrectionConfig(BaseModel):
    enabled: bool = True
    model: Optional[str] = None  # Ollama model name, e.g. "llama3.1:8b"
    base_url: str = "http://localhost:11434"  # Ollama server URL
    temperature: float = 0.1
    max_tokens: int = 2048
    conservative_for_non_high: bool = True


class TranslationConfig(BaseModel):
    enabled: bool = True
    model_path: Optional[str] = None  # Path to CTranslate2-converted model dir
    tokenizer_name: str = "facebook/nllb-200-distilled-1.3B"  # HF tokenizer ID
    target_language: str = "eng_Latn"
    max_length: int = 512
    beam_size: int = 4


class RefinementConfig(BaseModel):
    enabled: bool = True
    temperature: float = 0.2
    max_tokens: int = 2048


class TranslateGemmaConfig(BaseModel):
    """TranslateGemma translation model config (via HuggingFace Transformers)."""

    model_id: str = "google/translategemma-4b-it"
    batch_size: int = 8  # GPU batch size for HF pipeline
    max_new_tokens: int = 256
    quantize: Optional[Literal["4bit", "8bit"]] = "4bit"  # None = full precision


class QwenLLMConfig(BaseModel):
    """Shared local-LLM backend config for the Qwen post-processing
    stages. Two backends, both fully local/private:

      gguf (default)  — llama.cpp in-process, no daemon, pinned file.
                         Set gguf_path (local file) OR gguf_repo+
                         gguf_file (HF auto-download once).
      ollama          — external `ollama serve` daemon.
    """

    backend: Literal["gguf", "ollama"] = "gguf"
    # ── gguf backend ──
    gguf_path: Optional[str] = None      # explicit local .gguf (preferred)
    gguf_repo: Optional[str] = None      # e.g. "Qwen/Qwen3.5-9B-GGUF"
    gguf_file: Optional[str] = None      # e.g. "qwen3.5-9b-q4_k_m.gguf"
    n_gpu_layers: int = -1               # -1 = all on GPU, 0 = CPU
    n_ctx: int = 4096
    n_threads: int = 8
    # ── ollama backend ──
    ollama_model: str = "qwen3.5:9b"
    ollama_base_url: str = "http://localhost:11434"
    # ── generation ──
    temperature: float = 0.0             # deterministic — no creative drift
    max_tokens: int = 1024
    # Free-text domain/provenance context injected into the corrector
    # & translator prompts (e.g. "Transcript from a climate-disaster
    # survey in coastal Bangladesh (Satkhira); enumerator interviews a
    # respondent."). Disambiguates proper nouns the model already
    # knows (Amphan/Bulbul/Aila) WITHOUT a glossary. Empty = no line.
    context: str = ""


class QwenCorrectorConfig(QwenLLMConfig):
    """Qwen3.5-dense corrector for the SOURCE-language transcript.

    Runs BEFORE translation. Fixes ASR recognition errors and restores
    code-switched English written in Bengali script, WITHOUT inventing
    content or translating. Fully local (no cloud).

    Active by default now (Gemma replaced). Still must be measured on
    real Omnilingual output before production sign-off.
    """

    enabled: bool = True
    # True = keep speaker's colloquial/regional forms (verbatim survey).
    # False = standardize to literary Bengali.
    preserve_register: bool = True


class QwenTranslatorConfig(QwenLLMConfig):
    """Qwen3.5-dense translator (corrected Bengali -> English).

    Fully local (gguf or ollama). Faithful translation only — no
    summarizing, no added content; English/brand/number spans kept.
    Replaces the (broken) TranslateGemma stage.
    """

    enabled: bool = True


class FusionConfig(BaseModel):
    """Two-witness fusion + translation (the validated production post-ASR
    stage). titu (primary engine) ‖ Whisper-regional (second witness, CT2) →
    Gemma-3-12B reconciles into clean Bengali [bn] AND translates to English
    [en] in ONE call. Fully local, fully open, hallucination-resistant."""

    whisper_model: str = (
        "bengaliAI/tugstugi_bengaliai-regional-asr_whisper-medium")
    # Local CT2/faster-whisper conversion of whisper_model. Enables batched
    # N-best ~1.9x faster than HF transformers (A/B-validated 2026-06-25).
    # Missing path / "" → automatic fallback to the HF transformers backend.
    whisper_ct2_path: str = "models/tugstugi_ct2"
    # Chunks per Whisper GPU forward pass (intra-stage parallelism).
    # 8 fits the 8GB card with titu unloaded; per-item OOM fallback.
    whisper_batch_size: int = 8
    # N-best: Whisper returns this many candidate readings per segment
    # (beam search) so Gemma reconciles MULTIPLE hypotheses (SOTA GER) —
    # disambiguates borderline words (ward vs world). 1 = single witness.
    whisper_nbest: int = 3
    gguf_repo: str = "unsloth/gemma-3-12b-it-GGUF"
    gguf_file: str = "gemma-3-12b-it-UD-Q4_K_XL.gguf"
    gguf_n_gpu_layers: int = -1     # -1=all on GPU (fits 12B Q4 on 8GB)
    # Prompt-lookup speculative decoding: only helps when output echoes
    # the input tokens. Bengali→English TRANSLATION output is a different
    # language, so acceptance is ~0 and it adds overhead — OFF by default.
    # (A trained Gemma-3-1B *draft* model would help; prompt-lookup does
    # not.)
    prompt_lookup_tokens: int = 0

    # ── Fusion backend ─────────────────────────────────────────────────
    # "inprocess" = llama-cpp-python, one segment at a time (slow but no
    #   external binary). "server" = llama-server with continuous batching
    #   (~4.5x faster on the N-best workload; needs vendor/llamacpp/).
    # Falls back to in-process if the server binary/start fails.
    fusion_backend: Literal["inprocess", "server"] = "inprocess"
    n_parallel: int = 3              # llama-server slots (8GB fits ~3)
    server_port: int = 8081

    # ── Generation hyperparameters ────────────────────────────────────
    # temp=0 (GREEDY) — chosen on evidence over the Gemma-rec default (1.0).
    # A clean temp-only A/B showed temp=0 vs 1.0 is a WASH on accuracy once
    # repeat_penalty=1.0; then the decisive near-empty test settled it — on a
    # backchannel segment where BOTH witnesses said only "okay okay", temp=1.0
    # confabulated a fabricated survey turn (an invented year, then a Devanagari
    # token) in 2 of 3 runs, while temp=0 stayed grounded every time and gave the
    # lowest flag count. Greedy loses no accuracy and gains confabulation-
    # resistance. repeat_penalty=1.0 is load-bearing: llama-cpp's default 1.1
    # PENALISES legitimate repetition (repeated entities/months in dialectal
    # speech) and dropped সুরমা নদী — keep it 1.0. top_k/top_p/min_p are inert
    # at temp=0 (argmax) but kept for parity; min_p=0.0 overrides llama's 0.05.
    # The deterministic GER + foreign-script guards run AFTER generation as
    # defence-in-depth.
    temperature: float = 0.0        # greedy — evidence-based (NOT Gemma-rec 1.0)
    top_p: float = 0.95
    top_k: int = 64
    min_p: float = 0.0
    repeat_penalty: float = 1.0     # MUST be 1.0 — 1.1 drops repeated entities
    seed: int = 1234
    # n_ctx drives KV-cache size. The fusion prompt is ~554 in + ~384
    # out ≈ 1k tokens, so 8192 wildly over-allocated a 3GB KV cache that
    # overcommitted the 8GB card and forced PCIe paging (~4 tok/s decode,
    # the real cause of the 105-min stage). 2048 fits → ~3x faster.
    n_ctx: int = 2048
    # Flash attention: smaller KV + compute buffers, faster decode.
    flash_attn: bool = True
    # Quantize KV cache to q8_0 (half memory). ON: near-lossless per
    # benchmarks AND it fits 8GB fast, leaving room for the longer
    # N-best prompts. f16 KV pages on 8GB (~3x slower) for only a tiny
    # fidelity gain — accuracy comes from N-best, not KV precision.
    kv_quant: bool = True
    fusion_max_tokens: int = 1024   # bounded <think> + [bn] + [en] + headroom
    # (cap only bites on truncation; the per-run token p50/p90/max is logged
    #  so 1024 can be tuned to the measured distribution, not an assumption)
    # Ablation toggle: True = <think> reasoning arm (default, production);
    # False = matched think-off arm on the SAME audio. The think-off prompt
    # is derived from the same template (only the <think> block differs), so
    # the comparison is clean. Flip this for the second eval run.
    fusion_think: bool = True
    # GER guard: constrain the fused [bn] to the witness pool (titu + Whisper
    # N-best) — closest-to-N-best snap + fabricated-number abstain + agreement
    # flags. Training-free, deterministic, grapheme-level. Toggle off to A/B.
    ger_guard: bool = True
    # Snap is OFF by default: run once in observe-mode to collect the min_dist
    # distribution (written to .reasoning.txt), calibrate ger_snap_tau with
    # scripts/ger_calibrate.py, THEN enable snap. A tau picked by intuition
    # clobbers good reconciliations. (Number-abstain + flags work regardless.)
    ger_snap: bool = False
    ger_snap_tau: float = 0.72      # CALIBRATE from your slice before trusting
    # Roles (ENUMERATOR / RESPONDENT) are labelled per turn inside the
    # fusion call, from what is said (no diarization on this path).
    # ── Prompt templates (operational config, not code) ───────────────
    # Path (relative to repo root or absolute) to the YAML holding
    # `fusion_prompt`. Built-in default is used if the file is missing,
    # so existing configs keep working.
    prompts_file: str = "config/prompts/bn_survey.yaml"


class PostprocessingConfig(BaseModel):
    # "fusion" = the validated two-witness + NMT stage (recommended).
    # "qwen" = legacy single-LLM GGUF correction+translation.
    # TranslateGemma / ct2_nllb kept only as legacy fallbacks.
    translation_backend: Literal[
        "fusion", "qwen", "translategemma", "ct2_nllb"] = "qwen"
    fusion: FusionConfig = Field(default_factory=FusionConfig)
    translategemma: TranslateGemmaConfig = Field(default_factory=TranslateGemmaConfig)
    qwen_translator: QwenTranslatorConfig = Field(default_factory=QwenTranslatorConfig)
    # Source-text corrector (runs before translation).
    # "qwen" = local Qwen3.5 corrector (default). "none" = skip.
    correction_backend: Literal["qwen", "none"] = "qwen"
    qwen_corrector: QwenCorrectorConfig = Field(default_factory=QwenCorrectorConfig)
    # Below configs used only with ct2_nllb backend
    correction: CorrectionConfig = Field(default_factory=CorrectionConfig)
    translation: TranslationConfig = Field(default_factory=TranslationConfig)
    refinement: RefinementConfig = Field(default_factory=RefinementConfig)


class OutputConfig(BaseModel):
    format: str = "txt"
    transcription_style: str = "intelligent_verbatim"
    include_raw_text: bool = True
    include_translation: bool = True
    timestamp_format: str = "HH:MM:SS"


class LoggingConfig(BaseModel):
    level: str = "INFO"
    file: Optional[str] = None
    format: str = "rich"
    show_progress: bool = True


# =============================================================================
# Root configuration
# =============================================================================


class AppConfig(BaseModel):
    """Root configuration container for the entire ASR pipeline."""

    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    languages: dict[str, LanguageConfig] = Field(default_factory=dict)
    whisper: WhisperConfig = Field(default_factory=WhisperConfig)
    omnilingual: OmnilingualConfig = Field(default_factory=OmnilingualConfig)
    qwen: QwenConfig = Field(default_factory=QwenConfig)
    titu: TituConfig = Field(default_factory=TituConfig)
    diarization: DiarizationConfig = Field(default_factory=DiarizationConfig)
    alignment: AlignmentConfig = Field(default_factory=AlignmentConfig)
    postprocessing: PostprocessingConfig = Field(default_factory=PostprocessingConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    def get_language(self, code: str) -> LanguageConfig:
        """Look up a language by its ISO 639-3 code."""
        if code in self.languages:
            return self.languages[code]
        # Default: treat any unknown language as non-high resource
        return LanguageConfig(
            name=f"Unknown ({code})",
            tier=LanguageTier.NON_HIGH,
            bcp47=code,
            script="Latn",
            nllb_code=f"{code}_Latn",
        )

    def engine_for_language(self, code: str) -> str:
        """Return the ASR engine name for a given language code.

        Priority:
          1. High-resource → Whisper
          2. Otherwise → Qwen3-ASR (default; Omnilingual remains
             available via explicit config/override)
        """
        lang = self.get_language(code)
        if lang.tier == LanguageTier.HIGH:
            return "whisper"
        if self.pipeline.force_engine:
            return self.pipeline.force_engine
        # Default for non-high-resource: Omnilingual. Measured ~6.8% CER
        # (num-normalized) on FLEURS Bengali, fully local/private —
        # vs Qwen3-ASR-1.7B's 34% (no Bengali) and Qwen-Flash being
        # cloud-only (unusable for survey PII). Override with
        # pipeline.force_engine for A/B testing.
        return "omnilingual"


# =============================================================================
# Config loading
# =============================================================================


def _parse_languages(raw: dict[str, Any]) -> dict[str, LanguageConfig]:
    """Parse the languages section from raw YAML into LanguageConfig objects."""
    result: dict[str, LanguageConfig] = {}
    for code, lang_data in raw.items():
        result[code] = LanguageConfig(**lang_data)
    return result


def load_config(config_path: Optional[Path] = None) -> AppConfig:
    """
    Load and validate pipeline configuration from a YAML file.

    Falls back to the bundled default.yaml if no path is provided.
    """
    path = config_path or _DEFAULT_CONFIG
    if not path.exists():
        # Return default config if no file found
        return AppConfig()

    with open(path) as f:
        raw = yaml.safe_load(f)

    if raw is None:
        return AppConfig()

    # Parse languages separately because they need special handling.
    # If the user's custom config does not define `languages:`,
    # fall back to the bundled default.yaml's language registry so
    # things like Bengali still resolve correctly.
    languages_raw = raw.pop("languages", None)
    if not languages_raw and path != _DEFAULT_CONFIG \
            and _DEFAULT_CONFIG.exists():
        with open(_DEFAULT_CONFIG) as df:
            default_raw = yaml.safe_load(df) or {}
        languages_raw = default_raw.get("languages", {})
    languages = _parse_languages(languages_raw or {})

    config = AppConfig(**raw, languages=languages)

    # Environment variable override for HuggingFace token
    if config.diarization.auth_token is None:
        config.diarization.auth_token = os.environ.get("HF_TOKEN")

    return config
