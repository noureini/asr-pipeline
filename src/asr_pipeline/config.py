"""
Configuration management for the ASR pipeline.

Loads configuration from YAML files with environment variable overrides.
Uses Pydantic for validation and type coercion.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal, Optional

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


class PipelineConfig(BaseModel):
    """Top-level pipeline settings."""

    device: str = "cuda"
    compute_type: str = "float16"
    batch_size: int = 8
    num_workers: int = 4
    output_dir: str = "./outputs"
    low_vram: bool = False  # Sequential execution: load/unload one model at a time
    # Force the ASR engine for non-high-resource languages, overriding
    # the default tier routing. One of: "qwen", "omnilingual", "whisper".
    # null = use default routing (non-high -> qwen).
    force_engine: Optional[str] = None


class LoudnessConfig(BaseModel):
    enabled: bool = True
    target_lufs: float = -23.0


class VADConfig(BaseModel):
    enabled: bool = True
    threshold: float = 0.5
    min_speech_duration_ms: int = 250
    min_silence_duration_ms: int = 100


class ChunkingConfig(BaseModel):
    max_duration_s: int = 30
    overlap_s: int = 2


class NoiseReductionConfig(BaseModel):
    enabled: bool = True
    method: str = "spectral_gating"
    prop_decrease: float = 0.75


class PreprocessingConfig(BaseModel):
    target_sample_rate: int = 16000
    mono: bool = True
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
    backend: Literal["pyannote", "nemo_msdd"] = "pyannote"
    model: str = "pyannote/speaker-diarization-3.1"  # Only for pyannote backend
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


class PostprocessingConfig(BaseModel):
    translation_backend: Literal["translategemma", "ct2_nllb"] = "translategemma"
    translategemma: TranslateGemmaConfig = Field(default_factory=TranslateGemmaConfig)
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

    # Parse languages separately because they need special handling
    languages_raw = raw.pop("languages", {})
    languages = _parse_languages(languages_raw)

    config = AppConfig(**raw, languages=languages)

    # Environment variable override for HuggingFace token
    if config.diarization.auth_token is None:
        config.diarization.auth_token = os.environ.get("HF_TOKEN")

    return config
