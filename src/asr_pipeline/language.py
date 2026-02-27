"""
Language registry and detection.

Manages the two-tier language routing system:
  - HIGH tier → Whisper Large-v3
  - NON_HIGH tier → Omnilingual ASR

Also handles language detection for audio chunks and code-switching
detection.
"""

from __future__ import annotations

import logging
from typing import Optional

from asr_pipeline.config import AppConfig
from asr_pipeline.models import LanguageConfig, LanguageTier

logger = logging.getLogger("asr_pipeline")


class LanguageRegistry:
    """
    Central registry for language configuration and engine routing.

    Provides lookup, detection routing, and code-switching awareness.
    New languages default to non-high tier (Omnilingual).
    """

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._languages = config.languages
        self._high_codes: set[str] = {
            code for code, lang in self._languages.items() if lang.tier == LanguageTier.HIGH
        }
        self._non_high_codes: set[str] = {
            code for code, lang in self._languages.items() if lang.tier == LanguageTier.NON_HIGH
        }
        logger.debug(
            f"Language registry initialized: "
            f"{len(self._high_codes)} high-resource, "
            f"{len(self._non_high_codes)} non-high-resource languages"
        )

    def get(self, code: str) -> LanguageConfig:
        """Get language config by ISO 639-3 code. Unknown → non_high."""
        return self._config.get_language(code)

    def get_engine(self, code: str) -> str:
        """Return the ASR engine name for a language code."""
        return self._config.engine_for_language(code)

    def is_high_resource(self, code: str) -> bool:
        """Check if a language is in the high-resource tier."""
        return code in self._high_codes

    def get_nllb_code(self, code: str) -> str:
        """Get the NLLB-200 language code for translation."""
        lang = self.get(code)
        return lang.nllb_code

    @property
    def all_codes(self) -> set[str]:
        """All registered language codes."""
        return set(self._languages.keys())

    @property
    def high_resource_codes(self) -> set[str]:
        """Language codes in the high-resource tier."""
        return self._high_codes.copy()

    @property
    def non_high_resource_codes(self) -> set[str]:
        """Language codes in the non-high-resource tier."""
        return self._non_high_codes.copy()


def detect_language_from_whisper(
    model: object,
    audio_segment: object,
) -> tuple[str, float, bool]:
    """
    Use Whisper's built-in language detection on an audio segment.

    Returns:
        Tuple of (language_code, confidence, is_mixed).
        is_mixed is True if the top-2 language probabilities are close,
        indicating potential code-switching.
    """
    # This is a placeholder — actual implementation depends on
    # faster-whisper's detect_language API
    try:
        from faster_whisper import WhisperModel

        if not isinstance(model, WhisperModel):
            return ("eng", 0.0, False)

        # faster-whisper returns list of (language, probability)
        language_probs = model.detect_language(audio_segment)  # type: ignore[attr-defined]

        if not language_probs:
            return ("eng", 0.0, False)

        # Sort by probability descending
        sorted_probs = sorted(language_probs, key=lambda x: x[1], reverse=True)
        top_lang, top_prob = sorted_probs[0]

        # Detect code-switching: if top-2 are close in probability
        is_mixed = False
        if len(sorted_probs) > 1:
            second_prob = sorted_probs[1][1]
            if top_prob - second_prob < 0.3:
                is_mixed = True
                logger.debug(
                    f"Possible code-switching detected: "
                    f"{sorted_probs[0]} vs {sorted_probs[1]}"
                )

        return (top_lang, top_prob, is_mixed)

    except Exception as e:
        logger.warning(f"Language detection failed: {e}")
        return ("eng", 0.0, False)


def map_whisper_lang_to_iso639_3(whisper_code: str) -> str:
    """
    Map Whisper's BCP-47 / ISO 639-1 codes to ISO 639-3.

    Whisper uses short codes like 'en', 'es', 'hi'. We need ISO 639-3
    codes like 'eng', 'spa', 'hin' for the language registry.
    """
    _MAP: dict[str, str] = {
        "en": "eng", "es": "spa", "fr": "fra", "de": "deu",
        "pt": "por", "ru": "rus", "zh": "zho", "ja": "jpn",
        "ko": "kor", "it": "ita", "nl": "nld", "pl": "pol",
        "tr": "tur", "cs": "ces", "sv": "swe", "uk": "ukr",
        "ro": "ron", "ar": "ara", "hi": "hin", "bn": "ben",
        "ne": "nep", "sw": "swa", "am": "amh", "ha": "hau",
        "yo": "yor", "ig": "ibo", "tl": "tgl", "my": "mya",
        "km": "khm", "rw": "kin", "so": "som", "ti": "tir",
        "om": "orm",
    }
    return _MAP.get(whisper_code, whisper_code)


def map_iso639_3_to_omnilingual(
    code: str,
    registry: LanguageRegistry,
) -> str:
    """
    Map ISO 639-3 code to Omnilingual ASR's {code}_{script} format.

    Example: "hin" → "hin_Deva", "amh" → "amh_Ethi"
    """
    lang = registry.get(code)
    return f"{code}_{lang.script}"
