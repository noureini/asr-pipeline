"""
Data models for the ASR pipeline.

All intermediate and final data structures are defined here as Pydantic
models to enforce type safety, enable serialization, and provide clear
documentation of the data flowing through each pipeline stage.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


# =============================================================================
# Enumerations
# =============================================================================


class LanguageTier(str, Enum):
    """Two-tier language classification for engine routing."""

    HIGH = "high"
    NON_HIGH = "non_high"


class TranscriptionStyle(str, Enum):
    """Transcription output style."""

    VERBATIM = "verbatim"
    INTELLIGENT_VERBATIM = "intelligent_verbatim"


class OutputFormat(str, Enum):
    """Supported output file formats."""

    TXT = "txt"
    JSON = "json"
    SRT = "srt"
    ALL = "all"


# =============================================================================
# Language models
# =============================================================================


class LanguageConfig(BaseModel):
    """Configuration for a single language in the registry."""

    name: str
    tier: LanguageTier
    bcp47: str
    script: str
    nllb_code: str
    fine_tuned_checkpoint: Optional[str] = None  # Path to fine-tuned model


# =============================================================================
# Audio & preprocessing models
# =============================================================================


class AudioMetadata(BaseModel):
    """Metadata extracted from an audio file during preprocessing."""

    file_path: Path
    duration_s: float
    sample_rate: int
    channels: int
    format: str
    file_size_bytes: int


class AudioChunk(BaseModel):
    """A preprocessed audio segment ready for ASR inference."""

    chunk_id: int
    start_s: float
    end_s: float
    duration_s: float
    waveform_path: Optional[Path] = None  # Path to temp WAV chunk file

    class Config:
        arbitrary_types_allowed = True


# =============================================================================
# ASR output models
# =============================================================================


class ASRSegment(BaseModel):
    """A single transcription segment from the ASR engine."""

    segment_id: int
    start_s: float
    end_s: float
    text: str
    language: str  # ISO 639-3 code
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    words: list[WordSegment] = Field(default_factory=list)


class WordSegment(BaseModel):
    """Word-level timing from WhisperX alignment (high-resource only)."""

    word: str
    start_s: float
    end_s: float
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


# Fix forward references
ASRSegment.model_rebuild()


# =============================================================================
# Diarization models
# =============================================================================


class SpeakerSegment(BaseModel):
    """A speaker turn detected by the diarization engine."""

    speaker_id: str  # e.g., "SPEAKER_00"
    start_s: float
    end_s: float


class DiarizationResult(BaseModel):
    """Full diarization output for an audio file."""

    num_speakers: int
    segments: list[SpeakerSegment]


# =============================================================================
# Non-speech & audio quality models
# =============================================================================


class NonSpeechSegment(BaseModel):
    """A non-speech region detected by VAD (silence, noise, etc.)."""

    start_s: float
    end_s: float
    region_type: str = "non_speech"  # "silence", "non_speech", "inaudible"
    duration_s: float = 0.0


class AudioQualityMetrics(BaseModel):
    """Audio quality statistics derived from VAD analysis."""

    total_duration_s: float
    speech_duration_s: float
    non_speech_duration_s: float
    speech_ratio: float  # speech_duration / total_duration (0.0-1.0)
    num_speech_segments: int
    num_non_speech_segments: int
    avg_speech_segment_s: float
    longest_silence_s: float


# =============================================================================
# Aligned & post-processed models
# =============================================================================


class AlignedSegment(BaseModel):
    """An ASR segment merged with speaker identity."""

    segment_id: int
    start_s: float
    end_s: float
    speaker_id: str
    language: str
    raw_text: str
    confidence: float = 0.0
    words: list[WordSegment] = Field(default_factory=list)


class ProcessedSegment(BaseModel):
    """Fully processed segment after LLM correction and translation."""

    segment_id: int
    start_s: float
    end_s: float
    speaker_id: str
    language: str
    raw_text: str
    corrected_text: str
    english_translation: str
    refined_translation: str
    confidence: float = 0.0


# =============================================================================
# Pipeline result
# =============================================================================


class TranscriptMetadata(BaseModel):
    """Metadata block for the final transcript."""

    project_name: str = ""
    audio_file: str
    duration_s: float
    recording_date: str = ""
    languages_detected: list[str]
    num_speakers: int
    transcription_style: TranscriptionStyle
    asr_engines_used: list[str]
    postprocessing_stages: list[str]
    audio_quality: Optional[AudioQualityMetrics] = None


class TranscriptResult(BaseModel):
    """The complete pipeline output."""

    metadata: TranscriptMetadata
    segments: list[ProcessedSegment]
    non_speech_segments: list[NonSpeechSegment] = Field(default_factory=list)
