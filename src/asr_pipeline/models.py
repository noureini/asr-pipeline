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
    """Word-level timing (titu native CTC timestamps).

    `speaker_id` is filled in at alignment time by word-level speaker
    attribution (WhisperX-style): each word is tagged with the
    diarization turn it overlaps, which lets us split a chunk into
    speaker-pure segments instead of guessing one speaker per chunk.
    """

    word: str
    start_s: float
    end_s: float
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    speaker_id: Optional[str] = None


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
    source_file: Optional[str] = None
    absolute_start: Optional[str] = None  # ISO 8601 datetime
    absolute_end: Optional[str] = None  # ISO 8601 datetime


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
    source_file: Optional[str] = None
    absolute_start: Optional[str] = None  # ISO 8601 datetime
    absolute_end: Optional[str] = None  # ISO 8601 datetime
    # Targeted human-QA flag: set when the two witnesses disagreed,
    # a number needs verifying, or the fusion had to mark text unclear.
    needs_review: bool = False
    review_reason: str = ""

    # Fusion audit trail: the model's <think> reasoning for this segment
    # (why it trusted one witness over the other). Empty if the model
    # skipped the block. fusion_fallback=True means the LLM produced no
    # usable Bengali and the FLUENT raw line was substituted (coverage
    # save) — these segments need review and inflate WER-on-output.
    reasoning: str = ""
    fusion_fallback: bool = False
    # Raw witnesses for the snap audit: titu (LITERAL) and Whisper N-best
    # (FLUENT), the inputs the fusion saw. A vocab/proper-noun term that
    # appears in the output but in NEITHER witness is a context-snap
    # (model inserted it from the anchor, not the audio).
    witness_literal: str = ""
    witness_fluent: str = ""
    # GER guard metrics (calibration data): grapheme distance from the fused
    # [bn] to its NEAREST witness, and titu-vs-Whisper agreement. None when
    # the guard is off. Used by scripts/ger_calibrate.py to set snap_tau.
    ger_min_dist: Optional[float] = None
    ger_witness_agree: Optional[float] = None


# =============================================================================
# Batch / interview models
# =============================================================================


class AudioFileInfo(BaseModel):
    """Metadata for a single audio file within an interview."""

    filename: str
    recording_start: str  # ISO 8601 datetime
    duration_s: float = 0.0
    file_path: str = ""


class InterviewMetadata(BaseModel):
    """Metadata for a consolidated interview transcript."""

    interview_key: str
    interview_id: Optional[str] = None  # UUID from filenames
    source_files: list[AudioFileInfo] = Field(default_factory=list)
    total_duration_s: float = 0.0
    recording_start: str = ""  # ISO 8601, earliest file
    recording_end: str = ""  # ISO 8601, latest file end


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
    interview: Optional[InterviewMetadata] = None


class TranscriptResult(BaseModel):
    """The complete pipeline output."""

    metadata: TranscriptMetadata
    segments: list[ProcessedSegment]
    non_speech_segments: list[NonSpeechSegment] = Field(default_factory=list)


# =============================================================================
# Microphone testing models
# =============================================================================


class MicAudioMetrics(BaseModel):
    """Acoustic quality metrics for a single audio file."""

    file_path: str
    mic_name: str
    folder_key: str
    duration_s: float
    snr_db: float  # VAD-based signal-to-noise ratio
    clipped_samples: int  # samples with |amplitude| >= 0.99
    clipping_ratio: float  # clipped / total samples
    plosive_spike_count: int  # detected plosive energy spikes
    spectral_rolloff_hz: float  # freq below which 85% energy lies
    effective_bandwidth_hz: float  # bandwidth of significant energy
    crosstalk_ratio: float  # non-speech energy / speech energy
    peak_amplitude: float
    rms_dbfs: float
    speech_ratio: float  # fraction of audio that is speech


class MicSummary(BaseModel):
    """Aggregated metrics for one microphone across all test files."""

    mic_name: str
    num_files: int
    avg_snr_db: float
    avg_clipping_ratio: float
    total_plosive_spikes: int
    avg_spectral_rolloff_hz: float
    avg_effective_bandwidth_hz: float
    avg_crosstalk_ratio: float
    avg_rms_dbfs: float
    avg_speech_ratio: float
    score: float = 0.0  # weighted composite score
    files: list[MicAudioMetrics] = Field(default_factory=list)


class MicTestReport(BaseModel):
    """Full mic comparison report."""

    test_date: str
    language: str
    mic_summaries: list[MicSummary] = Field(default_factory=list)
    recommendation: str = ""
    transcriptions: dict[str, str] = Field(default_factory=dict)
