"""
Tests for configuration loading and data models.
"""

from pathlib import Path

import pytest

from asr_pipeline.config import AppConfig, load_config
from asr_pipeline.models import (
    AlignedSegment,
    ASRSegment,
    AudioChunk,
    AudioMetadata,
    AudioQualityMetrics,
    DiarizationResult,
    LanguageConfig,
    LanguageTier,
    NonSpeechSegment,
    ProcessedSegment,
    SpeakerSegment,
    TranscriptMetadata,
    TranscriptResult,
    TranscriptionStyle,
    WordSegment,
)


# =============================================================================
# Configuration tests
# =============================================================================


class TestConfig:
    """Test configuration loading and validation."""

    def test_default_config_loads(self) -> None:
        """Default config should load without errors."""
        config = load_config()
        assert isinstance(config, AppConfig)

    def test_config_from_yaml(self, tmp_path: Path) -> None:
        """Config should load from a YAML file."""
        yaml_content = """
pipeline:
  device: cpu
  compute_type: float32
languages:
  eng:
    name: English
    tier: high
    bcp47: en
    script: Latn
    nllb_code: eng_Latn
  hin:
    name: Hindi
    tier: non_high
    bcp47: hi
    script: Deva
    nllb_code: hin_Deva
"""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(yaml_content)

        config = load_config(config_file)
        assert config.pipeline.device == "cpu"
        assert "eng" in config.languages
        assert "hin" in config.languages
        assert config.languages["eng"].tier == LanguageTier.HIGH
        assert config.languages["hin"].tier == LanguageTier.NON_HIGH

    def test_engine_routing(self) -> None:
        """High-resource languages should route to Whisper."""
        config = load_config()
        # If the default config has these languages
        if "eng" in config.languages:
            assert config.engine_for_language("eng") == "whisper"
        if "hin" in config.languages:
            assert config.engine_for_language("hin") == "omnilingual"

    def test_unknown_language_defaults_non_high(self) -> None:
        """Unknown languages should default to non-high tier."""
        config = AppConfig()
        lang = config.get_language("xyz")
        assert lang.tier == LanguageTier.NON_HIGH
        assert config.engine_for_language("xyz") == "omnilingual"


# =============================================================================
# Data model tests
# =============================================================================


class TestDataModels:
    """Test Pydantic data model creation and validation."""

    def test_language_config(self) -> None:
        lang = LanguageConfig(
            name="Hindi",
            tier=LanguageTier.NON_HIGH,
            bcp47="hi",
            script="Deva",
            nllb_code="hin_Deva",
        )
        assert lang.name == "Hindi"
        assert lang.tier == LanguageTier.NON_HIGH

    def test_audio_chunk(self) -> None:
        chunk = AudioChunk(
            chunk_id=0,
            start_s=0.0,
            end_s=30.0,
            duration_s=30.0,
        )
        assert chunk.duration_s == 30.0

    def test_asr_segment(self) -> None:
        seg = ASRSegment(
            segment_id=0,
            start_s=1.2,
            end_s=4.5,
            text="Hello world",
            language="eng",
            confidence=0.95,
        )
        assert seg.text == "Hello world"
        assert seg.confidence == 0.95

    def test_speaker_segment(self) -> None:
        seg = SpeakerSegment(
            speaker_id="SPEAKER_00",
            start_s=0.0,
            end_s=5.0,
        )
        assert seg.speaker_id == "SPEAKER_00"

    def test_aligned_segment(self) -> None:
        seg = AlignedSegment(
            segment_id=0,
            start_s=1.2,
            end_s=4.5,
            speaker_id="SPEAKER_01",
            language="hin",
            raw_text="नमस्ते",
        )
        assert seg.speaker_id == "SPEAKER_01"
        assert seg.language == "hin"

    def test_processed_segment(self) -> None:
        seg = ProcessedSegment(
            segment_id=0,
            start_s=1.2,
            end_s=4.5,
            speaker_id="SPEAKER_00",
            language="swa",
            raw_text="habari",
            corrected_text="Habari.",
            english_translation="Hello.",
            refined_translation="Hello.",
        )
        assert seg.corrected_text == "Habari."
        assert seg.english_translation == "Hello."

    def test_transcript_result(self) -> None:
        result = TranscriptResult(
            metadata=TranscriptMetadata(
                audio_file="test.wav",
                duration_s=120.0,
                languages_detected=["Hindi"],
                num_speakers=2,
                transcription_style=TranscriptionStyle.INTELLIGENT_VERBATIM,
                asr_engines_used=["Omnilingual CTC 300M"],
                postprocessing_stages=["NLLB-200"],
            ),
            segments=[],
        )
        assert result.metadata.num_speakers == 2
        assert result.metadata.duration_s == 120.0


# =============================================================================
# Alignment tests
# =============================================================================


class TestAlignment:
    """Test segment alignment logic."""

    def test_compute_overlap(self) -> None:
        from asr_pipeline.alignment import compute_overlap

        # Full overlap
        assert compute_overlap(0, 10, 0, 10) == 10.0
        # Partial overlap
        assert compute_overlap(0, 10, 5, 15) == 5.0
        # No overlap
        assert compute_overlap(0, 5, 10, 15) == 0.0

    def test_align_segments_assigns_speakers(self) -> None:
        from asr_pipeline.alignment import align_segments

        asr_segs = [
            ASRSegment(
                segment_id=0, start_s=0.0, end_s=5.0,
                text="Hello", language="eng",
            ),
            ASRSegment(
                segment_id=1, start_s=5.5, end_s=10.0,
                text="World", language="eng",
            ),
        ]
        diar = DiarizationResult(
            num_speakers=2,
            segments=[
                SpeakerSegment(speaker_id="SPEAKER_00", start_s=0.0, end_s=5.0),
                SpeakerSegment(speaker_id="SPEAKER_01", start_s=5.0, end_s=10.0),
            ],
        )

        aligned = align_segments(asr_segs, diar)
        assert len(aligned) == 2
        assert aligned[0].speaker_id == "SPEAKER_00"
        assert aligned[1].speaker_id == "SPEAKER_01"

    def test_merge_consecutive_segments(self) -> None:
        from asr_pipeline.alignment import merge_consecutive_segments

        segments = [
            AlignedSegment(
                segment_id=0, start_s=0.0, end_s=2.0,
                speaker_id="SPEAKER_00", language="eng",
                raw_text="Hello",
            ),
            AlignedSegment(
                segment_id=1, start_s=2.1, end_s=4.0,
                speaker_id="SPEAKER_00", language="eng",
                raw_text="world",
            ),
        ]

        merged = merge_consecutive_segments(segments, max_gap_s=0.5)
        assert len(merged) == 1
        assert merged[0].raw_text == "Hello world"


# =============================================================================
# Formatter tests
# =============================================================================


class TestFormatter:
    """Test output formatting."""

    def test_format_timestamp(self) -> None:
        from asr_pipeline.formatter import format_timestamp

        assert format_timestamp(0.0) == "00:00:00"
        assert format_timestamp(3661.5) == "01:01:01"
        assert format_timestamp(3661.5, "HH:MM:SS.mmm") == "01:01:01.500"

    def test_format_txt(self) -> None:
        from asr_pipeline.formatter import format_txt

        result = TranscriptResult(
            metadata=TranscriptMetadata(
                audio_file="test.wav",
                duration_s=60.0,
                languages_detected=["Hindi"],
                num_speakers=1,
                transcription_style=TranscriptionStyle.INTELLIGENT_VERBATIM,
                asr_engines_used=["Omnilingual"],
                postprocessing_stages=["NLLB-200"],
            ),
            segments=[
                ProcessedSegment(
                    segment_id=0,
                    start_s=0.0,
                    end_s=5.0,
                    speaker_id="SPEAKER_00",
                    language="hin",
                    raw_text="नमस्ते",
                    corrected_text="नमस्ते।",
                    english_translation="Hello.",
                    refined_translation="Hello.",
                ),
            ],
        )

        txt = format_txt(result)
        assert "TRANSCRIPT" in txt
        assert "SPEAKER_00" in txt
        assert "नमस्ते" in txt
        assert "Hello." in txt
        assert "END OF TRANSCRIPT" in txt

    def test_format_srt(self) -> None:
        from asr_pipeline.formatter import format_srt

        result = TranscriptResult(
            metadata=TranscriptMetadata(
                audio_file="test.wav",
                duration_s=60.0,
                languages_detected=["English"],
                num_speakers=1,
                transcription_style=TranscriptionStyle.INTELLIGENT_VERBATIM,
                asr_engines_used=["Whisper"],
                postprocessing_stages=[],
            ),
            segments=[
                ProcessedSegment(
                    segment_id=0,
                    start_s=0.0,
                    end_s=5.0,
                    speaker_id="SPEAKER_00",
                    language="eng",
                    raw_text="Hello world",
                    corrected_text="Hello world.",
                    english_translation="Hello world.",
                    refined_translation="Hello world.",
                ),
            ],
        )

        srt = format_srt(result)
        assert "1\n" in srt
        assert "-->" in srt
        assert "SPEAKER_00" in srt


# =============================================================================
# Language registry tests
# =============================================================================


class TestLanguageRegistry:
    """Test language registry and routing."""

    def test_language_lookup(self) -> None:
        from asr_pipeline.language import LanguageRegistry

        config = load_config()
        registry = LanguageRegistry(config)

        if "eng" in config.languages:
            assert registry.is_high_resource("eng")
        if "hin" in config.languages:
            assert not registry.is_high_resource("hin")

    def test_unknown_language_is_non_high(self) -> None:
        from asr_pipeline.language import LanguageRegistry

        config = AppConfig()
        registry = LanguageRegistry(config)

        assert not registry.is_high_resource("xyz")
        assert registry.get_engine("xyz") == "omnilingual"

    def test_whisper_code_mapping(self) -> None:
        from asr_pipeline.language import map_whisper_lang_to_iso639_3

        assert map_whisper_lang_to_iso639_3("en") == "eng"
        assert map_whisper_lang_to_iso639_3("es") == "spa"
        assert map_whisper_lang_to_iso639_3("hi") == "hin"
        assert map_whisper_lang_to_iso639_3("sw") == "swa"


# =============================================================================
# VAD chunking tests
# =============================================================================


class TestVADChunking:
    """Test VAD-guided chunking logic."""

    def test_non_speech_extraction(self) -> None:
        """Non-speech regions should be the inverse of speech timestamps."""
        from asr_pipeline.preprocessor import AudioPreprocessor
        from asr_pipeline.config import PreprocessingConfig

        config = PreprocessingConfig()
        preprocessor = AudioPreprocessor(config, Path("/tmp/test_work"))

        speech_timestamps = [(1.0, 3.0), (5.0, 8.0)]
        total_duration = 10.0

        regions = preprocessor.extract_non_speech_regions(
            speech_timestamps, total_duration
        )

        # Expected: [(0, 1), (3, 5), (8, 10)]
        assert len(regions) == 3
        assert regions[0] == (0.0, 1.0, "non_speech")
        assert regions[1] == (3.0, 5.0, "non_speech")
        assert regions[2] == (8.0, 10.0, "non_speech")

    def test_non_speech_ignores_tiny_gaps(self) -> None:
        """Gaps smaller than min_gap_s should not be reported."""
        from asr_pipeline.preprocessor import AudioPreprocessor
        from asr_pipeline.config import PreprocessingConfig

        config = PreprocessingConfig()
        preprocessor = AudioPreprocessor(config, Path("/tmp/test_work"))

        # Gap of 0.1s between segments — should be ignored
        speech_timestamps = [(0.0, 3.0), (3.1, 6.0)]
        total_duration = 6.0

        regions = preprocessor.extract_non_speech_regions(
            speech_timestamps, total_duration, min_gap_s=0.3
        )

        # 0.1s gap is < 0.3 threshold, so no regions
        assert len(regions) == 0

    def test_non_speech_no_speech(self) -> None:
        """When no speech timestamps, empty list returned."""
        from asr_pipeline.preprocessor import AudioPreprocessor
        from asr_pipeline.config import PreprocessingConfig

        config = PreprocessingConfig()
        preprocessor = AudioPreprocessor(config, Path("/tmp/test_work"))

        regions = preprocessor.extract_non_speech_regions([], 10.0)
        # Entire file is non-speech
        assert len(regions) == 1
        assert regions[0] == (0.0, 10.0, "non_speech")


# =============================================================================
# Alignment with words tests
# =============================================================================


class TestAlignmentWords:
    """Test that word-level timestamps pass through alignment."""

    def test_aligned_segment_has_words(self) -> None:
        """AlignedSegment should accept and store words."""
        words = [
            WordSegment(word="hello", start_s=0.0, end_s=0.5, confidence=0.9),
            WordSegment(word="world", start_s=0.5, end_s=1.0, confidence=0.8),
        ]
        seg = AlignedSegment(
            segment_id=0,
            start_s=0.0,
            end_s=1.0,
            speaker_id="SPEAKER_00",
            language="eng",
            raw_text="hello world",
            words=words,
        )
        assert len(seg.words) == 2
        assert seg.words[0].word == "hello"
        assert seg.words[1].word == "world"

    def test_alignment_preserves_words(self) -> None:
        """align_segments should pass words from ASR to Aligned segments."""
        from asr_pipeline.alignment import align_segments

        words = [
            WordSegment(word="test", start_s=0.0, end_s=0.5, confidence=0.9),
        ]
        asr_segs = [
            ASRSegment(
                segment_id=0, start_s=0.0, end_s=5.0,
                text="test", language="eng", words=words,
            ),
        ]
        diar = DiarizationResult(
            num_speakers=1,
            segments=[
                SpeakerSegment(speaker_id="SPEAKER_00", start_s=0.0, end_s=5.0),
            ],
        )

        aligned = align_segments(asr_segs, diar)
        assert len(aligned) == 1
        assert len(aligned[0].words) == 1
        assert aligned[0].words[0].word == "test"

    def test_merge_consecutive_concatenates_words(self) -> None:
        """merge_consecutive_segments should combine word lists."""
        from asr_pipeline.alignment import merge_consecutive_segments

        segments = [
            AlignedSegment(
                segment_id=0, start_s=0.0, end_s=2.0,
                speaker_id="SPEAKER_00", language="eng",
                raw_text="Hello",
                words=[WordSegment(word="Hello", start_s=0.0, end_s=0.5, confidence=0.9)],
            ),
            AlignedSegment(
                segment_id=1, start_s=2.1, end_s=4.0,
                speaker_id="SPEAKER_00", language="eng",
                raw_text="world",
                words=[WordSegment(word="world", start_s=2.1, end_s=2.5, confidence=0.8)],
            ),
        ]

        merged = merge_consecutive_segments(segments, max_gap_s=0.5)
        assert len(merged) == 1
        assert len(merged[0].words) == 2
        assert merged[0].words[0].word == "Hello"
        assert merged[0].words[1].word == "world"


# =============================================================================
# Non-speech model tests
# =============================================================================


class TestNonSpeechModels:
    """Test non-speech data models."""

    def test_non_speech_segment(self) -> None:
        ns = NonSpeechSegment(
            start_s=3.0, end_s=5.0,
            region_type="non_speech", duration_s=2.0,
        )
        assert ns.duration_s == 2.0
        assert ns.region_type == "non_speech"

    def test_audio_quality_metrics(self) -> None:
        aq = AudioQualityMetrics(
            total_duration_s=100.0,
            speech_duration_s=85.0,
            non_speech_duration_s=15.0,
            speech_ratio=0.85,
            num_speech_segments=10,
            num_non_speech_segments=9,
            avg_speech_segment_s=8.5,
            longest_silence_s=4.2,
        )
        assert aq.speech_ratio == 0.85
        assert aq.longest_silence_s == 4.2

    def test_transcript_result_with_non_speech(self) -> None:
        """TranscriptResult should include non_speech_segments."""
        result = TranscriptResult(
            metadata=TranscriptMetadata(
                audio_file="test.wav",
                duration_s=60.0,
                languages_detected=["English"],
                num_speakers=1,
                transcription_style=TranscriptionStyle.INTELLIGENT_VERBATIM,
                asr_engines_used=["Whisper"],
                postprocessing_stages=[],
            ),
            segments=[],
            non_speech_segments=[
                NonSpeechSegment(
                    start_s=5.0, end_s=8.0,
                    region_type="non_speech", duration_s=3.0,
                ),
            ],
        )
        assert len(result.non_speech_segments) == 1
        assert result.non_speech_segments[0].duration_s == 3.0

    def test_alignment_config_defaults(self) -> None:
        """AlignmentConfig should default to enabled."""
        config = load_config()
        assert config.alignment.enabled is True


# =============================================================================
# Formatter with non-speech tests
# =============================================================================


class TestFormatterNonSpeech:
    """Test TXT formatter with non-speech placeholders."""

    def test_format_txt_with_non_speech(self) -> None:
        """TXT output should contain non-speech placeholders."""
        from asr_pipeline.formatter import format_txt

        result = TranscriptResult(
            metadata=TranscriptMetadata(
                audio_file="test.wav",
                duration_s=20.0,
                languages_detected=["English"],
                num_speakers=1,
                transcription_style=TranscriptionStyle.INTELLIGENT_VERBATIM,
                asr_engines_used=["Whisper"],
                postprocessing_stages=[],
                audio_quality=AudioQualityMetrics(
                    total_duration_s=20.0,
                    speech_duration_s=15.0,
                    non_speech_duration_s=5.0,
                    speech_ratio=0.75,
                    num_speech_segments=2,
                    num_non_speech_segments=1,
                    avg_speech_segment_s=7.5,
                    longest_silence_s=5.0,
                ),
            ),
            segments=[
                ProcessedSegment(
                    segment_id=0, start_s=0.0, end_s=7.0,
                    speaker_id="SPEAKER_00", language="eng",
                    raw_text="Hello world",
                    corrected_text="Hello world.",
                    english_translation="Hello world.",
                    refined_translation="Hello world.",
                ),
                ProcessedSegment(
                    segment_id=1, start_s=12.0, end_s=20.0,
                    speaker_id="SPEAKER_00", language="eng",
                    raw_text="Goodbye",
                    corrected_text="Goodbye.",
                    english_translation="Goodbye.",
                    refined_translation="Goodbye.",
                ),
            ],
            non_speech_segments=[
                NonSpeechSegment(
                    start_s=7.0, end_s=12.0,
                    region_type="non_speech", duration_s=5.0,
                ),
            ],
        )

        txt = format_txt(result)
        assert "NON-SPEECH" in txt
        assert "5.0s" in txt
        assert "Audio Quality:" in txt
        assert "75% speech" in txt
        assert "Hello world." in txt
        assert "Goodbye." in txt
