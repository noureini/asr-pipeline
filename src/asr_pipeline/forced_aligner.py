"""
Forced phoneme alignment using torchaudio MMS_FA.

Refines word-level timestamps from Whisper's cross-attention heuristics
(±200ms–1s drift) to precise phoneme-aligned timestamps (~20ms accuracy)
using Meta's Massively Multilingual Speech forced alignment model.

Supports 1,130 languages via ISO 639-3 codes. No new dependencies needed —
torchaudio is already part of the pipeline.

Inspired by WhisperX's alignment stage, but uses torchaudio's built-in
MMS_FA bundle instead of adding whisperx as a dependency.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

import torch
import torchaudio

from asr_pipeline.config import AlignmentConfig
from asr_pipeline.logging_config import create_progress
from asr_pipeline.models import ASRSegment, WordSegment

logger = logging.getLogger("asr_pipeline")

# Languages where forced alignment doesn't work well:
# CJK scripts have no space-delimited words, thousands of characters,
# and ambiguous word boundaries. Whisper's native timestamps are adequate.
_SKIP_LANGUAGES = {"zho", "jpn", "kor"}


class ForcedAligner:
    """
    Wav2vec2 forced alignment for precise word-level timestamps.

    Uses torchaudio.pipelines.MMS_FA (Meta's Massively Multilingual Speech)
    to align transcribed text to the audio waveform at the phoneme level,
    producing word timestamps accurate to ~20ms.

    Falls back gracefully: if alignment fails for any segment, the original
    Whisper timestamps are kept.
    """

    def __init__(
        self,
        config: AlignmentConfig,
        device: str = "cuda",
    ) -> None:
        self._config = config
        self._device = device
        self._model: Optional[torch.nn.Module] = None
        self._dictionary: Optional[dict[str, int]] = None
        self._sample_rate: int = 16000
        self._available: bool = False

    # ─────────────────────────────────────────────────────────────────
    # Lifecycle
    # ─────────────────────────────────────────────────────────────────

    def load(self) -> bool:
        """
        Load the MMS forced alignment model.

        Returns:
            True if model loaded successfully, False otherwise.
        """
        if self._model is not None:
            return self._available

        try:
            bundle = torchaudio.pipelines.MMS_FA
            self._sample_rate = bundle.sample_rate

            logger.info("  Loading MMS forced alignment model")
            self._model = bundle.get_model().to(self._device)
            self._model.eval()

            self._dictionary = bundle.get_dict()
            self._available = True
            logger.info(
                f"  ✓ MMS_FA loaded ({len(self._dictionary)} tokens, "
                f"sample_rate={self._sample_rate})"
            )
            return True

        except Exception as e:
            logger.warning(
                f"  ✗ Failed to load MMS_FA model: {e}. "
                f"Forced alignment will be skipped."
            )
            self._available = False
            return False

    def unload(self) -> None:
        """Release the alignment model from memory."""
        if self._model is not None:
            del self._model
            self._model = None
            self._available = False
            logger.debug("MMS_FA model unloaded")

    # ─────────────────────────────────────────────────────────────────
    # Main alignment entry point
    # ─────────────────────────────────────────────────────────────────

    def align_segments(
        self,
        segments: list[ASRSegment],
        wav_path: Path,
        language: str,
    ) -> list[ASRSegment]:
        """
        Refine word-level timestamps for all ASR segments.

        For each segment, re-aligns the transcribed text to the audio
        waveform using forced phoneme alignment. Updates word timestamps
        in-place on each segment.

        Args:
            segments: ASR segments with text and approximate timestamps.
            wav_path: Path to the full 16kHz mono WAV file.
            language: ISO 639-3 language code.

        Returns:
            The same segments with refined word timestamps.
        """
        if not self._available or self._model is None:
            logger.debug("Forced alignment not available, keeping original timestamps")
            return segments

        if language in _SKIP_LANGUAGES:
            logger.info(
                f"  Skipping forced alignment for {language} "
                f"(CJK script — Whisper timestamps are adequate)"
            )
            return segments

        if not segments:
            return segments

        aligned_count = 0
        skipped_count = 0

        progress = create_progress("Forced Alignment")
        with progress:
            task = progress.add_task("Aligning words", total=len(segments))

            for seg in segments:
                success = self._align_segment(seg, wav_path)
                if success:
                    aligned_count += 1
                else:
                    skipped_count += 1
                progress.advance(task)

        logger.info(
            f"  ✓ Forced alignment: {aligned_count} segments refined, "
            f"{skipped_count} kept original timestamps"
        )

        return segments

    # ─────────────────────────────────────────────────────────────────
    # Per-segment alignment
    # ─────────────────────────────────────────────────────────────────

    def _align_segment(
        self,
        seg: ASRSegment,
        wav_path: Path,
    ) -> bool:
        """
        Align a single ASR segment. Returns True if successful.

        Modifies seg.words in-place with refined timestamps.
        """
        # Skip empty or very short segments
        if not seg.text.strip() or (seg.end_s - seg.start_s) < 0.1:
            return False

        try:
            # Load the audio region for this segment
            waveform = self._load_audio_slice(wav_path, seg.start_s, seg.end_s)
            if waveform is None or waveform.shape[1] == 0:
                return False

            # Perform alignment
            words = self._align_single(waveform, seg.text, seg.start_s)
            if not words:
                return False

            # Update segment with refined word timestamps
            seg.words = words

            # Optionally tighten segment boundaries to match first/last word
            if words:
                seg.start_s = words[0].start_s
                seg.end_s = words[-1].end_s

            return True

        except Exception as e:
            logger.debug(
                f"  Alignment failed for segment {seg.segment_id}: {e}"
            )
            return False

    def _load_audio_slice(
        self,
        wav_path: Path,
        start_s: float,
        end_s: float,
    ) -> Optional[torch.Tensor]:
        """Load a slice of audio from the WAV file."""
        try:
            info = torchaudio.info(str(wav_path))
            sr = info.sample_rate

            frame_offset = int(start_s * sr)
            num_frames = int((end_s - start_s) * sr)

            # Clamp to file bounds
            frame_offset = max(0, frame_offset)
            num_frames = min(num_frames, info.num_frames - frame_offset)

            if num_frames <= 0:
                return None

            waveform, actual_sr = torchaudio.load(
                str(wav_path),
                frame_offset=frame_offset,
                num_frames=num_frames,
            )

            # Resample if needed (should already be 16kHz)
            if actual_sr != self._sample_rate:
                resampler = torchaudio.transforms.Resample(
                    actual_sr, self._sample_rate
                )
                waveform = resampler(waveform)

            return waveform.to(self._device)

        except Exception as e:
            logger.debug(f"  Failed to load audio slice: {e}")
            return None

    # ─────────────────────────────────────────────────────────────────
    # Core alignment logic
    # ─────────────────────────────────────────────────────────────────

    def _align_single(
        self,
        waveform: torch.Tensor,
        text: str,
        offset_s: float,
    ) -> list[WordSegment]:
        """
        Align text to waveform using CTC forced alignment.

        Args:
            waveform: Audio tensor [1, num_samples] at 16kHz.
            text: Transcribed text to align.
            offset_s: Time offset to add to all timestamps (segment start).

        Returns:
            List of WordSegment with precise timestamps, or empty list on failure.
        """
        assert self._model is not None
        assert self._dictionary is not None

        # Normalize and tokenize text
        normalized = self._normalize_for_alignment(text)
        if not normalized:
            return []

        # Convert characters to token IDs
        tokens = [
            self._dictionary[c]
            for c in normalized
            if c in self._dictionary
        ]
        if not tokens:
            return []

        # Run model to get emission probabilities
        with torch.no_grad():
            emission, _ = self._model(waveform)  # [1, T, C]

        # Prepare targets tensor
        targets = torch.tensor([tokens], dtype=torch.int32, device=self._device)
        input_lengths = torch.tensor([emission.shape[1]], device=self._device)
        target_lengths = torch.tensor([len(tokens)], device=self._device)

        # Run forced alignment
        aligned_tokens, scores = torchaudio.functional.forced_align(
            emission, targets, input_lengths, target_lengths, blank=0
        )

        # Convert from nested list/tensor to flat list
        token_spans = aligned_tokens[0]  # First (only) batch item
        token_scores = scores[0]

        # Compute emission rate (frames per second)
        num_frames = emission.shape[1]
        audio_duration = waveform.shape[1] / self._sample_rate
        emission_rate = num_frames / audio_duration if audio_duration > 0 else 1.0

        # Merge character tokens into words
        words = self._merge_tokens_to_words(
            token_spans, token_scores, normalized, offset_s, emission_rate
        )

        return words

    def _normalize_for_alignment(self, text: str) -> str:
        """
        Normalize text for forced alignment.

        - Lowercase
        - Remove punctuation (keep only characters in the dictionary)
        - Replace spaces with "|" (word boundary token)
        """
        if self._dictionary is None:
            return ""

        text = text.lower().strip()

        # Replace spaces/tabs with word boundary token
        text = re.sub(r"\s+", "|", text)

        # Keep only characters in the dictionary
        cleaned = ""
        for c in text:
            if c in self._dictionary:
                cleaned += c

        # Remove leading/trailing/duplicate word boundaries
        cleaned = re.sub(r"\|+", "|", cleaned)
        cleaned = cleaned.strip("|")

        return cleaned

    def _merge_tokens_to_words(
        self,
        token_spans: torch.Tensor,
        token_scores: torch.Tensor,
        normalized: str,
        offset_s: float,
        emission_rate: float,
    ) -> list[WordSegment]:
        """
        Merge character-level alignment spans into word-level segments.

        Characters are grouped by the "|" word boundary token.
        """
        words: list[WordSegment] = []
        current_word_chars: list[str] = []
        current_word_start: Optional[int] = None
        current_word_end: int = 0
        current_word_scores: list[float] = []

        chars = list(normalized)

        for i, (char_idx, frame_idx, score) in enumerate(
            zip(range(len(chars)), token_spans.tolist(), token_scores.tolist())
        ):
            char = chars[char_idx] if char_idx < len(chars) else ""

            if char == "|":
                # Word boundary — finalize current word
                if current_word_chars and current_word_start is not None:
                    word_text = "".join(current_word_chars)
                    start_s = offset_s + current_word_start / emission_rate
                    end_s = offset_s + current_word_end / emission_rate
                    avg_score = (
                        sum(current_word_scores) / len(current_word_scores)
                        if current_word_scores
                        else 0.0
                    )
                    words.append(
                        WordSegment(
                            word=word_text,
                            start_s=start_s,
                            end_s=end_s,
                            confidence=max(0.0, min(1.0, avg_score)),
                        )
                    )
                # Reset for next word
                current_word_chars = []
                current_word_start = None
                current_word_scores = []
            else:
                current_word_chars.append(char)
                if current_word_start is None:
                    current_word_start = frame_idx
                current_word_end = frame_idx
                current_word_scores.append(score)

        # Finalize last word
        if current_word_chars and current_word_start is not None:
            word_text = "".join(current_word_chars)
            start_s = offset_s + current_word_start / emission_rate
            end_s = offset_s + current_word_end / emission_rate
            avg_score = (
                sum(current_word_scores) / len(current_word_scores)
                if current_word_scores
                else 0.0
            )
            words.append(
                WordSegment(
                    word=word_text,
                    start_s=start_s,
                    end_s=end_s,
                    confidence=max(0.0, min(1.0, avg_score)),
                )
            )

        return words
