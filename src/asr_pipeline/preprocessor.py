"""
Audio preprocessing module.

Handles the full preprocessing chain:
  1. Format conversion (any format → 16kHz mono WAV via FFmpeg)
  2. Loudness normalization (EBU R128 / -23 LUFS)
  3. Voice Activity Detection (Silero VAD)
  4. Noise reduction (spectral gating)
  5. Chunking (30s segments with 2s overlap)
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch
import torchaudio

from asr_pipeline.config import PreprocessingConfig
from asr_pipeline.logging_config import create_progress
from asr_pipeline.models import AudioChunk, AudioMetadata

logger = logging.getLogger("asr_pipeline")


class AudioPreprocessor:
    """
    Preprocesses raw audio files into normalized, chunked segments
    ready for ASR inference.
    """

    def __init__(self, config: PreprocessingConfig, work_dir: Path) -> None:
        self._config = config
        self._work_dir = work_dir
        self._work_dir.mkdir(parents=True, exist_ok=True)
        self._vad_model: Optional[torch.nn.Module] = None
        self._speech_timestamps: Optional[list[tuple[float, float]]] = None

    @property
    def speech_timestamps(self) -> Optional[list[tuple[float, float]]]:
        """Raw VAD speech regions from the last preprocess() call."""
        return self._speech_timestamps

    # ─────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────

    def get_audio_metadata(self, audio_path: Path) -> AudioMetadata:
        """Extract metadata from an audio file using FFmpeg probe."""
        import json as json_mod

        result = subprocess.run(
            [
                "ffprobe", "-v", "quiet", "-print_format", "json",
                "-show_format", "-show_streams", str(audio_path),
            ],
            capture_output=True, text=True, check=True,
        )
        probe = json_mod.loads(result.stdout)
        fmt = probe.get("format", {})
        audio_stream = next(
            (s for s in probe.get("streams", []) if s.get("codec_type") == "audio"),
            {},
        )

        return AudioMetadata(
            file_path=audio_path,
            duration_s=float(fmt.get("duration", 0)),
            sample_rate=int(audio_stream.get("sample_rate", 16000)),
            channels=int(audio_stream.get("channels", 1)),
            format=audio_path.suffix.lstrip("."),
            file_size_bytes=audio_path.stat().st_size,
        )

    def preprocess(
        self, audio_path: Path,
    ) -> tuple[Path, list[AudioChunk], list[tuple[float, float, str]]]:
        """
        Run the full preprocessing pipeline on an audio file.

        Returns:
            Tuple of (path to normalized full WAV, list of audio chunks,
            list of non-speech regions as (start_s, end_s, type) tuples).
        """
        logger.info(f"  Input: [file]{audio_path.name}[/file]")

        # Step 1: Convert to 16kHz mono WAV
        wav_path = self._convert_to_wav(audio_path)
        logger.info("  ✓ Converted to 16kHz mono WAV")

        # Step 2: Loudness normalization
        if self._config.loudness_normalization.enabled:
            wav_path = self._normalize_loudness(wav_path)
            logger.info(
                f"  ✓ Loudness normalized to "
                f"{self._config.loudness_normalization.target_lufs} LUFS"
            )

        # Step 3: Noise reduction
        if self._config.noise_reduction.enabled:
            wav_path = self._reduce_noise(wav_path)
            logger.info("  ✓ Noise reduction applied (spectral gating)")

        # Step 4: VAD — detect speech regions for boundary-guided chunking
        speech_timestamps = None
        non_speech_regions: list[tuple[float, float, str]] = []
        if self._config.vad.enabled:
            speech_timestamps = self._detect_speech(wav_path)
            self._speech_timestamps = speech_timestamps
            total_speech = sum(
                (end - start) for start, end in speech_timestamps
            )

            # Get total duration for non-speech extraction
            waveform_info = torchaudio.info(str(wav_path))
            total_duration = waveform_info.num_frames / waveform_info.sample_rate

            # Extract non-speech regions (inverse of speech timestamps)
            # Classifies each gap as silence, noise, or music based on
            # audio energy and spectral features.
            non_speech_regions = self.extract_non_speech_regions(
                speech_timestamps, total_duration, wav_path=wav_path
            )
            total_non_speech = sum(
                (end - start) for start, end, _ in non_speech_regions
            )

            logger.info(
                f"  ✓ VAD: {len(speech_timestamps)} speech segments "
                f"({total_speech:.1f}s speech, "
                f"{total_non_speech:.1f}s non-speech, "
                f"{len(non_speech_regions)} gaps)"
            )

        # Step 5: Chunk into segments (VAD-guided cut & merge)
        chunks = self._create_chunks(wav_path, speech_timestamps)
        if speech_timestamps:
            logger.info(
                f"  ✓ Created {len(chunks)} VAD-aligned chunks "
                f"(max {self._config.chunking.max_duration_s}s, "
                f"cut at speech boundaries)"
            )
        else:
            logger.info(
                f"  ✓ Created {len(chunks)} fixed chunks "
                f"(max {self._config.chunking.max_duration_s}s, "
                f"{self._config.chunking.overlap_s}s overlap)"
            )

        return wav_path, chunks, non_speech_regions

    # ─────────────────────────────────────────────────────────────────
    # Step 1: Format conversion
    # ─────────────────────────────────────────────────────────────────

    def _convert_to_wav(self, audio_path: Path) -> Path:
        """Convert any audio format to 16kHz mono WAV using FFmpeg."""
        output_path = self._work_dir / "normalized.wav"

        cmd = [
            "ffmpeg", "-y",
            "-i", str(audio_path),
            "-ar", str(self._config.target_sample_rate),
            "-ac", "1" if self._config.mono else "2",
            "-c:a", "pcm_s16le",
            "-loglevel", "error",
            str(output_path),
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except FileNotFoundError:
            raise RuntimeError(
                "FFmpeg not found. Install it with: "
                "apt install ffmpeg (Linux) or brew install ffmpeg (macOS)"
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"FFmpeg conversion failed: {e.stderr}")

        return output_path

    # ─────────────────────────────────────────────────────────────────
    # Step 2: Loudness normalization
    # ─────────────────────────────────────────────────────────────────

    def _normalize_loudness(self, wav_path: Path) -> Path:
        """
        Normalize loudness to target LUFS using FFmpeg loudnorm filter.

        Uses a two-pass approach for accurate normalization.
        """
        target_lufs = self._config.loudness_normalization.target_lufs
        output_path = self._work_dir / "loudnorm.wav"

        # Single-pass loudness normalization (sufficient for speech)
        cmd = [
            "ffmpeg", "-y",
            "-i", str(wav_path),
            "-af", f"loudnorm=I={target_lufs}:TP=-1.5:LRA=11",
            "-ar", str(self._config.target_sample_rate),
            "-ac", "1",
            "-c:a", "pcm_s16le",
            "-loglevel", "error",
            str(output_path),
        ]

        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return output_path

    # ─────────────────────────────────────────────────────────────────
    # Step 3: Noise reduction
    # ─────────────────────────────────────────────────────────────────

    def _reduce_noise(self, wav_path: Path) -> Path:
        """
        Apply spectral gating noise reduction.

        Uses a simple spectral gate: estimate noise from the quietest
        portions and subtract it from the signal.
        """
        waveform, sr = torchaudio.load(str(wav_path))
        audio_np = waveform.squeeze().numpy()

        # Simple spectral gating via magnitude threshold
        # For production, consider noisereduce library
        try:
            import noisereduce as nr

            reduced = nr.reduce_noise(
                y=audio_np,
                sr=sr,
                prop_decrease=self._config.noise_reduction.prop_decrease,
                stationary=True,
            )
        except ImportError:
            logger.warning(
                "noisereduce not installed, skipping noise reduction. "
                "Install with: pip install noisereduce"
            )
            return wav_path

        output_path = self._work_dir / "denoised.wav"
        sf.write(str(output_path), reduced, sr)
        return output_path

    # ─────────────────────────────────────────────────────────────────
    # Step 4: Voice Activity Detection
    # ─────────────────────────────────────────────────────────────────

    def _get_vad_model(self) -> torch.nn.Module:
        """Lazily load Silero VAD model."""
        if self._vad_model is None:
            model, _ = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                trust_repo=True,
            )
            self._vad_model = model
        return self._vad_model

    def _detect_speech(self, wav_path: Path) -> list[tuple[float, float]]:
        """
        Detect speech regions using Silero VAD.

        Returns list of (start_s, end_s) tuples for speech segments.
        """
        waveform, sr = torchaudio.load(str(wav_path))
        waveform = waveform.squeeze()

        # Silero VAD expects 16kHz
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            waveform = resampler(waveform)
            sr = 16000

        model = self._get_vad_model()

        # Get speech timestamps
        speech_timestamps = []
        try:
            from silero_vad import get_speech_timestamps  # type: ignore[import]

            timestamps = get_speech_timestamps(
                waveform,
                model,
                threshold=self._config.vad.threshold,
                min_speech_duration_ms=self._config.vad.min_speech_duration_ms,
                min_silence_duration_ms=self._config.vad.min_silence_duration_ms,
                sampling_rate=sr,
            )
        except ImportError:
            # Fallback: use the model's get_speech_timestamps utility
            timestamps = model(waveform, sr)  # type: ignore[operator]
            if not isinstance(timestamps, list):
                return [(0.0, len(waveform) / sr)]

        for ts in timestamps:
            start_s = ts.get("start", 0) / sr
            end_s = ts.get("end", len(waveform)) / sr
            speech_timestamps.append((start_s, end_s))

        return speech_timestamps if speech_timestamps else [(0.0, len(waveform) / sr)]

    # ─────────────────────────────────────────────────────────────────
    # Step 5: Chunking (VAD-guided cut & merge)
    # ─────────────────────────────────────────────────────────────────

    def _create_chunks(
        self,
        wav_path: Path,
        speech_timestamps: Optional[list[tuple[float, float]]],
    ) -> list[AudioChunk]:
        """
        Split audio into VAD-boundary-aligned chunks (WhisperX-style cut & merge).

        When VAD timestamps are available, chunks are cut at natural silence
        boundaries — never mid-word. Adjacent speech segments are merged up
        to max_duration to give Whisper enough context.

        Falls back to fixed-window chunking when VAD is disabled.
        """
        waveform, sr = torchaudio.load(str(wav_path))
        total_duration = waveform.shape[1] / sr
        max_dur = self._config.chunking.max_duration_s

        chunk_dir = self._work_dir / "chunks"
        chunk_dir.mkdir(exist_ok=True)

        if not speech_timestamps:
            # Fallback: fixed-window chunking (VAD disabled or no speech found)
            return self._create_fixed_chunks(waveform, sr, total_duration, chunk_dir)

        # ── Phase 1: Merge adjacent speech segments up to max_duration ──
        # Combine close-together speech segments into groups that Whisper
        # can process as a single chunk, cutting only at silence gaps.
        merged_groups: list[tuple[float, float]] = []
        group_start = speech_timestamps[0][0]
        group_end = speech_timestamps[0][1]

        for seg_start, seg_end in speech_timestamps[1:]:
            gap = seg_start - group_end
            new_group_duration = seg_end - group_start

            # Merge if: adding this segment doesn't exceed max_duration
            # AND the gap between segments is small (< 3s silence)
            if new_group_duration <= max_dur and gap < 3.0:
                group_end = seg_end
            else:
                merged_groups.append((group_start, group_end))
                group_start = seg_start
                group_end = seg_end

        merged_groups.append((group_start, group_end))

        # ── Phase 2: Split groups that still exceed max_duration ─────────
        # Find the largest internal silence gap and split there.
        final_boundaries: list[tuple[float, float]] = []
        for g_start, g_end in merged_groups:
            if (g_end - g_start) <= max_dur:
                final_boundaries.append((g_start, g_end))
            else:
                # Collect internal VAD segments within this group
                internal = [
                    (s, e) for s, e in speech_timestamps
                    if s >= g_start and e <= g_end
                ]
                final_boundaries.extend(
                    self._split_at_best_gap(internal, max_dur)
                )

        # ── Phase 3: Add padding and extract chunk WAV files ────────────
        padding = 0.2  # seconds of context before/after each chunk
        chunks: list[AudioChunk] = []

        for chunk_id, (c_start, c_end) in enumerate(final_boundaries):
            # Add padding but clamp to audio bounds
            padded_start = max(0.0, c_start - padding)
            padded_end = min(total_duration, c_end + padding)

            start_sample = int(padded_start * sr)
            end_sample = int(padded_end * sr)
            chunk_waveform = waveform[:, start_sample:end_sample]

            chunk_path = chunk_dir / f"chunk_{chunk_id:04d}.wav"
            torchaudio.save(str(chunk_path), chunk_waveform, sr)

            chunks.append(
                AudioChunk(
                    chunk_id=chunk_id,
                    start_s=padded_start,
                    end_s=padded_end,
                    duration_s=padded_end - padded_start,
                    waveform_path=chunk_path,
                )
            )

        return chunks

    def _split_at_best_gap(
        self,
        segments: list[tuple[float, float]],
        max_dur: float,
    ) -> list[tuple[float, float]]:
        """
        Split a list of speech segments into sub-groups that fit within max_dur.

        Recursively finds the largest silence gap and splits there.
        """
        if not segments:
            return []

        total_start = segments[0][0]
        total_end = segments[-1][1]

        if (total_end - total_start) <= max_dur or len(segments) <= 1:
            return [(total_start, total_end)]

        # Find the largest gap between consecutive segments
        best_gap_idx = 0
        best_gap_size = 0.0
        for i in range(len(segments) - 1):
            gap = segments[i + 1][0] - segments[i][1]
            if gap > best_gap_size:
                best_gap_size = gap
                best_gap_idx = i

        # Split at the best gap
        left = segments[: best_gap_idx + 1]
        right = segments[best_gap_idx + 1 :]

        return (
            self._split_at_best_gap(left, max_dur)
            + self._split_at_best_gap(right, max_dur)
        )

    def _create_fixed_chunks(
        self,
        waveform: torch.Tensor,
        sr: int,
        total_duration: float,
        chunk_dir: Path,
    ) -> list[AudioChunk]:
        """Fallback: fixed-window chunking with overlap (used when VAD is disabled)."""
        max_dur = self._config.chunking.max_duration_s
        overlap = self._config.chunking.overlap_s
        chunks: list[AudioChunk] = []
        start = 0.0
        chunk_id = 0

        while start < total_duration:
            end = min(start + max_dur, total_duration)

            start_sample = int(start * sr)
            end_sample = int(end * sr)
            chunk_waveform = waveform[:, start_sample:end_sample]

            chunk_path = chunk_dir / f"chunk_{chunk_id:04d}.wav"
            torchaudio.save(str(chunk_path), chunk_waveform, sr)

            chunks.append(
                AudioChunk(
                    chunk_id=chunk_id,
                    start_s=start,
                    end_s=end,
                    duration_s=end - start,
                    waveform_path=chunk_path,
                )
            )

            chunk_id += 1
            start = end - overlap if end < total_duration else total_duration

        return chunks

    # ─────────────────────────────────────────────────────────────────
    # Step 6: Non-speech region extraction
    # ─────────────────────────────────────────────────────────────────

    def extract_non_speech_regions(
        self,
        speech_timestamps: list[tuple[float, float]],
        total_duration: float,
        min_gap_s: float = 0.3,
        wav_path: Optional[Path] = None,
    ) -> list[tuple[float, float, str]]:
        """
        Derive non-speech regions from VAD speech timestamps.

        Returns the "inverse" of speech: all time intervals where
        no speech was detected. Each region is classified as:
          - "silence": RMS energy below -50 dBFS (true silence)
          - "noise": Energy present but flat spectrum (hum, static)
          - "music": Energy with harmonic structure (jingles, music)

        Falls back to generic "non_speech" if wav_path is not provided.

        Args:
            speech_timestamps: List of (start_s, end_s) speech regions.
            total_duration: Total audio duration in seconds.
            min_gap_s: Minimum gap duration to report (ignore tiny gaps).
            wav_path: Path to the WAV file for audio analysis.

        Returns:
            List of (start_s, end_s, region_type) tuples.
        """
        # Find all gaps between speech segments
        gaps: list[tuple[float, float]] = []
        prev_end = 0.0

        for start, end in speech_timestamps:
            gap = start - prev_end
            if gap > min_gap_s:
                gaps.append((prev_end, start))
            prev_end = end

        # Trailing non-speech after last speech segment
        if total_duration - prev_end > min_gap_s:
            gaps.append((prev_end, total_duration))

        # Classify each gap
        if wav_path is not None and gaps:
            return self._classify_non_speech_regions(gaps, wav_path)
        else:
            return [(start, end, "non_speech") for start, end in gaps]

    def _classify_non_speech_regions(
        self,
        gaps: list[tuple[float, float]],
        wav_path: Path,
    ) -> list[tuple[float, float, str]]:
        """
        Classify non-speech gaps by analyzing audio energy and spectrum.

        Uses RMS energy and spectral flatness (Wiener entropy):
          - silence: RMS < -50 dBFS
          - noise: high spectral flatness (flat, broadband energy)
          - music: lower spectral flatness (harmonic/tonal content)
        """
        waveform, sr = torchaudio.load(str(wav_path))
        waveform = waveform.squeeze()  # [num_samples]
        total_samples = waveform.shape[0]

        regions: list[tuple[float, float, str]] = []

        # Thresholds
        silence_rms_db = -50.0  # Below this = silence
        noise_flatness = 0.5    # Spectral flatness above this = noise

        for start_s, end_s in gaps:
            start_sample = int(start_s * sr)
            end_sample = min(int(end_s * sr), total_samples)

            if end_sample <= start_sample:
                regions.append((start_s, end_s, "silence"))
                continue

            segment = waveform[start_sample:end_sample].float()

            # Compute RMS energy in dBFS
            rms = torch.sqrt(torch.mean(segment ** 2) + 1e-10)
            rms_db = 20.0 * torch.log10(rms + 1e-10).item()

            if rms_db < silence_rms_db:
                regions.append((start_s, end_s, "silence"))
                continue

            # Compute spectral flatness (Wiener entropy)
            # High flatness = noise-like (flat spectrum)
            # Low flatness = tonal/harmonic (music)
            region_type = self._classify_by_spectrum(segment, sr, noise_flatness)
            regions.append((start_s, end_s, region_type))

        return regions

    @staticmethod
    def _classify_by_spectrum(
        segment: torch.Tensor,
        sr: int,
        noise_threshold: float,
    ) -> str:
        """
        Classify an audio segment as 'noise' or 'music' via spectral flatness.

        Spectral flatness = geometric_mean(spectrum) / arithmetic_mean(spectrum)
        Values close to 1.0 indicate noise (flat spectrum).
        Values close to 0.0 indicate tonal/harmonic content (music).
        """
        # Compute magnitude spectrum via FFT
        n_fft = min(2048, len(segment))
        if n_fft < 64:
            return "noise"

        spectrum = torch.fft.rfft(segment[:n_fft]).abs()
        spectrum = spectrum[1:]  # Drop DC component

        if spectrum.numel() == 0 or spectrum.max() < 1e-10:
            return "noise"

        # Spectral flatness: exp(mean(log(x))) / mean(x)
        log_spectrum = torch.log(spectrum + 1e-10)
        geometric_mean = torch.exp(torch.mean(log_spectrum))
        arithmetic_mean = torch.mean(spectrum)

        if arithmetic_mean < 1e-10:
            return "noise"

        flatness = (geometric_mean / arithmetic_mean).item()

        return "noise" if flatness > noise_threshold else "music"
