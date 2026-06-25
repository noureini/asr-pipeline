"""
Audio preprocessing module — telephony-robust, shared-spine front end.

Chain:
  1. Format conversion → 16 kHz mono WAV (FFmpeg, high-quality soxr resample)
  2. Loudness normalization — OFF by default (whole-file gain shifts VAD)
  3. Noise reduction — OFF by default (ASR models are trained on noisy speech)
  4. Voice Activity Detection (Silero VAD), RATE-ADAPTIVE: native 8 kHz /
     256-sample profile for telephony, else 16 kHz / 512-sample; model-neutral,
     recall-leaning params so neither ASR witness is favored.
  5. Chunking: WhisperX-style greedy merge, cut only at silence, ≤15 s — the
     single shared spine both ASR witnesses (titu + Whisper) transcribe.
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
            # Rate-adaptive VAD runs on the ORIGINAL audio at its native rate
            # (8 kHz telephony stays 8 kHz for Silero's 256-win profile).
            native_sr = self.get_audio_metadata(audio_path).sample_rate
            speech_timestamps = self._detect_speech(audio_path, native_sr)
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

        cmd = ["ffmpeg", "-y", "-i", str(audio_path)]
        # High-quality soxr resampling (artifact-free up/downsample to 16 kHz).
        if self._config.resampler == "soxr":
            cmd += ["-af", "aresample=resampler=soxr:precision=28"]
        cmd += [
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

    def _detect_speech(
        self, audio_path: Path, native_sr: int
    ) -> list[tuple[float, float]]:
        """
        Detect speech regions using Silero VAD, rate-adaptively.

        Runs VAD at the source's NATIVE rate when it is 8 kHz (Silero's
        256-sample / 8 kHz profile), otherwise at 16 kHz (512-sample profile).
        Pushing 8 kHz telephony through the 16 kHz profile is a temporal
        mismatch that can drop valid speech frames. Timestamps are returned in
        SECONDS, so they apply unchanged to the 16 kHz wav used downstream.

        Returns list of (start_s, end_s) tuples for speech segments.
        """
        # Silero ships only 8 kHz and 16 kHz profiles; dispatch by source rate.
        vad_sr = 8000 if native_sr == 8000 else 16000

        # Decode the ORIGINAL at the VAD rate (passthrough when already that
        # rate; soxr downsample for anything higher). This gives true native
        # 8 kHz content for VAD rather than 8 kHz upsampled to 16 kHz.
        vad_wav = self._work_dir / f"vad_{vad_sr}.wav"
        cmd = ["ffmpeg", "-y", "-i", str(audio_path)]
        if self._config.resampler == "soxr":
            cmd += ["-af", "aresample=resampler=soxr:precision=28"]
        cmd += [
            "-ar", str(vad_sr), "-ac", "1", "-c:a", "pcm_s16le",
            "-loglevel", "error", str(vad_wav),
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True)

        waveform, sr = torchaudio.load(str(vad_wav))
        waveform = waveform.squeeze()

        model = self._get_vad_model()

        # Get speech timestamps (Silero auto-selects 256-win @8k / 512-win @16k)
        speech_timestamps: list[tuple[float, float]] = []
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

        # ── WhisperX-style greedy merge ─────────────────────────────────
        # Faithful port of whisperx/vads/vad.py:Vad.merge_chunks. Pack
        # VAD speech segments and close a chunk ONLY when adding the next
        # would make the span exceed max_dur — at the previous segment
        # boundary; never cut inside a speech segment; NO silence/gap
        # threshold. Our old "close on gap>3s" heuristic was homegrown
        # and unvalidated; WhisperX's minimal approach is the proven one.
        # Split any single VAD segment longer than max_dur at its
        # QUIETEST interior point (WhisperX splits an over-long segment
        # at the lowest-VAD-score frame, NOT evenly — an even split
        # would cut mid-word). We use min short-time energy as a robust
        # proxy for "most silence-like", restricted to the 30–70%
        # interior so pieces stay balanced, recursing until all ≤max_dur.
        mono_wf = waveform.mean(0) if waveform.dim() > 1 else waveform

        def _split_quiet(s: float, e: float) -> list[tuple[float, float]]:
            if e - s <= max_dur:
                return [(s, e)]
            a, b = int(s * sr), int(e * sr)
            seg = mono_wf[a:b]
            win = max(1, int(0.05 * sr))            # 50 ms frames
            lo, hi = int(len(seg) * 0.3), int(len(seg) * 0.7)
            best_i, best_v = None, None
            i = lo
            while i + win <= hi:
                v = float(seg[i:i + win].pow(2).mean())
                if best_v is None or v < best_v:
                    best_v, best_i = v, i
                i += win
            mid = (s + e) / 2.0 if best_i is None else \
                s + (best_i + win / 2) / sr
            return _split_quiet(s, mid) + _split_quiet(mid, e)

        norm_segs: list[tuple[float, float]] = []
        for s, e in speech_timestamps:
            norm_segs.extend(_split_quiet(s, e))

        final_boundaries: list[tuple[float, float]] = []
        cs, ce = norm_segs[0]
        for s, e in norm_segs[1:]:
            if e - cs > max_dur and ce - cs > 0:
                final_boundaries.append((cs, ce))
                cs = s
            ce = e
        final_boundaries.append((cs, ce))

        # ── Phase 3: Add padding and extract chunk WAV files ────────────
        # NO overlap (padding = 0): the Stage-1 decision is ONE clean shared
        # spine. titu reads each spine chunk individually (CTC, native word
        # timestamps); cross-boundary context for the Whisper witness is handled
        # at fusion via the merged-window read (Whisper reads merged runs of
        # consecutive spine chunks), not by overlapping+de-duping chunks here.
        padding = 0.0
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
