"""
Microphone quality testing module.

Analyzes acoustic properties of test recordings from multiple microphones
to provide a data-driven comparison for field interview equipment selection.

Metrics computed per audio file:
  - SNR (signal-to-noise ratio) via VAD-based speech/noise separation
  - Clipping and plosive spike detection
  - Spectral rolloff and effective bandwidth
  - Cross-talk bleed estimation
"""

from __future__ import annotations

import json
import logging
import re
import tempfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torchaudio

from asr_pipeline.config import AppConfig, PreprocessingConfig
from asr_pipeline.models import MicAudioMetrics, MicSummary, MicTestReport
from asr_pipeline.preprocessor import AudioPreprocessor

logger = logging.getLogger("asr_pipeline")


# ═══════════════════════════════════════════════════════════════════════
# Folder discovery & mic mapping
# ═══════════════════════════════════════════════════════════════════════


def parse_mic_mapping(readme_path: Path) -> dict[str, str]:
    """
    Parse a README.txt to extract folder_key -> mic_name mapping.

    Expected format (lines with ``# Mic Name`` headings followed by
    folder keys like ``35-36-03-87``):

        # DJI Mic 3
        35-36-03-87
        84-00-39-86

        # Hollyland Lark M2
        53-78-11-24
    """
    text = readme_path.read_text(encoding="utf-8")
    mapping: dict[str, str] = {}
    current_mic: Optional[str] = None

    folder_re = re.compile(r"(\d{2}-\d{2}-\d{2}-\d{2})")

    for line in text.splitlines():
        stripped = line.strip()
        # Detect mic heading (lines starting with #)
        if stripped.startswith("#"):
            current_mic = stripped.lstrip("#").strip()
            continue

        # Match folder keys under the current mic heading
        if current_mic:
            match = folder_re.search(stripped)
            if match:
                mapping[match.group(1)] = current_mic

    return mapping


def discover_test_audio(root_dir: Path) -> dict[str, list[Path]]:
    """
    Scan ``root_dir/{folder_key}/AudioAudit/*.m4a`` structure.

    Returns:
        Dict mapping folder_key -> list of audio file paths.
    """
    folder_re = re.compile(r"^\d{2}-\d{2}-\d{2}-\d{2}$")
    result: dict[str, list[Path]] = {}

    for entry in sorted(root_dir.iterdir()):
        if not entry.is_dir() or not folder_re.match(entry.name):
            continue

        audio_dir = entry / "AudioAudit"
        if not audio_dir.is_dir():
            continue

        audio_files = sorted(audio_dir.glob("*.m4a"))
        if audio_files:
            result[entry.name] = audio_files

    return result


# ═══════════════════════════════════════════════════════════════════════
# Core analysis engine
# ═══════════════════════════════════════════════════════════════════════


class MicTester:
    """Acoustic quality analyzer for microphone comparison."""

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._base_work_dir = Path(
            tempfile.mkdtemp(prefix="asr_mic_test_")
        )

    def analyze_file(
        self,
        audio_path: Path,
        mic_name: str,
        folder_key: str,
    ) -> MicAudioMetrics:
        """Run all acoustic metrics on a single audio file."""
        # Create a unique work dir per file to avoid overwrites
        file_work_dir = self._base_work_dir / folder_key / audio_path.stem
        file_work_dir.mkdir(parents=True, exist_ok=True)

        preprocessor = AudioPreprocessor(
            self._config.preprocessing, file_work_dir
        )

        # Step 1: Get metadata
        metadata = preprocessor.get_audio_metadata(audio_path)

        # Step 2: Convert to 16kHz mono WAV
        wav_path = preprocessor._convert_to_wav(audio_path)

        # Step 3: Load raw waveform (pre-normalization for accurate metrics)
        waveform, sr = torchaudio.load(str(wav_path))
        waveform_np = waveform.squeeze().numpy().astype(np.float64)
        total_samples = len(waveform_np)

        # Step 4: Run VAD to get speech regions
        speech_regions = preprocessor._detect_speech(wav_path)

        # Step 5: Compute all metrics
        snr_db = self._compute_snr(waveform_np, speech_regions, sr)
        clipped, clip_ratio = self._count_clipping(waveform_np)
        plosives = self._detect_plosive_spikes(waveform_np, sr)
        rolloff = self._compute_spectral_rolloff(waveform_np, sr)
        bandwidth = self._compute_effective_bandwidth(waveform_np, sr)
        crosstalk = self._estimate_crosstalk(waveform_np, speech_regions, sr)

        # Basic amplitude metrics
        peak = float(np.max(np.abs(waveform_np)))
        rms = float(np.sqrt(np.mean(waveform_np ** 2) + 1e-10))
        rms_dbfs = float(20.0 * np.log10(rms + 1e-10))

        # Speech ratio from VAD
        total_duration = total_samples / sr
        speech_duration = sum(e - s for s, e in speech_regions)
        speech_ratio = speech_duration / total_duration if total_duration > 0 else 0.0

        # Release VAD model from GPU
        preprocessor._vad_model = None
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        return MicAudioMetrics(
            file_path=str(audio_path),
            mic_name=mic_name,
            folder_key=folder_key,
            duration_s=metadata.duration_s,
            snr_db=round(snr_db, 2),
            clipped_samples=clipped,
            clipping_ratio=round(clip_ratio, 6),
            plosive_spike_count=plosives,
            spectral_rolloff_hz=round(rolloff, 1),
            effective_bandwidth_hz=round(bandwidth, 1),
            crosstalk_ratio=round(crosstalk, 4),
            peak_amplitude=round(peak, 4),
            rms_dbfs=round(rms_dbfs, 2),
            speech_ratio=round(speech_ratio, 3),
        )

    def analyze_all(
        self,
        file_map: dict[str, list[Path]],
        folder_mic_map: dict[str, str],
    ) -> list[MicSummary]:
        """Analyze all files and return per-mic summaries."""
        # Group results by mic
        mic_results: dict[str, list[MicAudioMetrics]] = defaultdict(list)

        total_files = sum(len(files) for files in file_map.values())
        processed = 0

        for folder_key, audio_files in sorted(file_map.items()):
            mic_name = folder_mic_map.get(folder_key, f"Unknown ({folder_key})")
            for audio_path in audio_files:
                processed += 1
                logger.info(
                    f"  [{processed}/{total_files}] Analyzing "
                    f"[file]{audio_path.name}[/file] ({mic_name})"
                )
                metrics = self.analyze_file(audio_path, mic_name, folder_key)
                mic_results[mic_name].append(metrics)

        # Build summaries
        summaries: list[MicSummary] = []
        for mic_name, files in sorted(mic_results.items()):
            n = len(files)
            summary = MicSummary(
                mic_name=mic_name,
                num_files=n,
                avg_snr_db=round(sum(f.snr_db for f in files) / n, 2),
                avg_clipping_ratio=round(
                    sum(f.clipping_ratio for f in files) / n, 6
                ),
                total_plosive_spikes=sum(f.plosive_spike_count for f in files),
                avg_spectral_rolloff_hz=round(
                    sum(f.spectral_rolloff_hz for f in files) / n, 1
                ),
                avg_effective_bandwidth_hz=round(
                    sum(f.effective_bandwidth_hz for f in files) / n, 1
                ),
                avg_crosstalk_ratio=round(
                    sum(f.crosstalk_ratio for f in files) / n, 4
                ),
                avg_rms_dbfs=round(sum(f.rms_dbfs for f in files) / n, 2),
                avg_speech_ratio=round(
                    sum(f.speech_ratio for f in files) / n, 3
                ),
                files=files,
            )
            summaries.append(summary)

        # Score each mic
        self._score_mics(summaries)

        return summaries

    def generate_recommendation(self, summaries: list[MicSummary]) -> str:
        """Generate a text recommendation based on scored summaries."""
        if not summaries:
            return "No data available for recommendation."

        ranked = sorted(summaries, key=lambda s: s.score, reverse=True)
        winner = ranked[0]

        lines = [
            f"RECOMMENDATION: {winner.mic_name}",
            f"  Composite score: {winner.score:.2f}/1.00",
            "",
        ]

        # Explain strengths
        for s in ranked:
            strengths = []
            weaknesses = []

            # Compare to others
            best_snr = max(x.avg_snr_db for x in summaries)
            best_xtalk = min(x.avg_crosstalk_ratio for x in summaries)
            best_clip = min(x.avg_clipping_ratio for x in summaries)
            best_bw = max(x.avg_effective_bandwidth_hz for x in summaries)
            fewest_plosives = min(x.total_plosive_spikes for x in summaries)

            if s.avg_snr_db == best_snr:
                strengths.append(f"best SNR ({s.avg_snr_db:.1f} dB)")
            if s.avg_crosstalk_ratio == best_xtalk:
                strengths.append(
                    f"lowest crosstalk ({s.avg_crosstalk_ratio:.4f})"
                )
            if s.avg_clipping_ratio == best_clip:
                strengths.append(
                    f"lowest clipping ({s.avg_clipping_ratio:.6f})"
                )
            if s.avg_effective_bandwidth_hz == best_bw:
                strengths.append(
                    f"widest bandwidth ({s.avg_effective_bandwidth_hz:.0f} Hz)"
                )
            if s.total_plosive_spikes == fewest_plosives:
                strengths.append(
                    f"fewest plosive spikes ({s.total_plosive_spikes})"
                )

            worst_snr = min(x.avg_snr_db for x in summaries)
            worst_xtalk = max(x.avg_crosstalk_ratio for x in summaries)
            worst_clip = max(x.avg_clipping_ratio for x in summaries)

            if s.avg_snr_db == worst_snr and len(summaries) > 1:
                weaknesses.append(f"lowest SNR ({s.avg_snr_db:.1f} dB)")
            if s.avg_crosstalk_ratio == worst_xtalk and len(summaries) > 1:
                weaknesses.append(
                    f"highest crosstalk ({s.avg_crosstalk_ratio:.4f})"
                )
            if s.avg_clipping_ratio == worst_clip and len(summaries) > 1:
                weaknesses.append(
                    f"most clipping ({s.avg_clipping_ratio:.6f})"
                )

            lines.append(
                f"  {s.mic_name} (score: {s.score:.2f}):"
            )
            if strengths:
                lines.append(f"    + {', '.join(strengths)}")
            if weaknesses:
                lines.append(f"    - {', '.join(weaknesses)}")
            lines.append("")

        return "\n".join(lines)

    # ─────────────────────────────────────────────────────────────────
    # Scoring
    # ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _score_mics(summaries: list[MicSummary]) -> None:
        """Assign a weighted composite score (0-1) to each mic."""
        if not summaries:
            return

        # Extract raw values
        snrs = [s.avg_snr_db for s in summaries]
        xtalks = [s.avg_crosstalk_ratio for s in summaries]
        clips = [s.avg_clipping_ratio for s in summaries]
        bws = [s.avg_effective_bandwidth_hz for s in summaries]
        plosives = [float(s.total_plosive_spikes) for s in summaries]

        def normalize(values: list[float], invert: bool = False) -> list[float]:
            """Normalize to 0-1 range. If invert, lower raw = higher score."""
            mn, mx = min(values), max(values)
            if mx == mn:
                return [1.0] * len(values)
            normed = [(v - mn) / (mx - mn) for v in values]
            if invert:
                normed = [1.0 - n for n in normed]
            return normed

        # Normalize (higher = better for all after inversion)
        n_snr = normalize(snrs)
        n_xtalk = normalize(xtalks, invert=True)
        n_clip = normalize(clips, invert=True)
        n_bw = normalize(bws)
        n_plosives = normalize(plosives, invert=True)

        # Weighted composite
        weights = {
            "snr": 0.35,
            "crosstalk": 0.25,
            "clipping": 0.15,
            "bandwidth": 0.15,
            "plosives": 0.10,
        }

        for i, s in enumerate(summaries):
            s.score = round(
                weights["snr"] * n_snr[i]
                + weights["crosstalk"] * n_xtalk[i]
                + weights["clipping"] * n_clip[i]
                + weights["bandwidth"] * n_bw[i]
                + weights["plosives"] * n_plosives[i],
                4,
            )

    # ─────────────────────────────────────────────────────────────────
    # Metric computation
    # ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _compute_snr(
        waveform: np.ndarray,
        speech_regions: list[tuple[float, float]],
        sr: int,
    ) -> float:
        """
        Compute SNR using VAD-based speech/noise separation.

        SNR = 10 * log10(speech_power / noise_power)
        """
        total_samples = len(waveform)
        speech_mask = np.zeros(total_samples, dtype=bool)

        for start_s, end_s in speech_regions:
            s = int(start_s * sr)
            e = min(int(end_s * sr), total_samples)
            speech_mask[s:e] = True

        speech_samples = waveform[speech_mask]
        noise_samples = waveform[~speech_mask]

        if len(speech_samples) == 0 or len(noise_samples) == 0:
            return 0.0

        speech_power = float(np.mean(speech_samples ** 2))
        noise_power = float(np.mean(noise_samples ** 2))

        if noise_power < 1e-12:
            return 60.0  # effectively infinite SNR, cap at 60 dB

        return float(10.0 * np.log10(speech_power / noise_power))

    @staticmethod
    def _count_clipping(waveform: np.ndarray) -> tuple[int, float]:
        """Count samples with |amplitude| >= 0.99."""
        clipped = int(np.sum(np.abs(waveform) >= 0.99))
        ratio = clipped / len(waveform) if len(waveform) > 0 else 0.0
        return clipped, float(ratio)

    @staticmethod
    def _detect_plosive_spikes(
        waveform: np.ndarray, sr: int
    ) -> int:
        """
        Detect sudden energy spikes characteristic of plosive consonants.

        Uses short-time energy in ~5ms frames. Flags frames where energy
        exceeds 3x the local median (computed over a 200ms window).
        """
        frame_length = int(0.005 * sr)  # 5ms frames
        if frame_length == 0:
            return 0

        # Compute short-time energy per frame
        n_frames = len(waveform) // frame_length
        if n_frames == 0:
            return 0

        frames = waveform[: n_frames * frame_length].reshape(n_frames, frame_length)
        frame_energy = np.mean(frames ** 2, axis=1)

        # Local median over ~200ms window
        window_frames = max(1, int(0.2 * sr / frame_length))
        spike_count = 0

        for i in range(n_frames):
            start = max(0, i - window_frames // 2)
            end = min(n_frames, i + window_frames // 2 + 1)
            local_median = np.median(frame_energy[start:end])

            if local_median > 1e-10 and frame_energy[i] > 3.0 * local_median:
                spike_count += 1

        return spike_count

    @staticmethod
    def _compute_spectral_rolloff(
        waveform: np.ndarray,
        sr: int,
        rolloff_pct: float = 0.85,
    ) -> float:
        """
        Frequency below which ``rolloff_pct`` of spectral energy lies.

        Uses the magnitude spectrum averaged over short frames.
        """
        n_fft = 2048
        hop = n_fft // 2
        n_frames = max(1, (len(waveform) - n_fft) // hop)

        # Accumulate average magnitude spectrum
        avg_spectrum = np.zeros(n_fft // 2 + 1)
        for i in range(n_frames):
            start = i * hop
            frame = waveform[start: start + n_fft]
            if len(frame) < n_fft:
                frame = np.pad(frame, (0, n_fft - len(frame)))
            spectrum = np.abs(np.fft.rfft(frame))
            avg_spectrum += spectrum

        avg_spectrum /= max(n_frames, 1)

        # Cumulative energy
        energy = avg_spectrum ** 2
        cumulative = np.cumsum(energy)
        total_energy = cumulative[-1]

        if total_energy < 1e-10:
            return 0.0

        # Find rolloff frequency
        threshold = rolloff_pct * total_energy
        rolloff_bin = int(np.searchsorted(cumulative, threshold))
        freq_resolution = sr / n_fft
        return float(rolloff_bin * freq_resolution)

    @staticmethod
    def _compute_effective_bandwidth(
        waveform: np.ndarray,
        sr: int,
        threshold_db: float = -20.0,
    ) -> float:
        """
        Bandwidth of the signal above ``threshold_db`` from the spectral peak.
        """
        n_fft = 2048
        hop = n_fft // 2
        n_frames = max(1, (len(waveform) - n_fft) // hop)

        avg_spectrum = np.zeros(n_fft // 2 + 1)
        for i in range(n_frames):
            start = i * hop
            frame = waveform[start: start + n_fft]
            if len(frame) < n_fft:
                frame = np.pad(frame, (0, n_fft - len(frame)))
            spectrum = np.abs(np.fft.rfft(frame))
            avg_spectrum += spectrum

        avg_spectrum /= max(n_frames, 1)

        if np.max(avg_spectrum) < 1e-10:
            return 0.0

        # Convert to dB
        spectrum_db = 20.0 * np.log10(avg_spectrum + 1e-10)
        peak_db = np.max(spectrum_db)
        threshold = peak_db + threshold_db  # e.g., peak - 20

        # Find bins above threshold
        above = np.where(spectrum_db >= threshold)[0]
        if len(above) == 0:
            return 0.0

        freq_resolution = sr / n_fft
        low_freq = float(above[0] * freq_resolution)
        high_freq = float(above[-1] * freq_resolution)
        return high_freq - low_freq

    @staticmethod
    def _estimate_crosstalk(
        waveform: np.ndarray,
        speech_regions: list[tuple[float, float]],
        sr: int,
    ) -> float:
        """
        Estimate cross-talk bleed by comparing energy in non-speech vs speech regions.

        For a clean mic with no bleed from another speaker, non-speech regions
        should be near-silent. Elevated energy in non-speech regions suggests
        the other speaker's voice is bleeding in.

        Returns ratio: mean_power(non_speech) / mean_power(speech).
        Lower is better (0.0 = no bleed).
        """
        total_samples = len(waveform)
        speech_mask = np.zeros(total_samples, dtype=bool)

        for start_s, end_s in speech_regions:
            s = int(start_s * sr)
            e = min(int(end_s * sr), total_samples)
            speech_mask[s:e] = True

        speech_samples = waveform[speech_mask]
        noise_samples = waveform[~speech_mask]

        if len(speech_samples) == 0 or len(noise_samples) == 0:
            return 0.0

        speech_power = float(np.mean(speech_samples ** 2))
        noise_power = float(np.mean(noise_samples ** 2))

        if speech_power < 1e-12:
            return 0.0

        return float(noise_power / speech_power)


# ═══════════════════════════════════════════════════════════════════════
# Rich display
# ═══════════════════════════════════════════════════════════════════════


def display_mic_report(report: MicTestReport) -> None:
    """Print Rich tables comparing microphone metrics."""
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    console = Console()

    # ── Table 1: Per-file metrics ──────────────────────────────────
    file_table = Table(
        title="Per-File Acoustic Metrics",
        show_lines=True,
        title_style="bold cyan",
    )
    file_table.add_column("Mic", style="bold")
    file_table.add_column("Folder", style="dim")
    file_table.add_column("Duration", justify="right")
    file_table.add_column("SNR (dB)", justify="right")
    file_table.add_column("Clip %", justify="right")
    file_table.add_column("Plosives", justify="right")
    file_table.add_column("Rolloff (Hz)", justify="right")
    file_table.add_column("Bandwidth (Hz)", justify="right")
    file_table.add_column("Crosstalk", justify="right")
    file_table.add_column("RMS (dBFS)", justify="right")
    file_table.add_column("Speech %", justify="right")

    # Collect all file metrics for color-coding
    all_files: list[MicAudioMetrics] = []
    for s in report.mic_summaries:
        all_files.extend(s.files)

    if all_files:
        best_snr = max(f.snr_db for f in all_files)
        worst_snr = min(f.snr_db for f in all_files)
        best_clip = min(f.clipping_ratio for f in all_files)
        worst_clip = max(f.clipping_ratio for f in all_files)
        best_xtalk = min(f.crosstalk_ratio for f in all_files)
        worst_xtalk = max(f.crosstalk_ratio for f in all_files)

    for s in report.mic_summaries:
        for f in s.files:

            def _color(val: float, best: float, worst: float, invert: bool = False) -> str:
                if best == worst:
                    return "white"
                if invert:
                    return "green" if val == best else ("red" if val == worst else "white")
                return "green" if val == best else ("red" if val == worst else "white")

            snr_color = _color(f.snr_db, best_snr, worst_snr)
            clip_color = _color(f.clipping_ratio, best_clip, worst_clip, invert=True)
            xtalk_color = _color(f.crosstalk_ratio, best_xtalk, worst_xtalk, invert=True)

            file_table.add_row(
                f.mic_name,
                f.folder_key,
                f"{f.duration_s:.1f}s",
                f"[{snr_color}]{f.snr_db:.1f}[/{snr_color}]",
                f"[{clip_color}]{f.clipping_ratio * 100:.4f}%[/{clip_color}]",
                str(f.plosive_spike_count),
                f"{f.spectral_rolloff_hz:.0f}",
                f"{f.effective_bandwidth_hz:.0f}",
                f"[{xtalk_color}]{f.crosstalk_ratio:.4f}[/{xtalk_color}]",
                f"{f.rms_dbfs:.1f}",
                f"{f.speech_ratio * 100:.1f}%",
            )

    console.print()
    console.print(file_table)

    # ── Table 2: Per-mic comparison ────────────────────────────────
    mic_table = Table(
        title="Microphone Comparison (Averages)",
        show_lines=True,
        title_style="bold cyan",
    )
    mic_table.add_column("Microphone", style="bold")
    mic_table.add_column("Files", justify="right")
    mic_table.add_column("Avg SNR (dB)", justify="right")
    mic_table.add_column("Avg Clip %", justify="right")
    mic_table.add_column("Plosive Spikes", justify="right")
    mic_table.add_column("Avg Rolloff (Hz)", justify="right")
    mic_table.add_column("Avg BW (Hz)", justify="right")
    mic_table.add_column("Avg Crosstalk", justify="right")
    mic_table.add_column("Avg RMS (dBFS)", justify="right")
    mic_table.add_column("Score", justify="right", style="bold magenta")

    ranked = sorted(report.mic_summaries, key=lambda s: s.score, reverse=True)
    for i, s in enumerate(ranked):
        score_style = "bold green" if i == 0 else "white"
        mic_table.add_row(
            s.mic_name,
            str(s.num_files),
            f"{s.avg_snr_db:.1f}",
            f"{s.avg_clipping_ratio * 100:.4f}%",
            str(s.total_plosive_spikes),
            f"{s.avg_spectral_rolloff_hz:.0f}",
            f"{s.avg_effective_bandwidth_hz:.0f}",
            f"{s.avg_crosstalk_ratio:.4f}",
            f"{s.avg_rms_dbfs:.1f}",
            f"[{score_style}]{s.score:.4f}[/{score_style}]",
        )

    console.print()
    console.print(mic_table)

    # ── Recommendation panel ───────────────────────────────────────
    if report.recommendation:
        console.print()
        console.print(
            Panel(
                report.recommendation,
                title="Recommendation",
                border_style="green",
                padding=(1, 2),
            )
        )

    # ── Transcriptions ─────────────────────────────────────────────
    if report.transcriptions:
        console.print()
        tx_table = Table(
            title="Transcriptions",
            show_lines=True,
            title_style="bold cyan",
        )
        tx_table.add_column("File", style="dim", max_width=40)
        tx_table.add_column("Transcript", max_width=80)

        for fpath, text in sorted(report.transcriptions.items()):
            short_path = Path(fpath).parent.parent.name + "/" + Path(fpath).name
            # Truncate long transcripts
            display_text = text[:200] + "..." if len(text) > 200 else text
            tx_table.add_row(short_path, display_text)

        console.print(tx_table)


def save_report(report: MicTestReport, output_dir: Path) -> Path:
    """Save the mic test report as JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "mic-test-report.json"
    report_path.write_text(
        report.model_dump_json(indent=2),
        encoding="utf-8",
    )
    return report_path
