"""
Speaker diarization using NVIDIA NeMo.

Alternative backend to pyannote, using NeMo's ClusteringDiarizer
or NeuralDiarizer (MSDD) with TitaNet speaker embeddings.

NeMo is an optional dependency — only imported when backend="nemo_msdd".
When Silero VAD segments are provided, NeMo skips its own MarbleNet VAD
and uses the pre-computed speech regions directly.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from asr_pipeline.config import DiarizationConfig
from asr_pipeline.models import DiarizationResult, SpeakerSegment

logger = logging.getLogger("asr_pipeline")


class NemoSpeakerDiarizer:
    """
    Speaker diarization using NVIDIA NeMo (TitaNet + Clustering + optional MSDD).

    Pipeline:
        External Silero VAD → TitaNet embeddings → Clustering → [MSDD refinement]

    When vad_segments are provided, writes them as RTTM and skips NeMo's
    internal MarbleNet VAD entirely, avoiding redundant computation.
    """

    def __init__(
        self,
        config: DiarizationConfig,
        device: str = "cuda",
        work_dir: Optional[Path] = None,
    ) -> None:
        self._config = config
        self._device = device
        self._work_dir = work_dir or Path("/tmp/nemo_diar")
        self._diarizer: Optional[object] = None

    # ─────────────────────────────────────────────────────────────────
    # Lifecycle
    # ─────────────────────────────────────────────────────────────────

    def load(self) -> None:
        """Validate NeMo availability. Actual model init happens in diarize()."""
        try:
            from nemo.collections.asr.models import ClusteringDiarizer  # noqa: F401

            logger.info("  NeMo diarization backend available")
        except ImportError:
            raise ImportError(
                "NeMo toolkit not installed. Install with:\n"
                "  pip install nemo_toolkit[asr]\n"
                "Or add the [nemo] extra: pip install -e '.[nemo]'"
            )

    def unload(self) -> None:
        """Release NeMo diarizer from memory."""
        if self._diarizer is not None:
            del self._diarizer
            self._diarizer = None
            logger.debug("NeMo diarizer unloaded")

    # ─────────────────────────────────────────────────────────────────
    # Main diarization entry point
    # ─────────────────────────────────────────────────────────────────

    def diarize(
        self,
        audio_path: Path,
        vad_segments: Optional[list[tuple[float, float]]] = None,
    ) -> DiarizationResult:
        """
        Run NeMo speaker diarization.

        Args:
            audio_path: Path to 16kHz mono WAV file.
            vad_segments: Pre-computed VAD speech regions from Silero.
                         If provided, NeMo skips its own MarbleNet VAD.

        Returns:
            DiarizationResult with speaker segments and count.
        """
        from omegaconf import OmegaConf

        # Set up working directories
        nemo_dir = self._work_dir / "nemo_diar"
        nemo_dir.mkdir(parents=True, exist_ok=True)
        out_dir = nemo_dir / "output"
        out_dir.mkdir(exist_ok=True)

        # Step 1: Write NeMo input manifest
        vad_rttm_path = None
        if vad_segments:
            vad_rttm_path = nemo_dir / "external_vad.rttm"
            self._write_vad_rttm(audio_path.stem, vad_segments, vad_rttm_path)
            logger.info(
                f"  Using external Silero VAD ({len(vad_segments)} segments)"
            )

        manifest_path = nemo_dir / "input_manifest.json"
        self._write_manifest(audio_path, manifest_path, vad_rttm_path)

        # Step 2: Build NeMo config
        cfg = self._build_nemo_config(manifest_path, out_dir, vad_rttm_path)
        cfg = OmegaConf.create(cfg)

        # Step 3: Run diarization
        nemo_cfg = self._config.nemo
        if nemo_cfg.use_msdd:
            from nemo.collections.asr.models.msdd_models import NeuralDiarizer

            logger.info("  Running NeMo NeuralDiarizer (MSDD, overlap-aware)")
            diarizer = NeuralDiarizer(cfg=cfg)
        else:
            from nemo.collections.asr.models import ClusteringDiarizer

            logger.info("  Running NeMo ClusteringDiarizer (TitaNet)")
            diarizer = ClusteringDiarizer(cfg=cfg)

        self._diarizer = diarizer
        diarizer.diarize()

        # Step 4: Parse output RTTM
        rttm_path = out_dir / "pred_rttms" / f"{audio_path.stem}.rttm"
        result = self._parse_rttm(rttm_path)

        logger.info(
            f"  \u2713 NeMo diarization complete: {result.num_speakers} speakers, "
            f"{len(result.segments)} segments"
        )

        return result

    # ─────────────────────────────────────────────────────────────────
    # NeMo I/O helpers
    # ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _write_manifest(
        audio_path: Path,
        manifest_path: Path,
        vad_rttm_path: Optional[Path] = None,
    ) -> None:
        """Write NeMo-format input manifest (JSON lines)."""
        import torchaudio

        info = torchaudio.info(str(audio_path))
        duration = info.num_frames / info.sample_rate

        entry = {
            "audio_filepath": str(audio_path),
            "offset": 0,
            "duration": duration,
            "label": "infer",
            "text": "-",
            "rttm_filepath": str(vad_rttm_path) if vad_rttm_path else None,
            "uem_filepath": None,
        }
        with open(manifest_path, "w") as f:
            f.write(json.dumps(entry) + "\n")

    @staticmethod
    def _write_vad_rttm(
        file_id: str,
        vad_segments: list[tuple[float, float]],
        rttm_path: Path,
    ) -> None:
        """
        Convert Silero VAD segments to RTTM format for NeMo.

        RTTM format:
            SPEAKER <file_id> 1 <start_s> <duration_s> <NA> <NA> <label> <NA> <NA>
        """
        with open(rttm_path, "w") as f:
            for start_s, end_s in vad_segments:
                duration = end_s - start_s
                if duration > 0:
                    f.write(
                        f"SPEAKER {file_id} 1 {start_s:.3f} {duration:.3f} "
                        f"<NA> <NA> speech <NA> <NA>\n"
                    )

    def _build_nemo_config(
        self,
        manifest_path: Path,
        out_dir: Path,
        vad_rttm_path: Optional[Path],
    ) -> dict:
        """Build NeMo OmegaConf-compatible config dict."""
        nemo_cfg = self._config.nemo
        use_oracle_vad = vad_rttm_path is not None

        cfg: dict = {
            # Top-level keys required by ClusteringDiarizer.__init__
            "device": self._device if self._device != "cuda" else None,
            "sample_rate": 16000,
            "batch_size": 64,
            "num_workers": 0,
            "verbose": True,
            "diarizer": {
                "manifest_filepath": str(manifest_path),
                "out_dir": str(out_dir),
                "oracle_vad": use_oracle_vad,
                "collar": 0.25,
                "ignore_overlap": not nemo_cfg.use_msdd,
                "vad": {
                    "model_path": "vad_multilingual_marblenet",
                    "external_vad_manifest": None,
                    "parameters": {
                        "window_length_in_sec": 0.31,
                        "shift_length_in_sec": 0.01,
                        "smoothing": "median",
                        "overlap": 0.5,
                        "onset": 0.8,
                        "offset": 0.6,
                        "pad_offset": -0.05,
                        "min_duration_on": 0.2,
                        "min_duration_off": 0.2,
                    },
                },
                "speaker_embeddings": {
                    "model_path": nemo_cfg.speaker_embeddings_model,
                    "parameters": {
                        "window_length_in_sec": [1.5, 1.25, 1.0, 0.75, 0.5],
                        "shift_length_in_sec": [0.75, 0.625, 0.5, 0.375, 0.25],
                        "multiscale_weights": [1, 1, 1, 1, 1],
                        "save_embeddings": False,
                    },
                },
                "clustering": {
                    "parameters": {
                        "oracle_num_speakers": (
                            self._config.min_speakers
                            if self._config.min_speakers == self._config.max_speakers
                            and self._config.min_speakers is not None
                            else None
                        ),
                        "max_num_speakers": self._config.max_speakers or 8,
                        "enhanced_count_thres": 80,
                        "max_rp_threshold": nemo_cfg.max_rp_threshold,
                        "sparse_search_volume": nemo_cfg.sparse_search_volume,
                    },
                },
            },
        }

        # Add MSDD config when enabled
        if nemo_cfg.use_msdd:
            cfg["diarizer"]["msdd_model"] = {
                "model_path": "diar_msdd_telephonic",
                "parameters": {
                    "use_speaker_model_from_ckpt": True,
                    "infer_batch_size": 25,
                    "sigmoid_threshold": [0.7],
                    "seq_eval_mode": False,
                    "split_infer": True,
                    "diar_window_length": 50,
                    "overlap_infer_spk_limit": 5,
                },
            }

        return cfg

    def _parse_rttm(self, rttm_path: Path) -> DiarizationResult:
        """
        Parse NeMo RTTM output into DiarizationResult.

        RTTM line format:
            SPEAKER <file_id> 1 <start_s> <duration_s> <NA> <NA> <speaker_id> <NA> <NA>
        """
        segments: list[SpeakerSegment] = []
        speakers_seen: set[str] = set()

        if not rttm_path.exists():
            logger.warning(f"  NeMo RTTM output not found at {rttm_path}")
            return DiarizationResult(num_speakers=0, segments=[])

        with open(rttm_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 8 or parts[0] != "SPEAKER":
                    continue

                start_s = float(parts[3])
                duration = float(parts[4])
                end_s = start_s + duration
                speaker_id = parts[7]

                if duration < self._config.min_segment_duration:
                    continue

                segments.append(
                    SpeakerSegment(
                        speaker_id=speaker_id,
                        start_s=start_s,
                        end_s=end_s,
                    )
                )
                speakers_seen.add(speaker_id)

        # Normalize to SPEAKER_00, SPEAKER_01, ...
        speaker_map = {
            old_id: f"SPEAKER_{i:02d}"
            for i, old_id in enumerate(sorted(speakers_seen))
        }
        for seg in segments:
            seg.speaker_id = speaker_map[seg.speaker_id]

        # Sort by start time
        segments.sort(key=lambda s: s.start_s)

        return DiarizationResult(
            num_speakers=len(speakers_seen),
            segments=segments,
        )
