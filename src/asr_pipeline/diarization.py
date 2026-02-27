"""
Speaker diarization module.

Provides a factory function to create the appropriate diarization
backend (pyannote or NeMo MSDD) based on configuration.

Identifies who speaks when in an audio file, independent of the
ASR transcription. Results are later merged with ASR segments
in the alignment stage.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from asr_pipeline.config import DiarizationConfig
from asr_pipeline.models import DiarizationResult, SpeakerSegment

logger = logging.getLogger("asr_pipeline")


class SpeakerDiarizer:
    """
    Speaker diarization using pyannote.audio (3.x or 4.x).

    Detects speaker turns and assigns consistent speaker IDs
    (SPEAKER_00, SPEAKER_01, ...) across the entire audio file.

    When running on pyannote 4.x with community-1, uses exclusive
    speaker diarization which assigns each frame to one speaker.
    Falls back to standard itertracks on pyannote 3.x.
    """

    def __init__(
        self,
        config: DiarizationConfig,
        device: str = "cuda",
    ) -> None:
        self._config = config
        self._device = device
        self._pipeline: Optional[object] = None

    # ─────────────────────────────────────────────────────────────────
    # Lifecycle
    # ─────────────────────────────────────────────────────────────────

    def load(self) -> None:
        """Load the pyannote diarization pipeline."""
        if self._pipeline is not None:
            logger.debug("Diarization pipeline already loaded")
            return

        import torch
        from pyannote.audio import Pipeline

        logger.info(
            f"  Loading diarization model [bold]{self._config.model}[/bold]"
        )

        pipeline_kwargs = {}
        if self._config.auth_token:
            # pyannote 3.x uses "use_auth_token", 4.x uses "token"
            try:
                import pyannote.audio
                major = int(pyannote.audio.__version__.split(".")[0])
                key = "token" if major >= 4 else "use_auth_token"
            except Exception:
                key = "use_auth_token"
            pipeline_kwargs[key] = self._config.auth_token

        # PyTorch 2.6+ defaults weights_only=True in torch.load().
        # pyannote checkpoints contain custom classes (TorchVersion,
        # Specifications, etc.) that aren't in the safe-globals allowlist.
        # Temporarily patch torch.load to allow these trusted checkpoints.
        _original_torch_load = torch.load
        torch.load = lambda *args, **kwargs: _original_torch_load(
            *args, **{**kwargs, "weights_only": False}
        )
        try:
            self._pipeline = Pipeline.from_pretrained(
                self._config.model,
                **pipeline_kwargs,
            )
        finally:
            torch.load = _original_torch_load

        # Move to device
        if self._device == "cuda" and torch.cuda.is_available():
            self._pipeline.to(torch.device("cuda"))  # type: ignore[union-attr]

        logger.info("  ✓ Diarization pipeline loaded")

    def unload(self) -> None:
        """Release diarization model from memory."""
        if self._pipeline is not None:
            del self._pipeline
            self._pipeline = None
            logger.debug("Diarization pipeline unloaded")

    # ─────────────────────────────────────────────────────────────────
    # Diarization
    # ─────────────────────────────────────────────────────────────────

    def diarize(
        self,
        audio_path: Path,
        vad_segments: Optional[list[tuple[float, float]]] = None,
    ) -> DiarizationResult:
        """
        Run speaker diarization on the full audio file.

        Args:
            audio_path: Path to the preprocessed WAV file.
            vad_segments: Pre-computed VAD speech regions (ignored by pyannote,
                         used by NeMo backend to skip its internal VAD).

        Returns:
            DiarizationResult with speaker segments and count.
        """
        if self._pipeline is None:
            raise RuntimeError("Diarization pipeline not loaded. Call load() first.")

        logger.info(f"  Running diarization on [file]{audio_path.name}[/file]")

        # Prepare pipeline parameters
        params = {}
        if self._config.min_speakers is not None:
            params["min_speakers"] = self._config.min_speakers
        if self._config.max_speakers is not None:
            params["max_speakers"] = self._config.max_speakers

        # Run diarization
        diarization_output = self._pipeline(  # type: ignore[misc]
            str(audio_path),
            **params,
        )

        # Parse pyannote output into our data model.
        # Prefer exclusive_speaker_diarization (community-1 / pyannote 4.0+)
        # which assigns each frame to exactly one speaker (no overlaps).
        # Fall back to regular itertracks for older pipelines.
        annotation = getattr(
            diarization_output, "exclusive_speaker_diarization", None
        )
        if annotation is None:
            annotation = diarization_output

        segments: list[SpeakerSegment] = []
        speakers_seen: set[str] = set()

        for turn, _, speaker in annotation.itertracks(yield_label=True):
            # Filter very short segments
            duration = turn.end - turn.start
            if duration < self._config.min_segment_duration:
                continue

            segments.append(
                SpeakerSegment(
                    speaker_id=speaker,
                    start_s=turn.start,
                    end_s=turn.end,
                )
            )
            speakers_seen.add(speaker)

        # Normalize speaker IDs to SPEAKER_00, SPEAKER_01, etc.
        speaker_map = {
            old_id: f"SPEAKER_{i:02d}"
            for i, old_id in enumerate(sorted(speakers_seen))
        }
        for seg in segments:
            seg.speaker_id = speaker_map[seg.speaker_id]

        num_speakers = len(speakers_seen)
        logger.info(
            f"  ✓ Diarization complete: {num_speakers} speakers, "
            f"{len(segments)} segments"
        )

        return DiarizationResult(
            num_speakers=num_speakers,
            segments=segments,
        )


# =============================================================================
# Factory
# =============================================================================


def create_diarizer(
    config: DiarizationConfig,
    device: str = "cuda",
    work_dir: Optional[Path] = None,
) -> SpeakerDiarizer:
    """
    Create the appropriate diarization backend based on config.

    Returns a pyannote SpeakerDiarizer (default) or NemoSpeakerDiarizer.
    Both share the same load/unload/diarize interface.
    """
    if config.backend == "nemo_msdd":
        from asr_pipeline.nemo_diarization import NemoSpeakerDiarizer

        return NemoSpeakerDiarizer(config, device=device, work_dir=work_dir)  # type: ignore[return-value]
    else:
        return SpeakerDiarizer(config, device=device)
