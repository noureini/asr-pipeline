"""
ASR engine interface and implementations.

Defines the BaseASREngine abstract base class that all ASR engines
must implement, providing a consistent interface for the pipeline
orchestrator.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from asr_pipeline.models import ASRSegment, AudioChunk


class BaseASREngine(ABC):
    """
    Abstract base class for ASR engines.

    All engines (Whisper, Omnilingual, etc.) must implement this interface
    so the pipeline can swap engines transparently based on language tier.
    """

    @abstractmethod
    def load(self) -> None:
        """Load the model into memory."""
        ...

    @abstractmethod
    def unload(self) -> None:
        """Release the model from memory."""
        ...

    @abstractmethod
    def transcribe_chunk(
        self,
        chunk: AudioChunk,
        language: str,
    ) -> list[ASRSegment]:
        """Transcribe a single audio chunk."""
        ...

    @abstractmethod
    def transcribe_batch(
        self,
        chunks: list[AudioChunk],
        language: str,
    ) -> list[ASRSegment]:
        """Transcribe a batch of audio chunks."""
        ...

    @abstractmethod
    def detect_language(
        self,
        audio_path: Path,
    ) -> tuple[str, float]:
        """Detect the language of an audio file. Returns (code, confidence)."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable engine name."""
        ...

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """Whether the model is currently loaded in memory."""
        ...
