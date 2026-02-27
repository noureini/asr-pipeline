"""
ASR Pipeline — Production-ready multilingual automatic speech recognition.

A two-tier ASR pipeline routing high-resource languages to Whisper and
non-high-resource languages to Omnilingual ASR, with speaker diarization
via pyannote, NLLB-200 translation, and LLM post-processing.
"""

__version__ = "0.1.0"
