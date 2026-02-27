"""
Entry point for running the ASR pipeline as a module.

Usage:
    python -m asr_pipeline transcribe audio.m4a --language hin
    python -m asr_pipeline list-languages
    python -m asr_pipeline check-deps
"""

from asr_pipeline.cli import main

if __name__ == "__main__":
    main()
