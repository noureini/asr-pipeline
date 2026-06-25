"""
Main pipeline orchestrator.

Coordinates all stages of the ASR pipeline in sequence:
  1. Preprocessing (VAD-guided chunking)
  2. Language detection & routing
  3+4. Transcription + Diarization (parallel)
  3b. Forced alignment (wav2vec2 word timestamp refinement)
  5. Alignment (merge ASR + speaker labels)
  6. Post-processing (correction → translation → refinement)
  7. Output formatting

Each stage logs clearly so the user always knows what is happening.
"""

from __future__ import annotations

import logging
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from asr_pipeline.alignment import align_segments, merge_consecutive_segments
from asr_pipeline.config import VALID_ASR_ENGINES, AppConfig
from asr_pipeline.diarization import SpeakerDiarizer, create_diarizer
from asr_pipeline.engines.omnilingual_engine import OmnilingualEngine
from asr_pipeline.engines.qwen_engine import QwenEngine
from asr_pipeline.engines.titu_engine import TituEngine
from asr_pipeline.engines.indic_conformer_engine import IndicConformerEngine
from asr_pipeline.engines.whisper_engine import WhisperEngine
from asr_pipeline.forced_aligner import ForcedAligner
from asr_pipeline.formatter import write_transcript
from asr_pipeline.language import LanguageRegistry, map_iso639_3_to_omnilingual
from asr_pipeline.logging_config import (
    console,
    create_progress,
    get_pipeline_stages,
    print_audio_info,
    print_completion_summary,
    print_pipeline_plan,
    print_routing_decision,
    stage_log,
)
from asr_pipeline.models import (
    ASRSegment,
    AudioChunk,
    AudioQualityMetrics,
    LanguageTier,
    NonSpeechSegment,
    ProcessedSegment,
    TranscriptMetadata,
    TranscriptResult,
    TranscriptionStyle,
)
from asr_pipeline.postprocessor import PostProcessor
from asr_pipeline.fusion import FusionTranslator
from asr_pipeline.preprocessor import AudioPreprocessor

logger = logging.getLogger("asr_pipeline")


def _set_global_seeds(seed: int) -> None:
    """Seed every RNG the pipeline touches so a run is reproducible from ONE
    knob (pipeline.seed): Python, NumPy, torch (CPU+CUDA), and CTranslate2 (the
    CT2 Whisper sampler). Deliberately does NOT enable
    torch.use_deterministic_algorithms — it errors on some NeMo/pyannote ops and
    is unnecessary (the sampling stages, now seeded, are the real variation)."""
    import random
    random.seed(seed)
    try:
        import numpy as _np
        _np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
    try:
        import ctranslate2
        ctranslate2.set_random_seed(seed)
    except Exception:
        pass


class ASRPipeline:
    """
    Main orchestrator for the multilingual ASR pipeline.

    Coordinates preprocessing, ASR, diarization, alignment,
    post-processing, and output formatting. Routes languages
    to the appropriate engine based on the two-tier system.
    """

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._language_registry = LanguageRegistry(config)

        # Work directory for intermediate files
        self._work_dir: Optional[Path] = None

        # Pipeline components (initialized lazily)
        self._preprocessor: Optional[AudioPreprocessor] = None
        self._whisper: Optional[WhisperEngine] = None
        self._omnilingual: Optional[OmnilingualEngine] = None
        self._qwen: Optional[QwenEngine] = None
        self._titu: Optional[TituEngine] = None
        self._indic_conformer: Optional[IndicConformerEngine] = None
        self._diarizer: Optional[SpeakerDiarizer] = None
        self._aligner: Optional[ForcedAligner] = None
        self._postprocessor: Optional[PostProcessor] = None

    # ─────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────

    def transcribe(
        self,
        audio_path: str | Path,
        language: str,
        output_dir: Optional[str | Path] = None,
        project_name: str = "",
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        write_output: bool = True,
    ) -> TranscriptResult:
        """
        Run the full transcription pipeline on an audio file.

        Args:
            audio_path: Path to the input audio file.
            language: ISO 639-3 language code (e.g., "hin", "spa").
            output_dir: Output directory (defaults to config).
            project_name: Optional project name for the header.
            min_speakers: Override minimum speaker count.
            max_speakers: Override maximum speaker count.
            write_output: Whether to write output files. Set to False
                for batch mode where output is written after merging.

        Returns:
            TranscriptResult with all segments and metadata.
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        output_dir = Path(output_dir or self._config.pipeline.output_dir)
        pipeline_start = time.perf_counter()

        # Reproducibility: seed every RNG from the single pipeline.seed knob.
        _set_global_seeds(self._config.pipeline.seed)

        # Override speaker counts if provided
        if min_speakers is not None:
            self._config.diarization.min_speakers = min_speakers
        if max_speakers is not None:
            self._config.diarization.max_speakers = max_speakers

        # Create temp work directory
        with tempfile.TemporaryDirectory(prefix="asr_pipeline_") as tmp:
            self._work_dir = Path(tmp)
            logger.debug(f"Work directory: {self._work_dir}")

            # Show pipeline plan (after work_dir exists)
            self._print_header(audio_path, language)

            try:
                # ── Stage 1: Preprocessing ──────────────────────
                with stage_log("1", "Preprocessing", "Audio normalization, VAD, chunking"):
                    wav_path, chunks, non_speech_regions = self._run_preprocessing(
                        audio_path
                    )
                    # Capture raw VAD timestamps for NeMo diarization
                    speech_timestamps = (
                        self._preprocessor.speech_timestamps
                        if self._preprocessor
                        else None
                    )

                # ── Stage 2: Language Detection & Routing ────────
                with stage_log("2", "Language Detection", "Routing to ASR engine"):
                    engine_name, omni_lang = self._route_language(language)

                # ── Stages 3 & 4: Transcription + Diarization (parallel) ──
                asr_segments, diarization_result = self._run_transcription_and_diarization(
                    chunks, language, engine_name, omni_lang, wav_path,
                    speech_timestamps=speech_timestamps,
                )

                # ── Stage 3b: Forced Alignment ─────────────────
                with stage_log("3b", "Word Timestamps", "titu native CTC word timestamps (forced alignment skipped)"):
                    asr_segments = self._run_forced_alignment(
                        asr_segments, wav_path, language
                    )
                    # De-dup the chunk overlap: each word in an overlap
                    # region is owned by exactly one chunk (midpoint split),
                    # so boundary words are kept once, transcribed with
                    # context. Needs the word timestamps from 3b.
                    asr_segments = self._dedup_overlap_words(asr_segments)

                # ── Stage 5: Alignment ───────────────────────────
                # Word-level speaker attribution → speaker-pure segments
                # (still labelled by diarization cluster SPEAKER_xx). The
                # cluster→role decision (ENUMERATOR / RESPONDENT) is made
                # once inside the fusion stage by an LLM classifier, where
                # Gemma is already loaded — so the raw transcript shows the
                # honest pre-LLM clusters.
                with stage_log("5", "Alignment", "Word-level speaker attribution"):
                    aligned = align_segments(asr_segments, diarization_result)
                    aligned = merge_consecutive_segments(aligned)

                # ── Stage 5b: Raw transcript snapshot ──────────
                # Save a transcript built ONLY from titu + diarization
                # (no LLM correction, no translation). Lets the reviewer
                # compare raw ASR vs. LLM-cleaned output side by side,
                # and gives a usable deliverable even if stage 6 fails
                # or is killed.
                with stage_log("5b", "Raw Output",
                               "Writing raw titu+diarization transcript"):
                    raw_processed = [
                        ProcessedSegment(
                            segment_id=s.segment_id,
                            start_s=s.start_s,
                            end_s=s.end_s,
                            speaker_id=s.speaker_id,
                            language=s.language,
                            raw_text=s.raw_text,
                            corrected_text=s.raw_text,
                            english_translation="",
                            refined_translation="",
                            confidence=s.confidence,
                        )
                        for s in aligned
                    ]
                    raw_result = self._build_result(
                        audio_path, language, engine_name, raw_processed,
                        diarization_result.num_speakers, project_name,
                        non_speech_regions,
                    )
                    if write_output:
                        raw_written = write_transcript(
                            raw_result,
                            output_dir,
                            f"{audio_path.stem}.raw",
                            output_format=self._config.output.format,
                            timestamp_fmt=self._config.output.timestamp_format,
                            include_raw=True,
                            include_translation=False,
                        )
                        for p in raw_written:
                            logger.info(f"  ✓ raw transcript: {p.name}")

                # Free GPU memory before loading translation/LLM models
                self._free_gpu_for_postprocessing()

                # ── Stage 6a: Translation ───────────────────────
                # ── Stage 6b: Joint Refinement ──────────────────
                with stage_log("6", "Post-Processing", "Translation → Joint Refinement"):
                    processed = self._run_postprocessing(aligned, wav_path)

                # ── Stage 7: Output ──────────────────────────────
                with stage_log("7", "Output", "Formatting and writing transcript"):
                    result = self._build_result(
                        audio_path, language, engine_name, processed,
                        diarization_result.num_speakers, project_name,
                        non_speech_regions,
                    )
                    written_files: list[Path] = []
                    if write_output:
                        written_files = write_transcript(
                            result,
                            output_dir,
                            audio_path.stem,
                            output_format=self._config.output.format,
                            timestamp_fmt=self._config.output.timestamp_format,
                            include_raw=self._config.output.include_raw_text,
                            include_translation=self._config.output.include_translation,
                        )
                        # Targeted human-QA worklist: only the segments
                        # the fusion flagged (witness disagreement, a
                        # number to verify, or unclear text). Lets a
                        # reviewer check ~handful of spots, not all 200+.
                        self._write_review_file(
                            result.segments, output_dir, audio_path.stem)
                        # Fusion audit trail (the <think> reasoning per
                        # segment). On by default: free at generation time
                        # and the evidence that the model is doing genuine
                        # likelihood-vs-prior reconciliation — also where
                        # the dropped/fallback segments are inspected.
                        self._write_reasoning_file(
                            result.segments, output_dir, audio_path.stem)

            finally:
                self._cleanup()

        # Summary
        total_elapsed = time.perf_counter() - pipeline_start
        metadata = self._preprocessor_metadata if hasattr(self, '_preprocessor_metadata') else None
        print_completion_summary(
            audio_file=str(audio_path),
            duration_s=result.metadata.duration_s,
            num_speakers=result.metadata.num_speakers,
            num_segments=len(result.segments),
            languages=result.metadata.languages_detected,
            output_file=str(written_files[0]) if written_files else "N/A",
            total_elapsed_s=total_elapsed,
        )

        return result

    # ─────────────────────────────────────────────────────────────────
    # Stage implementations
    # ─────────────────────────────────────────────────────────────────

    def _print_header(self, audio_path: Path, language: str) -> None:
        """Print the pipeline header with audio info and execution plan."""
        console.print()
        console.print(
            "[bold cyan]╔══════════════════════════════════════════════╗[/bold cyan]"
        )
        console.print(
            "[bold cyan]║     ASR Pipeline — Multilingual Transcriber  ║[/bold cyan]"
        )
        console.print(
            "[bold cyan]╚══════════════════════════════════════════════╝[/bold cyan]"
        )
        console.print()

        # Show audio info
        self._preprocessor = AudioPreprocessor(
            self._config.preprocessing, self._work_dir  # type: ignore[arg-type]
        )
        meta = self._preprocessor.get_audio_metadata(audio_path)
        print_audio_info(
            file_path=str(audio_path),
            duration_s=meta.duration_s,
            sample_rate=meta.sample_rate,
            channels=meta.channels,
            file_size_mb=meta.file_size_bytes / (1024 * 1024),
        )

        # Store for later
        self._preprocessor_metadata = meta

        # Language info
        lang_config = self._language_registry.get(language)
        print_routing_decision(
            language=lang_config.name,
            tier=lang_config.tier.value,
            engine=self._config.engine_for_language(language),
        )

        # Pipeline plan (adapts to active translation backend)
        active_stages = get_pipeline_stages(
            self._config.postprocessing.translation_backend
        )
        print_pipeline_plan(active_stages)

    def _run_preprocessing(
        self, audio_path: Path,
    ) -> tuple[Path, list[AudioChunk], list[tuple[float, float, str]]]:
        """Stage 1: Preprocess the audio file."""
        if self._preprocessor is None:
            self._preprocessor = AudioPreprocessor(
                self._config.preprocessing, self._work_dir  # type: ignore[arg-type]
            )
        return self._preprocessor.preprocess(audio_path)

    def _route_language(self, language: str) -> tuple[str, str]:
        """
        Stage 2: Determine which engine to use based on language tier.

        Returns:
            Tuple of (engine_name, omnilingual_lang_code).
            omnilingual_lang_code is only used if engine is omnilingual.
        """
        lang_config = self._language_registry.get(language)
        engine_name = self._config.engine_for_language(language)
        omni_lang = map_iso639_3_to_omnilingual(language, self._language_registry)

        # Guard: routing must resolve to a buildable engine. force_engine is
        # already validated at config load, but assert here too so the hardcoded
        # tier paths (and any future routing change) can never silently pick an
        # engine the pipeline can't construct — fail loudly, not mid-run.
        if engine_name not in VALID_ASR_ENGINES:
            raise ValueError(
                f"Stage 2 routing resolved language '{language}' to unknown "
                f"engine '{engine_name}'. Valid engines: "
                f"{sorted(VALID_ASR_ENGINES)}. Check pipeline.force_engine and "
                f"the tier mapping in your config."
            )

        # Why this engine? Make the routing reason explicit — never a silent
        # surprise about which model transcribed the survey.
        forced = self._config.pipeline.force_engine
        if lang_config.tier == LanguageTier.HIGH:
            reason = "tier=HIGH → whisper"
        elif forced:
            reason = f"force_engine={forced}"
        else:
            reason = "tier default → omnilingual"

        logger.info(
            f"  Language: [bold]{lang_config.name}[/bold] ({language})"
        )
        logger.info(
            f"  Tier: [bold]{lang_config.tier.value}[/bold] → "
            f"Engine: [bold]{engine_name}[/bold]  ([dim]{reason}[/dim])"
        )
        if engine_name == "omnilingual":
            logger.info(f"  Omnilingual code: {omni_lang}")

        return engine_name, omni_lang

    def _run_transcription_and_diarization(
        self,
        chunks: list[AudioChunk],
        language: str,
        engine_name: str,
        omni_lang: str,
        wav_path: Path,
        speech_timestamps: Optional[list[tuple[float, float]]] = None,
    ) -> tuple[list[ASRSegment], object]:
        """
        Stages 3 & 4: Run transcription and diarization in parallel.

        Both stages are independent — they take the preprocessed audio
        and produce separate outputs that are merged later in alignment.
        Running them concurrently saves wall-clock time equal to the
        shorter of the two stages.

        Args:
            speech_timestamps: Pre-computed VAD regions passed to NeMo
                              diarization backend (pyannote ignores these).
        """
        console.print()

        parallel_start = time.perf_counter()

        if True:  # phoneme_first engine removed \u2014 standard path only
            # \u2500\u2500 Standard: transcription + diarization in parallel \u2500\u2500
            console.rule(
                "[stage]Stages 3 & 4: Transcription + Diarization (parallel)[/stage]",
                style="cyan",
            )
            logger.info(
                f"  Transcription ({engine_name}) and diarization "
                f"running concurrently"
            )

            # We use threads (not processes) because the heavy lifting
            # happens in C/CUDA extensions that release the GIL.
            with ThreadPoolExecutor(max_workers=2) as executor:
                transcription_future = executor.submit(
                    self._run_transcription,
                    chunks, language, engine_name, omni_lang,
                    wav_path,
                )
                diarization_future = executor.submit(
                    self._run_diarization, wav_path, speech_timestamps,
                )

                # Wait for both and propagate any exceptions
                asr_segments = transcription_future.result()
                diarization_result = diarization_future.result()

        parallel_elapsed = time.perf_counter() - parallel_start
        logger.info(
            f"[success]\u2713 Stages 3 & 4 completed[/success] "
            f"in {parallel_elapsed:.1f}s"
        )

        return asr_segments, diarization_result

    def _run_transcription(
        self,
        chunks: list[AudioChunk],
        language: str,
        engine_name: str,
        omni_lang: str,
        wav_path: Optional[Path] = None,
    ) -> list[ASRSegment]:
        """Stage 3: Run ASR transcription on all chunks."""
        all_segments: list[ASRSegment] = []

        if engine_name == "whisper":
            if self._whisper is None or not self._whisper.is_loaded:
                self._whisper = WhisperEngine(
                    self._config.whisper,
                    device=self._config.pipeline.device,
                    compute_type=self._config.pipeline.compute_type,
                )
                self._whisper.load()

            if self._config.whisper.batch_inference and wav_path is not None:
                # Batched: feed full WAV to BatchedInferencePipeline
                # Pass VAD-aligned chunk boundaries as clip_timestamps.
                # BatchedInferencePipeline expects List[dict] with "start"/"end"
                # keys in seconds — NOT a flat list of floats.
                clip_timestamps: list[dict[str, float]] = [
                    {"start": chunk.start_s, "end": chunk.end_s}
                    for chunk in chunks
                ]

                logger.info(
                    f"  Using batched inference "
                    f"(batch_size={self._config.whisper.batch_size})"
                )
                progress = create_progress("Transcription")
                with progress:
                    task = progress.add_task(
                        f"Whisper ({self._config.whisper.model_size}) — batched",
                        total=None,
                    )
                    all_segments = self._whisper.transcribe_full_audio(
                        wav_path, language, clip_timestamps=clip_timestamps
                    )
                    progress.update(task, completed=len(all_segments))
            else:
                # Sequential: process pre-cut chunks one by one
                progress = create_progress("Transcription")
                with progress:
                    task = progress.add_task(
                        f"Whisper ({self._config.whisper.model_size})",
                        total=len(chunks),
                    )
                    for chunk in chunks:
                        segments = self._whisper.transcribe_chunk(chunk, language)
                        all_segments.extend(segments)
                        progress.advance(task)

        elif engine_name == "qwen":
            # ── Qwen3-ASR engine (default for non-high-resource) ──
            # VAD chunking supplies segment boundaries; this engine
            # only produces text per chunk.
            if self._qwen is None or not self._qwen.is_loaded:
                self._qwen = QwenEngine(
                    self._config.qwen,
                    device=self._config.pipeline.device,
                )
                self._qwen.load()

            logger.info(
                f"  Transcribing {len(chunks)} chunks with "
                f"{self._qwen.name}"
            )
            progress = create_progress("Transcription")
            with progress:
                task = progress.add_task(
                    "Qwen3-ASR", total=len(chunks),
                )
                for chunk in chunks:
                    segments = self._qwen.transcribe_chunk(chunk, language)
                    all_segments.extend(segments)
                    progress.advance(task)

        elif engine_name == "titu":
            # ── titu Bengali FastConformer-CTC engine ──
            # NeMo model. Batched inference is significantly faster
            # than per-chunk; we use transcribe_batch.
            if self._titu is None or not self._titu.is_loaded:
                self._titu = TituEngine(
                    device=self._config.pipeline.device,
                    batch_size=self._config.titu.batch_size,
                    model_id=self._config.titu.model_id,
                )
                self._titu.load()

            logger.info(
                f"  Transcribing {len(chunks)} chunks with "
                f"{self._titu.name}"
            )
            progress = create_progress("Transcription")
            with progress:
                task = progress.add_task(
                    "titu (FastConformer-CTC)", total=len(chunks),
                )
                # Batched: NeMo's model.transcribe handles batching
                # internally. Update progress proportionally.
                batch_size = self._config.titu.batch_size
                for i in range(0, len(chunks), batch_size):
                    batch = chunks[i:i + batch_size]
                    segs = self._titu.transcribe_batch(batch, language)
                    all_segments.extend(segs)
                    for _ in batch:
                        progress.advance(task)

        elif engine_name == "indic_conformer":
            # ── AI4Bharat IndicConformer-600M (open, local) ──
            # Loads via transformers trust_remote_code; per-utterance
            # call interface, so we loop chunks. decoder configurable
            # via pipeline.indic_conformer_decoder (ctc|rnnt).
            decoder = getattr(
                self._config.pipeline, "indic_conformer_decoder", "rnnt")
            if self._indic_conformer is None or not self._indic_conformer.is_loaded:
                self._indic_conformer = IndicConformerEngine(
                    device=self._config.pipeline.device,
                    decoder=decoder,
                )
                self._indic_conformer.load()

            logger.info(
                f"  Transcribing {len(chunks)} chunks with "
                f"{self._indic_conformer.name}"
            )
            progress = create_progress("Transcription")
            with progress:
                task = progress.add_task(
                    self._indic_conformer.name, total=len(chunks),
                )
                for chunk in chunks:
                    segs = self._indic_conformer.transcribe_chunk(
                        chunk, language)
                    all_segments.extend(segs)
                    progress.advance(task)

        else:  # omnilingual
            if self._omnilingual is None:
                self._omnilingual = OmnilingualEngine(
                    self._config.omnilingual,
                    device=self._config.pipeline.device,
                )
                self._omnilingual.load()
            elif self._omnilingual.is_loaded:
                # Model exists but may be offloaded to CPU — reload to GPU
                self._omnilingual.reload_to_gpu()
            else:
                self._omnilingual.load()

            logger.info(
                f"  Transcribing {len(chunks)} chunks with Omnilingual ASR"
            )

            progress = create_progress("Transcription")
            with progress:
                task = progress.add_task(
                    f"Omnilingual ({self._config.omnilingual.model_card})",
                    total=len(chunks),
                )
                for chunk in chunks:
                    segments = self._omnilingual.transcribe_chunk(
                        chunk, omni_lang
                    )
                    all_segments.extend(segments)
                    progress.advance(task)

        logger.info(
            f"  ✓ Transcribed {len(all_segments)} segments from "
            f"{len(chunks)} chunks"
        )

        return all_segments

    def _run_diarization(
        self,
        wav_path: Path,
        vad_segments: Optional[list[tuple[float, float]]] = None,
    ) -> object:
        """Stage 4: Run speaker diarization."""
        self._diarizer = create_diarizer(
            self._config.diarization,
            device=self._config.pipeline.device,
            work_dir=self._work_dir,
        )
        self._diarizer.load()
        return self._diarizer.diarize(wav_path, vad_segments=vad_segments)

    def _dedup_overlap_words(
        self, segments: list[ASRSegment]
    ) -> list[ASRSegment]:
        """Remove the duplicate text in overlapping chunks by word-ownership.

        Chunks overlap by ~overlap_s (set in the preprocessor) so every
        boundary word is transcribed WITH context in at least one chunk.
        Two consecutive segments that overlap in time share that region's
        words twice; assign each word to exactly one segment by splitting
        the overlap at its MIDPOINT (word center < midpoint → left segment,
        else → right), then rebuild each segment's text from its kept words.

        No-op when segments don't overlap or lack word timestamps (e.g.
        alignment was skipped) — safe to call unconditionally.
        """
        if len(segments) < 2:
            return segments
        n_trimmed = 0
        for i in range(len(segments) - 1):
            a, b = segments[i], segments[i + 1]
            if not a.words or not b.words:
                continue
            if b.start_s >= a.end_s:          # no temporal overlap
                continue
            mid = (b.start_s + a.end_s) / 2.0
            a_keep = [w for w in a.words if (w.start_s + w.end_s) / 2.0 < mid]
            b_keep = [w for w in b.words if (w.start_s + w.end_s) / 2.0 >= mid]
            # only commit if both sides keep something (never blank a segment)
            if a_keep and len(a_keep) != len(a.words):
                a.words = a_keep
                a.text = " ".join(w.word for w in a_keep).strip()
                a.end_s = a_keep[-1].end_s
                n_trimmed += 1
            if b_keep and len(b_keep) != len(b.words):
                b.words = b_keep
                b.text = " ".join(w.word for w in b_keep).strip()
                b.start_s = b_keep[0].start_s
                n_trimmed += 1
        if n_trimmed:
            logger.info("  ✓ Overlap de-dup: trimmed %d segment edges",
                        n_trimmed)
        return segments

    def _run_forced_alignment(
        self,
        segments: list[ASRSegment],
        wav_path: Path,
        language: str,
    ) -> list[ASRSegment]:
        """
        Stage 3b: Refine word-level timestamps via wav2vec2 forced alignment.

        Uses torchaudio's MMS_FA model to align transcribed words to the
        audio waveform with ~20ms precision. Skips gracefully if disabled
        or for CJK languages.
        """
        if not self._config.alignment.enabled:
            logger.info("  Forced alignment disabled in config, skipping")
            return segments

        # titu (CTC) now emits native word timestamps; when present they are
        # consistent with the text, so the MMS_FA pass is redundant AND would
        # CORRUPT them (MMS_FA is a no-op on Bengali — it strips non-Latin —
        # and would only partially align code-switched Latin words). Skip it.
        if any(s.words for s in segments):
            n = sum(1 for s in segments if s.words)
            logger.info("  Word timestamps already present (titu CTC) on "
                        "%d/%d segments — skipping MMS_FA", n, len(segments))
            return segments

        self._aligner = ForcedAligner(
            self._config.alignment,
            device=self._config.pipeline.device,
        )

        if not self._aligner.load():
            logger.warning("  Forced alignment model unavailable, skipping")
            return segments

        return self._aligner.align_segments(segments, wav_path, language)

    def _run_postprocessing(self, aligned: list, wav_path=None) -> list:
        """Stage 6: Run post-processing (translation + optional refinement)."""
        # ── Validated two-witness fusion + NMT backend ──────────────
        if self._config.postprocessing.translation_backend == "fusion":
            fc = self._config.postprocessing.fusion
            logger.info(
                "  Fusion stage: titu ‖ Whisper → Gemma-3 fusion + "
                "translation (one model, fully local, open)")
            logger.info(f"    2nd witness        : {fc.whisper_model}")
            logger.info(f"    fusion+translation : {fc.gguf_file}")
            translator = FusionTranslator(
                device=self._config.pipeline.device,
                whisper_model=fc.whisper_model,
                whisper_ct2_path=fc.whisper_ct2_path,
                gguf_repo=fc.gguf_repo,
                gguf_file=fc.gguf_file,
                gguf_n_gpu_layers=fc.gguf_n_gpu_layers,
                prompt_lookup_tokens=fc.prompt_lookup_tokens,
                whisper_batch_size=fc.whisper_batch_size,
                whisper_nbest=fc.whisper_nbest,
                temperature=fc.temperature,
                top_p=fc.top_p,
                top_k=fc.top_k,
                min_p=fc.min_p,
                repeat_penalty=fc.repeat_penalty,
                seed=fc.seed,
                n_ctx=fc.n_ctx,
                flash_attn=fc.flash_attn,
                kv_quant=fc.kv_quant,
                fusion_max_tokens=fc.fusion_max_tokens,
                fusion_think=fc.fusion_think,
                ger_guard=fc.ger_guard,
                ger_snap=fc.ger_snap,
                ger_snap_tau=fc.ger_snap_tau,
                prompts_file=fc.prompts_file,
                fusion_backend=fc.fusion_backend,
                n_parallel=fc.n_parallel,
                server_port=fc.server_port,
            )
            return translator.process(aligned, wav_path)

        self._postprocessor = PostProcessor(
            self._config.postprocessing,
            self._language_registry,
            device=self._config.pipeline.device,
        )
        self._postprocessor.load()

        # Log which stages are active
        backend = self._config.postprocessing.translation_backend
        active: list[str] = []

        if backend == "translategemma":
            tg_model = self._config.postprocessing.translategemma.model_id
            tg_quant = self._config.postprocessing.translategemma.quantize
            quant_label = f", {tg_quant}" if tg_quant else ", full precision"
            active.append(
                f"6: TranslateGemma Translation + Cleanup ({tg_model}{quant_label})"
            )
        elif backend == "qwen":
            # Local LLM (GGUF) for both correction and translation.
            # In our config this is TigerLLM-9B for Bengali ASR cleanup
            # + Bengali → English translation.
            corr = self._config.postprocessing.qwen_corrector
            trans = self._config.postprocessing.qwen_translator
            if corr.enabled:
                model_label = corr.gguf_file or corr.gguf_repo or "GGUF"
                active.append(
                    f"6a: Source Correction (local LLM: {model_label})"
                )
            if trans.enabled:
                model_label = trans.gguf_file or trans.gguf_repo or "GGUF"
                active.append(
                    f"6b: Translation (local LLM: {model_label})"
                )
        else:
            if self._config.postprocessing.translation.enabled:
                if self._config.postprocessing.translation.model_path:
                    active.append("6a: CTranslate2 NLLB Translation (batched)")
                else:
                    active.append("6a: Translation (skipped — no CT2 model path configured)")
            if self._config.postprocessing.refinement.enabled:
                if self._config.postprocessing.correction.model:
                    active.append(
                        f"6b: Ollama Joint Refinement ({self._config.postprocessing.correction.model})"
                    )
                else:
                    active.append("6b: Joint Refinement (skipped — no Ollama model configured)")

        for stage in active:
            logger.info(f"  {stage}")

        return self._postprocessor.process(aligned)

    def _build_result(
        self,
        audio_path: Path,
        language: str,
        engine_name: str,
        processed: list,
        num_speakers: int,
        project_name: str,
        non_speech_regions: list[tuple[float, float, str]],
    ) -> TranscriptResult:
        """Stage 7: Build the final transcript result."""
        lang_config = self._language_registry.get(language)
        meta = self._preprocessor_metadata

        # Collect detected languages
        languages_detected = list(
            {lang_config.name}
            | {
                self._language_registry.get(s.language).name
                for s in processed
                if s.language != language
            }
        )

        # Engines used
        engines_used = []
        if engine_name == "whisper":
            engines_used.append(f"Whisper {self._config.whisper.model_size}")
        elif engine_name == "qwen":
            engines_used.append(f"Qwen3-ASR ({self._config.qwen.model})")
        elif engine_name == "titu":
            engines_used.append("titu (FastConformer-CTC Bengali)")
        elif engine_name == "indic_conformer":
            dec = getattr(
                self._config.pipeline, "indic_conformer_decoder", "rnnt")
            engines_used.append(
                f"IndicConformer-600M ({dec.upper()}, AI4Bharat)")
        else:
            engines_used.append(f"Omnilingual {self._config.omnilingual.model_card}")

        # Post-processing stages
        pp_stages = []
        backend = self._config.postprocessing.translation_backend
        if backend == "fusion":
            fc = self._config.postprocessing.fusion
            pp_stages.append(
                f"Two-witness fusion + translation "
                f"(titu ‖ Whisper → {fc.gguf_file})")
        elif backend == "translategemma":
            pp_stages.append("TranslateGemma Translation")
        elif backend == "qwen":
            corr = self._config.postprocessing.qwen_corrector
            trans = self._config.postprocessing.qwen_translator
            if corr.enabled:
                pp_stages.append(
                    f"Source Correction ({corr.gguf_file or corr.gguf_repo or 'local LLM'})"
                )
            if trans.enabled:
                pp_stages.append(
                    f"Translation ({trans.gguf_file or trans.gguf_repo or 'local LLM'})"
                )
        else:
            if self._config.postprocessing.translation.enabled:
                pp_stages.append("NLLB-200 Translation")
            if self._config.postprocessing.refinement.enabled:
                pp_stages.append("Ollama Joint Refinement")

        # Build audio quality metrics from non-speech regions
        audio_quality = self._compute_audio_quality(
            meta.duration_s, non_speech_regions
        )

        # Build non-speech segment models
        ns_segments = [
            NonSpeechSegment(
                start_s=start,
                end_s=end,
                region_type=rtype,
                duration_s=end - start,
            )
            for start, end, rtype in non_speech_regions
        ]

        metadata = TranscriptMetadata(
            project_name=project_name,
            audio_file=str(audio_path.name),
            duration_s=meta.duration_s,
            languages_detected=languages_detected,
            num_speakers=num_speakers,
            transcription_style=TranscriptionStyle(
                self._config.output.transcription_style
            ),
            asr_engines_used=engines_used,
            postprocessing_stages=pp_stages,
            audio_quality=audio_quality,
        )

        return TranscriptResult(
            metadata=metadata,
            segments=processed,
            non_speech_segments=ns_segments,
        )

    def _compute_audio_quality(
        self,
        total_duration: float,
        non_speech_regions: list[tuple[float, float, str]],
    ) -> Optional[AudioQualityMetrics]:
        """Compute audio quality metrics from VAD results."""
        if not non_speech_regions and total_duration <= 0:
            return None

        non_speech_duration = sum(
            end - start for start, end, _ in non_speech_regions
        )
        speech_duration = max(0.0, total_duration - non_speech_duration)
        speech_ratio = speech_duration / total_duration if total_duration > 0 else 0.0

        # Count speech segments (inverse of non-speech count + 1)
        num_speech = len(non_speech_regions) + 1 if non_speech_regions else 1
        avg_speech = speech_duration / num_speech if num_speech > 0 else 0.0

        longest_silence = max(
            (end - start for start, end, _ in non_speech_regions),
            default=0.0,
        )

        return AudioQualityMetrics(
            total_duration_s=total_duration,
            speech_duration_s=round(speech_duration, 1),
            non_speech_duration_s=round(non_speech_duration, 1),
            speech_ratio=round(speech_ratio, 3),
            num_speech_segments=num_speech,
            num_non_speech_segments=len(non_speech_regions),
            avg_speech_segment_s=round(avg_speech, 1),
            longest_silence_s=round(longest_silence, 1),
        )

    # ─────────────────────────────────────────────────────────────────
    # GPU memory management
    # ─────────────────────────────────────────────────────────────────

    def _write_review_file(self, segments, output_dir, stem) -> None:
        """Write a QA worklist of only the flagged segments."""
        flagged = [s for s in segments
                   if getattr(s, "needs_review", False)]
        path = Path(output_dir) / f"{stem}.review.txt"

        def hhmmss(sec: float) -> str:
            sec = int(sec or 0)
            return f"{sec // 3600:02d}:{(sec % 3600) // 60:02d}:{sec % 60:02d}"

        lines = [
            "HUMAN-QA WORKLIST — review only these segments",
            f"{len(flagged)} of {len(segments)} segments flagged "
            f"({100 * len(flagged) // max(1, len(segments))}%)",
            "Reasons: number=verify the figure | witness-disagree=ASR "
            "engines diverged | unclear=fusion couldn't recover | "
            "empty-output=no translation | fusion-fallback=LLM gave no "
            "Bengali, FLUENT raw substituted | fusion-truncated=hit the "
            "token cap (the hard/long segments)",
            "=" * 70,
            "",
        ]
        for s in flagged:
            lines.append(f"[{hhmmss(s.start_s)}] {s.speaker_id}  "
                         f"⚠ {s.review_reason}")
            if s.raw_text:
                lines.append(f"  raw: {s.raw_text}")
            lines.append(f"  bn : {s.corrected_text}")
            lines.append(f"  en : {s.english_translation}")
            lines.append("")
        Path(path).write_text("\n".join(lines), encoding="utf-8")
        logger.info("  ✓ review worklist: %s (%d flagged)",
                    path.name, len(flagged))

    def _write_reasoning_file(self, segments, output_dir, stem) -> None:
        """Write the fusion <think> reasoning per segment — the audit
        trail for the witness-reconciliation claim. Marks fallback
        segments (LLM produced no Bengali → FLUENT raw substituted) so the
        coverage hole is inspectable, not hidden in the transcript."""
        path = Path(output_dir) / f"{stem}.reasoning.txt"

        def hhmmss(sec: float) -> str:
            sec = int(sec or 0)
            return f"{sec // 3600:02d}:{(sec % 3600) // 60:02d}:{sec % 60:02d}"

        fb = sum(1 for s in segments
                 if getattr(s, "fusion_fallback", False))
        lines = [
            "FUSION REASONING TRACE — model's <think> per segment",
            f"{len(segments)} segments | {fb} fusion-fallback "
            "(LLM empty → FLUENT raw substituted)",
            "=" * 70,
            "",
        ]
        for s in segments:
            tag = "  ⚠ FALLBACK" if getattr(s, "fusion_fallback", False) else ""
            lines.append(f"[{hhmmss(s.start_s)}] {s.speaker_id}{tag}")
            think = (getattr(s, "reasoning", "") or "").strip()
            # raw witnesses (snap audit): a term in bn/en but in NEITHER of
            # these came from the vocabulary anchor, not the audio.
            lit = (getattr(s, "witness_literal", "") or "").strip()
            flu = (getattr(s, "witness_fluent", "") or "").strip()
            if lit or flu:
                lines.append(f"  LITERAL (titu)   : {lit}")
                lines.append(f"  FLUENT (whisper) : {flu}")
            lines.append(f"  think: {think or '(model skipped <think>)'}")
            lines.append(f"  bn   : {s.corrected_text}")
            lines.append(f"  en   : {s.english_translation}")
            md = getattr(s, "ger_min_dist", None)
            if md is not None:
                ag = getattr(s, "ger_witness_agree", None)
                rr = (getattr(s, "review_reason", "") or "").strip()
                lines.append(
                    f"  ger  : min_dist={md:.3f}"
                    + (f" agree={ag:.3f}" if ag is not None else "")
                    + (f" [{rr}]" if rr else ""))
            lines.append("")
        Path(path).write_text("\n".join(lines), encoding="utf-8")
        logger.info("  ✓ reasoning trace: %s (%d fallback)",
                    path.name, fb)

    def _free_gpu_for_postprocessing(self) -> None:
        """
        Unload ASR/diarization/alignment models before post-processing.

        Whisper large-v3 alone uses ~3-4 GB VRAM. Freeing it before loading
        CTranslate2 NLLB prevents CUDA OOM errors during translation.
        On small GPUs (e.g., 6 GB), aggressive cleanup is essential.
        """
        import gc

        if self._whisper is not None:
            self._whisper.unload()
            self._whisper = None
        if self._titu is not None:
            # titu (NeMo FastConformer) is the primary engine on this
            # path and holds a large CUDA workspace. Not freeing it left
            # the 8GB card full → the fusion Whisper OOM'd even one clip.
            self._titu.unload()
            self._titu = None
        if self._omnilingual is not None:
            # Offload to CPU instead of destroying — avoids fairseq2
            # thread-local gang context corruption on reload.
            self._omnilingual.offload_to_cpu()
        if self._diarizer is not None:
            self._diarizer.unload()
            self._diarizer = None
        if self._aligner is not None:
            self._aligner.unload()
            self._aligner = None

        # Aggressive cleanup: multiple gc passes + CUDA cache clear
        gc.collect()
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except ImportError:
            pass

        logger.debug("Freed GPU memory for post-processing stage")

    # ─────────────────────────────────────────────────────────────────
    # Cleanup
    # ─────────────────────────────────────────────────────────────────

    def _cleanup(self) -> None:
        """Release post-processing models from memory.

        ASR and diarization models are kept loaded between batch files
        to avoid fairseq2 thread-local gang context corruption. They
        are freed explicitly by _free_gpu_for_postprocessing() within
        each file's pipeline run.
        """
        if self._diarizer is not None:
            self._diarizer.unload()
            self._diarizer = None
        if self._aligner is not None:
            self._aligner.unload()
            self._aligner = None
        if self._postprocessor is not None:
            self._postprocessor.unload()
            self._postprocessor = None
        logger.debug("Post-processing models unloaded from memory")

    def cleanup_all(self) -> None:
        """Release ALL models from memory (call at end of batch)."""
        self._cleanup()
        if self._whisper is not None:
            self._whisper.unload()
            self._whisper = None
        if self._omnilingual is not None:
            self._omnilingual.unload()
            self._omnilingual = None
        logger.debug("All models unloaded from memory")
