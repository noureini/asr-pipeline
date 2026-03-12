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
from asr_pipeline.config import AppConfig
from asr_pipeline.diarization import SpeakerDiarizer, create_diarizer
from asr_pipeline.engines.omnilingual_engine import OmnilingualEngine
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
    TranscriptMetadata,
    TranscriptResult,
    TranscriptionStyle,
)
from asr_pipeline.postprocessor import PostProcessor
from asr_pipeline.preprocessor import AudioPreprocessor

logger = logging.getLogger("asr_pipeline")


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
                with stage_log("3b", "Forced Alignment", "Refining word timestamps via wav2vec2"):
                    asr_segments = self._run_forced_alignment(
                        asr_segments, wav_path, language
                    )

                # ── Stage 5: Alignment ───────────────────────────
                with stage_log("5", "Alignment", "Merging ASR segments with speaker labels"):
                    aligned = align_segments(asr_segments, diarization_result)
                    aligned = merge_consecutive_segments(aligned)

                # Free GPU memory before loading translation/LLM models
                self._free_gpu_for_postprocessing()

                # ── Stage 6a: Translation ───────────────────────
                # ── Stage 6b: Joint Refinement ──────────────────
                with stage_log("6", "Post-Processing", "Translation → Joint Refinement"):
                    processed = self._run_postprocessing(aligned)

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

        logger.info(
            f"  Language: [bold]{lang_config.name}[/bold] ({language})"
        )
        logger.info(
            f"  Tier: [bold]{lang_config.tier.value}[/bold] → "
            f"Engine: [bold]{engine_name}[/bold]"
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
        console.rule(
            "[stage]Stages 3 & 4: Transcription + Diarization (parallel)[/stage]",
            style="cyan",
        )
        logger.info(
            f"  Transcription ({engine_name}) and diarization "
            f"running concurrently"
        )

        parallel_start = time.perf_counter()

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
            f"in {parallel_elapsed:.1f}s (parallel)"
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

        self._aligner = ForcedAligner(
            self._config.alignment,
            device=self._config.pipeline.device,
        )

        if not self._aligner.load():
            logger.warning("  Forced alignment model unavailable, skipping")
            return segments

        return self._aligner.align_segments(segments, wav_path, language)

    def _run_postprocessing(self, aligned: list) -> list:
        """Stage 6: Run post-processing (translation + optional refinement)."""
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
        else:
            engines_used.append(f"Omnilingual {self._config.omnilingual.model_card}")

        # Post-processing stages
        pp_stages = []
        backend = self._config.postprocessing.translation_backend
        if backend == "translategemma":
            pp_stages.append("TranslateGemma Translation")
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
