#!/usr/bin/env bash
# End-to-end interview processing using the EXISTING project pipeline:
#   diarize_parts.py     (pyannote, working torch.load patch)
#   assemble_transcript.py (formatter.write_transcript → TXT/JSON/SRT)
#
# Produces TWO deliverable sets per interview:
#   outputs/v1_deliverable/<stem>/raw/             — Pipeline 1 (titu + diar)
#   outputs/v1_deliverable/<stem>/cleaned_tigerllm/ — Pipeline 2A
#
# Pipeline 2B (Claude web app): user pastes Pipeline 1 transcript.txt
#                               into Claude.ai inside the secure infra.
#
# Usage:
#   bash scripts/run_interview_pipeline.sh \
#     15_Sunamganj_CA-MMI_URBAN_Male_15_0602_1635 \
#     "EXCELLENT (SNR 27 dB)"

set -euo pipefail
# Silence the torchaudio/pyannote/speechbrain deprecation warnings
# that flood the output. These are informational (about the 2.9
# release migration to torchcodec) and don't affect functionality.
export PYTHONWARNINGS="ignore::UserWarning:torchaudio,ignore::UserWarning:pyannote,ignore::UserWarning:speechbrain,ignore::UserWarning:torio,ignore::FutureWarning"

STEM="${1:?usage: $0 <STEM> [quality_string]}"
QUALITY="${2:-}"

ROOT="$(dirname "$(realpath "$0")")/.."
cd "$ROOT"

# titu data lives here (outputs of dump_vad_chunks.py +
# titu_transcribe_chunks.py)
WORK_DIR="outputs/real_eval_titu_excellent"
OUT_BASE="outputs/v1_deliverable/$STEM"
CLEANUP_DIR="data/p2g/cleanup_${STEM}__tigerllm_9b_it"
TRANSLATE_DIR="data/p2g/translated_${STEM}__tigerllm_9b_it"

echo "================================================================"
echo "Interview: $STEM"
echo "Quality:   $QUALITY"
echo "================================================================"

# -1. BUILD SURVEY WIKI (one-time per project, but cheap to redo)
echo
echo "[-1/6] Building/refreshing survey wiki from instrument"
if [[ -f "for testing (Noureini)/survey_instrument.docx" ]]; then
  uv run python scripts/build_survey_wiki.py
else
  echo "  NOTE: no survey_instrument.docx found; cleanup will run"
  echo "        without domain vocabulary anchor (lower quality)"
fi

# 0. PREPARE SOURCE FOLDER (audio copies + GT docx text)
echo
echo "[0/6] Preparing source/ (audio + GT docx + quality summary)"
uv run python scripts/prepare_deliverable.py --stem "$STEM"

# 1. SPEAKER DIARIZATION — existing diarize_parts.py
#    Writes <partstem>.diar.jsonl alongside chunks.jsonl
echo
echo "[1/6] Speaker diarization (pyannote — existing diarize_parts.py)"
uv run python scripts/diarize_parts.py \
  --stem "$STEM" \
  --out-dir "$WORK_DIR"

# 1b. VERIFY ENUMERATOR ATTRIBUTION via consent keywords
echo
echo "[1b/6] Verifying ENUMERATOR attribution against consent script"
uv run python scripts/verify_speaker_attribution.py \
  --stem "$STEM" \
  --titu-root "$WORK_DIR" \
  || echo "  (verification produced warnings; spot-check recommended)"

# 1c. WORD-LEVEL FORCED ALIGNMENT + PER-WORD SPEAKER ATTRIBUTION
#     (best practice: WhisperX-style)
#     Produces <partstem>.aligned_segments.jsonl with clean
#     speaker-pure segments split at exact word boundaries.
echo
echo "[1c/6] Word-level alignment + per-word speaker attribution (MMS_FA)"
uv run python scripts/word_align_and_split.py \
  --stem "$STEM"

# 2. ASSEMBLE RAW TRANSCRIPT — existing assemble_transcript.py
#    Uses formatter.write_transcript → TXT + JSON + SRT
#    Writes alongside the chunks/diar jsonl, then we copy to the
#    deliverable folder.
echo
echo "[2/6] Assembling raw transcript (Pipeline 1 deliverable, word-aligned)"
# Ensure no stale corrected/translated jsonl from a previous run
rm -f "$WORK_DIR/$STEM"/*.corrected.jsonl
rm -f "$WORK_DIR/$STEM"/*.translated.jsonl
uv run python scripts/assemble_aligned.py \
  --stem "$STEM" \
  --out-dir "$WORK_DIR" \
  --asr-model titu_stt_bn_fastconformer
mkdir -p "$OUT_BASE/raw"
for ext in txt json srt; do
  cp "$WORK_DIR/$STEM/$STEM.$ext" "$OUT_BASE/raw/transcript.$ext" \
    2>/dev/null || true
done
echo "  -> $OUT_BASE/raw/"

# 3. LLM CLEANUP — TigerLLM (our new script)
echo
echo "[3/6] LLM cleanup (TigerLLM)"
uv run python scripts/cleanup_titu.py \
  --stem "$STEM" \
  --backend tiger \
  --gguf-repo mradermacher/TigerLLM-9B-it-GGUF \
  --gguf-file TigerLLM-9B-it.Q4_K_M.gguf \
  --titu-suffix .chunks.jsonl

# 4. TRANSLATION — TigerLLM
echo
echo "[4/6] Bengali → English translation (TigerLLM)"
uv run python scripts/translate_chunks.py \
  --stem "$STEM" \
  --cleaned-dir "$CLEANUP_DIR" \
  --out-dir "$TRANSLATE_DIR"

# 5. CONVERT cleanup+translate outputs into the format
#    assemble_transcript.py reads (*.corrected.jsonl + *.translated.jsonl)
echo
echo "[5/6] Converting cleanup + translation to assemble format"
uv run python scripts/to_assemble_format.py \
  --stem "$STEM" \
  --cleaned-dir "$CLEANUP_DIR" \
  --translated-dir "$TRANSLATE_DIR" \
  --out-dir "$WORK_DIR"

# 6. ASSEMBLE CLEANED + TRANSLATED — re-run existing assembler
#    Now it picks up the *.corrected.jsonl and *.translated.jsonl too
echo
echo "[6/6] Assembling cleaned + translated transcript (Pipeline 2A, word-aligned)"
uv run python scripts/assemble_aligned.py \
  --stem "$STEM" \
  --out-dir "$WORK_DIR" \
  --asr-model titu_stt_bn_fastconformer \
  --use-cleaned
mkdir -p "$OUT_BASE/cleaned_tigerllm"
for ext in txt json srt; do
  cp "$WORK_DIR/$STEM/$STEM.$ext" \
    "$OUT_BASE/cleaned_tigerllm/transcript.$ext" 2>/dev/null || true
done
echo "  -> $OUT_BASE/cleaned_tigerllm/"

echo
echo "================================================================"
echo "DONE: $STEM"
echo "  Pipeline 1 (raw)            -> $OUT_BASE/raw/"
echo "  Pipeline 2A (TigerLLM)      -> $OUT_BASE/cleaned_tigerllm/"
echo "  Pipeline 2B (Claude web)    -> paste $OUT_BASE/raw/transcript.txt"
echo "================================================================"
