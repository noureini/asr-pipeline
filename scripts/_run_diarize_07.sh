#!/usr/bin/env bash
set -euo pipefail
cd /home/noureini/asr-pipeline
STEM=07_Shatkhira_CA-MMI_URBAN_Male_07_0202_1716
LOG="outputs/real_eval_inproc/$STEM/diarize.log"
uv run python scripts/diarize_parts.py --stem "$STEM" \
  --min-speakers 2 --max-speakers 4 > "$LOG" 2>&1
echo "DONE -> $LOG"
