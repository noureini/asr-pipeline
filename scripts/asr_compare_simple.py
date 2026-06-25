"""Simple titu vs regional-Whisper transcription comparison — NO GT.

The longform jsonl's gt_text is unreliable (mispaired), but the utt WAVS
are real audio. So we just transcribe the same wavs with both engines
and compare output quality directly: coherence, fluency, completeness,
hallucination/repetition. No ground truth needed.

Picks medium-length utterances (3-15s) so both models have enough audio
for a fair read (sub-2s clips are unfair to both).

Whisper: bare pipeline() exactly as the model card shows — NO forced
language/task (forcing language on a fine-tune can trigger empty output).
titu: per-utterance (order-safe vs NeMo's internal batch reordering).

Usage:
    uv run python scripts/asr_compare_simple.py --n 14
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S")
logger = logging.getLogger("cmp")

DATA_DIR = Path("data/w2v_train_longform")
JSONLS = [
    ("07_Shatkhira",
     DATA_DIR / "07_Shatkhira_CA-MMI_URBAN_Male_07_0202_1716.longform.jsonl"),
    ("15_Sunamganj",
     DATA_DIR / "15_Sunamganj_CA-MMI_URBAN_Male_15_0602_1635.longform.jsonl"),
]


def select(n, seed=0):
    import random
    rng = random.Random(seed)
    picks = []
    per = max(1, n // len(JSONLS))
    for group, path in JSONLS:
        recs = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                dur = r.get("end", 0) - r.get("start", 0)
                if 3.0 <= dur <= 15.0:        # fair-length only
                    recs.append(r)
        chosen = rng.sample(recs, min(per, len(recs)))
        for r in chosen:
            r["_group"] = group
        picks += chosen
    return picks


def run_titu(recs, device):
    import torch
    logger.info("Loading titu ...")
    import nemo.collections.asr as nemo_asr
    m = nemo_asr.models.ASRModel.from_pretrained(
        "hishab/titu_stt_bn_fastconformer")
    if device == "cuda" and torch.cuda.is_available():
        m = m.to(torch.device("cuda"))
    m.eval()
    out = []
    for i, r in enumerate(recs):
        o = m.transcribe([r["wav"]], batch_size=1, verbose=False)[0]
        out.append(o if isinstance(o, str) else getattr(o, "text", str(o)))
        if (i + 1) % 10 == 0:
            logger.info("  titu %d/%d", i + 1, len(recs))
    del m
    torch.cuda.empty_cache()
    return out


def run_whisper(hf_id, recs, device):
    import torch
    from transformers import pipeline
    logger.info("Loading Whisper %s (bare pipeline) ...", hf_id)
    dev = 0 if (device == "cuda" and torch.cuda.is_available()) else -1
    asr = pipeline("automatic-speech-recognition", model=hf_id,
                   device=dev, chunk_length_s=30)
    out = []
    for i, r in enumerate(recs):
        try:
            res = asr(r["wav"])           # path; pipeline loads + resamples
            out.append((res.get("text") or "").strip())
        except Exception as e:
            logger.warning("  whisper failed utt %d: %s", i, e)
            out.append(f"[ERROR: {type(e).__name__}]")
        if (i + 1) % 5 == 0:
            logger.info("  whisper %d/%d", i + 1, len(recs))
    del asr
    torch.cuda.empty_cache()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=14)
    ap.add_argument("--whisper", default=
                    "bengaliAI/tugstugi_bengaliai-regional-asr_whisper-medium")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output", type=Path,
                    default=Path("outputs/asr_compare_simple.md"))
    args = ap.parse_args()

    recs = select(args.n)
    logger.info("Selected %d utterances (3-15s)", len(recs))
    titu = run_titu(recs, args.device)
    whisper = run_whisper(args.whisper, recs, args.device)

    lines = ["# titu vs regional-Whisper — transcription quality (no GT)\n"]
    lines.append(f"Candidate Whisper: `{args.whisper}`. Same real-audio "
                 "utterances, both engines. Judge coherence, fluency, "
                 "completeness, hallucination/repetition.\n")
    for i, (r, t, w) in enumerate(zip(recs, titu, whisper), 1):
        dur = r.get("end", 0) - r.get("start", 0)
        lines.append(f"## {i}. [{r['_group']}]  ({dur:.1f}s)")
        lines.append(f"- **titu**    : {t}")
        lines.append(f"- **whisper** : {w}")
        lines.append("")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    logger.info("Report -> %s", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
