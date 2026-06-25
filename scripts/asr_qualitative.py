"""Qualitative ASR comparison on GT field utterances — titu vs a
candidate Whisper model, read side-by-side against ground truth.

WER is deliberately NOT used: it counts any deviation as error and
rewards deletion (the failure mode that made IndicConformer look good
while dropping content). This emits GT / titu / candidate per utterance
so quality is judged on what actually matters for survey data:
content preservation, named entities, numbers, code-switch, and
whether the model DROPS audio vs transcribes it.

Selects utterances biased toward informative ones (containing digits,
Latin code-switch, or longer spans) plus some random, so the read
surfaces the hard cases rather than trivial short ones.

Usage:
    uv run python scripts/asr_qualitative.py --n 15
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("qual")

DATA_DIR = Path("data/w2v_train_longform")
JSONLS = [
    ("07_Shatkhira",
     DATA_DIR / "07_Shatkhira_CA-MMI_URBAN_Male_07_0202_1716.longform.jsonl"),
    ("15_Sunamganj",
     DATA_DIR / "15_Sunamganj_CA-MMI_URBAN_Male_15_0602_1635.longform.jsonl"),
]

_HAS_DIGIT = re.compile(r"[০-৯0-9]")
_HAS_LATIN = re.compile(r"[A-Za-z]")


def informative_score(text: str) -> int:
    s = 0
    if _HAS_DIGIT.search(text):
        s += 2          # numbers = high-value survey content
    if _HAS_LATIN.search(text):
        s += 2          # code-switch
    s += min(len(text.split()) // 5, 3)   # longer = more to drop
    return s


def select(n: int, seed: int = 0):
    import random
    rng = random.Random(seed)
    picks = []
    per = max(1, n // len(JSONLS))
    for group, path in JSONLS:
        recs = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if r.get("gt_text", "").strip():
                    recs.append(r)
        # Select by INDEX (not object equality, which breaks on
        # duplicate-content utterances). Half informative, half random.
        order = sorted(range(len(recs)),
                       key=lambda i: -informative_score(recs[i]["gt_text"]))
        n_info = per // 2
        chosen_idx = set(order[:n_info])
        rest = [i for i in range(len(recs)) if i not in chosen_idx]
        chosen_idx |= set(rng.sample(rest, min(per - n_info, len(rest))))
        for i in sorted(chosen_idx):
            r = recs[i]
            r["_group"] = group
            picks.append(r)
    return picks


def run_whisper(hf_id, recs, device):
    import torch
    import torchaudio
    from transformers import pipeline
    logger.info("Loading Whisper %s (pipeline, per model card) ...", hf_id)
    dev = 0 if (device == "cuda" and torch.cuda.is_available()) else -1
    asr = pipeline("automatic-speech-recognition", model=hf_id, device=dev)
    gkw = {"task": "transcribe", "no_repeat_ngram_size": 3,
           "max_new_tokens": 128}
    hyps = []
    for i, r in enumerate(recs):
        wav, sr = torchaudio.load(r["wav"])
        if wav.size(0) > 1:
            wav = wav.mean(0, keepdim=True)
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
        arr = wav.squeeze(0).numpy()
        try:
            out = asr(arr, generate_kwargs={**gkw, "language": "bn"})
        except Exception:
            try:                       # some fine-tunes reject language kw
                out = asr(arr, generate_kwargs=gkw)
            except Exception as e:
                logger.warning("  whisper failed utt %d: %s", i, e)
                hyps.append("")
                continue
        hyps.append((out.get("text") or "").strip())
        if (i + 1) % 10 == 0:
            logger.info("  whisper %d/%d", i + 1, len(recs))
    del asr
    torch.cuda.empty_cache()
    return hyps


def run_titu(recs, device):
    """Per-utterance transcription — NeMo's batched transcribe() sorts
    by duration internally and does not reliably restore input order,
    which silently mis-pairs outputs to records. One-at-a-time is
    order-safe (fine for a small qualitative sample)."""
    import torch
    logger.info("Loading titu (NeMo) ...")
    import nemo.collections.asr as nemo_asr
    model = nemo_asr.models.ASRModel.from_pretrained(
        "hishab/titu_stt_bn_fastconformer")
    if device == "cuda" and torch.cuda.is_available():
        model = model.to(torch.device("cuda"))
    model.eval()
    hyps = []
    for i, r in enumerate(recs):
        out = model.transcribe([r["wav"]], batch_size=1, verbose=False)
        o = out[0]
        hyps.append(o if isinstance(o, str) else getattr(o, "text", str(o)))
        if (i + 1) % 10 == 0:
            logger.info("  titu %d/%d", i + 1, len(recs))
    del model
    torch.cuda.empty_cache()
    return hyps


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--whisper", default=
                    "bengaliAI/tugstugi_bengaliai-regional-asr_whisper-medium")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output", type=Path,
                    default=Path("outputs/asr_qualitative.md"))
    args = ap.parse_args()

    recs = select(args.n)
    logger.info("Selected %d utterances", len(recs))
    titu = run_titu(recs, args.device)
    whisper = run_whisper(args.whisper, recs, args.device)

    lines = ["# Qualitative ASR comparison — titu vs regional-Whisper\n"]
    lines.append(f"Candidate: `{args.whisper}`. {len(recs)} GT "
                 "utterances, biased toward informative ones (digits / "
                 "code-switch / longer). Judge: content preservation, "
                 "numbers, names, code-switch, drops vs transcribes — "
                 "NOT WER.\n")
    for i, (r, t, w) in enumerate(zip(recs, titu, whisper), 1):
        lines.append(f"## {i}. [{r['_group']}]  ({r['start']:.1f}–"
                     f"{r['end']:.1f}s)")
        lines.append(f"- **GT**    : {r['gt_text']}")
        lines.append(f"- **titu**  : {t}")
        lines.append(f"- **whisper**: {w}")
        lines.append("")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    logger.info("Full report: %s", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
