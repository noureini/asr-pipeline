"""Quantify wav<->gt_text pairing quality vs time-into-audio.

The longform eval data was built by align_mms_longform.py using a LINEAR
PROPORTIONAL assignment of transcript characters to audio windows
(cs = w*len(concat)/num_windows). That assumes constant char-rate over
the whole interview; real survey audio (silences, consent script,
untranscribed cross-talk) violates it, so text<->audio drifts with time.

This measures the damage: transcribe each utt wav with titu (order-safe,
per-utt) and compute char-error-rate between titu output and the stored
gt_text. Correctly-paired utts cluster LOW (titu's own error, ~0.1-0.3);
mispaired utts cluster HIGH (~0.7-1.0, different text entirely).

Outputs (outputs/pairing_quality.md):
  - CER distribution + bimodality
  - CER vs time-into-audio bins (the drift signature)
  - clean-subset size at CER thresholds
  - writes outputs/clean_utts.<stem>.json (idx list with CER <= thresh)
    for re-running Stage 0 / ASR eval on trustworthy pairs only

Usage:
    uv run python scripts/check_pairing_quality.py --sample 300
    uv run python scripts/check_pairing_quality.py --all   # all 2,341
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S")
logger = logging.getLogger("pairq")

DATA_DIR = Path("data/w2v_train_longform")
JSONLS = [
    ("07_Shatkhira",
     DATA_DIR / "07_Shatkhira_CA-MMI_URBAN_Male_07_0202_1716.longform.jsonl"),
    ("15_Sunamganj",
     DATA_DIR / "15_Sunamganj_CA-MMI_URBAN_Male_15_0602_1635.longform.jsonl"),
]

_BN_DIGIT = {ord(c): str(i) for i, c in enumerate("০১২৩৪৫৬৭৮৯")}
_PUNCT = re.compile(r"[।!?,.;:\"'`()\[\]{}…\-—–_/\\|*~]")


def norm(t: str) -> str:
    t = unicodedata.normalize("NFC", t or "").translate(_BN_DIGIT)
    t = _PUNCT.sub(" ", t).lower()
    return re.sub(r"\s+", "", t)        # char-level: drop spaces too


def cer(a: str, b: str) -> float:
    a, b = list(a), list(b)
    if not a and not b:
        return 0.0
    if not a or not b:
        return 1.0
    prev = list(range(len(b) + 1))
    for i, ai in enumerate(a, 1):
        cur = [i]
        for j, bj in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1,
                           prev[j - 1] + (ai != bj)))
        prev = cur
    return prev[-1] / max(len(a), len(b))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=300)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--clean-thresh", type=float, default=0.45)
    args = ap.parse_args()

    import torch
    import nemo.collections.asr as nemo_asr
    logger.info("Loading titu ...")
    model = nemo_asr.models.ASRModel.from_pretrained(
        "hishab/titu_stt_bn_fastconformer")
    if args.device == "cuda" and torch.cuda.is_available():
        model = model.to(torch.device("cuda"))
    model.eval()

    lines = ["# wav<->gt_text pairing quality (titu-vs-GT CER)\n"]
    lines.append("Low CER = titu agrees with GT = correctly paired. "
                 "High CER (~0.7-1.0) = different audio = mispaired. "
                 "Per-utt transcription (order-safe).\n")
    all_clean = {}
    for group, path in JSONLS:
        with open(path, encoding="utf-8") as f:
            recs = [json.loads(line) for line in f
                    if json.loads(line).get("gt_text", "").strip()]
        if not args.all and args.sample:
            step = max(1, len(recs) // args.sample)
            recs = recs[::step][: args.sample]
        logger.info("%s: %d utts", group, len(recs))

        cers, starts, idxs = [], [], []
        for i, r in enumerate(recs):
            try:
                out = model.transcribe([r["wav"]], batch_size=1,
                                       verbose=False)[0]
                hyp = out if isinstance(out, str) else getattr(
                    out, "text", str(out))
            except Exception:
                hyp = ""
            c = cer(norm(hyp), norm(r["gt_text"]))
            cers.append(c)
            starts.append(r.get("start", 0.0))
            idxs.append(r.get("idx", i))
            if (i + 1) % 50 == 0:
                logger.info("  %d/%d", i + 1, len(recs))

        import numpy as np
        cers = np.array(cers)
        starts = np.array(starts)
        clean_mask = cers <= args.clean_thresh
        clean_idx = [int(idxs[k]) for k in range(len(idxs))
                     if clean_mask[k]]
        all_clean[group] = clean_idx

        lines.append(f"## {group}  (n={len(recs)})")
        lines.append(f"- median CER: **{np.median(cers):.2f}** | "
                     f"clean (CER≤{args.clean_thresh}): "
                     f"**{clean_mask.mean():.0%}** "
                     f"({clean_mask.sum()}/{len(recs)})")
        # CER histogram (bimodality check)
        hist = np.histogram(cers, bins=[0, .2, .4, .6, .8, 1.01])[0]
        lines.append("- CER hist [0-.2,.2-.4,.4-.6,.6-.8,.8-1]: "
                     + " ".join(str(int(h)) for h in hist))
        # drift: CER by time-into-audio quartile
        lines.append("- CER by time-into-audio bin (drift signature):")
        order = np.argsort(starts)
        q = np.array_split(order, 5)
        for bi, qi in enumerate(q):
            if len(qi) == 0:
                continue
            t0, t1 = starts[qi].min() / 60, starts[qi].max() / 60
            lines.append(f"    {t0:5.0f}-{t1:>3.0f} min: "
                         f"median CER {np.median(cers[qi]):.2f} "
                         f"(clean {np.mean(cers[qi] <= args.clean_thresh):.0%})")
        lines.append("")

    out = Path("outputs")
    out.mkdir(exist_ok=True)
    (out / "clean_utts.json").write_text(
        json.dumps(all_clean, ensure_ascii=False), encoding="utf-8")
    lines.append("Clean-utterance idx lists -> outputs/clean_utts.json "
                 "(use to re-run Stage 0 / ASR eval on trustworthy pairs).")
    (out / "pairing_quality.md").write_text(
        "\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    logger.info("Report -> outputs/pairing_quality.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
