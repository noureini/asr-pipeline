"""ASR WER/CER bake-off on GT-transcribed field utterances.

Quantitative head-to-head between ASR engines on the existing
GT-aligned asset (data/w2v_train_longform/), no full pipeline needed:
2,341 short utterances with ground-truth text from two interviews
(07_Shatkhira, 15_Sunamganj) of regional coastal-Bangladesh Bengali.

Backends:
  whisper_hf  : any HF Whisper checkpoint (e.g. the regional fine-tune
                bengaliAI/tugstugi_bengaliai-regional-asr_whisper-medium)
  titu        : hishab/titu_stt_bn_fastconformer (the current engine)

Metrics: token-WER and char-CER on normalized text. Normalization
strips punctuation/danda, collapses whitespace, lowercases Latin
code-switch, and unifies Bengali/ASCII digits — so neither model is
penalized for punctuation or digit-script choices. Code-switched
English words are KEPT (both should produce them).

Usage:
    uv pip install jiwer        # optional; falls back to builtin WER
    uv run python scripts/asr_wer_bakeoff.py \
        --models whisper_regional,titu --sample 200
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
import unicodedata
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("wer")

DATA_DIR = Path("data/w2v_train_longform")
JSONLS = [
    ("07_Shatkhira",
     DATA_DIR / "07_Shatkhira_CA-MMI_URBAN_Male_07_0202_1716.longform.jsonl"),
    ("15_Sunamganj",
     DATA_DIR / "15_Sunamganj_CA-MMI_URBAN_Male_15_0602_1635.longform.jsonl"),
]

# Known model specs (label -> (backend, hf_id_or_none))
MODEL_SPECS = {
    "whisper_regional": (
        "whisper_hf",
        "bengaliAI/tugstugi_bengaliai-regional-asr_whisper-medium"),
    "titu": ("titu", "hishab/titu_stt_bn_fastconformer"),
}

_BN_DIGIT = {ord(c): str(i) for i, c in enumerate("০১২৩৪৫৬৭৮৯")}
_PUNCT = re.compile(r"[।!?,.;:\"'`()\[\]{}…\-—–_/\\|*~]")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "")
    text = text.translate(_BN_DIGIT)          # ০-৯ -> 0-9
    text = _PUNCT.sub(" ", text)
    text = text.lower()                        # Latin code-switch
    return re.sub(r"\s+", " ", text).strip()


def edit_distance(a: list, b: list) -> int:
    """Levenshtein over token (or char) sequences."""
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ai in enumerate(a, 1):
        cur = [i]
        for j, bj in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1,
                           prev[j - 1] + (ai != bj)))
        prev = cur
    return prev[-1]


def score(refs: list[str], hyps: list[str]) -> tuple[float, float]:
    """Aggregate WER (token) and CER (char) over a corpus."""
    w_err = w_tot = c_err = c_tot = 0
    for r, h in zip(refs, hyps):
        rt, ht = r.split(), h.split()
        w_err += edit_distance(rt, ht)
        w_tot += len(rt)
        rc, hc = list(r.replace(" ", "")), list(h.replace(" ", ""))
        c_err += edit_distance(rc, hc)
        c_tot += len(rc)
    return (w_err / max(1, w_tot), c_err / max(1, c_tot))


def load_utts(sample: int, seed: int = 0):
    import random
    rng = random.Random(seed)
    groups = {}
    for group, path in JSONLS:
        with open(path, encoding="utf-8") as f:
            recs = [json.loads(line) for line in f]
        recs = [r for r in recs if r.get("gt_text", "").strip()]
        if sample:
            per = sample // len(JSONLS)
            recs = recs[:: max(1, len(recs) // per)][:per]
        groups[group] = recs
    return groups


def run_whisper(hf_id, recs, device):
    import torch
    import torchaudio
    from transformers import (WhisperProcessor,
                              WhisperForConditionalGeneration)
    logger.info("Loading Whisper %s ...", hf_id)
    proc = WhisperProcessor.from_pretrained(hf_id)
    model = WhisperForConditionalGeneration.from_pretrained(
        hf_id).to(device).eval()
    try:
        forced = proc.get_decoder_prompt_ids(
            language="bengali", task="transcribe")
    except Exception:
        forced = None
    hyps = []
    for i, r in enumerate(recs):
        wav, sr = torchaudio.load(r["wav"])
        if wav.size(0) > 1:
            wav = wav.mean(0, keepdim=True)
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
        feats = proc(wav.squeeze(0).numpy(), sampling_rate=16000,
                     return_tensors="pt").input_features.to(device)
        with torch.inference_mode():
            ids = model.generate(
                feats, forced_decoder_ids=forced,
                max_new_tokens=128, num_beams=1,
                no_repeat_ngram_size=3)
        hyps.append(proc.batch_decode(ids, skip_special_tokens=True)[0])
        if (i + 1) % 50 == 0:
            logger.info("  whisper %d/%d", i + 1, len(recs))
    del model
    torch.cuda.empty_cache()
    return hyps


def run_titu(recs, device):
    import torch
    logger.info("Loading titu (NeMo) ...")
    import nemo.collections.asr as nemo_asr
    model = nemo_asr.models.ASRModel.from_pretrained(
        "hishab/titu_stt_bn_fastconformer")
    if device == "cuda" and torch.cuda.is_available():
        model = model.to(torch.device("cuda"))
    model.eval()
    paths = [r["wav"] for r in recs]
    outs = model.transcribe(paths, batch_size=8)
    hyps = []
    for o in outs:
        hyps.append(o if isinstance(o, str) else getattr(o, "text", str(o)))
    del model
    torch.cuda.empty_cache()
    return hyps


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="whisper_regional,titu",
                    help="comma list from: " + ", ".join(MODEL_SPECS))
    ap.add_argument("--sample", type=int, default=200,
                    help="utterances total (0 = all 2,341)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output", type=Path,
                    default=Path("outputs/asr_wer_bakeoff.md"))
    args = ap.parse_args()

    groups = load_utts(args.sample)
    n_total = sum(len(v) for v in groups.values())
    logger.info("Evaluating on %d utterances (%s)", n_total,
                ", ".join(f"{g}:{len(v)}" for g, v in groups.items()))

    results = {}            # label -> {group -> (wer, cer), overall, secs}
    for label in [m.strip() for m in args.models.split(",")]:
        if label not in MODEL_SPECS:
            logger.warning("unknown model %s, skipping", label)
            continue
        backend, hf_id = MODEL_SPECS[label]
        t0 = time.perf_counter()
        per_group = {}
        all_ref, all_hyp = [], []
        for group, recs in groups.items():
            try:
                if backend == "whisper_hf":
                    hyps = run_whisper(hf_id, recs, args.device)
                else:
                    hyps = run_titu(recs, args.device)
            except Exception as e:
                logger.error("  %s failed on %s: %s", label, group, e)
                hyps = [""] * len(recs)
            refs = [normalize(r["gt_text"]) for r in recs]
            hyps = [normalize(h) for h in hyps]
            per_group[group] = score(refs, hyps)
            all_ref += refs
            all_hyp += hyps
        results[label] = dict(
            per_group=per_group,
            overall=score(all_ref, all_hyp),
            secs=time.perf_counter() - t0,
            sample_hyps=list(zip(all_ref[:8], all_hyp[:8])))

    # ── report ──
    lines = ["# ASR WER/CER bake-off on GT field utterances\n"]
    lines.append(f"Eval set: **{n_total}** GT-transcribed utterances "
                 "(regional coastal-Bangladesh Bengali). Normalized "
                 "(punct/danda stripped, digits unified, Latin "
                 "lowercased; code-switch kept). Lower = better.\n")
    lines.append("| model | overall WER | overall CER | "
                 "Shatkhira WER | Sunamganj WER | sec |")
    lines.append("|---|---|---|---|---|---|")
    for label, r in results.items():
        ow, oc = r["overall"]
        sw = r["per_group"].get("07_Shatkhira", (float("nan"),) * 2)[0]
        nw = r["per_group"].get("15_Sunamganj", (float("nan"),) * 2)[0]
        lines.append(f"| {label} | {ow:.3f} | {oc:.3f} | "
                     f"{sw:.3f} | {nw:.3f} | {r['secs']:.0f} |")

    for label, r in results.items():
        lines.append(f"\n## {label} — first 8 (ref → hyp)")
        for ref, hyp in r["sample_hyps"]:
            lines.append(f"- ref: {ref}")
            lines.append(f"  hyp: {hyp}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:6 + 2 * len(results)]))
    logger.info("Full report: %s", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
