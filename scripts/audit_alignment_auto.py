"""Programmatically verify long-form MMS-FA alignment quality without
needing Bengali knowledge.

Two checks per utterance:

  1. Statistical: words/sec and chars/sec must fall in plausible
     Bengali speech rate ranges. Bad ranges almost certainly mean
     audio doesn't match GT.

  2. Whisper-small oracle: transcribe the audio with a known-good
     multilingual ASR (Whisper-small supports Bengali), then compute
     character-level overlap between Whisper's hypothesis and the
     GT text. High overlap = alignment correct. Near-zero overlap =
     audio doesn't contain the GT text.

The verdict at the end tells you whether alignment is trustworthy
for SLAM-ASR training. No human listening or Bengali required.

Usage:
  uv run python scripts/audit_alignment_auto.py \\
      --manifest data/w2v_train_longform/07_*.longform.jsonl
"""
from __future__ import annotations
import argparse, json, random, sys
from pathlib import Path
from collections import Counter

REPO = Path(__file__).resolve().parents[1]


def char_overlap(a: str, b: str) -> float:
    """Fraction of GT chars (a) that appear in same-frequency window in
    hyp (b). Multiset Jaccard-like: |intersect| / |a|."""
    if not a:
        return 0.0
    ca = Counter(a.replace(" ", ""))
    cb = Counter(b.replace(" ", ""))
    inter = sum((ca & cb).values())
    return inter / max(1, sum(ca.values()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--n", type=int, default=20,
                    help="number of utterances to audit with Whisper")
    ap.add_argument("--skip-whisper", action="store_true",
                    help="only run statistical check")
    ap.add_argument("--whisper-model", default="openai/whisper-small")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not args.manifest.exists():
        sys.exit(f"missing manifest: {args.manifest}")

    rows = [json.loads(l) for l in
            args.manifest.read_text(encoding="utf-8").splitlines()
            if l.strip()]
    print(f"loaded {len(rows)} utts from {args.manifest.name}\n")

    # --- CHECK 1: statistical -----------------------------------------
    print("=" * 70)
    print("CHECK 1 — statistical sanity (words/sec + chars/sec)")
    print("=" * 70)
    plausible_wps = (1.5, 6.0)     # Bengali ~3-5 typical
    plausible_cps = (8.0, 30.0)    # chars per sec
    flags = 0
    for r in rows:
        dur = r["end"] - r["start"]
        if dur <= 0:
            continue
        wps = len(r["gt_text"].split()) / dur
        cps = len(r["gt_text"]) / dur
        is_bad = not (plausible_wps[0] <= wps <= plausible_wps[1]
                       and plausible_cps[0] <= cps <= plausible_cps[1])
        if is_bad:
            flags += 1
    pct = 100 * flags / max(1, len(rows))
    verdict_stat = "OK" if pct < 10 else ("SUSPICIOUS" if pct < 30
                                           else "BROKEN")
    print(f"  {flags}/{len(rows)} utts ({pct:.1f}%) outside plausible "
          f"speech rates")
    print(f"  → statistical verdict: {verdict_stat}\n")

    if args.skip_whisper:
        return

    # --- CHECK 2: Whisper oracle --------------------------------------
    print("=" * 70)
    print(f"CHECK 2 — {args.whisper_model} oracle on {args.n} random utts")
    print("=" * 70)
    print("loading Whisper (~1 GB download, cached after first run) ...",
          flush=True)
    import torch, torchaudio
    from transformers import (WhisperForConditionalGeneration,
                              WhisperProcessor)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    proc = WhisperProcessor.from_pretrained(args.whisper_model)
    mdl = WhisperForConditionalGeneration.from_pretrained(
        args.whisper_model, torch_dtype=torch.float16
        if device.type == "cuda" else torch.float32).to(device).eval()
    forced = proc.get_decoder_prompt_ids(language="bn",
                                          task="transcribe")
    print(f"  Whisper loaded on {device}\n")

    rng = random.Random(args.seed)
    rows_with_wav = [r for r in rows if Path(r["wav"]).exists()]
    picks = rng.sample(rows_with_wav, min(args.n, len(rows_with_wav)))

    overlaps = []
    print(f"  {'idx':>5} {'dur':>5}s {'wps':>4} {'overlap':>7}  "
          f"verdict")
    print(f"  {'-'*5} {'-'*6} {'-'*4} {'-'*7}  {'-'*15}")
    for r in picks:
        try:
            wav, sr = torchaudio.load(r["wav"])
            if sr != 16000:
                wav = torchaudio.functional.resample(wav, sr, 16000)
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
            inp = proc(wav.squeeze(0).numpy(), sampling_rate=16000,
                        return_tensors="pt").input_features.to(
                device, dtype=torch.float16
                if device.type == "cuda" else torch.float32)
            with torch.no_grad():
                ids = mdl.generate(inp, forced_decoder_ids=forced,
                                    max_new_tokens=200, num_beams=1)
            hyp = proc.batch_decode(ids,
                                      skip_special_tokens=True)[0]
        except Exception as e:
            print(f"  {r['idx']:5d}  ERR: {type(e).__name__}")
            continue
        ov = char_overlap(r["gt_text"], hyp)
        overlaps.append(ov)
        dur = r["end"] - r["start"]
        wps = len(r["gt_text"].split()) / max(0.1, dur)
        if ov >= 0.5:
            v = "MATCH"
        elif ov >= 0.25:
            v = "partial"
        else:
            v = "MISMATCH"
        print(f"  {r['idx']:5d}  {dur:5.1f}  {wps:4.1f}  "
              f"{ov:7.2f}  {v}")

    if overlaps:
        good = sum(1 for o in overlaps if o >= 0.5)
        bad = sum(1 for o in overlaps if o < 0.25)
        mean = sum(overlaps) / len(overlaps)
        print()
        print(f"  mean char-overlap: {mean:.2f}")
        print(f"  MATCH (≥0.50): {good}/{len(overlaps)} "
              f"({100*good/len(overlaps):.0f}%)")
        print(f"  MISMATCH (<0.25): {bad}/{len(overlaps)} "
              f"({100*bad/len(overlaps):.0f}%)")
        if good / len(overlaps) >= 0.6:
            verdict = "ALIGNMENT GOOD — safe to train"
        elif bad / len(overlaps) >= 0.5:
            verdict = "ALIGNMENT BROKEN — fix before training"
        else:
            verdict = "ALIGNMENT MIXED — risky; investigate"
        print(f"\n  → Whisper verdict: {verdict}")


if __name__ == "__main__":
    main()
