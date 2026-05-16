"""
Validate Omnilingual on all 10 FLEURS Bengali clips — one command.

Runs the full pipeline (Omnilingual forced via omni_test.yaml,
post-processing disabled so it's fast and we measure raw ASR) on each
clip in test_data/fleurs_demo_10/, then computes CER vs the gold
references with the SAME pre-registered M1/M2 normalization used for
the Qwen / cloud-Flash comparisons.

This is the "does the 6.8% single-clip result hold at n=10" check.
Still FLEURS clean read-speech — NOT a substitute for real-audio
ground truth, but the necessary scale check before that.

Usage:
    uv run python scripts/eval_omni_fleurs10.py
    # different config / model variant:
    uv run python scripts/eval_omni_fleurs10.py --config omni_test.yaml

Output:
    outputs/omni_eval/NNNN.json   (per-clip pipeline output)
    results/omni_fleurs10.json    (scores)
    console: per-clip + mean CER table, M1 and M2
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
import unicodedata
from pathlib import Path

DEMO = Path("test_data/fleurs_demo_10")

_BN = {"০": "0", "১": "1", "২": "2", "৩": "3", "৪": "4",
       "৫": "5", "৬": "6", "৭": "7", "৮": "8", "৯": "9"}
_WN = {"একশ": "100", "একশো": "100", "এক হাজার": "1000", "একহাজার": "1000"}


def base_norm(s: str) -> str:
    s = unicodedata.normalize("NFC", s).strip()
    s = re.sub(r"[।,.!?‌‍]", "", s)
    return re.sub(r"\s+", " ", s)


def num_norm(s: str) -> str:
    for w, d in _WN.items():
        s = s.replace(w, d)
    for b, e in _BN.items():
        s = s.replace(b, e)
    return re.sub(r"\s+", " ", s).strip()


def cer(ref: str, hyp: str) -> float:
    r, h = list(ref), list(hyp)
    n, m = len(r), len(h)
    d = list(range(m + 1))
    for i in range(1, n + 1):
        p = d[0]
        d[0] = i
        for j in range(1, m + 1):
            c = d[j]
            co = 0 if r[i - 1] == h[j - 1] else 1
            d[j] = min(d[j] + 1, d[j - 1] + 1, p + co)
            p = c
    return d[m] / max(n, 1)


def extract_text(json_path: Path) -> str:
    """Join Bengali ASR text from a pipeline JSON output. Post-processing
    is disabled so corrected_text == raw ASR; fall back across fields."""
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return ""
    segs = data.get("segments", []) or []
    parts = []
    for s in segs:
        t = (s.get("corrected_text") or s.get("raw_text")
             or s.get("text") or "").strip()
        if t:
            parts.append(t)
    return " ".join(parts).strip()


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--demo-dir", type=Path, default=DEMO)
    p.add_argument("--config", default="omni_test.yaml",
                   help="Pipeline config (default omni_test.yaml: "
                        "Omnilingual forced, post-processing off)")
    p.add_argument("--out-dir", type=Path, default=Path("outputs/omni_eval"))
    p.add_argument("--results", type=Path,
                   default=Path("results/omni_fleurs10.json"))
    p.add_argument("--skip-run", action="store_true",
                   help="Don't re-run the pipeline; just score existing "
                        "JSON in --out-dir (for re-scoring)")
    p.add_argument("--max", type=int, default=0,
                   help="Quick test: only the first N clips (0 = all 10)")
    args = p.parse_args()

    refs_path = args.demo_dir / "references.json"
    audio_dir = args.demo_dir / "audio"
    if not refs_path.exists() or not audio_dir.is_dir():
        print(f"ERROR: {args.demo_dir} missing references.json or audio/")
        sys.exit(1)
    refs = {r["file"]: r["transcript"]
            for r in json.loads(refs_path.read_text(encoding="utf-8"))}
    args.out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(refs)
    if args.max and args.max > 0:
        files = files[:args.max]
        print(f"QUICK TEST: first {len(files)} clip(s) only\n")

    rows = []
    t0 = time.time()
    for i, fname in enumerate(files, 1):
        wav = audio_dir / fname
        stem = Path(fname).stem
        out_json = args.out_dir / f"{stem}.json"

        if not args.skip_run:
            if not wav.exists():
                print(f"  [{i}/{len(files)}] {fname}: MISSING audio, skip")
                continue
            print(f"  [{i}/{len(files)}] transcribing {fname} ...", flush=True)
            cmd = [
                "uv", "run", "asr-pipeline", "transcribe", str(wav),
                "-l", "ben", "-c", args.config,
                "-f", "json", "-o", str(args.out_dir),
            ]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                print(f"      pipeline failed (rc={r.returncode}); "
                      f"stderr tail:\n{r.stderr[-400:]}")
                continue

        if not out_json.exists():
            print(f"      no JSON at {out_json}, skip")
            continue
        hyp = extract_text(out_json)
        ref = refs[fname]
        m1 = cer(base_norm(ref), base_norm(hyp)) * 100
        m2 = cer(num_norm(base_norm(ref)), num_norm(base_norm(hyp))) * 100
        rows.append({"file": fname, "ref": ref, "hyp": hyp,
                     "cer_m1": m1, "cer_m2": m2})
        print(f"      M1={m1:5.1f}%  M2={m2:5.1f}%  | {hyp[:60]}")

    if not rows:
        print("No clips scored.")
        sys.exit(1)

    n = len(rows)
    mean_m1 = sum(r["cer_m1"] for r in rows) / n
    mean_m2 = sum(r["cer_m2"] for r in rows) / n
    elapsed = time.time() - t0

    print(f"\n{'=' * 60}")
    print(f"Omnilingual — FLEURS Bengali, n={n}  ({elapsed/60:.1f} min)")
    print(f"{'=' * 60}")
    print(f"{'file':>10} {'M1':>8} {'M2(+num)':>10}")
    for r in rows:
        print(f"{r['file']:>10} {r['cer_m1']:7.1f}% {r['cer_m2']:9.1f}%")
    print(f"{'MEAN':>10} {mean_m1:7.1f}% {mean_m2:9.1f}%")
    print(f"\nReference points (single clip 0001 earlier):")
    print(f"  cloud Qwen-Flash    M2 ~3.4%   (cloud-only, not usable for survey)")
    print(f"  Omnilingual 0001    M2  6.8%")
    print(f"  Qwen-1.7B translit  M2 33.9%")
    print(f"\nThis is clean FLEURS read-speech. Real Bangladesh field")
    print(f"audio will be worse — hand-transcribed ground truth still")
    print(f"required before production sign-off.")

    args.results.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"n": n, "mean_cer_m1": mean_m1, "mean_cer_m2": mean_m2,
               "rows": rows, "config": args.config},
              open(args.results, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"\nSaved -> {args.results}")


if __name__ == "__main__":
    main()
