"""
Load-bearing check: does Qwen3-ASR-1.7B produce usable Bengali output,
even though Bengali is NOT in its officially supported language list?

The whole proposed "Qwen3-ASR -> Qwen3.5-Omni correct" pipeline rests
on this. If Qwen3-ASR emits garbage for Bengali, the right ASR front
end is Omnilingual (already in the pipeline, supports Bengali) or
Qwen3.5-Omni doing ASR directly — and we redesign accordingly.

Runs Qwen3-ASR-1.7B locally (fits ~4GB VRAM at bf16) on the 10 FLEURS
Bengali clips in test_data/fleurs_demo_10/, computes CER vs the gold
references with the SAME pre-registered normalization used in the
clean Omni pilot (M1 = NFC+punct stripped, M2 = +number norm), and
saves raw outputs so the Omni-correction step can consume them.

Usage:
    pip install -U qwen-asr
    uv run python scripts/check_qwen_asr_bengali.py
    # force a language hint (not officially supported, but test it):
    uv run python scripts/check_qwen_asr_bengali.py --language Bengali
    # CPU fallback if 6GB OOMs:
    uv run python scripts/check_qwen_asr_bengali.py --device cpu

Outputs:
    results/qwen3asr_bengali_raw.json   {file: {asr_text, detected_lang}}
    console: per-clip detected language + CER table vs references
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import unicodedata
from pathlib import Path

DEMO_DIR = Path("test_data/fleurs_demo_10")

# ─── Pre-registered normalization (identical to the Omni pilot) ──────────

_BN_DIGIT = {"০": "0", "১": "1", "২": "2", "৩": "3", "৪": "4",
             "৫": "5", "৬": "6", "৭": "7", "৮": "8", "৯": "9"}
_WORD_NUM = {"একশ": "100", "একশো": "100",
             "এক হাজার": "1000", "একহাজার": "1000"}


def base_norm(s: str) -> str:
    s = unicodedata.normalize("NFC", s).strip()
    s = re.sub(r"[।,.!?‌‍]", "", s)        # danda, latin punct, ZWNJ/ZWJ
    s = re.sub(r"\s+", " ", s)
    return s


def num_norm(s: str) -> str:
    for w, d in _WORD_NUM.items():
        s = s.replace(w, d)
    for bn, en in _BN_DIGIT.items():
        s = s.replace(bn, en)
    return re.sub(r"\s+", " ", s).strip()


def cer(ref: str, hyp: str) -> float:
    r, h = list(ref), list(hyp)
    n, m = len(r), len(h)
    d = list(range(m + 1))
    for i in range(1, n + 1):
        prev = d[0]
        d[0] = i
        for j in range(1, m + 1):
            cur = d[j]
            cost = 0 if r[i - 1] == h[j - 1] else 1
            d[j] = min(d[j] + 1, d[j - 1] + 1, prev + cost)
            prev = cur
    return d[m] / max(n, 1)


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--demo-dir", type=Path, default=DEMO_DIR,
                   help="Folder with audio/ + references.json")
    p.add_argument("--model", default="Qwen/Qwen3-ASR-1.7B")
    p.add_argument("--language", default=None,
                   help="None=auto-detect (the honest test, since Bengali "
                        "is unsupported). Or force e.g. 'Bengali'.")
    p.add_argument("--device", default="cuda:0",
                   help="cuda:0 (default) or cpu if 6GB OOMs")
    p.add_argument("--dtype", default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])
    p.add_argument("--out", type=Path,
                   default=Path("results/qwen3asr_bengali_raw.json"))
    args = p.parse_args()

    refs_path = args.demo_dir / "references.json"
    audio_dir = args.demo_dir / "audio"
    if not refs_path.exists() or not audio_dir.is_dir():
        print(f"ERROR: {args.demo_dir} missing references.json or audio/.")
        print("Run: just fetch-fleurs-demo 10")
        sys.exit(1)
    refs = {r["file"]: r["transcript"]
            for r in json.load(open(refs_path, encoding="utf-8"))}

    try:
        import torch
        from qwen_asr import Qwen3ASRModel
    except ImportError as e:
        print(f"ERROR: {e}")
        print("Install: pip install -U qwen-asr")
        sys.exit(1)

    dtype = getattr(torch, args.dtype)
    print(f"Loading {args.model} ({args.dtype}, {args.device})...")
    t0 = time.time()
    try:
        model = Qwen3ASRModel.from_pretrained(
            args.model, dtype=dtype, device_map=args.device,
        )
    except Exception as e:
        print(f"ERROR loading model: {e}")
        if "out of memory" in str(e).lower():
            print("6GB OOM — retry with --device cpu")
        sys.exit(1)
    print(f"  loaded in {time.time() - t0:.1f}s")

    raw = {}
    rows = []
    print(f"\nTranscribing {len(refs)} clips "
          f"(language={args.language or 'auto-detect'})...\n")
    for fname in sorted(refs):
        wav = audio_dir / fname
        if not wav.exists():
            print(f"  {fname}: MISSING audio, skipped")
            continue
        try:
            res = model.transcribe(audio=str(wav), language=args.language)
            text = res[0].text
            det = getattr(res[0], "language", "?")
        except Exception as e:
            print(f"  {fname}: transcribe error: {e}")
            text, det = "", "ERR"
        raw[fname] = {"asr_text": text, "detected_lang": det}

        ref = refs[fname]
        m1 = cer(base_norm(ref), base_norm(text)) * 100
        m2 = cer(num_norm(base_norm(ref)),
                 num_norm(base_norm(text))) * 100
        rows.append((fname, det, m1, m2))
        print(f"  {fname}  lang={det:<10}  CER M1={m1:5.1f}%  M2={m2:5.1f}%")
        print(f"     ref: {ref[:70]}")
        print(f"     asr: {text[:70]}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(raw, open(args.out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    if rows:
        n = len(rows)
        mean_m1 = sum(r[2] for r in rows) / n
        mean_m2 = sum(r[3] for r in rows) / n
        langs = {}
        for r in rows:
            langs[r[1]] = langs.get(r[1], 0) + 1
        print(f"\n{'=' * 60}")
        print(f"Qwen3-ASR-1.7B on FLEURS Bengali, n={n}")
        print(f"{'=' * 60}")
        print(f"  Mean CER  M1 (NFC+punct):  {mean_m1:5.1f}%")
        print(f"  Mean CER  M2 (+num norm):  {mean_m2:5.1f}%")
        print(f"  Detected languages: {langs}")
        print(f"\n  Raw outputs -> {args.out}")
        print(f"\nInterpretation:")
        print(f"  CER < 25%  : Qwen3-ASR IS usable for Bengali despite no")
        print(f"               official support; Omni-correct chain viable.")
        print(f"  CER 25-50% : marginal — Omni would carry most of the load.")
        print(f"  CER > 50%  : Qwen3-ASR not a viable Bengali front end;")
        print(f"               use Omnilingual or Qwen3.5-Omni-direct ASR.")
        print(f"\n  Compare to: Omnilingual CTC (in-pipeline, supports BN)")
        print(f"  and your clean Omni-pilot raw-ASR baseline (8.5% M1).")


if __name__ == "__main__":
    main()
