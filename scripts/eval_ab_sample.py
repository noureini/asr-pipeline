"""
Read-only A/B sampler for qualitative evaluation.

Joins, by segment idx, the cached artifacts of one interview:
  raw            <- *.chunks.jsonl        (Omnilingual ASR)
  A_corrected    <- *.corrected.jsonl     (Path A: per-chunk QwenCorrector)
  A_en           <- *.translated.jsonl    (Path A: per-chunk QwenTranslator)
  B_speaker/bn/en<- polished.jsonl         (Path B: whole-doc polisher)

Prints a STRATIFIED sample (skips the leading mic-test/setup noise,
then spreads across the real interview in both parts) as aligned
blocks, so a reviewer/LLM scores like-for-like.

Usage:
  uv run python scripts/eval_ab_sample.py --stem <STEM> [--n 16]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

OUT = Path("outputs/real_eval_inproc")


def _load(path: Path) -> dict:
    m = {}
    if path.exists():
        for ln in path.read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if not ln:
                continue
            try:
                r = json.loads(ln)
                m[r["idx"]] = r
            except Exception:
                pass
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", required=True)
    ap.add_argument("--out-dir", type=Path, default=OUT)
    ap.add_argument("--n", type=int, default=16,
                    help="approx number of sampled segments")
    ap.add_argument("--skip-head", type=int, default=16,
                    help="skip first N segments of part 1 (setup noise)")
    ap.add_argument("--cap", type=int, default=400,
                    help="max chars printed per field")
    args = ap.parse_args()

    d = args.out_dir / args.stem
    chunk_files = sorted(d.glob("*.chunks.jsonl"))
    if not chunk_files:
        raise SystemExit(f"no *.chunks.jsonl in {d}")
    polished = _load(d / "polished.jsonl")

    # Build the global ordered segment list (part p -> idx base p*1e5).
    segs = []
    for p, cf in enumerate(chunk_files):
        stem = cf.name[: -len(".chunks.jsonl")]
        raw = _load(cf)
        a_c = _load(d / f"{stem}.corrected.jsonl")
        a_t = _load(d / f"{stem}.translated.jsonl")
        for i in sorted(raw):
            g = p * 100000 + i
            segs.append({
                "g": g, "part": p + 1, "i": i,
                "start": raw[i].get("start"),
                "raw": (raw[i].get("text") or "").strip(),
                "Ac": (a_c.get(i, {}).get("text") or "").strip(),
                "Ae": (a_t.get(i, {}).get("text") or "").strip(),
                "Bsp": polished.get(g, {}).get("speaker", "?"),
                "Bc": (polished.get(g, {}).get("bn") or "").strip(),
                "Be": (polished.get(g, {}).get("en") or "").strip(),
            })

    # Stratified sample: drop the first skip-head (part-1 setup noise),
    # then evenly space n picks across the remainder.
    body = segs[args.skip_head:]
    if not body:
        raise SystemExit("nothing to sample after skip-head")
    step = max(1, len(body) // args.n)
    picks = body[::step][: args.n]

    cap = args.cap

    def c(s):
        s = s or ""
        return s if len(s) <= cap else s[:cap] + " …[trunc]"

    print(f"# A/B sample — {args.stem}")
    print(f"# total segments={len(segs)} sampled={len(picks)} "
          f"(skipped first {args.skip_head} of part1 as setup)")
    for s in picks:
        st = "" if s["start"] is None else f"{s['start']:.1f}s"
        print("\n" + "=" * 70)
        print(f"[g{s['g']} part{s['part']} #{s['i']} @{st}] "
              f"B-speaker={s['Bsp']}")
        print(f"RAW : {c(s['raw'])}")
        print(f"A_bn: {c(s['Ac'])}")
        print(f"B_bn: {c(s['Bc'])}")
        print(f"A_en: {c(s['Ae'])}")
        print(f"B_en: {c(s['Be'])}")


if __name__ == "__main__":
    main()
