"""Convert cleanup_titu.py and translate_chunks.py outputs into the
*.corrected.jsonl / *.translated.jsonl files that
assemble_transcript.py expects (one file per audio part, schema
{idx, text}).

Usage:
  uv run python scripts/to_assemble_format.py \\
      --stem 15_Sunamganj_CA-MMI_URBAN_Male_15_0602_1635 \\
      --cleaned-dir data/p2g/cleanup_15_..._tigerllm_9b_it \\
      --translated-dir data/p2g/translated_15_..._tigerllm_9b_it \\
      --out-dir outputs/real_eval_titu_excellent
"""
from __future__ import annotations
import argparse, json, sys
from collections import defaultdict
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", required=True)
    ap.add_argument("--cleaned-dir", type=Path, required=True)
    ap.add_argument("--translated-dir", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, required=True,
                    help="dir containing the chunks.jsonl files; "
                         "corrected/translated jsonl will be written "
                         "alongside them")
    args = ap.parse_args()

    stem_dir = args.out_dir / args.stem
    if not stem_dir.exists():
        sys.exit(f"missing {stem_dir}")

    # --- cleaned.jsonl -> *.corrected.jsonl per part ---
    cleaned_path = args.cleaned_dir / "cleaned.jsonl"
    if not cleaned_path.exists():
        sys.exit(f"missing {cleaned_path}")
    by_part: dict[str, list[dict]] = defaultdict(list)
    for ln in cleaned_path.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if ln:
            r = json.loads(ln)
            by_part[r["partstem"]].append({
                "idx": r["idx"],
                "text": r.get("cleaned", "") or r.get("raw", ""),
            })
    for partstem, rows in by_part.items():
        rows.sort(key=lambda x: x["idx"])
        out = stem_dir / f"{partstem}.corrected.jsonl"
        with out.open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  wrote {len(rows)} rows -> {out}")

    # --- translated.jsonl -> *.translated.jsonl per part ---
    if args.translated_dir:
        trans_path = args.translated_dir / "translated.jsonl"
        if trans_path.exists():
            by_part2: dict[str, list[dict]] = defaultdict(list)
            for ln in trans_path.read_text(
                    encoding="utf-8").splitlines():
                ln = ln.strip()
                if ln:
                    r = json.loads(ln)
                    by_part2[r["partstem"]].append({
                        "idx": r["idx"],
                        "text": r.get("en", ""),
                    })
            for partstem, rows in by_part2.items():
                rows.sort(key=lambda x: x["idx"])
                out = stem_dir / f"{partstem}.translated.jsonl"
                with out.open("w", encoding="utf-8") as fh:
                    for r in rows:
                        fh.write(json.dumps(
                            r, ensure_ascii=False) + "\n")
                print(f"  wrote {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
