"""Print (raw_titu, cleaned, GT) triples for manual quality judgment.
String-match metrics (WER/CER) penalize verbatim ASR against edited
GT. Human eyeball tells us whether cleanup actually improves
readability/coherence."""
from __future__ import annotations
import argparse, json, random
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

ap = argparse.ArgumentParser()
ap.add_argument("--tsv", type=Path,
                default=REPO / "data" / "p2g"
                        / "cleanup_15_Sunamganj_CA-MMI_URBAN"
                          "_Male_15_0602_1635"
                        / "per_chunk.tsv")
ap.add_argument("--n", type=int, default=10)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

if not args.tsv.exists():
    raise SystemExit(f"missing {args.tsv}\n"
                     f"run cleanup_titu.py first")

# Load TSV (header + rows)
lines = args.tsv.read_text(encoding="utf-8").splitlines()
header = lines[0].split("\t")
rows = []
for ln in lines[1:]:
    parts = ln.split("\t")
    if len(parts) >= len(header):
        rows.append(dict(zip(header, parts)))

random.seed(args.seed)
pick = random.sample(rows, min(args.n, len(rows)))

for i, r in enumerate(pick, 1):
    print("=" * 76)
    print(f"[{i}] {r['partstem']} idx={r['idx']} "
          f"t={float(r['start']):.1f}s")
    print(f"    CER raw={float(r['cer_raw']):.2f}  "
          f"cleaned={float(r['cer_cleaned']):.2f}")
    print()
    print(f"GT      : {r['ref']}")
    print()
    print(f"RAW     : {r['raw']}")
    print()
    print(f"CLEANED : {r['cleaned']}")
    print()

print("=" * 76)
print("\nQuestions to answer for each triple:")
print("  (1) Is CLEANED more coherent/grammatical Bengali than RAW?")
print("  (2) Does CLEANED preserve more of the actual spoken content?")
print("  (3) Are dialect forms preserved (নাই, করতেছি, তাইনে)?")
print("  (4) Are code-switched English words preserved correctly?")
