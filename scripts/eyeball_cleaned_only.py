"""Pull random (raw, cleaned) pairs from cleaned.jsonl for human
eyeball. Works WITHOUT the judge pass having run."""
from __future__ import annotations
import argparse, json, random
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

ap = argparse.ArgumentParser()
ap.add_argument("--out-dir", type=Path,
                default=REPO / "data" / "p2g"
                        / "cleanup_15_Sunamganj_CA-MMI_URBAN"
                          "_Male_15_0602_1635__tigerllm_9b_it")
ap.add_argument("--n", type=int, default=30)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

p = args.out_dir / "cleaned.jsonl"
if not p.exists():
    raise SystemExit(f"missing {p}")
rows = [json.loads(l) for l in
        p.read_text(encoding="utf-8").splitlines() if l.strip()]
print(f"loaded {len(rows)} cleaned chunks\n")

# Filter to chunks where cleaned differs from raw (interesting ones)
diff = [r for r in rows if r.get("raw", "").strip()
                            != r.get("cleaned", "").strip()]
print(f"  {len(diff)} chunks where cleanup changed something\n")

random.seed(args.seed)
sample = random.sample(diff, min(args.n, len(diff)))

for i, r in enumerate(sample, 1):
    print("=" * 76)
    print(f"[{i}/{len(sample)}] {r['partstem']} idx={r['idx']} "
          f"t={r.get('start', 0):.1f}s")
    print(f"  RAW     : {r.get('raw', '')[:220]}")
    print(f"  CLEANED : {r.get('cleaned', '')[:220]}")
print()
