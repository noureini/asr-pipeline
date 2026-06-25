"""Stratified eyeball sampler for the multi-aspect judge results.

Picks 50 chunks across four strata:
  - 15 cleanup wins (fluency=cleaned AND faithfulness in {cleaned,tie})
  - 15 raw wins    (fluency=raw)
  - 10 ties        (both aspects = tie)
  - 10 hallucinations (hallucination_B=yes)

Output is formatted for an external Bengali reader to verify
independently. The reader rates each pair and we compare to the
LLM-judge verdict per aspect.
"""
from __future__ import annotations
import argparse, json, random
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

ap = argparse.ArgumentParser()
ap.add_argument("--out-dir", type=Path,
                default=REPO / "data" / "p2g"
                        / "cleanup_15_Sunamganj_CA-MMI_URBAN"
                          "_Male_15_0602_1635")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

judge_path = args.out_dir / "judge.jsonl"
if not judge_path.exists():
    raise SystemExit(f"missing {judge_path}\n"
                     f"run cleanup_titu.py first")

rows = [json.loads(l) for l in
        judge_path.read_text(encoding="utf-8").splitlines() if l.strip()]
print(f"loaded {len(rows)} judged chunks\n")

# Headline tally
from collections import Counter
flu = Counter(r.get("fluency", "?") for r in rows)
fai = Counter(r.get("faithfulness", "?") for r in rows)
hal = Counter(r.get("hallucination_B", "?") for r in rows)
print(f"FLUENCY tally     : "
      f"cleaned={flu['cleaned']}  raw={flu['raw']}  tie={flu['tie']}")
print(f"FAITHFULNESS tally: "
      f"cleaned={fai['cleaned']}  raw={fai['raw']}  tie={fai['tie']}")
print(f"HALLUCINATION tally: "
      f"yes={hal['yes']}  no={hal['no']}\n")

# Stratified pools
cleanup_wins = [r for r in rows
                if r.get("fluency") == "cleaned"
                and r.get("faithfulness") in ("cleaned", "tie")]
raw_wins = [r for r in rows if r.get("fluency") == "raw"]
ties = [r for r in rows
        if r.get("fluency") == "tie"
        and r.get("faithfulness") == "tie"]
hallucs = [r for r in rows if r.get("hallucination_B") == "yes"]

random.seed(args.seed)


def show(label: str, pool: list, n: int) -> None:
    print("=" * 76)
    print(f"=== {label}  ({min(n, len(pool))} of {len(pool)} "
          f"in pool) ===")
    print("=" * 76)
    if not pool:
        print("  (none)\n")
        return
    sample = random.sample(pool, min(n, len(pool)))
    for i, r in enumerate(sample, 1):
        verdicts = (
            f"fluency={r.get('fluency')}  "
            f"faithfulness={r.get('faithfulness')}  "
            f"halluc={r.get('hallucination_B')}  "
            f"cs={r.get('code_switch')}  "
            f"dial={r.get('dialect')}")
        print(f"\n[{label} {i}] {r['partstem']} idx={r['idx']} "
              f"t={r.get('start', 0):.1f}s")
        print(f"  judge: {verdicts}")
        print(f"  RAW     : {r['raw'][:220]}")
        print(f"  CLEANED : {r['cleaned'][:220]}")
    print()


show("CLEANUP WINS", cleanup_wins, 15)
show("RAW WINS",     raw_wins,     15)
show("TIES",         ties,         10)
show("HALLUCINATIONS (cleaned has unsupported additions)",
     hallucs, 10)

print("\n" + "=" * 76)
print("Paste this output back to your AI assistant. They will read")
print("each pair and produce an independent reader verdict on")
print("fluency / faithfulness / hallucination per chunk, then compare")
print("to the LLM-judge verdict to compute agreement rate.")
