"""Three-way eyeball: raw titu vs Tiger-cleaned vs Claude-cleaned.

For each chunk, shows:
  - RAW    (titu output, unchanged baseline)
  - TIGER  (TigerLLM cleanup of that chunk)
  - CLAUDE (Claude cleanup of that chunk)

Lets you read 30 random chunks and judge quality differences across
the three deliverable variants at a glance.
"""
from __future__ import annotations
import argparse, json, random
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

ap = argparse.ArgumentParser()
ap.add_argument("--stem", required=True)
ap.add_argument("--tiger-dir", type=Path, default=None,
                help="defaults to cleanup_<stem>__tigerllm_9b_it/")
ap.add_argument("--claude-dir", type=Path, default=None,
                help="defaults to cleanup_<stem>__claude_sonnet_4_5/")
ap.add_argument("--n", type=int, default=30)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--show-identical", action="store_true",
                help="include chunks where Tiger==Claude==raw "
                     "(usually skipped for readability)")
args = ap.parse_args()

tiger_dir = (args.tiger_dir or
             REPO / "data" / "p2g"
             / f"cleanup_{args.stem}__tigerllm_9b_it")
claude_dir = (args.claude_dir or
              REPO / "data" / "p2g"
              / f"cleanup_{args.stem}__claude_sonnet_4_5")

def load(p: Path) -> dict:
    if not p.exists():
        return {}
    out = {}
    for ln in p.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            r = json.loads(ln)
            out[(r["partstem"], r["idx"])] = r
        except (json.JSONDecodeError, KeyError):
            continue
    return out

tiger = load(tiger_dir / "cleaned.jsonl")
claude = load(claude_dir / "cleaned.jsonl")

print(f"loaded Tiger : {len(tiger)} chunks from {tiger_dir.name}")
print(f"loaded Claude: {len(claude)} chunks from {claude_dir.name}")

# Use chunks present in BOTH
common = sorted(set(tiger) & set(claude))
print(f"common chunks: {len(common)}\n")

if not common:
    raise SystemExit("no overlap between Tiger and Claude outputs — "
                     "did you run both backends?")

random.seed(args.seed)

# Filter to chunks where at least one cleanup changed something
if not args.show_identical:
    interesting = []
    for k in common:
        raw = tiger[k].get("raw", "").strip()
        t = tiger[k].get("cleaned", "").strip()
        c = claude[k].get("cleaned", "").strip()
        if raw != t or raw != c or t != c:
            interesting.append(k)
    pool = interesting
    print(f"  {len(pool)} chunks where at least one cleanup "
          f"changed something\n")
else:
    pool = common

picks = random.sample(pool, min(args.n, len(pool)))

for i, key in enumerate(picks, 1):
    t = tiger[key]
    c = claude[key]
    raw = t.get("raw", "")
    t_cln = t.get("cleaned", "")
    c_cln = c.get("cleaned", "")
    print("=" * 78)
    print(f"[{i}/{len(picks)}] {key[0]}  idx={key[1]}  "
          f"t={t.get('start', 0):.1f}s")
    # Mark which paths differ
    flags = []
    if t_cln != raw:
        flags.append("Tiger changed")
    if c_cln != raw:
        flags.append("Claude changed")
    if t_cln != c_cln:
        flags.append("Tiger ≠ Claude")
    print(f"  flags: {', '.join(flags) if flags else 'all identical'}")
    print()
    print(f"  RAW    : {raw[:220]}")
    print(f"  TIGER  : {t_cln[:220]}")
    print(f"  CLAUDE : {c_cln[:220]}")
    print()
