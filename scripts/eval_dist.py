"""Distribution stats on zero_shot_eval/results.tsv.

Mean is a lying summary statistic when the failure mode is bimodal
(some utts well-recognized, some catastrophic). Median + percentiles
tell the truth."""
from __future__ import annotations
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TSV = REPO / "data" / "zero_shot_eval" / "results.tsv"

rows = TSV.read_text(encoding="utf-8").splitlines()[1:]
by_model: dict[str, list[float]] = {}
for ln in rows:
    parts = ln.split("\t")
    if len(parts) < 5:
        continue
    name = parts[0]
    try:
        cer = float(parts[4])
    except ValueError:
        continue
    by_model.setdefault(name, []).append(cer)

def pct(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    if not xs:
        return 0.0
    i = max(0, min(len(xs) - 1, int(q * (len(xs) - 1))))
    return xs[i]

print(f"{'model':<25} {'n':>4} {'p10':>6} {'p25':>6} {'p50':>6} "
      f"{'p75':>6} {'p90':>6} {'mean':>6} {'<0.3':>5} {'>0.8':>5}")
print("-" * 90)
for name, vs in by_model.items():
    n = len(vs)
    mean = sum(vs) / n
    good = sum(1 for v in vs if v < 0.3)
    bad = sum(1 for v in vs if v > 0.8)
    print(f"{name:<25} {n:4d} {pct(vs,.1):6.2f} {pct(vs,.25):6.2f} "
          f"{pct(vs,.5):6.2f} {pct(vs,.75):6.2f} {pct(vs,.9):6.2f} "
          f"{mean:6.2f} {good:4d}  {bad:4d}")
