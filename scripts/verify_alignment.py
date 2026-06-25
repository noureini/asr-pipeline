"""Hand-verify whether long-form MMS-FA alignment is actually correct.

Picks 8 utterances spread across the timeline, copies each WAV to
data/verify_alignment/ with a filename that includes the GT text
truncated, so you can play them back and EAR-CHECK whether the audio
contains the claimed transcript.

This is the cheapest, most important diagnostic before anything else:
if alignment is broken, every model eval downstream is meaningless.
"""
from __future__ import annotations
import json, shutil, re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TEST = (REPO / "data" / "w2v_train_longform"
        / "15_Sunamganj_CA-MMI_URBAN_Male_15_0602_1635.longform.jsonl")
TRAIN = (REPO / "data" / "w2v_train_longform"
         / "07_Shatkhira_CA-MMI_URBAN_Male_07_0202_1716.longform.jsonl")
OUT = REPO / "data" / "verify_alignment"
OUT.mkdir(parents=True, exist_ok=True)
# Clear previous
for f in OUT.iterdir():
    f.unlink()


def _safe(s: str, n: int = 40) -> str:
    s = re.sub(r"[^\w\sঀ-৿]", "", s)[:n]
    return re.sub(r"\s+", "_", s).strip("_")


for tag, path in [("test_15", TEST), ("train_07", TRAIN)]:
    rows = [json.loads(l) for l in
            path.read_text(encoding="utf-8").splitlines() if l.strip()]
    rows.sort(key=lambda r: r["start"])
    # 8 spread evenly across the timeline (avoid first/last 5%)
    idx = [int((0.05 + 0.9 * i / 7) * len(rows)) for i in range(8)]
    picks = [rows[i] for i in idx]
    print(f"\n=== {tag} ({len(rows)} utts) ===")
    for i, r in enumerate(picks):
        dur = r["end"] - r["start"]
        out_name = (
            f"{tag}_{i:02d}_t{int(r['start']):04d}s_d{dur:.1f}s_"
            f"{_safe(r['gt_text'])}.wav")
        out_path = OUT / out_name
        shutil.copy(r["wav"], out_path)
        print(f"  [{i}] t={r['start']:7.1f}s  dur={dur:.1f}s")
        print(f"      GT:  {r['gt_text'][:110]}")
        print(f"      WAV: {out_path}")

print(f"\nPlay these and confirm each WAV contains the GT text:")
print(f"  {OUT}")
