"""
Day-0b: pair each titu VAD chunk (audio + start/end + titu prediction)
with the closest GT sentence via edit-distance similarity, with
monotonic pointer to respect temporal order.

GT sentences are TRANSLITERATED via /tmp/translit.json before
matching → labels are bilingual (Bengali script + Latin for
code-switched English).

titu's role: alignment bridge only. The label is GT, not titu.

Output (per --out-manifest):
  {idx, partstem, wav, gt_text, titu_text, score, start, end}

Usage:
  uv run python scripts/align_titu_gt.py \
      --stem 07_Shatkhira_CA-MMI_URBAN_Male_07_0202_1716 \
      --titu-dir outputs/real_eval_titu \
      --docx "for testing (Noureini)/07_..._transcription.docx" \
      --out-manifest /tmp/w2v_train/07.train.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
from difflib import SequenceMatcher
from pathlib import Path

_RE_SPEAKER_TAG = re.compile(r"^(প্র:|উ:|Q:|A:)\s*", re.M)
_RE_PARENS = re.compile(r"\([^)]{0,200}\)")
_RE_BRACK = re.compile(r"\[[^\]]{0,200}\]")
_RE_WS = re.compile(r"\s+")
_RE_SENT = re.compile(r"[।.?!]+\s*")
_RE_TOK = re.compile(r"\S+")


def docx_sents(p: Path) -> list[str]:
    import docx
    out = []
    d = docx.Document(str(p))
    for para in d.paragraphs:
        t = para.text or ""
        t = _RE_SPEAKER_TAG.sub("", t)
        t = _RE_PARENS.sub(" ", t)
        t = _RE_BRACK.sub(" ", t)
        t = _RE_WS.sub(" ", t).strip()
        for s in _RE_SENT.split(t):
            s = s.strip()
            if s:
                out.append(s)
    return out


def apply_translit(sent: str, table: dict[str, str]) -> str:
    """Whole-word match; substitute Bengali-script English with Latin."""
    return " ".join(table.get(w, w) for w in sent.split())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", required=True)
    ap.add_argument("--titu-dir", type=Path, required=True)
    ap.add_argument("--docx", type=Path, required=True)
    ap.add_argument("--translit-json", type=Path,
                    default=Path("/tmp/translit.json"))
    ap.add_argument("--out-manifest", type=Path, required=True)
    ap.add_argument("--threshold", type=float, default=0.35)
    ap.add_argument("--search-window", type=int, default=10)
    args = ap.parse_args()

    table = json.loads(args.translit_json.read_text(encoding="utf-8"))
    print(f"translit table: {len(table)} entries")
    sents_raw = docx_sents(args.docx)
    sents = [apply_translit(s, table) for s in sents_raw]
    n_tr = sum(1 for r, t in zip(sents_raw, sents) if r != t)
    print(f"GT sentences: {len(sents)} ({n_tr} contained translit hits)")

    d = args.titu_dir / args.stem
    chunks_files = sorted(d.glob("*.chunks.jsonl"))
    if not chunks_files:
        raise SystemExit(f"no *.chunks.jsonl in {d}")

    args.out_manifest.parent.mkdir(parents=True, exist_ok=True)
    n_total = n_kept = 0
    hist = [0] * 10
    ptr = 0  # monotonic position
    with args.out_manifest.open("w", encoding="utf-8") as fh:
        for cf in chunks_files:
            partstem = cf.name[: -len(".chunks.jsonl")]
            wavdir = d / "_chunkwav" / partstem
            for ln in cf.read_text(encoding="utf-8").splitlines():
                ln = ln.strip()
                if not ln:
                    continue
                ch = json.loads(ln)
                n_total += 1
                titu_text = (ch.get("text") or "").strip()
                if not titu_text:
                    continue
                lo = max(0, ptr - 2)
                hi = min(len(sents), ptr + args.search_window)
                best = (0.0, -1, "")
                for j in range(lo, hi):
                    r = SequenceMatcher(None, titu_text, sents[j]).ratio()
                    if r > best[0]:
                        best = (r, j, sents[j])
                hist[min(9, int(best[0] * 10))] += 1
                if best[0] < args.threshold:
                    continue
                wav = wavdir / f"{ch['idx']:05d}.wav"
                if not wav.exists():
                    continue
                fh.write(json.dumps({
                    "idx": ch["idx"],
                    "partstem": partstem,
                    "wav": str(wav),
                    "gt_text": best[2],
                    "titu_text": titu_text,
                    "score": round(best[0], 3),
                    "start": ch.get("start"),
                    "end": ch.get("end"),
                }, ensure_ascii=False) + "\n")
                n_kept += 1
                ptr = best[1] + 1

    rate = 100 * n_kept / max(n_total, 1)
    print(f"chunks total: {n_total}  kept: {n_kept} ({rate:.1f}%)")
    print("score histogram (each bin width=0.1):")
    for i, c in enumerate(hist):
        bar = "#" * min(60, c // 5)
        print(f"  [{i/10:.1f}-{(i+1)/10:.1f}] {c:4d}  {bar}")
    print(f"-> {args.out_manifest}")


if __name__ == "__main__":
    main()
