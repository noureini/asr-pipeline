"""
Stage-2 bridge (part 2/2) — ISOLATED titu venv (NeMo).

Reads the manifests written by dump_vad_chunks.py and transcribes
the IDENTICAL VAD chunks with hishab/titu_stt_bn_fastconformer,
emitting <partstem>.chunks.jsonl in the EXACT pipeline schema
({idx,start,end,text}) so it flows through the same
eval_real_survey_inproc --score-only --correct --translate and
assemble path as the Omnilingual baseline. RESUMABLE.

Run with the isolated venv:
  /tmp/titu-venv/bin/python scripts/titu_transcribe_chunks.py \
      --stem <STEM> [--limit N] [--batch 8]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load_done(p: Path) -> dict:
    m = {}
    if p.exists():
        for ln in p.read_text(encoding="utf-8").splitlines():
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
    ap.add_argument("--out-dir", type=Path,
                    default=Path("outputs/real_eval_titu"))
    ap.add_argument("--limit", type=int, default=None,
                    help="first N chunks PER PART (fast slice test)")
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    d = args.out_dir / args.stem
    mans = sorted(d.glob("*.manifest.jsonl"))
    if not mans:
        sys.exit(f"no *.manifest.jsonl in {d} — run dump_vad_chunks.py "
                 f"(main venv) first")

    import nemo.collections.asr as nemo_asr
    print("loading hishab/titu_stt_bn_fastconformer ...", flush=True)
    model = nemo_asr.models.ASRModel.from_pretrained(
        "hishab/titu_stt_bn_fastconformer")

    for man in mans:
        partstem = man.name[: -len(".manifest.jsonl")]
        rows = []
        for ln in man.read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if ln:
                rows.append(json.loads(ln))
        rows.sort(key=lambda r: r["idx"])
        if args.limit is not None:
            rows = rows[: args.limit]

        ckpt = d / f"{partstem}.chunks.jsonl"
        done = _load_done(ckpt)
        todo = [r for r in rows if r["idx"] not in done]
        print(f"  {partstem}: {len(rows)} chunks "
              f"({len(done)} done) -> {len(todo)} to do", flush=True)
        if not todo:
            continue

        with ckpt.open("a", encoding="utf-8") as fh:
            for i in range(0, len(todo), args.batch):
                batch = todo[i: i + args.batch]
                wavs = [r["wav"] for r in batch]
                outs = model.transcribe(wavs, batch_size=len(wavs))
                for r, o in zip(batch, outs):
                    txt = o if isinstance(o, str) else getattr(
                        o, "text", str(o))
                    fh.write(json.dumps(
                        {"idx": r["idx"],
                         "start": r["start"],
                         "end": r["end"],
                         "text": (txt or "").strip()},
                        ensure_ascii=False) + "\n")
                    fh.flush()
                print(f"    {min(i + args.batch, len(todo))}"
                      f"/{len(todo)}", flush=True)
        print(f"  -> {ckpt}", flush=True)

    print("done. Next (MAIN venv): "
          "just eval-real-survey-inproc <stem> "
          "--out-dir outputs/real_eval_titu --score-only "
          "--correct --translate")


if __name__ == "__main__":
    main()
