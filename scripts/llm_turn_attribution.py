"""
Speaker attribution by DISCOURSE, not acoustics.

pyannote cannot separate the two same-gender voices on this single-
mic field audio (empirically: ~95% of speech collapses into one
cluster, both parts). But this is a SCRIPTED survey interview: the
enumerator asks (consent script, guide questions, probes), the
respondent answers (his life / livelihood / disasters). The TEXT
carries the speaker signal the AUDIO does not.

This labels each chunk ENUMERATOR / RESPONDENT with the local Qwen
GGUF (the same model already used for correct/translate — no new
infra), via a sliding window with role continuity. RESUMABLE.

Output (per part): <partstem>.speakers.jsonl  {idx, speaker}
assemble_transcript.py prefers this over *.diar.jsonl when present.

  uv run python scripts/llm_turn_attribution.py --stem <STEM> \
      [--limit N] [--window 14]
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

_LBL = re.compile(r"(\d+)\s*[:=]\s*([ER])", re.I)


def _read_jsonl(p: Path) -> dict:
    m = {}
    if p.exists():
        for ln in p.read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if ln:
                r = json.loads(ln)
                m[r["idx"]] = r
    return m


_PROMPT = (
    "You are labelling speaker turns in a transcript of a SCRIPTED "
    "qualitative survey interview from coastal Bangladesh.\n"
    "Two speakers:\n"
    "  E = ENUMERATOR — reads the consent script, asks the guide's "
    "questions and probes (\"আপনি … মনে করেন?\", \"বলেন তো\", "
    "\"কেন\", asks about preparation/cyclone/flood, thanks, moves on).\n"
    "  R = RESPONDENT — answers about his own life, household, "
    "shrimp gher, livelihood, experiences; narrates, gives opinions.\n"
    "Turns are STICKY: a speaker usually continues for several "
    "consecutive lines (a long answer, or a multi-part question). "
    "Only switch when the content clearly changes role.\n"
    "{cont}"
    "Label EVERY numbered line. Output ONLY one line per item as "
    "`<n>:E` or `<n>:R` — no other text.\n\n"
    "{lines}\n\nLabels:"
)


def _label_window(gen, rows, prev_role):
    cont = ""
    if prev_role:
        cont = (f"The line just before #{rows[0][0]} was spoken by "
                f"{prev_role}. Continue consistently.\n")
    lines = "\n".join(f"{i}: {(t or '').strip()[:200]}"
                       for i, t in rows)
    out = gen(_PROMPT.format(cont=cont, lines=lines),
              temperature=0.0, max_tokens=900) or ""
    got = {}
    for m in _LBL.finditer(out):
        got[int(m.group(1))] = ("ENUMERATOR" if m.group(2).upper() == "E"
                                else "RESPONDENT")
    return got


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", required=True)
    ap.add_argument("--out-dir", type=Path,
                    default=Path("outputs/real_eval_titu"))
    ap.add_argument("--config", type=Path,
                    default=Path("omni_test_1b.yaml"))
    ap.add_argument("--gguf-repo", default="unsloth/Qwen3.5-9B-GGUF")
    ap.add_argument("--gguf-file",
                    default="Qwen3.5-9B-UD-Q4_K_XL.gguf")
    ap.add_argument("--gguf-path", default=None)
    ap.add_argument("--window", type=int, default=14)
    ap.add_argument("--limit", type=int, default=None,
                    help="first N chunks PER PART (validation slice)")
    args = ap.parse_args()

    from asr_pipeline.config import load_config
    from asr_pipeline.postprocessor import QwenCorrector

    cfg = load_config(args.config)
    d = args.out_dir / args.stem
    cfiles = sorted(d.glob("*.chunks.jsonl"))
    if not cfiles:
        raise SystemExit(f"no *.chunks.jsonl in {d}")

    qcfg = cfg.postprocessing.qwen_corrector
    qcfg.backend = "gguf"
    qcfg.gguf_path = args.gguf_path
    qcfg.gguf_repo = None if args.gguf_path else args.gguf_repo
    qcfg.gguf_file = None if args.gguf_path else args.gguf_file
    qc = QwenCorrector(qcfg)
    if not qc.load():
        raise SystemExit("Qwen GGUF failed to load")
    gen = qc._llm.generate  # noqa: SLF001 — reuse loaded model

    for cf in cfiles:
        part = cf.name[: -len(".chunks.jsonl")]
        corr = _read_jsonl(d / f"{part}.corrected.jsonl")
        raw = _read_jsonl(cf)
        idxs = sorted(raw)
        if args.limit is not None:
            idxs = idxs[: args.limit]
        out = d / f"{part}.speakers.jsonl"
        done = _read_jsonl(out)
        prev_role = None
        # rebuild prev_role from last done label (resume continuity)
        if done:
            prev_role = done[max(done)]["speaker"]
        todo = [i for i in idxs if i not in done]
        print(f"  {part}: {len(idxs)} chunks "
              f"({len(done)} done) -> {len(todo)} to label", flush=True)
        with out.open("a", encoding="utf-8") as fh:
            for w in range(0, len(todo), args.window):
                win = todo[w: w + args.window]
                rows = [(i, (corr.get(i, {}).get("text")
                             or raw.get(i, {}).get("text") or ""))
                        for i in win]
                got = _label_window(gen, rows, prev_role)
                for i in win:
                    role = got.get(i, prev_role or "RESPONDENT")
                    fh.write(json.dumps({"idx": i, "speaker": role},
                                        ensure_ascii=False) + "\n")
                    fh.flush()
                    prev_role = role
                print(f"    {min(w + args.window, len(todo))}"
                      f"/{len(todo)}", flush=True)
        print(f"  -> {out}", flush=True)
    qc.unload()
    print("done. Re-run assemble (it prefers *.speakers.jsonl).")


if __name__ == "__main__":
    main()
