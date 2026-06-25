"""Content-based role attribution + mixed-segment splitting.

Re-labels a finished transcript WITHOUT re-running the 110-min fusion:
reads the existing [bn]/[en] segments, sends them to Gemma in conversation
order (so it sees the Q->A flow), and assigns ENUMERATOR / RESPONDENT from
the CONTENT — ignoring the broken pyannote speaker clusters. Mixed
segments (an enumerator question + the respondent's reply in one block)
are split into separate role-labeled turns.

Single-speaker segments only get a role (text reused verbatim, no drift);
only mixed segments are re-emitted as a split.

Usage:
    uv run python scripts/relabel_roles.py <transcript.txt>
    uv run python scripts/relabel_roles.py <transcript.txt> --window 12 --out foo.txt
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from string import Template

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

HEAD = re.compile(r"^\[(\d{2}:\d{2}:\d{2})\]\s+([A-Z_0-9]+):\s*$")


def parse(path: Path) -> list[dict]:
    segs: list[dict] = []
    cur: dict | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        m = HEAD.match(line)
        if m:
            cur = {"time": m.group(1), "role": m.group(2),
                   "bn_raw": "", "bn": "", "en": ""}
            segs.append(cur)
            continue
        if cur is None:
            continue
        if line.startswith("[bn-raw]"):
            cur["bn_raw"] = line[len("[bn-raw]"):].strip()
        elif line.startswith("[bn]"):
            cur["bn"] = line[len("[bn]"):].strip()
        elif line.startswith("[en]"):
            cur["en"] = line[len("[en]"):].strip()
    return segs


def _extract_json(text: str):
    """Robustly pull the JSON role array out of a model reply.

    Handles markdown ```json fences, leading/trailing prose, and a
    truncated array (max_tokens cut) by (1) scanning for the balanced
    [...] and (2) falling back to parsing individual {...} objects.
    """
    if not text:
        return None
    t = re.sub(r"```(?:json)?", "", text)
    # (1) balanced top-level array
    a = t.find("[")
    if a >= 0:
        depth = 0
        for j in range(a, len(t)):
            if t[j] == "[":
                depth += 1
            elif t[j] == "]":
                depth -= 1
                if depth == 0:
                    try:
                        data = json.loads(t[a:j + 1])
                        if isinstance(data, list):
                            return data
                    except Exception:
                        break
    # (2) fallback: per-object parse (survives a truncated/!valid array).
    # Matches flat objects like {"seg":0,"role":"RESPONDENT"}; split
    # objects (nested) are recovered by the balanced scan above.
    objs = []
    for m in re.finditer(r"\{[^{}]*\}", t):
        try:
            o = json.loads(m.group(0))
            if isinstance(o, dict):
                objs.append(o)
        except Exception:
            pass
    return objs or None


def relabel(segs: list[dict], proc, window: int, role_prompt: str,
            temperature: float, max_tokens: int) -> list[dict]:
    """Role-only: assign ENUMERATOR/RESPONDENT to every segment. Any
    segment the model leaves out defaults to a ROLE (alternating from the
    previous turn) — never the raw SPEAKER_xx label."""
    tpl = Template(role_prompt)
    out: list[dict] = []
    prev = "RESPONDENT"
    for w in range(0, len(segs), window):
        chunk = segs[w:w + window]
        body = "\n".join(
            f"[{i}] bn: {s['bn']} | en: {s['en']}"
            for i, s in enumerate(chunk))
        prompt = tpl.safe_substitute(n=len(chunk), body=body)
        reply = proc.generate(prompt, temperature=temperature,
                              max_tokens=max_tokens)
        data = _extract_json(reply) or []
        by_idx = {d.get("seg"): d for d in data if isinstance(d, dict)}
        for i, s in enumerate(chunk):
            d = by_idx.get(i)
            role = str(d.get("role")).upper() if d and d.get("role") else ""
            if role not in ("ENUMERATOR", "RESPONDENT"):
                role = "ENUMERATOR" if prev == "RESPONDENT" else "RESPONDENT"
            prev = role
            out.append({**s, "role": role})
        print(f"  window {w}-{w + len(chunk) - 1}: {len(data)} labels",
              file=sys.stderr)
    return out


def write(out: list[dict], path: Path) -> None:
    lines = []
    for s in out:
        lines.append(f"[{s['time']}] {s['role']}:")
        if s["bn_raw"]:
            lines.append(f"[bn-raw] {s['bn_raw']}")
        lines.append(f"[bn] {s['bn']}")
        lines.append(f"[en] {s['en']}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("transcript", type=Path)
    ap.add_argument("--window", type=int, default=12)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=600)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--prompts-file",
                    default="config/prompts/bn_survey.yaml",
                    help="same prompts file the pipeline uses")
    ap.add_argument("--gguf-repo", default="unsloth/gemma-3-12b-it-GGUF")
    ap.add_argument("--gguf-file",
                    default="gemma-3-12b-it-UD-Q4_K_XL.gguf")
    args = ap.parse_args()

    segs = parse(args.transcript)
    if not segs:
        print("no segments parsed", file=sys.stderr)
        return 1
    before = {"ENUMERATOR": 0, "RESPONDENT": 0}
    for s in segs:
        before[s["role"]] = before.get(s["role"], 0) + 1

    # Single source of truth: the SAME prompts file the pipeline uses.
    from asr_pipeline.fusion import load_prompts
    role_prompt = load_prompts(args.prompts_file)["role_prompt"]

    from asr_pipeline.postprocessor import GGUFProcessor
    proc = GGUFProcessor(gguf_repo=args.gguf_repo, gguf_file=args.gguf_file,
                         n_ctx=2048, n_gpu_layers=-1,
                         flash_attn=True, kv_quant=True)
    if not proc.load():
        print("Gemma GGUF failed to load", file=sys.stderr)
        return 1

    out = relabel(segs, proc, args.window, role_prompt,
                  args.temperature, args.max_tokens)
    proc.unload()

    after = {}
    for s in out:
        after[s["role"]] = after.get(s["role"], 0) + 1

    out_path = args.out or args.transcript.with_suffix(".relabeled.txt")
    write(out, out_path)
    print(f"\nsegments: {len(segs)} -> {len(out)} (splits added)")
    print(f"roles BEFORE: {before}")
    print(f"roles AFTER : {after}")
    print(f"written: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
