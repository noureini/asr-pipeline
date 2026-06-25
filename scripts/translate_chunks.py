"""Translate cleaned Bengali chunks → English using TigerLLM-9B
(or Qwen3.5-9B), with cross-chunk context for continuity.

Reads: data/p2g/cleanup_<STEM>__<backend>/cleaned.jsonl
Writes: data/p2g/translated_<STEM>__<backend>/translated.jsonl
        {partstem, idx, start, end, bn, en}

Usage:
  uv run python scripts/translate_chunks.py \\
      --stem <STEM> \\
      --cleaned-dir data/p2g/cleanup_<STEM>__tigerllm_9b_it \\
      --out-dir    data/p2g/translated_<STEM>__tigerllm_9b_it
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_GGUF_REPO = "mradermacher/TigerLLM-9B-it-GGUF"
DEFAULT_GGUF_FILE = "TigerLLM-9B-it.Q4_K_M.gguf"


TRANSLATE_PREAMBLE = (
    "You are translating spoken Bengali from a household survey in "
    "coastal Bangladesh into clear, natural English.\n\n"
    "DOMAIN: Survey speakers discuss family, agriculture (চাষ, "
    "ফসল), livelihood (মাছ, মজুর), water/flooding (বন্যা, "
    "ঘূর্ণিঝড়), administrative units (ward, NGO, পৌরসভা, "
    "ইউনিয়ন), money/loans (টাকা, loan).\n\n"
    "RULES:\n"
    "1. Translate the meaning. Do NOT keep Bengali script in the "
    "output unless it is a proper noun.\n"
    "2. Keep English code-switch words from the Bengali input "
    "verbatim (e.g. 'ward', 'NGO', 'loan' stay English).\n"
    "3. Preserve the speaker's tone — colloquial speech, "
    "incomplete sentences, repetitions can be lightly smoothed.\n"
    "4. Do NOT invent content not present in the input.\n"
    "5. Output ONLY the English translation on one line. No "
    "explanations, no quotes, no preamble.\n\n"
)


def render_qwen_prompt(msg: str) -> str:
    return (
        "<|im_start|>user\n" + msg + "<|im_end|>\n"
        "<|im_start|>assistant\n"
        "<think>\n\n</think>\n\n"
    )


def translate_prompt(bn: str, prev_en: str) -> str:
    ctx = (prev_en[-200:] if prev_en else "").strip()
    msg = (TRANSLATE_PREAMBLE
           + (f"PREVIOUS ENGLISH (continuity reference — do NOT "
              f"repeat in your output):\n{ctx}\n\n" if ctx else "")
           + f"BENGALI:\n{bn}\n\n"
           + "ENGLISH: ")
    return render_qwen_prompt(msg)


def strip_artifacts(s: str) -> str:
    """Light artifact cleaning."""
    import re
    s = re.sub(r"</?think\s*>", "", s, flags=re.IGNORECASE)
    s = re.sub(r"<\|.*?\|>", "", s)
    s = re.sub(r"^\s*(ENGLISH|English|english)\s*[:：]\s*", "",
                s, flags=re.MULTILINE)
    s = s.strip().strip("\"'`").strip()
    # Take first line containing Latin chars (avoid model commentary)
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    if not lines:
        return ""
    latin = [ln for ln in lines if any(c.isascii() and c.isalpha()
                                         for c in ln)]
    return latin[0] if latin else lines[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", required=True)
    ap.add_argument("--cleaned-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--gguf-repo", default=DEFAULT_GGUF_REPO)
    ap.add_argument("--gguf-file", default=DEFAULT_GGUF_FILE)
    ap.add_argument("--n-gpu-layers", type=int, default=-1)
    ap.add_argument("--n-ctx", type=int, default=4096)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cleaned_path = args.cleaned_dir / "cleaned.jsonl"
    if not cleaned_path.exists():
        sys.exit(f"missing {cleaned_path}")

    chunks = []
    for ln in cleaned_path.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if ln:
            chunks.append(json.loads(ln))
    chunks.sort(key=lambda c: (c["partstem"], float(c.get("start", 0)),
                                c["idx"]))
    if args.limit:
        chunks = chunks[: args.limit]
    print(f"loaded {len(chunks)} cleaned chunks", flush=True)

    # Resume
    out_path = args.out_dir / "translated.jsonl"
    done: set[tuple[str, int]] = set()
    en_by_part: dict[str, str] = {}
    if out_path.exists():
        for ln in out_path.read_text(
                encoding="utf-8").splitlines():
            ln = ln.strip()
            if ln:
                try:
                    r = json.loads(ln)
                    done.add((r["partstem"], r["idx"]))
                    en_by_part[r["partstem"]] = r.get("en", "")
                except (json.JSONDecodeError, KeyError):
                    continue
    todo = [c for c in chunks
            if (c["partstem"], c["idx"]) not in done]
    print(f"  {len(done)} done, {len(todo)} to translate",
          flush=True)
    if not todo:
        print("nothing to do.")
        return

    from huggingface_hub import hf_hub_download
    from llama_cpp import Llama
    print(f"loading {args.gguf_file} ...", flush=True)
    path = hf_hub_download(repo_id=args.gguf_repo,
                             filename=args.gguf_file)
    llm = Llama(model_path=path, n_ctx=args.n_ctx,
                n_gpu_layers=args.n_gpu_layers, verbose=False)
    print("  model loaded", flush=True)

    t0 = time.time()
    with out_path.open("a", encoding="utf-8") as fh:
        for n, c in enumerate(todo, 1):
            bn = (c.get("cleaned") or c.get("text")
                  or c.get("raw") or "").strip()
            if not bn:
                en = ""
            else:
                prev = en_by_part.get(c["partstem"], "")
                try:
                    out = llm.create_completion(
                        prompt=translate_prompt(bn, prev),
                        max_tokens=args.max_tokens,
                        temperature=0.2, top_p=0.9, top_k=40,
                        repeat_penalty=1.05,
                        stop=["<|im_end|>", "<|endoftext|>",
                              "\nBENGALI:", "\nENGLISH:"],
                    )
                    en = strip_artifacts(
                        out["choices"][0]["text"])
                    if not en:
                        en = "[translation failed]"
                except Exception as e:
                    en = f"[ERR: {type(e).__name__}: {e}]"
            row = {
                "partstem": c["partstem"], "idx": c["idx"],
                "start": float(c.get("start", 0)),
                "end": float(c.get("end", 0)),
                "bn": bn,
                "en": en,
            }
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            en_by_part[c["partstem"]] = en
            if n % 20 == 0 or n == len(todo):
                el = time.time() - t0
                rate = n / max(el, 1e-6)
                eta = (len(todo) - n) / rate if rate else 0
                print(f"  {n}/{len(todo)} ({rate:.1f}/s, "
                      f"eta {eta:.0f}s)", flush=True)

    print(f"\ndone. -> {out_path}")


if __name__ == "__main__":
    main()
