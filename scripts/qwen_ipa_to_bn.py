"""
Convert ZIPA IPA -> Bengali text via local Qwen GGUF.

Reads <partstem>.zipa.jsonl (per-chunk IPA written by zipa_chunks.py),
prompts Qwen with the IPA->Bengali instruction below, writes
<partstem>.ipa2bn.jsonl {idx,start,end,text}. RESUMABLE.

  uv run python scripts/qwen_ipa_to_bn.py --stem <STEM> \
      --out-dir outputs/real_eval_titu_excellent [--limit N]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


_PROMPT = (
    "TASK: Decode a continuous IPA phoneme stream into Bengali "
    "with proper word boundaries.\n\n"
    "Input: an IPA phoneme stream from a Bengali audio recording. "
    "The phonemes are FLAT (no word boundaries). Your job is to "
    "segment them into Bengali WORDS separated by SPACES, using "
    "your Bengali vocabulary and grammar.\n\n"
    "Domain hint (use to disambiguate): household survey interview "
    "from coastal Bangladesh (Shatkhira / Sunamganj). Topics: "
    "family, fishing, shrimp gher, farming, cyclones (Aila, "
    "Amphan, Bulbul), floods, salinity, livelihood, government "
    "aid. Speaker is a regional male in colloquial Bengali.\n\n"
    "Rules:\n"
    "1. Output Bengali sentences with a SPACE between every word. "
    "Never concatenate words.\n"
    "2. Match phoneme stretches against Bengali vocabulary; e.g. "
    "the stretch 'g o r u' decodes to গরু (one word), "
    "'k i n l a m' decodes to কিনলাম, "
    "'b o n n a' decodes to বন্যা, "
    "'a m a r n a m' decodes to আমার নাম (two words).\n"
    "3. Code-switched English words (NGO, loan, ward, number, ok, "
    "family) must be written in ENGLISH (Latin script).\n"
    "4. If a phoneme stretch is genuinely unrecognizable, write "
    "[unclear] — never invent words.\n"
    "5. Output ONLY the decoded Bengali transcript on a SINGLE "
    "LINE — no explanation, no 'IPA:' echo, no preamble, no "
    "comments. Do NOT copy any example fragments from the rules; "
    "decode ONLY the IPA stream given on the IPA line below.\n\n"
    "IPA: {ipa}\n"
    "Bengali (decoded, single line, words space-separated):"
)


def _read_jsonl(p: Path) -> dict:
    m = {}
    if p.exists():
        for ln in p.read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if ln:
                r = json.loads(ln)
                m[r["idx"]] = r
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", required=True)
    ap.add_argument("--out-dir", type=Path,
                    default=Path("outputs/real_eval_titu_excellent"))
    ap.add_argument("--config", type=Path,
                    default=Path("omni_test_1b.yaml"))
    ap.add_argument("--gguf-repo", default="unsloth/Qwen3.5-9B-GGUF")
    ap.add_argument("--gguf-file",
                    default="Qwen3.5-9B-UD-Q4_K_XL.gguf")
    ap.add_argument("--gguf-path", default=None)
    ap.add_argument("--limit", type=int, default=None,
                    help="first N chunks PER PART (validation slice)")
    args = ap.parse_args()

    from asr_pipeline.config import load_config
    from asr_pipeline.postprocessor import QwenCorrector

    cfg = load_config(args.config)
    d = args.out_dir / args.stem
    zfiles = sorted(d.glob("*.zipa.jsonl"))
    if not zfiles:
        sys.exit(f"no *.zipa.jsonl in {d}")

    qcfg = cfg.postprocessing.qwen_corrector
    qcfg.backend = "gguf"
    qcfg.gguf_path = args.gguf_path
    qcfg.gguf_repo = None if args.gguf_path else args.gguf_repo
    qcfg.gguf_file = None if args.gguf_path else args.gguf_file
    qc = QwenCorrector(qcfg)
    if not qc.load():
        sys.exit("Qwen GGUF failed to load")
    gen = qc._llm.generate  # noqa: SLF001 — reuse loaded model

    for zf in zfiles:
        part = zf.name[: -len(".zipa.jsonl")]
        src = _read_jsonl(zf)
        idxs = sorted(src)
        if args.limit is not None:
            idxs = idxs[: args.limit]
        out = d / f"{part}.ipa2bn.jsonl"
        done = _read_jsonl(out)
        todo = [i for i in idxs if i not in done]
        print(f"  {part}: {len(idxs)} chunks "
              f"({len(done)} done) -> {len(todo)} to convert",
              flush=True)
        with out.open("a", encoding="utf-8") as fh:
            for n, i in enumerate(todo, 1):
                r = src[i]
                ipa = (r.get("ipa") or "").strip()
                if not ipa:
                    bn = ""
                else:
                    try:
                        bn = (gen(_PROMPT.format(ipa=ipa),
                                  temperature=0.0,
                                  max_tokens=400) or "").strip()
                    except Exception as e:
                        bn = ""
                        print(f"    WARN idx {i}: {e}", flush=True)
                fh.write(json.dumps(
                    {"idx": i,
                     "start": r.get("start"),
                     "end": r.get("end"),
                     "text": bn}, ensure_ascii=False) + "\n")
                fh.flush()
                if n % 20 == 0 or n == len(todo):
                    print(f"    {n}/{len(todo)} "
                          f"(last: \"{bn[:60]}\")", flush=True)
        print(f"  -> {out}", flush=True)

    qc.unload()
    print("done.")


if __name__ == "__main__":
    main()
