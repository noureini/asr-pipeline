"""Re-translate the [en] lines of a finished transcript with the corrected
prompt — fixes Bengali-script LEAKS in the English (খাজ→work) WITHOUT a
full pipeline re-run. Reuses the existing [bn]; preserves everything else
(timestamps, roles, bn-raw, bn, non-speech markers).

(Limit: words already mis-romanized inside [bn] — e.g. a wrong "PROJECT" —
can't be repaired here; that needs a full re-fusion from the witnesses.)

Usage:
    uv run python scripts/retranslate_en.py <transcript.txt>
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

_PROMPT = """\
Translate this Bengali survey utterance (coastal Bangladesh, colloquial \
regional dialect) into natural, fluent English.
CRITICAL: the output must be 100% ENGLISH — it may NOT contain ANY \
Bengali-script characters. Translate the MEANING of EVERY word, including \
dialectal forms (খাজ/কাম = work; মাইর যায় = the crop is destroyed; কুনু = \
some/any; ট্যালার = tractor; আধি = sharecropping; নেউক্কা = boat; চাল/সাউল \
= rice). Read Bengali compound numbers exactly: সাড়ে X = X-and-a-half, \
দেড় = 1.5, আড়াই = 2.5, but -শো/-শত = hundreds (আড়াইশো = 250, দেড়শো = 150) \
and হাজার = thousand. The ONLY tokens that may stay in Latin are words the \
speaker genuinely said in English (ward, NGO, loan, account, taka, bank \
names like Pubali Bank / Grameen). If a Bengali word is genuinely \
unrecoverable, write [unclear] — NEVER leave the raw Bengali word and \
NEVER guess a number. Output ONLY the English, nothing else.

Bengali: $bn
English:"""


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: retranslate_en.py <transcript.txt>", file=sys.stderr)
        return 1
    path = Path(sys.argv[1])
    from string import Template
    from asr_pipeline.postprocessor import GGUFProcessor
    tpl = Template(_PROMPT)

    proc = GGUFProcessor(
        gguf_repo="unsloth/gemma-3-12b-it-GGUF",
        gguf_file="gemma-3-12b-it-UD-Q4_K_XL.gguf",
        n_ctx=2048, n_gpu_layers=-1, flash_attn=True, kv_quant=True)
    if not proc.load():
        print("Gemma failed to load", file=sys.stderr)
        return 1

    lines = path.read_text(encoding="utf-8").splitlines()
    out, cur_bn, n, fixed = [], None, 0, 0
    for line in lines:
        if line.startswith("[bn]"):
            cur_bn = line[4:].strip()
            out.append(line)
        elif line.startswith("[en]"):
            n += 1
            old = line[4:].strip()
            if cur_bn:
                new = proc.generate(tpl.safe_substitute(bn=cur_bn),
                                    temperature=0.0,
                                    max_tokens=384).strip()
                # strip any stray "English:" echo
                new = re.sub(r"^English:\s*", "", new).strip()
                out.append("[en] " + (new or old))
                if re.search(r"[ঀ-৿]", old) and not re.search(
                        r"[ঀ-৿]", new):
                    fixed += 1
            else:
                out.append(line)
            if n % 25 == 0:
                print(f"  re-translated {n} en lines", file=sys.stderr)
        else:
            out.append(line)
    proc.unload()

    out_path = path.with_suffix(".cleaned.txt")
    out_path.write_text("\n".join(out), encoding="utf-8")
    print(f"\n{n} [en] lines re-translated; {fixed} had Bengali leaks "
          f"now fixed → {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
