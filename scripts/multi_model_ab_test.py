"""N-way local LLM comparison on the same Bengali survey segments.

Runs an arbitrary list of GGUFs (TigerLLM Q4, TigerLLM Q5, Gemma 4 12B, …)
sequentially against the same N segments from a pipeline raw transcript,
then writes a side-by-side markdown report.

Edit MODELS below to add/remove models. The harness handles per-family
chat templates (Gemma vs raw completion) and unloads cleanly between
models so 8 GB VRAM is enough.

Usage:
    uv run python scripts/multi_model_ab_test.py \
        --input outputs/v1_deliverable/15_Sunamganj_CA-MMI_URBAN_Male_15_0602_1635/15_Sunamganj_CA-MMI_URBAN_Male_15_0602_1635.raw.txt \
        --n 15 \
        --output outputs/multi_model_ab_test.md
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("multi_ab")


# ─────────────────────────────────────────────────────────────────────
# Edit this list to control which models the harness runs.
# Each entry: (label, hf_repo, gguf_filename, n_ctx, n_gpu_layers)
#   n_ctx        : per-model context (lower → smaller KV cache).
#   n_gpu_layers : -1 = all layers on GPU (fits 8 GB). A positive int
#                  offloads only that many layers to GPU and runs the
#                  rest on CPU/RAM — needed for the 24B sarvam-m, which
#                  is far bigger than 8 GB VRAM. Slow but works for a
#                  small quality test.
# ─────────────────────────────────────────────────────────────────────
MODELS: list[tuple[str, str, str, int, int]] = [
    # ── 24B Sarvam-M — strongest open Indic model. Won't fit 8 GB, so
    #    partial offload: ~16 layers on GPU, rest on CPU RAM. SLOW
    #    (~2-5 tok/s) but fine for a few test segments. Needs ~16 GB RAM.
    ("Sarvam-M-24B-Q4",
     "bartowski/sarvamai_sarvam-m-GGUF",
     "sarvamai_sarvam-m-Q4_K_M.gguf",
     2048,
     16),
    # ── Local-fit baselines (re-run for direct comparison) ──
    ("Gemma4-12B-UD-Q4",
     "unsloth/gemma-4-12b-it-GGUF",
     "gemma-4-12b-it-UD-Q4_K_XL.gguf",
     1536,
     -1),
    ("TigerLLM-Q5",
     "mradermacher/TigerLLM-9B-it-GGUF",
     "TigerLLM-9B-it.Q5_K_M.gguf",
     4096,
     -1),
]


CORRECT_PROMPT = (
    "You are correcting a Bengali ASR transcript from an IPA climate-"
    "risk survey in coastal Bangladesh (Satkhira / Sunamganj). Speakers "
    "are ENUMERATOR and RESPONDENT discussing demographics, livelihood, "
    "cyclones, NGOs, loans.\n\n"
    "Fix obvious recognition errors using your knowledge of standard "
    "and colloquial Bengali. Restore code-switched English (ward, NGO, "
    "loan, family) in Latin script. Preserve register, hedging, "
    "repetition. If you genuinely can't recover a span, mark it "
    "[অস্পষ্ট] — do NOT invent numbers, names, or specifics. Output "
    "only the corrected Bengali, nothing else.\n\n"
    "Raw: {text}\n\nCorrected:"
)

TRANSLATE_PROMPT = (
    "Translate this Bengali survey response to faithful English. "
    "Preserve hedging, uncertainty, and code-switched English terms. "
    "Render numbers as digits. Mark unclear spans as [unclear]. Do not "
    "invent specifics. Output only the English translation.\n\n"
    "Bengali: {text}\n\nEnglish:"
)


def load_segments(path: Path, n: int) -> list[dict]:
    """Parse a .raw.txt or .raw.json and sample N substantive segments."""
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".json" or text.lstrip().startswith("{"):
        data = json.loads(text)
        segs = data.get("segments", [])
    else:
        pattern = re.compile(
            r"^\[(\d{2}):(\d{2}):(\d{2})\]\s+(\S+):\s*\n\[bn\]\s+(.+?)"
            r"(?=\n\s*\n|\n\[|\Z)",
            re.MULTILINE | re.DOTALL,
        )
        segs = []
        for m in pattern.finditer(text):
            hh, mm, ss, spk, body = m.groups()
            segs.append({
                "raw_text": " ".join(body.split()),
                "start_s": int(hh) * 3600 + int(mm) * 60 + int(ss),
                "speaker_id": spk,
            })
    substantive = [s for s in segs if len(s.get("raw_text", "").strip()) >= 5]
    if not substantive:
        raise RuntimeError(f"No segments parsed from {path}")
    if len(substantive) <= n:
        return substantive
    step = len(substantive) // n
    return [substantive[i * step] for i in range(n)]


def detect_family(repo: str, file: str) -> str:
    """Pick chat-template strategy from repo/file naming."""
    name = (repo + " " + file).lower()
    if "tiger" in name or "gemma" in name:
        return "gemma"
    if "qwen" in name:
        return "qwen"
    return "generic"


def run_model(
    label: str,
    repo: str,
    file: str,
    n_ctx: int,
    n_gpu_layers: int,
    segments: list[dict],
) -> tuple[list[str], list[str], float, float]:
    """Load → correct + translate each segment → unload. Returns
    (corrections, translations, load_seconds, inference_seconds)."""
    from llama_cpp import Llama

    logger.info(f"=== {label} ({repo} / {file}) ===")
    logger.info(f"Loading (n_ctx={n_ctx}, n_gpu_layers={n_gpu_layers}) ...")
    t0 = time.perf_counter()
    try:
        llm = Llama.from_pretrained(
            repo_id=repo,
            filename=file,
            n_ctx=n_ctx,
            n_threads=8,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )
    except Exception as e:
        logger.error(f"  LOAD FAILED for {label}: {e}")
        return [], [], 0.0, 0.0
    load_s = time.perf_counter() - t0
    logger.info(f"Loaded in {load_s:.1f}s")

    family = detect_family(repo, file)
    has_chat = bool(
        (getattr(llm, "metadata", {}) or {}).get("tokenizer.chat_template")
    )
    use_chat = has_chat and family in ("gemma", "generic")
    logger.info(f"family={family} chat_template={has_chat} → use_chat={use_chat}")

    def gen(prompt: str) -> str:
        try:
            if use_chat:
                out = llm.create_chat_completion(
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=256,
                )
                return (out["choices"][0]["message"].get("content") or "").strip()
            elif family == "qwen":
                out = llm.create_completion(
                    prompt + "<think>\n\n</think>\n\n",
                    temperature=0.0,
                    max_tokens=256,
                    stop=["</output>"],
                )
                text = out["choices"][0]["text"]
                text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
                text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL)
                return text.strip()
            else:
                out = llm.create_completion(
                    prompt, temperature=0.0, max_tokens=256, stop=["\n\n\n"]
                )
                return out["choices"][0]["text"].strip()
        except Exception as e:
            logger.warning(f"  gen fail: {e}")
            return ""

    corrections, translations = [], []
    seg_times = []
    for i, seg in enumerate(segments, 1):
        raw = seg["raw_text"].strip()
        t_seg = time.perf_counter()
        corrected = gen(CORRECT_PROMPT.format(text=raw)) or raw
        translated = gen(TRANSLATE_PROMPT.format(text=corrected))
        corrections.append(corrected)
        translations.append(translated)
        seg_times.append(time.perf_counter() - t_seg)
        logger.info(f"  [{i}/{len(segments)}] {seg_times[-1]:.1f}s")

    del llm
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass

    return corrections, translations, load_s, sum(seg_times)


def write_report(
    out_path: Path,
    segments: list[dict],
    results: dict[str, dict],
) -> None:
    """Side-by-side markdown for all models."""
    labels = list(results.keys())
    lines = [f"# Multi-model A/B on Bengali survey segments\n"]
    lines.append(f"- Segments: **{len(segments)}**")
    lines.append("- Per-model timing (excluding load):\n")
    for lbl in labels:
        r = results[lbl]
        if not r["corrections"]:
            lines.append(f"  - **{lbl}**: LOAD FAILED")
            continue
        avg = r["inference_s"] / len(segments)
        lines.append(
            f"  - **{lbl}**: load {r['load_s']:.0f}s · "
            f"inference {r['inference_s']:.0f}s "
            f"(avg {avg:.1f}s/seg)"
        )
    lines.append("\n---\n")

    for i, seg in enumerate(segments):
        ts = seg.get("start_s", 0)
        mm, ss = divmod(int(ts), 60)
        hh, mm = divmod(mm, 60)
        spk = seg.get("speaker_id", "?")
        lines.append(f"## Segment {i+1} — [{hh:02d}:{mm:02d}:{ss:02d}] {spk}\n")
        lines.append(f"**[bn-raw]** {seg['raw_text'].strip()}\n")
        for lbl in labels:
            r = results[lbl]
            if not r["corrections"]:
                continue
            lines.append(f"### {lbl}")
            lines.append(f"**[bn]** {r['corrections'][i]}")
            lines.append(f"**[en]** {r['translations'][i]}\n")
        lines.append("---\n")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Wrote report: {out_path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--output", type=Path,
                    default=Path("outputs/multi_model_ab_test.md"))
    args = ap.parse_args()

    segments = load_segments(args.input, args.n)
    logger.info(f"Sampled {len(segments)} segments from {args.input.name}")

    results: dict[str, dict] = {}
    for label, repo, file, n_ctx, n_gpu_layers in MODELS:
        corr, trans, load_s, inf_s = run_model(
            label, repo, file, n_ctx, n_gpu_layers, segments)
        results[label] = {
            "corrections": corr,
            "translations": trans,
            "load_s": load_s,
            "inference_s": inf_s,
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_report(args.output, segments, results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
