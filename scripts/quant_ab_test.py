"""Quick A/B test: TigerLLM-9B Q4_K_M vs Q5_K_M on the same segments.

Reads N segments from an existing .raw.json output, runs both quantizations
through correction + translation, writes a side-by-side markdown file.

Usage:
    uv run python scripts/quant_ab_test.py \
        --input outputs/v1_deliverable/15_Sunamganj_CA-MMI_URBAN_Male_15_0602_1635/15_Sunamganj_CA-MMI_URBAN_Male_15_0602_1635.raw.json \
        --n 15 \
        --output outputs/quant_ab_test.md
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
import time
from pathlib import Path

# Make src/ importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("quant_ab")


def load_segments(path: Path, n: int) -> list[dict]:
    """Pull N evenly-spaced non-trivial segments from a pipeline output.

    Supports both .raw.json (preferred, structured) and .raw.txt (parsed
    via regex). The .txt format from formatter.py looks like:

        [HH:MM:SS] SPEAKER_XX:
        [bn] <bengali text>

    with indented [NOISE]/[SILENCE]/[MUSIC] markers and a header block.
    """
    import re

    text = path.read_text(encoding="utf-8")

    if path.suffix == ".json" or text.lstrip().startswith("{"):
        data = json.loads(text)
        segs = data.get("segments", [])
    else:
        # Parse the .txt format
        segs = []
        # Pattern: [HH:MM:SS] SPEAKER_XX:\n[bn] text
        pattern = re.compile(
            r"^\[(\d{2}):(\d{2}):(\d{2})\]\s+(\S+):\s*\n\[bn\]\s+(.+?)(?=\n\s*\n|\n\[|\Z)",
            re.MULTILINE | re.DOTALL,
        )
        for m in pattern.finditer(text):
            hh, mm, ss, spk, body = m.groups()
            start_s = int(hh) * 3600 + int(mm) * 60 + int(ss)
            # body may span multiple lines; collapse whitespace
            raw_text = " ".join(body.split())
            segs.append({
                "raw_text": raw_text,
                "start_s": float(start_s),
                "speaker_id": spk,
            })

    # Filter to substantive segments (≥ 5 chars), then sample evenly
    substantive = [s for s in segs if len(s.get("raw_text", "").strip()) >= 5]
    if not substantive:
        raise RuntimeError(
            f"No segments found in {path}. First 500 chars:\n{text[:500]}")
    if len(substantive) <= n:
        return substantive
    step = len(substantive) // n
    return [substantive[i * step] for i in range(n)]


def run_one_quant(
    gguf_file: str,
    segments: list[dict],
    n_ctx: int = 4096,
    n_gpu_layers: int = -1,
) -> tuple[list[str], list[str], float]:
    """Load model, run correct + translate on all segments, unload, return results."""
    from llama_cpp import Llama

    logger.info(f"Loading {gguf_file} ...")
    t0 = time.perf_counter()
    llm = Llama.from_pretrained(
        repo_id="mradermacher/TigerLLM-9B-it-GGUF",
        filename=gguf_file,
        n_ctx=n_ctx,
        n_threads=8,
        n_gpu_layers=n_gpu_layers,
        verbose=False,
    )
    load_s = time.perf_counter() - t0
    logger.info(f"Loaded in {load_s:.1f}s")

    # Tight, lean prompts — same shape as the pipeline uses but inlined here
    CORRECT_PROMPT = (
        "You are correcting a Bengali ASR transcript from an IPA climate-"
        "risk survey in coastal Bangladesh. Fix obvious recognition errors "
        "using your knowledge of standard and colloquial Bengali. Preserve "
        "the speaker's register, hedging, and code-switched English. Do "
        "not add or remove content. Output only the corrected Bengali "
        "text, nothing else.\n\nRaw: {text}\n\nCorrected:"
    )
    TRANSLATE_PROMPT = (
        "Translate this Bengali survey response to faithful English. "
        "Preserve hedging, uncertainty, and code-switched English terms. "
        "Render numbers as digits. Output only the English translation.\n\n"
        "Bengali: {text}\n\nEnglish:"
    )

    corrections, translations = [], []
    seg_times = []
    for i, seg in enumerate(segments, 1):
        raw = seg["raw_text"].strip()
        t_seg = time.perf_counter()

        # Correct
        try:
            out = llm.create_chat_completion(
                messages=[{"role": "user",
                           "content": CORRECT_PROMPT.format(text=raw)}],
                temperature=0.0,
                max_tokens=256,
            )
            corrected = (out["choices"][0]["message"].get("content") or "").strip()
        except Exception as e:
            logger.warning(f"  correct fail seg {i}: {e}")
            corrected = ""
        corrections.append(corrected or raw)

        # Translate the CORRECTED text (or raw if correction failed)
        try:
            out = llm.create_chat_completion(
                messages=[{"role": "user",
                           "content": TRANSLATE_PROMPT.format(
                               text=corrections[-1])}],
                temperature=0.0,
                max_tokens=256,
            )
            translated = (out["choices"][0]["message"].get("content") or "").strip()
        except Exception as e:
            logger.warning(f"  translate fail seg {i}: {e}")
            translated = ""
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

    total_s = sum(seg_times)
    return corrections, translations, total_s


def write_report(
    out_path: Path,
    segments: list[dict],
    q4_corr: list[str],
    q4_trans: list[str],
    q4_time: float,
    q5_corr: list[str],
    q5_trans: list[str],
    q5_time: float,
) -> None:
    """Side-by-side markdown so you can eyeball the differences."""
    lines = []
    lines.append("# TigerLLM-9B Quantization A/B: Q4_K_M vs Q5_K_M\n")
    lines.append(f"- Segments: **{len(segments)}**")
    lines.append(f"- Q4_K_M total: **{q4_time:.1f}s** ({q4_time/len(segments):.1f}s/seg)")
    lines.append(f"- Q5_K_M total: **{q5_time:.1f}s** ({q5_time/len(segments):.1f}s/seg)")
    lines.append(f"- Speed delta: **{(q5_time-q4_time)/q4_time*100:+.0f}%** for Q5\n")
    lines.append("---\n")

    for i, seg in enumerate(segments):
        ts = seg.get("start_s", 0)
        mm, ss = divmod(int(ts), 60)
        hh, mm = divmod(mm, 60)
        spk = seg.get("speaker_id", "?")
        raw = seg["raw_text"].strip()

        lines.append(f"## Segment {i+1} — [{hh:02d}:{mm:02d}:{ss:02d}] {spk}\n")
        lines.append(f"**[bn-raw]** {raw}\n")
        lines.append(f"### Q4_K_M")
        lines.append(f"**[bn]** {q4_corr[i]}")
        lines.append(f"**[en]** {q4_trans[i]}\n")
        lines.append(f"### Q5_K_M")
        lines.append(f"**[bn]** {q5_corr[i]}")
        lines.append(f"**[en]** {q5_trans[i]}\n")
        lines.append("---\n")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Wrote report: {out_path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path,
                    help="Path to .raw.json from a prior pipeline run")
    ap.add_argument("--n", type=int, default=15,
                    help="Number of segments to sample (default 15)")
    ap.add_argument("--output", type=Path,
                    default=Path("outputs/quant_ab_test.md"))
    args = ap.parse_args()

    segments = load_segments(args.input, args.n)
    logger.info(f"Sampled {len(segments)} segments from {args.input.name}")

    logger.info("=== Q4_K_M pass ===")
    q4_corr, q4_trans, q4_time = run_one_quant(
        "TigerLLM-9B-it.Q4_K_M.gguf", segments)

    logger.info("=== Q5_K_M pass ===")
    q5_corr, q5_trans, q5_time = run_one_quant(
        "TigerLLM-9B-it.Q5_K_M.gguf", segments)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_report(
        args.output, segments,
        q4_corr, q4_trans, q4_time,
        q5_corr, q5_trans, q5_time,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
