"""
Part B — standalone whole-document transcript polisher (NOT pipeline).

Takes the cached ASR chunks of an interview (the *.chunks.jsonl
written by eval_real_survey_inproc.py — idx + per-part start/end + raw
text) and runs a LOCAL Qwen GGUF over SLIDING WINDOWS of consecutive
segments to jointly produce, per segment:

  - speaker    : enumerator | respondent  (LLM text-based turn tagging
                 — heuristic, for human review; NOT acoustic diarization)
  - corrected  : ASR/phonetic fixes, code-switched English restored,
                 dialect standardized; no additions/translation
  - english    : faithful English translation

GLOBAL TIMESTAMPS: a survey interview is recorded in multiple parts
(…​.m4a, …​_1.m4a). Each part's VAD timestamps restart at 0, so they
are NOT directly joinable to survey metadata. This script ffprobes
each part's true duration and offsets later parts, emitting
interview-relative `start_s/end_s` (seconds from interview start) plus
`HH:MM:SS.mmm`, plus the originating `part` + within-part time for
audit. Those global seconds are the join key for downstream metadata.

Timestamp granularity = VAD segment (≤ ~30 s), not word-level
(forced alignment is intentionally off for speed).

Windows carry the previous turns as rolling context (speaker
continuity). On-device (PII-safe). RESUMABLE: per-segment results are
checkpointed; a re-run skips finished windows.

Primary deliverable: an ALIGNED REVIEW TABLE (one row per segment)
as CSV + XLSX for a human to qualitatively correct, plus
turn-segmented, timestamped polished_corrected.txt /
polished_translation.txt.

Usage:
  uv run python scripts/polish_transcript.py \
      --stem 07_Shatkhira_CA-MMI_URBAN_Male_07_0202_1716
  # resume after an interruption: run the same command again.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_OUT = Path("outputs/real_eval_inproc")
DEFAULT_AUDIO = Path("for testing (Noureini)")


def _ffprobe_dur(audio: Path) -> float | None:
    """True duration (s) of an audio file via ffprobe, or None."""
    if not audio.exists():
        return None
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries",
             "format=duration", "-of", "default=nk=1:nw=1",
             str(audio)],
            capture_output=True, text=True, check=True)
        return float(r.stdout.strip())
    except Exception:
        return None


def _hms(s):
    if s is None:
        return ""
    s = float(s)
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    sec = s % 60
    return f"{h:02d}:{m:02d}:{sec:06.3f}"


def load_segments(stem_dir: Path, txt: Path | None,
                  audio_folder: Path):
    """Ordered segments with GLOBAL (interview-relative) timestamps.

    Each: {idx, part, p_start, p_end, g_start, g_end, raw}.
    Global time = sum of true durations of preceding parts + the
    chunk's within-part time. Falls back to the max within-part
    end of preceding parts if a part's audio can't be probed.
    """
    if txt is not None:
        segs = []
        for i, line in enumerate(
                txt.read_text(encoding="utf-8").splitlines()):
            line = line.strip()
            if line:
                segs.append({"idx": i, "part": txt.name,
                             "p_start": None, "p_end": None,
                             "g_start": None, "g_end": None,
                             "raw": line})
        return segs

    parts = sorted(stem_dir.glob("*.chunks.jsonl"))
    if not parts:
        print(f"ERROR: no *.chunks.jsonl in {stem_dir} "
              f"(run eval_real_survey_inproc.py first), or pass --txt")
        sys.exit(1)

    segs = []
    base = 0
    offset = 0.0          # global seconds before the current part
    for pf in parts:                       # '.'<'_' => main part first
        audiostem = pf.name[:-len(".chunks.jsonl")]
        rows = []
        for line in pf.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
        rows.sort(key=lambda x: x["idx"])

        audio = audio_folder / f"{audiostem}.m4a"
        dur = _ffprobe_dur(audio)
        max_end = max((r.get("end") or 0.0) for r in rows) if rows \
            else 0.0
        if dur is None:
            dur = max_end
            print(f"  WARN: could not probe {audio.name}; using last "
                  f"chunk end ({dur:.1f}s) as part length — global "
                  f"offset may be slightly short.")
        else:
            print(f"  part {audiostem}: audio {dur:.1f}s, "
                  f"offset +{offset:.1f}s")

        for r in rows:
            ps, pe = r.get("start"), r.get("end")
            segs.append({
                "idx": base + r["idx"],
                "part": audiostem,
                "p_start": ps, "p_end": pe,
                "g_start": None if ps is None else offset + ps,
                "g_end": None if pe is None else offset + pe,
                "raw": (r.get("text") or "").strip(),
            })
        offset += dur
        base += 100000
    return segs


def make_windows(segs, budget_chars: int):
    windows, cur, n = [], [], 0
    for s in segs:
        ln = len(s["raw"]) + 1
        if cur and n + ln > budget_chars:
            windows.append(cur)
            cur, n = [], 0
        cur.append(s)
        n += ln
    if cur:
        windows.append(cur)
    return windows


_PROMPT = """\
You clean a Bengali survey interview transcript produced by speech
recognition. It is a conversation between an ENUMERATOR (asks scripted
questions) and a RESPONDENT (answers).

For EACH numbered segment below, decide:
- speaker: "enumerator" or "respondent" (use the conversation flow;
  questions/instructions = enumerator, answers/stories = respondent;
  keep the same speaker across a continued turn).
- bn: INTELLIGENT-VERBATIM correction. Fix ONLY clear speech-
  recognition errors — garbled/non-word tokens and obvious phonetic
  slips the recognizer made. PRESERVE the speaker's words exactly as
  spoken: keep dialect and regional/colloquial grammar and forms
  (e.g. keep মানুহ, এতিয়া, নাই, দিছি, পাঠাইছি), keep hedges, false
  starts and repetitions. Do NOT standardize dialect, do NOT smooth
  grammar, do NOT rephrase / condense / "improve" into book-like
  prose, do NOT add or invent. CODE-SWITCHED ENGLISH: ONLY when you
  are CONFIDENT a token is an English word the recognizer wrote
  phonetically in Bengali script, output it in correct Latin English
  (e.g. ভেকসিন->vaccine, স্পোর্টস->sports, সিস্টেম->system,
  স্যালারি->salary, লোন->loan, ইন্টারেস্ট->interest, এনজিও->NGO,
  বিকাশ->bKash). Keep genuine Bengali words and Bengali
  administrative terms in Bengali (e.g. ইউনিয়ন পরিষদ, ওয়ার্ড). If
  a span is garbled / unintelligible / you are NOT confident what
  was said, leave it EXACTLY as-is — the raw ASR is kept for human
  audio review; never turn noise into a confident word.
- en: a faithful English translation of bn. PRESERVE the speaker's
  hedging, uncertainty and register (e.g. "maybe", "God willing",
  "I can't say") — do NOT smooth into formal/essay prose, interpret,
  or summarize. Keep English words, numbers, proper nouns. If bn is
  garbled/unintelligible, output "" for en.

{context}Segments:
{segments}

Output ONLY a JSON array, one object per segment, in order, each
exactly: {{"id": <int>, "speaker": "...", "bn": "...", "en": "..."}}
No reasoning, no preamble, no markdown — just the JSON array."""


def _extract_json_array(text: str):
    text = re.sub(r"<think>.*?</think>", "", text,
                  flags=re.DOTALL).strip()
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(),
                  flags=re.MULTILINE).strip()
    i, j = text.find("["), text.rfind("]")
    if i == -1 or j == -1 or j < i:
        return None
    try:
        return json.loads(text[i:j + 1])
    except Exception:
        return None


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stem", required=True)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT,
                   help="Base dir; reads/writes <out-dir>/<stem>/")
    p.add_argument("--audio-folder", type=Path, default=DEFAULT_AUDIO,
                   help="Folder with the .m4a parts (for true "
                        "per-part durations -> global timestamps).")
    p.add_argument("--txt", type=Path, default=None,
                   help="Alt input: a plain transcript .txt (one "
                        "segment per line, NO timestamps).")
    p.add_argument("--gguf-repo", default="unsloth/Qwen3.5-9B-GGUF")
    p.add_argument("--gguf-file", default="Qwen3.5-9B-UD-Q4_K_XL.gguf",
                   help="Unsloth-dynamic ~lossless 6GB. OOM on 8GB? use "
                        "Qwen3.5-9B-UD-Q3_K_XL.gguf and/or lower --n-ctx.")
    p.add_argument("--gguf-path", default=None)
    p.add_argument("--n-ctx", type=int, default=6144,
                   help="Lower (e.g. 4096) if VRAM-tight with the 6GB "
                        "model; raise to 8192 if you have headroom.")
    p.add_argument("--window-chars", type=int, default=1200,
                   help="Max raw chars of source per window.")
    p.add_argument("--ctx-turns", type=int, default=2,
                   help="Previous corrected turns carried as context.")
    p.add_argument("--max-tokens", type=int, default=2048)
    args = p.parse_args()

    stem_dir = args.out_dir / args.stem
    stem_dir.mkdir(parents=True, exist_ok=True)
    segs = load_segments(stem_dir, args.txt, args.audio_folder)
    print(f"Loaded {len(segs)} segments for {args.stem}")
    windows = make_windows(segs, args.window_chars)
    print(f"{len(windows)} windows (~{args.window_chars} chars each)")

    ck = stem_dir / "polished.jsonl"
    done: dict[int, dict] = {}
    if ck.exists():
        for line in ck.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                done[r["idx"]] = r
            except Exception:
                pass
    print(f"  {len(done)} segments already polished (resuming)")

    need = [w for w in windows
            if any(s["idx"] not in done for s in w)]
    if need:
        from llama_cpp import Llama
        src = args.gguf_path or f"{args.gguf_repo}/{args.gguf_file}"
        print(f"Loading GGUF [{src}] n_ctx={args.n_ctx} ...", flush=True)
        common = dict(n_ctx=args.n_ctx, n_gpu_layers=-1,
                      n_threads=8, verbose=False)
        if args.gguf_path:
            llm = Llama(model_path=str(args.gguf_path), **common)
        else:
            llm = Llama.from_pretrained(
                repo_id=args.gguf_repo, filename=args.gguf_file,
                **common)
        print("  GGUF loaded.", flush=True)

        t0 = time.time()
        with ck.open("a", encoding="utf-8") as fh:
            for wi, w in enumerate(windows, 1):
                if all(s["idx"] in done for s in w):
                    continue
                prev = [done[s["idx"]] for s in segs
                        if s["idx"] in done][-args.ctx_turns:]
                ctx = ""
                if prev:
                    ctx = "Previous turns (context, do NOT re-output):\n"
                    for r in prev:
                        ctx += (f'  {r.get("speaker","?")}: '
                                f'{r.get("bn","")[:200]}\n')
                    ctx += "\n"
                seg_lines = "\n".join(
                    f'{s["idx"]}: {s["raw"]}' for s in w if s["raw"])
                if not seg_lines.strip():
                    for s in w:
                        rec = {"idx": s["idx"], "speaker": "?",
                               "bn": s["raw"], "en": ""}
                        fh.write(json.dumps(rec,
                                 ensure_ascii=False) + "\n")
                        fh.flush()
                        done[s["idx"]] = rec
                    continue
                prompt = _PROMPT.format(context=ctx, segments=seg_lines)
                try:
                    out = llm.create_completion(
                        prompt + "<think>\n\n</think>\n\n",
                        temperature=0.0,
                        max_tokens=args.max_tokens, stop=["</s>"])
                    raw_out = out["choices"][0]["text"]
                except Exception as e:
                    print(f"  window {wi}: gen failed ({e}); "
                          f"raw fallback", flush=True)
                    raw_out = ""
                arr = _extract_json_array(raw_out) or []
                by_id = {}
                for o in arr:
                    try:
                        by_id[int(o["id"])] = o
                    except Exception:
                        pass
                for s in w:
                    if s["idx"] in done:
                        continue
                    o = by_id.get(s["idx"], {})
                    rec = {
                        "idx": s["idx"],
                        "speaker": str(o.get("speaker", "?")),
                        "bn": (o.get("bn") or s["raw"]).strip(),
                        "en": (o.get("en") or "").strip(),
                    }
                    fh.write(json.dumps(rec,
                             ensure_ascii=False) + "\n")
                    fh.flush()
                    done[s["idx"]] = rec
                if wi % 5 == 0 or wi == len(windows):
                    el = (time.time() - t0) / 60
                    print(f"  window {wi}/{len(windows)} "
                          f"({len(done)}/{len(segs)} segs, "
                          f"{el:.1f} min)", flush=True)

    # ── Build a pipeline-native TranscriptResult and write the SAME
    #    JSON + TXT (+ SRT) the production pipeline emits. Global
    #    interview time goes in start_s/end_s (numeric join key);
    #    source_file = originating part for audit. ─────────────────────
    from asr_pipeline.formatter import write_transcript
    from asr_pipeline.models import (
        AudioQualityMetrics,
        NonSpeechSegment,
        ProcessedSegment,
        TranscriptionStyle,
        TranscriptMetadata,
        TranscriptResult,
    )

    _SPK = {"enumerator": "ENUMERATOR", "respondent": "RESPONDENT"}

    def spk(v: str) -> str:
        return _SPK.get(str(v).strip().lower(), "UNKNOWN")

    psegs: list[ProcessedSegment] = []
    for s in segs:
        r = done.get(s["idx"], {})
        gs = 0.0 if s["g_start"] is None else round(s["g_start"], 3)
        ge = gs if s["g_end"] is None else round(s["g_end"], 3)
        en = (r.get("en") or "").strip()
        psegs.append(ProcessedSegment(
            segment_id=s["idx"],
            start_s=gs,                 # GLOBAL interview seconds
            end_s=ge,
            speaker_id=spk(r.get("speaker", "?")),
            language="ben",
            raw_text=s["raw"],
            corrected_text=(r.get("bn") or s["raw"]).strip(),
            english_translation=en,
            refined_translation=en,     # -> formatter prints [en] line
            confidence=0.0,
            source_file=s["part"],
            absolute_start=None,        # None -> formatter uses
            absolute_end=None,          #          global HH:MM:SS
        ))

    # Inter-segment silence (same part) as non-speech, global time.
    ns: list[NonSpeechSegment] = []
    for a, b in zip(psegs, psegs[1:]):
        if (b.source_file == a.source_file
                and b.start_s - a.end_s > 1.0):
            ns.append(NonSpeechSegment(
                start_s=a.end_s, end_s=b.start_s,
                region_type="silence",
                duration_s=round(b.start_s - a.end_s, 3),
                source_file=a.source_file))

    total = max((p.end_s for p in psegs), default=0.0)
    speech = sum(p.end_s - p.start_s for p in psegs)
    nspeech = sum(n.duration_s for n in ns)
    aq = AudioQualityMetrics(
        total_duration_s=round(total, 1),
        speech_duration_s=round(speech, 1),
        non_speech_duration_s=round(nspeech, 1),
        speech_ratio=round(speech / total, 3) if total else 0.0,
        num_speech_segments=len(psegs),
        num_non_speech_segments=len(ns),
        avg_speech_segment_s=round(speech / len(psegs), 1)
        if psegs else 0.0,
        longest_silence_s=round(
            max((n.duration_s for n in ns), default=0.0), 1),
    )
    meta = TranscriptMetadata(
        project_name="",
        audio_file=args.stem,
        duration_s=round(total, 1),
        recording_date="",
        languages_detected=["Bengali"],
        num_speakers=len({p.speaker_id for p in psegs}),
        transcription_style=TranscriptionStyle.INTELLIGENT_VERBATIM,
        asr_engines_used=["Omnilingual (cached ASR)"],
        postprocessing_stages=[
            "Qwen GGUF correction",
            "Qwen GGUF translation",
            "LLM text speaker turn-tagging",
        ],
        audio_quality=aq,
        interview=None,
    )
    result = TranscriptResult(metadata=meta, segments=psegs,
                              non_speech_segments=ns)

    written = write_transcript(
        result, stem_dir, args.stem, output_format="all",
        timestamp_fmt="HH:MM:SS", include_raw=True,
        include_translation=True)

    print(f"\n{'=' * 60}")
    print(f"Polished {len(segs)} segments -> {stem_dir}/")
    for w in written:
        print(f"  {w.name:<24} (pipeline-native format)")
    print(f"  polished.jsonl           (resumable checkpoint)")
    print("NOTE: segments[].start_s/end_s are GLOBAL interview seconds "
          "(numeric join key for survey metadata); source_file = "
          "originating part. speaker_id is an LLM text turn-guess for "
          "review, NOT acoustic diarization.")


if __name__ == "__main__":
    main()
