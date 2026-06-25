#!/usr/bin/env python3
"""
Stage 4 A/B (diarizer) + titu word-timestamp validation, on the same shared spine.

Two things in one run, in the right order:

  1. TIMESTAMP CHECK (the linchpin). titu fastconformer word timestamps must be
     structurally sound — in-bounds (inside their chunk), monotonic, sensible
     durations — because Stage 5 attributes each WORD to a speaker BY its
     timestamp. Bad timestamps => bad attribution regardless of the diarizer.

  2. DIARIZER A/B. Run pyannote 3.1 vs community-1 (and community-1 with a
     min_speakers=2 floor). For each, attribute the titu words to speakers and
     print a SPEAKER-ATTRIBUTED transcript, so we can eyeball whether each model
     cleanly separates enumerator-question from respondent-answer (the Q&A flow).

Run:
    uv run --no-sync python scripts/ab_diarization.py test_data/15_part1_mid5min.m4a
"""

from __future__ import annotations

import argparse
import statistics
import tempfile
from pathlib import Path

# (label, model_id, min_speakers)  — max stays uncapped (None) for 3rd voices
DIARIZERS = [
    ("3.1",             "pyannote/speaker-diarization-3.1",         None),
    ("community-1",     "pyannote/speaker-diarization-community-1", None),
    ("community-1+min2", "pyannote/speaker-diarization-community-1", 2),
]


def get_titu(chunks):
    """Transcribe the spine with fastconformer; return per-chunk ASRSegments."""
    import torch
    from asr_pipeline.engines.titu_engine import TituEngine

    eng = TituEngine()  # fastconformer default, timestamps on
    eng.load()
    segs = eng.transcribe_batch(chunks, "ben")
    eng.unload()
    torch.cuda.empty_cache()
    return segs


def check_timestamps(segs, chunks) -> list[str]:
    """Structural validation of titu word timestamps."""
    L = ["TIMESTAMP CHECK (titu fastconformer word timestamps)"]
    total_words = 0
    with_ts = 0
    durs: list[float] = []
    issues: list[str] = []
    chunks_no_words = 0

    for seg, ch in zip(segs, chunks):
        if not seg.words:
            chunks_no_words += 1
            continue
        prev_end = None
        for w in seg.words:
            total_words += 1
            if w.start_s is None or w.end_s is None:
                continue
            with_ts += 1
            durs.append(w.end_s - w.start_s)
            if w.start_s < ch.start_s - 0.10 or w.end_s > ch.end_s + 0.10:
                issues.append(
                    f"  out-of-bounds: '{w.word}' [{w.start_s:.2f},{w.end_s:.2f}]"
                    f" outside chunk [{ch.start_s:.2f},{ch.end_s:.2f}]")
            if w.end_s < w.start_s:
                issues.append(f"  negative-duration: '{w.word}'")
            if prev_end is not None and w.start_s < prev_end - 0.10:
                issues.append(
                    f"  non-monotonic: '{w.word}' start {w.start_s:.2f}"
                    f" < prev_end {prev_end:.2f}")
            prev_end = w.end_s

    L.append(f"  chunks: {len(chunks)} ({chunks_no_words} with NO words)")
    L.append(f"  words: {total_words} total, {with_ts} with timestamps")
    if durs:
        L.append(f"  word duration s — min {min(durs):.3f} / "
                 f"median {statistics.median(durs):.3f} / max {max(durs):.3f}")
    L.append(f"  structural issues: {len(issues)}")
    L.extend(issues[:15])
    if len(issues) > 15:
        L.append(f"  ... (+{len(issues) - 15} more)")
    L.append("  VERDICT: " + ("CLEAN — timestamps usable for attribution"
                              if not issues else
                              "ISSUES FOUND — investigate before trusting attribution"))
    return L


def run_diarizer(model: str, min_spk, wav_path: Path, work: Path, token):
    import torch
    from asr_pipeline.config import DiarizationConfig
    from asr_pipeline.diarization import create_diarizer

    cfg = DiarizationConfig(model=model, min_speakers=min_spk, auth_token=token)
    d = create_diarizer(cfg, device="cuda", work_dir=work)
    d.load()
    res = d.diarize(wav_path)
    d.unload()
    torch.cuda.empty_cache()
    return res


def attribute(words, segments):
    """Assign each (word, s, e) to the speaker whose segment holds its midpoint
    (nearest segment if it falls in a gap)."""
    out = []
    for word, s, e in words:
        mid = (s + e) / 2.0
        spk = None
        for sg in segments:
            if sg.start_s <= mid <= sg.end_s:
                spk = sg.speaker_id
                break
        if spk is None and segments:
            spk = min(
                segments,
                key=lambda sg: min(abs(mid - sg.start_s), abs(mid - sg.end_s)),
            ).speaker_id
        out.append((spk or "?", word))
    return out


def group_turns(attr):
    turns = []
    cur, buf = None, []
    for spk, word in attr:
        if spk != cur:
            if buf:
                turns.append((cur, " ".join(buf)))
            cur, buf = spk, [word]
        else:
            buf.append(word)
    if buf:
        turns.append((cur, " ".join(buf)))
    return turns


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", type=Path)
    args = ap.parse_args()
    if not args.audio.exists():
        print(f"!! not found: {args.audio}")
        return 1

    import os
    from dotenv import load_dotenv
    load_dotenv()
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("!! HF_TOKEN not set (.env) — pyannote will fail to download")

    from asr_pipeline.config import PreprocessingConfig
    from asr_pipeline.preprocessor import AudioPreprocessor

    work = Path(tempfile.mkdtemp(prefix="ab_diar_"))
    pp = AudioPreprocessor(PreprocessingConfig(), work)
    wav_path, chunks, _ns = pp.preprocess(args.audio)
    print(f"shared spine: {len(chunks)} chunks")

    # ── 1. titu + timestamp check ────────────────────────────────────────
    print("[titu] transcribing spine (fastconformer, timestamps) ...")
    segs = get_titu(chunks)
    ts_report = check_timestamps(segs, chunks)
    print("\n".join("  " + l for l in ts_report))

    words = [(w.word, w.start_s, w.end_s)
             for s in segs for w in s.words
             if w.start_s is not None and w.end_s is not None]

    # ── 2. diarizer A/B ──────────────────────────────────────────────────
    diar_blocks = []
    for label, model, min_spk in DIARIZERS:
        print(f"[diar:{label}] running {model} (min_speakers={min_spk}) ...")
        try:
            res = run_diarizer(model, min_spk, wav_path, work, token)
        except Exception as e:
            diar_blocks.append((label, f"!! FAILED: {type(e).__name__}: {e}", []))
            print(f"   FAILED: {type(e).__name__}: {e}")
            continue
        turns = group_turns(attribute(words, res.segments))
        head = (f"{res.num_speakers} speakers, {len(res.segments)} segments, "
                f"{len(turns)} attributed turns")
        diar_blocks.append((label, head, turns))
        print(f"   {head}")

    # ── report ───────────────────────────────────────────────────────────
    rpt = args.audio.with_suffix(".ab_diar.txt")
    L = [f"Stage 4 A/B + timestamp check  —  {args.audio.name}",
         f"shared spine: {len(chunks)} chunks", "=" * 74, ""]
    L += ts_report
    L += ["", "=" * 74,
          "DIARIZER A/B  — speaker-attributed transcript (titu words by speaker)",
          "=" * 74]
    for label, head, turns in diar_blocks:
        L += ["", f"### {label}   [{head}]", "-" * 74]
        for spk, text in turns:
            L.append(f"{spk}: {text}")
    rpt.write_text("\n".join(L), encoding="utf-8")
    print(f"\n✓ report: {rpt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
