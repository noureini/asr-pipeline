"""
Deterministic transcript assembler (NO LLM, NO windowing).

Joins Path A's per-segment artifacts by idx, computes GLOBAL
interview timestamps (ffprobe part offsets), and emits the
pipeline-native JSON / TXT / SRT via the project's own
formatter.write_transcript — byte-compatible with the production
pipeline output.

Inputs (in <out-dir>/<stem>/, produced by eval_real_survey_inproc):
  *.chunks.jsonl      raw ASR        (idx,start,end,text)   [required]
  *.corrected.jsonl   Qwen corrected (idx,text)             [optional]
  *.translated.jsonl  Qwen English   (idx,text)             [optional]
  *.diar.jsonl        pyannote turns (speaker,start,end)    [optional]

Speaker labelling (when *.diar.jsonl is present, produced by
scripts/diarize_parts.py):
  - Each ASR segment is assigned the diarized speaker with the
    largest time-overlap (dominant-overlap).
  - Clusters -> roles PER PART (pyannote labels are NOT consistent
    across separately-diarized parts):
      * Anchor part (first part): the speaker dominating the
        scripted consent-intro window (--anchor-window s, default
        360 = ~6 min, matching the ~5.55 min GT consent monologue)
        = ENUMERATOR; next-most-speech = RESPONDENT; rest = OTHER.
      * Later parts (no scripted anchor): speech-share heuristic —
        most-speech = RESPONDENT, next = ENUMERATOR, rest = OTHER.
        Documented as heuristic; survey team should spot-check.
  - Turn-straddling chunks (a runner-up speaker covers >=30% of
    the segment) get a trailing "?" (e.g. RESPONDENT?) so the
    survey team can see ambiguous attributions at a glance.
  - Segments with no diarized overlap -> UNKNOWN.
If no *.diar.jsonl exists, speaker_id falls back to UNKNOWN.

Usage:
  uv run python scripts/diarize_parts.py  --stem <STEM>   # first
  uv run python scripts/assemble_transcript.py --stem <STEM>
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

OUT = Path("outputs/real_eval_inproc")
AUDIO = Path("for testing (Noureini)")


def _jsonl(p: Path) -> dict:
    m: dict[int, dict] = {}
    if not p.exists():
        return m
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


def _diar(p: Path) -> list[dict]:
    """Load pyannote turns (LOCAL part seconds), sorted by start."""
    turns: list[dict] = []
    if not p.exists():
        return turns
    for ln in p.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            r = json.loads(ln)
            turns.append({"speaker": r["speaker"],
                          "start": float(r["start"]),
                          "end": float(r["end"])})
        except Exception:
            pass
    turns.sort(key=lambda t: (t["start"], t["end"]))
    return turns


def _assign_speakers(
    local: list[tuple],
    turns: list[dict],
    is_anchor_part: bool,
    anchor_window: float,
    straddle_frac: float = 0.30,
) -> list[str]:
    """
    Map diarized clusters -> ENUMERATOR/RESPONDENT/OTHER for ONE part
    and label each ASR segment by dominant time-overlap.

    `local` items are (idx, lstart, lend, ...) in LOCAL part seconds.
    Returns one label per `local` item (parallel list). Turn-straddle
    segments get a trailing "?".
    """
    if not turns:
        return ["UNKNOWN"] * len(local)

    total: dict[str, float] = defaultdict(float)
    anchor: dict[str, float] = defaultdict(float)
    for t in turns:
        total[t["speaker"]] += t["end"] - t["start"]
        if t["start"] < anchor_window:
            anchor[t["speaker"]] += (
                min(t["end"], anchor_window) - t["start"])
    by_speech = sorted(total, key=lambda s: -total[s])

    role: dict[str, str] = {}
    if is_anchor_part and anchor:
        enum = max(anchor, key=lambda s: anchor[s])
        rest = [s for s in by_speech if s != enum]
        role[enum] = "ENUMERATOR"
        if rest:
            role[rest[0]] = "RESPONDENT"
        for s in rest[1:]:
            role[s] = "OTHER"
    else:
        # No scripted anchor: in a survey the respondent speaks the
        # most, the enumerator asks (shorter) questions. Heuristic.
        if by_speech:
            role[by_speech[0]] = "RESPONDENT"
        if len(by_speech) > 1:
            role[by_speech[1]] = "ENUMERATOR"
        for s in by_speech[2:]:
            role[s] = "OTHER"

    out: list[str] = []
    for item in local:
        ls, le = item[1], item[2]
        ov: dict[str, float] = defaultdict(float)
        for t in turns:
            o = min(le, t["end"]) - max(ls, t["start"])
            if o > 0:
                ov[t["speaker"]] += o
        if not ov:
            out.append("UNKNOWN")
            continue
        ranked = sorted(ov, key=lambda s: -ov[s])
        lab = role.get(ranked[0], "OTHER")
        segdur = max(le - ls, 1e-6)
        runner = ov[ranked[1]] if len(ranked) > 1 else 0.0
        if runner >= straddle_frac * segdur:
            lab += "?"  # turn-straddle / ambiguous attribution
        out.append(lab)
    return out


def _dur(audio: Path) -> float | None:
    if not audio.exists():
        return None
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries",
             "format=duration", "-of", "default=nk=1:nw=1",
             str(audio)], capture_output=True, text=True, check=True)
        return float(r.stdout.strip())
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", required=True)
    ap.add_argument("--out-dir", type=Path, default=OUT)
    ap.add_argument("--audio-folder", type=Path, default=AUDIO)
    ap.add_argument("--asr-model", default="omniASR_LLM_1B_v2")
    ap.add_argument("--anchor-window", type=float, default=360.0,
                    help="scripted consent-intro window (s) used to "
                         "identify the ENUMERATOR in the first part")
    ap.add_argument("--no-diar", action="store_true",
                    help="ignore *.diar.jsonl; speaker_id=UNKNOWN")
    args = ap.parse_args()

    d = args.out_dir / args.stem
    cfiles = sorted(d.glob("*.chunks.jsonl"))
    if not cfiles:
        sys.exit(f"no *.chunks.jsonl in {d} — run eval_real_survey_"
                 f"inproc (ASR) first")

    from asr_pipeline.formatter import write_transcript
    from asr_pipeline.models import (
        AudioQualityMetrics,
        NonSpeechSegment,
        ProcessedSegment,
        TranscriptionStyle,
        TranscriptMetadata,
        TranscriptResult,
    )

    psegs: list[ProcessedSegment] = []
    offset = 0.0
    diar_seen = False
    for p, cf in enumerate(cfiles):
        stem = cf.name[: -len(".chunks.jsonl")]
        raw = _jsonl(cf)
        corr = _jsonl(d / f"{stem}.corrected.jsonl")
        tran = _jsonl(d / f"{stem}.translated.jsonl")
        turns = [] if args.no_diar else _diar(d / f"{stem}.diar.jsonl")
        if turns:
            diar_seen = True
        audio = args.audio_folder / f"{stem}.m4a"
        pdur = _dur(audio)
        maxend = max((r.get("end") or 0.0)
                     for r in raw.values()) if raw else 0.0
        if pdur is None:
            pdur = maxend
            print(f"  WARN: could not probe {audio.name}; using last "
                  f"chunk end {pdur:.1f}s as part length")
        else:
            print(f"  part {stem}: {pdur:.1f}s, offset +{offset:.1f}s"
                  f"{'  (no diar.jsonl -> UNKNOWN)' if not turns else ''}")

        # local (idx, lstart, lend, raw, corr, eng) in part seconds
        local: list[tuple] = []
        for i in sorted(raw):
            r = raw[i]
            rt = (r.get("text") or "").strip()
            ct = (corr.get(i, {}).get("text") or rt).strip()
            en = (tran.get(i, {}).get("text") or "").strip()
            ls = float(r.get("start") or 0.0)
            le = float(r.get("end") or r.get("start") or 0.0)
            local.append((i, ls, le, rt, ct, en))

        labels = _assign_speakers(
            local, turns, is_anchor_part=(p == 0),
            anchor_window=args.anchor_window)
        if turns:
            cnt: dict[str, int] = defaultdict(int)
            for lb in labels:
                cnt[lb.rstrip("?")] += 1
            amb = sum(1 for lb in labels if lb.endswith("?"))
            print(f"    diar: {len(turns)} turns -> "
                  f"{dict(sorted(cnt.items()))}; {amb} straddle/ambig")

        # LLM discourse turn-attribution overrides acoustic diar when
        # present (pyannote fails on same-gender single-mic audio).
        spk = _jsonl(d / f"{stem}.speakers.jsonl")
        if spk:
            labels = [(spk.get(i, {}).get("speaker") or lab)
                      for (i, *_rest), lab in zip(local, labels)]
            scnt: dict[str, int] = defaultdict(int)
            for lb in labels:
                scnt[lb.rstrip("?")] += 1
            print(f"    speakers.jsonl (LLM discourse) -> "
                  f"{dict(sorted(scnt.items()))}")

        for (i, ls, le, rt, ct, en), lab in zip(local, labels):
            psegs.append(ProcessedSegment(
                segment_id=p * 100000 + i,
                start_s=round(offset + ls, 3),
                end_s=round(offset + le, 3),
                speaker_id=lab,
                language="ben",
                raw_text=rt,
                corrected_text=ct,
                english_translation=en,
                refined_translation=en,
                confidence=0.0,
                source_file=stem,
                absolute_start=None, absolute_end=None,
            ))
        offset += pdur

    if not psegs:
        sys.exit("no segments assembled")

    ns: list[NonSpeechSegment] = []
    for a, b in zip(psegs, psegs[1:]):
        if (b.source_file == a.source_file
                and b.start_s - a.end_s > 1.0):
            ns.append(NonSpeechSegment(
                start_s=a.end_s, end_s=b.start_s,
                region_type="silence",
                duration_s=round(b.start_s - a.end_s, 3),
                source_file=a.source_file))

    total = max(p.end_s for p in psegs)
    speech = sum(p.end_s - p.start_s for p in psegs)
    nsp = sum(n.duration_s for n in ns)
    aq = AudioQualityMetrics(
        total_duration_s=round(total, 1),
        speech_duration_s=round(speech, 1),
        non_speech_duration_s=round(nsp, 1),
        speech_ratio=round(speech / total, 3) if total else 0.0,
        num_speech_segments=len(psegs),
        num_non_speech_segments=len(ns),
        avg_speech_segment_s=round(speech / len(psegs), 1),
        longest_silence_s=round(
            max((n.duration_s for n in ns), default=0.0), 1),
    )
    meta = TranscriptMetadata(
        project_name="",
        audio_file=args.stem,
        duration_s=round(total, 1),
        recording_date="",
        languages_detected=["Bengali"],
        num_speakers=len({p.speaker_id.rstrip("?") for p in psegs}),
        transcription_style=TranscriptionStyle.INTELLIGENT_VERBATIM,
        asr_engines_used=[f"Omnilingual {args.asr_model}"],
        postprocessing_stages=[
            "Qwen intelligent-verbatim correction (per-segment)",
            "Qwen translation (per-segment)",
        ] + ([
            "pyannote 3.1 diarization + dominant-overlap + "
            "scripted-intro role map (per-part)"
        ] if diar_seen else []),
        audio_quality=aq,
        interview=None,
    )
    result = TranscriptResult(metadata=meta, segments=psegs,
                              non_speech_segments=ns)
    written = write_transcript(
        result, d, args.stem, output_format="all",
        timestamp_fmt="HH:MM:SS", include_raw=True,
        include_translation=True)

    print(f"\nAssembled {len(psegs)} segments "
          f"({total/60:.1f} min) -> {d}/")
    for w in written:
        print(f"  {w.name}")
    if diar_seen:
        spk_tot: dict[str, int] = defaultdict(int)
        for s in psegs:
            spk_tot[s.speaker_id.rstrip("?")] += 1
        amb_tot = sum(1 for s in psegs if s.speaker_id.endswith("?"))
        print(f"  speakers: {dict(sorted(spk_tot.items()))}; "
              f"{amb_tot} segments flagged ambiguous (trailing '?')")
        print("NOTE: start_s/end_s = GLOBAL interview seconds "
              "(metadata join key); source_file = originating part. "
              "speaker_id from pyannote 3.1 + dominant-overlap + "
              "scripted-intro role map (per-part; later parts use a "
              "speech-share heuristic — spot-check). Trailing '?' = "
              "turn-straddling/ambiguous chunk. [bn-raw] shows only "
              "where the corrector changed the ASR.")
    else:
        print("NOTE: start_s/end_s = GLOBAL interview seconds "
              "(metadata join key); source_file = originating part. "
              "speaker_id = UNKNOWN (run scripts/diarize_parts.py "
              "first for ENUMERATOR/RESPONDENT labels). [bn-raw] "
              "shows only where the corrector changed the ASR.")


if __name__ == "__main__":
    main()
