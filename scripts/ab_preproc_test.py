"""
A/B preprocessing isolation test.

Hypothesis: the pipeline's aggressive spectral-gating denoise (and/or
loudness norm) degrades Omnilingual ASR on real field audio — which is
why FLEURS (fed RAW to the model) scored ~2.7% CER while the
real-survey run (model + full preprocessing) produced word-salad.

Same model (Omnilingual LLM 1B), same audio content, ONLY the
preprocessing varies:

  A raw            : decode -> 16k mono            (FLEURS-equivalent)
  B loudness       : A + EBU R128 loudness norm
  C loud+denoise   : B + spectral-gating denoise   (CURRENT PIPELINE)
  D denoise only   : A + spectral-gating denoise

Reuses the pipeline's OWN _convert_to_wav / _normalize_loudness /
_reduce_noise so we test the real code, not an approximation. VAD
chunking is bypassed (we cut fixed spans) so preprocessing is the
only variable.

Usage:
  uv run python scripts/ab_preproc_test.py --stem <STEM> \
      --spans part1:76,part1:136,part2:99,part2:119 --fleurs 3
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
import unicodedata
from pathlib import Path

OUT = Path("outputs/real_eval_inproc")
AUDIO = Path("for testing (Noureini)")
FLEURS = Path("test_data/fleurs_demo_10")


def _norm(s):
    s = unicodedata.normalize("NFC", s or "").strip()
    s = re.sub(r"[।,.!?;:\"'()\[\]{}—–\-‌‍]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def cer(ref, hyp):
    r, h = list(_norm(ref)), list(_norm(hyp))
    n, m = len(r), len(h)
    d = list(range(m + 1))
    for i in range(1, n + 1):
        p = d[0]
        d[0] = i
        for j in range(1, m + 1):
            c = d[j]
            d[j] = min(d[j] + 1, d[j - 1] + 1,
                       p + (0 if r[i - 1] == h[j - 1] else 1))
            p = c
    return d[m] / max(n, 1) * 100


def _jsonl(p):
    m = {}
    if p.exists():
        for ln in p.read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if ln:
                try:
                    r = json.loads(ln)
                    m[r["idx"]] = r
                except Exception:
                    pass
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", required=True)
    ap.add_argument("--spans", default="part1:76,part1:136,"
                    "part2:99,part2:119",
                    help="partN:chunkidx comma list (mix good/bad)")
    ap.add_argument("--fleurs", type=int, default=3,
                    help="also test N FLEURS clips (0=skip)")
    ap.add_argument("--omni-lang", default="ben_Beng")
    args = ap.parse_args()

    import torch  # noqa

    from asr_pipeline.config import load_config
    from asr_pipeline.engines.omnilingual_engine import OmnilingualEngine
    from asr_pipeline.models import AudioChunk
    from asr_pipeline.preprocessor import AudioPreprocessor

    cfg = load_config(Path("omni_test_1b.yaml"))
    cfg.omnilingual.max_audio_length_s = 60

    d = OUT / args.stem
    chunk_files = sorted(d.glob("*.chunks.jsonl"))
    if not chunk_files:
        sys.exit(f"no *.chunks.jsonl in {d}")
    part_jsonl = {f"part{i+1}": chunk_files[i]
                  for i in range(len(chunk_files))}
    part_stem = {f"part{i+1}": chunk_files[i].name[:-len(".chunks.jsonl")]
                 for i in range(len(chunk_files))}

    print("Loading Omnilingual 1B ...", flush=True)
    eng = OmnilingualEngine(cfg.omnilingual, device="cuda")
    eng.load()
    dev = str(getattr(eng._pipeline, "device", "?"))
    print(f"  device={dev}", flush=True)

    tmp = Path(tempfile.mkdtemp(prefix="abprep_"))
    pp = AudioPreprocessor(cfg.preprocessing, tmp)

    def transcribe(wav: Path, lang: str) -> str:
        import soundfile as sf
        dur = sf.info(str(wav)).duration
        ch = AudioChunk(chunk_id=0, start_s=0.0, end_s=dur,
                        duration_s=dur, waveform_path=wav)
        segs = eng.transcribe_chunk(ch, lang)
        return " ".join(s.text for s in segs).strip()

    def variants(src_wav: Path, tag: str):
        # A raw (already 16k mono via _convert_to_wav)
        A = pp._convert_to_wav(src_wav)
        B = pp._normalize_loudness(A)
        C = pp._reduce_noise(B)
        D = pp._reduce_noise(A)
        return {"A raw": A, "B loud": B,
                "C loud+denoise(PIPELINE)": C, "D denoise-only": D}

    # ---- real-audio spans ----
    print("\n" + "#" * 72)
    print("# REAL SURVEY AUDIO — known good & bad spans")
    print("#" * 72)
    for spec in args.spans.split(","):
        spec = spec.strip()
        if not spec:
            continue
        part, idx = spec.split(":")
        idx = int(idx)
        rec = _jsonl(part_jsonl[part]).get(idx)
        if not rec:
            print(f"\n[{spec}] not found, skip")
            continue
        st, en = float(rec["start"]), float(rec["end"])
        m4a = AUDIO / f"{part_stem[part]}.m4a"
        cut = tmp / f"{part}_{idx}.wav"
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-ss", str(st),
             "-to", str(en), "-i", str(m4a), "-ac", "1",
             "-ar", "16000", str(cut)], check=True)
        print("\n" + "=" * 72)
        print(f"[{spec}]  {st:.1f}-{en:.1f}s ({en-st:.1f}s)")
        print(f"PIPELINE-CACHED (full preproc) : "
              f"{(rec.get('text') or '').strip()[:300]}")
        for name, w in variants(cut, spec).items():
            print(f"{name:<28}: {transcribe(w, args.omni_lang)[:300]}")

    # ---- FLEURS clips (clean speech control) ----
    if args.fleurs > 0:
        refs = {r["file"]: r["transcript"] for r in json.loads(
            (FLEURS / "references.json").read_text(encoding="utf-8"))}
        files = sorted(refs)[: args.fleurs]
        print("\n" + "#" * 72)
        print("# FLEURS (clean read speech) — does denoise hurt even here?")
        print("#" * 72)
        for fn in files:
            wav = FLEURS / "audio" / fn
            v = variants(wav, fn)
            ref = refs[fn]
            print("\n" + "=" * 72)
            print(f"[{fn}] GOLD: {ref[:200]}")
            for name in ("A raw", "C loud+denoise(PIPELINE)"):
                h = transcribe(v[name], args.omni_lang)
                print(f"{name:<28} CER={cer(ref, h):5.1f}% : {h[:200]}")

    eng.unload()
    print("\nDONE. Compare A (raw) vs C (pipeline) — if A is dramatically "
          "cleaner, preprocessing (denoise) is what we were missing.")


if __name__ == "__main__":
    main()
