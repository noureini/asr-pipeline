"""
Measure how COMPLEMENTARY a profile's two witnesses are, before using them.

The fusion judge can only fix a word if at least one witness heard it right.
So a good witness pair is not just two accurate models: their errors must
fall on DIFFERENT words. On a reference test set (scripts/fetch_fleurs.py)
this reports, per witness and for the pair:

  WER            each witness alone (raw, and with Latin-script words and
                 digits dropped so spelling conventions don't count)
  oracle WER     per clip, pick the better witness — a floor for selection
  BOTH wrong     share of all reference words that NEITHER witness got
                 right: the floor the judge cannot go below (lower = better)
  shared errors  of the words EITHER witness got wrong, the share BOTH got
                 wrong. Rises as both witnesses get better (the leftovers are
                 the hard words), so compare pairs on BOTH wrong first.
  rescuable      share of each witness's errors that the other got right
                 (for Whisper also counting any of its N-best readings)

Transcripts are cached per model in OUT_DIR, so candidates can be mixed and
matched cheaply: write a candidate profile, run, compare.

Usage:
    uv run --frozen python scripts/witness_complementarity.py \
        -p languages/nep.yaml ~/fleurs_ne OUT_DIR
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import tempfile
import unicodedata
from pathlib import Path

import numpy as np
import soundfile as sf

from asr_pipeline.engines.ctc_engine import CTCEngine
from asr_pipeline.fusion import FusionTranslator
from asr_pipeline.models import AlignedSegment, AudioChunk
from asr_pipeline.profiles import LanguageProfile, load_profile_file

_SKIP = re.compile(r"[A-Za-z0-9०-९০-৯]")


def toks(s: str, norm: bool = False) -> list[str]:
    """Words with punctuation/symbols removed. Strips by Unicode category,
    NOT with [^\\w\\s]: Python's \\w excludes combining vowel signs
    (Devanagari े ो, Bengali া ি …), which would split every word."""
    s = "".join(" " if unicodedata.category(c)[0] in "PS" else c
                for c in (s or "").lower())
    t = s.split()
    return [x for x in t if not _SKIP.search(x)] if norm else t


def correct_positions(hyp: list[str], ref: list[str]) -> set[int]:
    """Reference word positions matched exactly by `hyp` in a minimum
    edit-distance alignment."""
    n, m = len(ref), len(hyp)
    d = np.zeros((n + 1, m + 1), dtype=np.int32)
    d[:, 0] = np.arange(n + 1)
    d[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1,
                          d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]))
    ok, i, j = set(), n, m
    while i > 0 and j > 0:
        if ref[i - 1] == hyp[j - 1] and d[i, j] == d[i - 1, j - 1]:
            ok.add(i - 1)
            i, j = i - 1, j - 1
        elif d[i, j] == d[i - 1, j - 1] + 1:
            i, j = i - 1, j - 1
        elif d[i, j] == d[i - 1, j] + 1:
            i -= 1
        else:
            j -= 1
    return ok


def wer(hyp: list[str], ref: list[str]) -> float:
    n, m = len(ref), len(hyp)
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        for j in range(1, m + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1,
                         prev[j - 1] + (ref[i - 1] != hyp[j - 1]))
        prev = cur
    return prev[m]


def cache_name(model: str, extra: str = "") -> str:
    return re.sub(r"[^\w.-]", "_", model + extra) + ".json"


def run_ctc(p: LanguageProfile, clips: list[Path]) -> dict:
    w = p.witness_ctc
    eng = CTCEngine(model_id=w.model, language=p.code, backend=w.backend,
                    revision=w.revision, nemo_file=w.nemo_file,
                    decoder=w.decoder, language_head=w.language_head,
                    target_lang=w.target_lang,
                    batch_size=w.batch_size)
    eng.load()
    out = {}
    for i in range(0, len(clips), w.batch_size):
        batch = clips[i:i + w.batch_size]
        chunks = [AudioChunk(chunk_id=k, start_s=0.0,
                             end_s=sf.info(str(c)).duration,
                             duration_s=sf.info(str(c)).duration,
                             waveform_path=c) for k, c in enumerate(batch)]
        for c, seg in zip(batch, eng.transcribe_batch(chunks, p.code)):
            out[c.name] = [seg.text]
    eng.unload()
    return out


def run_whisper(p: LanguageProfile, clips: list[Path]) -> dict:
    # join clips (1 s gaps) so the witness runs exactly as in the pipeline
    pieces, segs, t = [], [], 0.0
    gap = np.zeros(16000, dtype=np.float32)
    for c in clips:
        audio, sr = sf.read(str(c), dtype="float32")
        if sr != 16000:
            raise ValueError(f"{c.name}: expected 16 kHz, got {sr}")
        segs.append(AlignedSegment(segment_id=len(segs), start_s=t,
                                   end_s=t + len(audio) / sr,
                                   speaker_id="SPEAKER_00",
                                   language=p.code, raw_text=""))
        pieces += [audio, gap]
        t += (len(audio) + len(gap)) / sr
    joined = Path(tempfile.mkdtemp()) / "joined.wav"
    sf.write(str(joined), np.concatenate(pieces), 16000)
    nbest = FusionTranslator(profile=p)._run_whisper(segs, joined)
    return {c.name: hyps for c, hyps in zip(clips, nbest)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("-p", "--profile", type=Path, required=True,
                    help="profile YAML defining the witness pair")
    ap.add_argument("data", type=Path, help="dir from fetch_fleurs.py")
    ap.add_argument("out_dir", type=Path, help="cache + report dir")
    args = ap.parse_args()
    p = load_profile_file(args.profile)
    data, out = args.data.expanduser(), args.out_dir.expanduser()
    out.mkdir(parents=True, exist_ok=True)

    refs = {}
    with open(data / "ref.tsv", encoding="utf-8") as f:
        for row in csv.reader(f, delimiter="\t"):
            refs[row[0]] = row[1]
    clips = [data / n for n in refs]

    w1, w2 = p.witness_ctc, p.witness_whisper
    hyps = {}
    for key, runner, name in (
            ("w1", run_ctc, cache_name(w1.model, f"_{w1.decoder or ''}")),
            ("w2", run_whisper, cache_name(w2.model))):
        cache = out / name
        if cache.exists():
            hyps[key] = json.loads(cache.read_text(encoding="utf-8"))
        else:
            hyps[key] = runner(p, clips)
            cache.write_text(json.dumps(hyps[key], ensure_ascii=False),
                             encoding="utf-8")

    lines = [f"Pair: W1 {w1.model} ({w1.backend}, {w1.decoder or 'default'})",
             f"      W2 {w2.model}",
             f"Test set: {data} ({len(refs)} clips)", ""]
    for norm in (False, True):
        e1 = e2 = eo = n = 0
        shared = either = res1 = res2 = bad1_n = bad2_n = 0
        for name, ref_text in refs.items():
            ref = toks(ref_text, norm)
            h1 = toks(hyps["w1"][name][0], norm)
            h2s = [toks(h, norm) for h in hyps["w2"][name] or [""]]
            a, b = wer(h1, ref), wer(h2s[0], ref)
            e1, e2, eo, n = e1 + a, e2 + b, eo + min(a, b), n + len(ref)
            ok1 = correct_positions(h1, ref)
            ok2 = correct_positions(h2s[0], ref)
            ok2n = set().union(*(correct_positions(h, ref) for h in h2s))
            bad1 = set(range(len(ref))) - ok1
            bad2 = set(range(len(ref))) - ok2
            shared += len(bad1 & bad2)
            either += len(bad1 | bad2)
            res1 += len(bad1 & ok2n)       # W1 errors W2 (any N-best) got
            res2 += len(bad2 & ok1)        # W2 errors W1 got
            bad1_n += len(bad1)
            bad2_n += len(bad2)
        tag = "normalised" if norm else "raw"
        lines += [
            f"[{tag}]",
            f"  WER  W1 {e1 / n:6.1%}   W2 {e2 / n:6.1%}   "
            f"oracle (best per clip) {eo / n:6.1%}",
            f"  BOTH wrong (share of all reference words): "
            f"{shared / max(n, 1):6.1%}   <- the judge's floor; lower = better",
            f"  shared errors (both wrong / either wrong): "
            f"{shared / max(either, 1):6.1%}   (rises as both witnesses improve)",
            f"  rescuable: W1 errors W2 got right (any N-best) "
            f"{res1 / max(bad1_n, 1):6.1%} | W2 errors W1 got right "
            f"{res2 / max(bad2_n, 1):6.1%}", ""]
    report = "\n".join(lines)
    print(report)
    (out / f"report_{args.profile.stem}.txt").write_text(report,
                                                          encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
