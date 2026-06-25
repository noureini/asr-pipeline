#!/usr/bin/env python3
"""
A/B: titu native CTC timestamps  vs  arijitx wav2vec2 forced-aligned timestamps.

Same shared spine, same titu TEXT, same community-1 diarization — the ONLY
variable is where each word sits in time. We then attribute words to speakers
both ways and compare the speaker-attributed transcripts, plus report how much
arijitx's timing diverges from titu's (and whether it aligned at all).

Why this design: forced alignment only re-times EXISTING text; it can't fix ASR
errors. So we align titu's words with arijitx and check whether the sharper
timing changes who-said-what. If arijitx silently mis-aligns on hard 8 kHz audio,
the divergence diagnostics will show it (so we don't mistake a broken align for
"no improvement").

Run:
    uv run --no-sync python scripts/ab_aligner.py test_data/15_part1_mid5min.m4a
"""

from __future__ import annotations

import argparse
import statistics
import tempfile
from pathlib import Path

ARIJITX = "arijitx/wav2vec2-xls-r-300m-bengali"
DIAR_MODEL = "pyannote/speaker-diarization-community-1"


# ── titu witness (text + native timestamps) ──────────────────────────────
def titu_words_per_chunk(chunks):
    import torch
    from asr_pipeline.engines.titu_engine import TituEngine

    eng = TituEngine()
    eng.load()
    segs = eng.transcribe_batch(chunks, "ben")
    eng.unload()
    torch.cuda.empty_cache()
    # list per chunk of (word, abs_start, abs_end)
    return [[(w.word, w.start_s, w.end_s) for w in s.words] for s in segs]


# ── arijitx forced alignment (re-time titu's words) ──────────────────────
def arijitx_realign(chunks, titu_per_chunk):
    import torch
    import torchaudio
    import torchaudio.functional as AF
    from transformers import AutoModelForCTC, AutoProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    proc = AutoProcessor.from_pretrained(ARIJITX)
    model = AutoModelForCTC.from_pretrained(ARIJITX).to(device).eval()
    vocab = proc.tokenizer.get_vocab()
    blank_id = proc.tokenizer.pad_token_id or 0

    out_per_chunk = []
    diag = {"words_in": 0, "words_aligned": 0, "oov_chars": 0, "tot_chars": 0,
            "oob": 0, "deltas": []}

    for ch, titu_words in zip(chunks, titu_per_chunk):
        if not titu_words:
            out_per_chunk.append([])
            continue
        wave, sr = torchaudio.load(str(ch.waveform_path))
        if sr != 16000:
            wave = torchaudio.functional.resample(wave, sr, 16000)
        with torch.no_grad():
            iv = proc(wave.squeeze().cpu().numpy(), sampling_rate=16000,
                      return_tensors="pt").input_values.to(device)
            logits = model(iv).logits
        emission = torch.log_softmax(logits, dim=-1)  # (1,T,C)
        T = emission.shape[1]
        dur = wave.shape[1] / 16000.0
        tpf = dur / T if T else 0.0

        flat, lengths, kept = [], [], []
        for (wtext, _s, _e) in titu_words:
            diag["words_in"] += 1
            ids = []
            for c in wtext:
                diag["tot_chars"] += 1
                if c in vocab:
                    ids.append(vocab[c])
                else:
                    diag["oov_chars"] += 1
            if ids:
                kept.append((wtext, _s, _e))
                lengths.append(len(ids))
                flat.extend(ids)

        if not flat or len(flat) > T:
            # can't align (no in-vocab chars, or more tokens than frames)
            out_per_chunk.append([])
            continue

        targets = torch.tensor([flat], dtype=torch.int32, device=device)
        try:
            aligned, scores = AF.forced_align(targets=targets, blank=blank_id,
                                              log_probs=emission)
        except TypeError:
            aligned, scores = AF.forced_align(emission, targets, blank=blank_id)
        spans = AF.merge_tokens(aligned[0], scores[0])

        words, i = [], 0
        for (wtext, ns, _ne), L in zip(kept, lengths):
            ws, we = spans[i], spans[i + L - 1]
            i += L
            a_start = ch.start_s + ws.start * tpf
            a_end = ch.start_s + we.end * tpf
            words.append((wtext, a_start, a_end))
            diag["words_aligned"] += 1
            if a_start < ch.start_s - 0.1 or a_end > ch.end_s + 0.1:
                diag["oob"] += 1
            diag["deltas"].append(abs(a_start - ns))  # vs titu native start
        out_per_chunk.append(words)

    del model
    torch.cuda.empty_cache()
    return out_per_chunk, diag


# ── community-1 diarization (run once) ───────────────────────────────────
def diarize(wav_path, work, token):
    import torch
    from asr_pipeline.config import DiarizationConfig
    from asr_pipeline.diarization import create_diarizer

    cfg = DiarizationConfig(model=DIAR_MODEL, min_speakers=2, auth_token=token)
    d = create_diarizer(cfg, device="cuda", work_dir=work)
    d.load()
    res = d.diarize(wav_path)
    d.unload()
    torch.cuda.empty_cache()
    return res


def attribute(words, segments):
    out = []
    for word, s, e in words:
        mid = (s + e) / 2.0
        spk = None
        for sg in segments:
            if sg.start_s <= mid <= sg.end_s:
                spk = sg.speaker_id
                break
        if spk is None and segments:
            spk = min(segments, key=lambda sg: min(
                abs(mid - sg.start_s), abs(mid - sg.end_s))).speaker_id
        out.append((spk or "?", word))
    return out


def group_turns(attr):
    turns, cur, buf = [], None, []
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

    from asr_pipeline.config import PreprocessingConfig
    from asr_pipeline.preprocessor import AudioPreprocessor

    work = Path(tempfile.mkdtemp(prefix="ab_align_"))
    pp = AudioPreprocessor(PreprocessingConfig(), work)
    wav_path, chunks, _ = pp.preprocess(args.audio)
    print(f"shared spine: {len(chunks)} chunks")

    print("[titu] transcribe + native timestamps ...")
    titu_pc = titu_words_per_chunk(chunks)
    print("[arijitx] forced-align titu's text ...")
    arij_pc, diag = arijitx_realign(chunks, titu_pc)
    print("[diar] community-1 ...")
    res = diarize(wav_path, work, token)

    titu_words = [w for c in titu_pc for w in c]
    arij_words = [w for c in arij_pc for w in c]

    titu_turns = group_turns(attribute(titu_words, res.segments))
    arij_turns = group_turns(attribute(arij_words, res.segments))

    # ── report ───────────────────────────────────────────────────────────
    L = [f"Aligner A/B — titu native vs arijitx forced-align  —  {args.audio.name}",
         f"shared spine: {len(chunks)} chunks | diarization: {res.num_speakers} speakers",
         "=" * 74, "", "ARIJITX ALIGNMENT DIAGNOSTICS (did it actually align?)"]
    L.append(f"  titu words: {diag['words_in']} | arijitx aligned: {diag['words_aligned']}")
    oov_pct = 100.0 * diag["oov_chars"] / max(1, diag["tot_chars"])
    L.append(f"  OOV chars: {diag['oov_chars']}/{diag['tot_chars']} ({oov_pct:.1f}%)")
    L.append(f"  out-of-bounds aligned words: {diag['oob']}")
    if diag["deltas"]:
        ds = diag["deltas"]
        L.append(f"  |arijitx_start - titu_start| s — median {statistics.median(ds):.3f} "
                 f"/ mean {statistics.mean(ds):.3f} / max {max(ds):.3f}")
    L.append("  (large OOV% / many OOB / huge deltas => arijitx mis-aligned, "
             "not a fair 'no-improvement')")
    L += ["", "=" * 74, "ATTRIBUTION A/B (same community-1 diarization)", "=" * 74]
    L += ["", f"### titu-native timestamps  [{len(titu_turns)} turns]", "-" * 74]
    for spk, t in titu_turns:
        L.append(f"{spk}: {t}")
    L += ["", f"### arijitx forced-aligned timestamps  [{len(arij_turns)} turns]", "-" * 74]
    for spk, t in arij_turns:
        L.append(f"{spk}: {t}")

    rpt = args.audio.with_suffix(".ab_aligner.txt")
    rpt.write_text("\n".join(L), encoding="utf-8")
    print(f"\n✓ report: {rpt}")
    print(f"  titu turns: {len(titu_turns)} | arijitx turns: {len(arij_turns)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
