"""PROTOTYPE — WhisperX backbone + 2-model MBR consensus + (optional) Gemma.

Goal: test the proposed pipeline end-to-end on the 5-min slice and TIME each
stage, to compare against the current titu‖Whisper→Gemma stack (~3m42s/5min).

    WhisperX(backbone) → VAD + transcribe + word-align + diarize   [segmentation
                        + word timestamps + speakers; needs a CT2 model]
    N-best (HF models) → beam-search n-best per segment from each MBR model
                        (original transformers format, no CT2 conversion)
    MBR consensus     → per segment, the hypothesis minimising expected risk
                        Σ P(h')·grapheme-distance(h,h') over the pooled n-best,
                        P = softmax of model log-probs (reuses ger_guard.ndist)
    Gemma (optional)  → clean + translate with a FAITHFULNESS-scaffolded prompt
                        (domain priors, names/PII, numbers, anti-invention) →
                        fluent [bn] + [en]. The prompt's rules, NOT a translate-
                        only restriction, are what stop the hallucination; the
                        number-abstain guard backstops digit slips.

This is a PROTOTYPE: standalone, no pipeline wiring, so we can measure speed +
quality before committing to the rewrite.

    uv pip install whisperx
    uv run --no-sync python scripts/proto_whisperx_mbr.py \\
        test_data/15_part1_open5min.m4a [--gemma]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, "src")

# ── BACKBONE: WhisperX (segmentation + word-align + diarize). Needs CT2. ──
# TESTED 2026-06-24: base large-v3-turbo is FAR WORSE on this dialectal 8 kHz
# audio — 922s (46× slower, repetition loops "এসো এসো এসো…") + garbled output.
# Dialect adaptation > raw capacity here. Stay on the bengali.ai-regional medium.
#   models/tugstugi_ct2   medium, DIALECT-adapted (bengali.ai) — BEST so far
# A bigger model would only help if fine-tuned ON dialectal data (not off-shelf).
BACKBONE = "models/tugstugi_ct2"
# ── MBR pool: HF Whisper models we draw N-BEST from (beam search). ORIGINAL
# transformers format — NO CT2 conversion needed. For real MBR signal these
# should be DIVERSE (different group/data), not near-duplicates. Candidates:
#   bengaliAI/tugstugi_bengaliai-regional-asr_whisper-medium  (bengali.ai regional)
#   bangla-speech-processing/BanglaASR                        (CommonVoice, diverse)
#   ashrafulparan/whisper-small-bangla                        (bengali.ai dialects)
#   anuragshas/whisper-large-v2-bn                            (large-v2, diverse)
MBR_MODELS = [
    "bengaliAI/tugstugi_bengaliai-regional-asr_whisper-medium",
    "bangla-speech-processing/BanglaASR",
]
NBEST = 5                               # beams / hypotheses per model per segment
# titu = NeMo CTC literal witness(es) — independent architecture from Whisper
# (real MBR diversity; CTC can't hallucinate loops). Listed models run
# side-by-side with --titu so we can assess quality + pick the best one.
TITU_MODELS = [
    "hishab/titu_stt_bn_conformer_large",   # 4.4K-hr, diverse, non-telephony
    "hishab/titu_stt_bn_fastconformer",     # 18K-hr, proven on this audio
]
# Bengali wav2vec2 forced-align model. These output BENGALI GRAPHEMES (unlike
# MMS_FA's Latin/IPA dict that stripped Bengali to empty) — that's why WhisperX
# aligns Bengali with them. Candidates (xls-r-300m is the newer/stronger arch):
#   arijitx/wav2vec2-xls-r-300m-bengali           (default)
#   shahruk10/wav2vec2-xls-r-300m-bengali-commonvoice
#   arijitx/wav2vec2-large-xlsr-bengali
#   tanmoyio/wav2vec2-large-xlsr-bengali
ALIGN_MODEL = "arijitx/wav2vec2-xls-r-300m-bengali"
LANG = "bn"
# community-1 (2025) supersedes 3.1: same segmentation, but markedly better
# speaker assignment/counting (less speaker CONFUSION) — strongest gains on
# noisy real-world audio, i.e. exactly our 8 kHz survey case. This directly
# targets the "question's words in the wrong cluster" residual. Needs HF
# access accepted (done). Fall back to 3.1 if access ever lapses.
DIARIZE_MODEL = "pyannote/speaker-diarization-community-1"
# ─────────────────────────────────────────────────────────────────────────


class Timer:
    def __init__(self): self.t = {}
    def __call__(self, name): self._n = name; return self
    def __enter__(self): self._s = time.perf_counter(); return self
    def __exit__(self, *a):
        self.t[self._n] = time.perf_counter() - self._s
        print(f"  ⏱ {self._n}: {self.t[self._n]:.1f}s")


def _gpu_flush():
    """Release freed-model VRAM so the next model fits on the 8 GB card."""
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _overlap(a0, a1, b0, b1):
    """Seconds of temporal overlap between [a0,a1] and [b0,b1]."""
    return max(0.0, min(a1, b1) - max(a0, b0))


def _words_in_span(words, s, e):
    """Join the words [(word, start, end), …] whose midpoint falls inside the
    span [s, e] (or that overlap it) → the text for that span. Used to map one
    model's word-timestamped output onto ANOTHER segmentation, for comparison
    only (alignment-for-scoring)."""
    out = []
    for w, ws, we in words:
        if _overlap(ws, we, s, e) > 0 or (s <= 0.5 * (ws + we) <= e):
            out.append(w)
    return " ".join(out).strip()


def _vad_chunks(audio, sr, max_s=28.0):
    """titu's OWN segmentation: model-neutral standalone Silero VAD → speech
    regions greedy-merged into ≤max_s chunks at silence. NOT Whisper's VAD —
    feeding titu Whisper's cut points starves it (boundary bleed). Falls back
    to fixed windows if Silero is unavailable."""
    import numpy as np
    try:
        import torch
        model, utils = torch.hub.load("snakers4/silero-vad", "silero_vad",
                                      trust_repo=True)
        get_ts = utils[0]
        wav = torch.from_numpy(np.asarray(audio, dtype=np.float32))
        regions = [(t["start"] / sr, t["end"] / sr)
                   for t in get_ts(wav, model, sampling_rate=sr)]
    except Exception as e:
        print(f"  (Silero VAD unavailable: {type(e).__name__} — fixed windows)")
        dur = len(audio) / sr
        regions = [(i * max_s, min((i + 1) * max_s, dur))
                   for i in range(int(dur // max_s) + 1)]
    chunks, cs, ce = [], None, None
    for s, e in regions:
        if cs is None:
            cs, ce = s, e
        elif e - cs <= max_s:
            ce = e
        else:
            chunks.append((cs, ce))
            cs, ce = s, e
    if cs is not None:
        chunks.append((cs, ce))
    return chunks


def _titu_transcribe_native(model_id, audio, chunks, sr):
    """Transcribe a NeMo titu CTC model on ITS OWN chunks (from _vad_chunks),
    with native word timestamps. Returns a flat [(word, abs_start, abs_end)]
    list — used to map titu words into ANY segmentation for comparison
    (alignment-for-scoring), without constraining titu's transcription input."""
    import os
    import tempfile

    import nemo.collections.asr as nemo_asr
    import numpy as np
    import soundfile as sf
    import torch

    m = nemo_asr.models.ASRModel.from_pretrained(model_id)
    if torch.cuda.is_available():
        m = m.to("cuda")
    m.eval()
    tmp = tempfile.mkdtemp(prefix="titu_native_")
    paths = []
    for i, (s, e) in enumerate(chunks):
        clip = np.asarray(audio[int(s * sr):int(e * sr)], dtype=np.float32)
        if clip.size == 0:
            clip = np.zeros(int(0.1 * sr), dtype=np.float32)
        p = os.path.join(tmp, f"{i:04d}.wav")
        sf.write(p, clip, sr)
        paths.append(p)
    try:
        hyps = m.transcribe(paths, batch_size=8, timestamps=True)
    except Exception:
        hyps = m.transcribe(paths, batch_size=8)   # no word timestamps
    words = []
    for (cs, ce), h in zip(chunks, hyps):
        ts = getattr(h, "timestamp", None)
        if isinstance(ts, dict) and ts.get("word"):
            for w in ts["word"]:
                ww = str(w.get("word") or "").strip()
                ws, we = w.get("start"), w.get("end")
                if ww and ws is not None and we is not None:
                    words.append((ww, cs + float(ws), cs + float(we)))
        else:                                       # no timestamps → chunk text
            txt = (h if isinstance(h, str)
                   else getattr(h, "text", "") or "").strip()
            if txt:
                words.append((txt, cs, ce))
    del m
    for p in paths:
        try:
            os.remove(p)
        except OSError:
            pass
    return words


def align_word_streams(a, b, *, time_tau=3.0, time_w=0.40, gap=0.70):
    """Needleman-Wunsch alignment of two word streams a, b = [(word, t_mid)],
    robust to mismatched segmentation. Substitution cost blends grapheme
    distance and time proximity so an exact text match overrides a ~1s
    timestamp offset (the thing that mis-binned the age). Returns a list of
    (ai|None, bi|None) index pairs — None = a gap (one witness has the token,
    the other doesn't). O(|a|·|b|); fine for a few hundred words each."""
    from asr_pipeline.ger_guard import ndist
    na, nb = len(a), len(b)
    dp = [[0.0] * (nb + 1) for _ in range(na + 1)]
    bt = [[""] * (nb + 1) for _ in range(na + 1)]
    for i in range(1, na + 1):
        dp[i][0] = i * gap
        bt[i][0] = "u"
    for j in range(1, nb + 1):
        dp[0][j] = j * gap
        bt[0][j] = "l"
    for i in range(1, na + 1):
        wa, ta = a[i - 1]
        for j in range(1, nb + 1):
            wb, tb = b[j - 1]
            sub = ((1 - time_w) * ndist(wa, wb)
                   + time_w * min(1.0, abs(ta - tb) / time_tau))
            diag, up, left = (dp[i - 1][j - 1] + sub,
                              dp[i - 1][j] + gap, dp[i][j - 1] + gap)
            best = min(diag, up, left)
            dp[i][j] = best
            bt[i][j] = "d" if best == diag else ("u" if best == up else "l")
    i, j, pairs = na, nb, []
    while i > 0 or j > 0:
        step = bt[i][j]
        if step == "d":
            pairs.append((i - 1, j - 1)); i -= 1; j -= 1
        elif step == "u":
            pairs.append((i - 1, None)); i -= 1
        else:
            pairs.append((None, j - 1)); j -= 1
    pairs.reverse()
    return pairs


def _mid_stream(words):
    """[(word, start, end)] → [(word, midpoint)] for the aligner."""
    return [(w, 0.5 * (s + e)) for (w, s, e) in words]


def _whisper_word_stream(segs):
    """Flat [(word, start, end)] from WhisperX segments' aligned words."""
    out = []
    for s in segs:
        for w in s.get("words") or []:
            ww = (w.get("word") or "").strip()
            ws, we = w.get("start"), w.get("end")
            if ww and ws is not None and we is not None:
                out.append((ww, float(ws), float(we)))
    return out


def majority_combine(segs, titu_streams, tau=0.34):
    """Per-word FUZZY majority vote: align each titu stream to Whisper's words,
    then for each Whisper word cluster the readings (whisper + each titu) by
    GRAPHEME similarity before voting — so dialectal spelling variants of the
    same word (কতো ≈ খতো ≈ কত) count as ONE vote, not three disagreements
    (exact-match voting is far too brittle for Bengali). Largest cluster ≥2 →
    use it (prefer Whisper's spelling); all readings genuinely differ → keep
    Whisper and flag. Returns (combined_text, flags)."""
    from asr_pipeline.ger_guard import (edit_distance, grapheme_clusters,
                                        _normalize)

    def _close(a, b):
        """Same word? Allow ONE grapheme edit for short words (normalized
        distance is too harsh: 1 edit in a 2-grapheme word = 0.50), else the
        ratio. So কতো≈কত≈খতো merge; ফাহা/খা/ভাই (≥2 edits) don't."""
        ga, gb = grapheme_clusters(_normalize(a)), grapheme_clusters(_normalize(b))
        d, m = edit_distance(ga, gb), max(len(ga), len(gb), 1)
        return (m >= 2 and d <= 1) or d / m <= tau

    wstream = _whisper_word_stream(segs)
    w_mid = _mid_stream(wstream)
    aligned = {}
    for name, tw in titu_streams.items():
        pairs = align_word_streams(_mid_stream(tw), w_mid)
        aligned[name] = {bj: _mid_stream(tw)[ai][0]
                         for ai, bj in pairs if ai is not None and bj is not None}
    out, flags = [], []
    for i, (ww, _) in enumerate(w_mid):
        cands = [ww] + [aligned[n].get(i, "") for n in titu_streams]
        cands = [c for c in cands if c.strip()]
        if not cands:
            continue
        clusters = []                               # fuzzy-group by grapheme dist
        for c in cands:
            for cl in clusters:
                if _close(c, cl[0]):
                    cl.append(c)
                    break
            else:
                clusters.append([c])
        best = max(clusters, key=len)
        if len(best) >= 2:                           # a (fuzzy) majority exists
            out.append(ww if ww in best else best[0])   # prefer Whisper spelling
        else:                                        # all genuinely different
            out.append(ww)
            flags.append((i, cands))
    return " ".join(out), flags


def mbr_select(candidates):
    """Expected-risk MBR over an n-best pool.

    candidates: list of (text, logprob). The MBR pick minimises expected risk
    h* = argmin_h Σ_h' P(h')·risk(h, h'), with P from softmax of the model
    log-probs (the pseudo-posterior) and risk = grapheme distance (≈ CER, the
    right unit for Bengali — word-level WER is unreliable on run-together
    Bengali). Returns (winner_text, spread) where spread = posterior-weighted
    mean pairwise distance (0 = the pool agrees, high = genuine disagreement).
    """
    import math

    from asr_pipeline.ger_guard import ndist
    cands = [(t.strip(), lp) for t, lp in candidates if (t or "").strip()]
    if not cands:
        return "", 0.0
    if len(cands) == 1:
        return cands[0][0], 0.0
    texts = [t for t, _ in cands]
    lps = [lp for _, lp in cands]
    mx = max(lps)
    w = [math.exp(lp - mx) for lp in lps]
    z = sum(w) or 1.0
    w = [x / z for x in w]
    # precompute pairwise distances once
    n = len(texts)
    d = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d[i][j] = d[j][i] = ndist(texts[i], texts[j])
    best, best_risk = texts[0], None
    for i in range(n):
        risk = sum(w[j] * d[i][j] for j in range(n) if j != i)
        if best_risk is None or risk < best_risk:
            best, best_risk = texts[i], risk
    spread = sum(w[i] * w[j] * d[i][j] for i in range(n) for j in range(n)
                 if i != j)
    return best, spread


def _nbest_for_model(model_id, audio, spans, nbest, device, sr, batch=8):
    """Generate `nbest` hypotheses (beam search) per segment span from a HF
    Whisper model, BATCHED across segments so it stays fast (Whisper pads each
    clip to 30 s; processing 8 at once amortises that on the GPU). Returns
    list-per-segment of [(text, logprob), …] — original transformers model,
    no CT2. Sequence scores (length-normalised log-probs) feed the MBR
    pseudo-posterior."""
    import torch
    from transformers import (WhisperForConditionalGeneration,
                              WhisperProcessor)

    dtype = torch.float16 if device == "cuda" else torch.float32
    proc = WhisperProcessor.from_pretrained(model_id)
    model = WhisperForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=dtype).to(device).eval()
    # Newer transformers dropped forced_decoder_ids from generate() — pass
    # language/task instead. Clear any baked-in forced ids to avoid conflict.
    try:
        model.generation_config.forced_decoder_ids = None
    except Exception:
        pass

    out = [[] for _ in spans]
    clips, idx = [], []                  # non-empty clips + their seg index
    for i, (a, b) in enumerate(spans):
        clip = audio[a:b]
        if len(clip) >= int(0.2 * sr):
            clips.append(clip)
            idx.append(i)

    for bs in range(0, len(clips), batch):
        chunk = clips[bs:bs + batch]
        feats = proc(chunk, sampling_rate=sr,
                     return_tensors="pt").input_features.to(device, dtype)
        with torch.no_grad():
            gen = model.generate(
                feats, language="bn", task="transcribe",
                num_beams=nbest, num_return_sequences=nbest, max_new_tokens=200,
                output_scores=True, return_dict_in_generate=True)
        texts = proc.batch_decode(gen.sequences, skip_special_tokens=True)
        sc = (gen.sequences_scores.tolist()
              if getattr(gen, "sequences_scores", None) is not None
              else [0.0] * len(texts))
        # generate returns nbest sequences per input, contiguous per input j
        for j in range(len(chunk)):
            seg_i = idx[bs + j]
            for k in range(nbest):
                t = texts[j * nbest + k].strip()
                if t:
                    out[seg_i].append((t, float(sc[j * nbest + k])))
    del model
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", type=Path)
    ap.add_argument("--gemma", action="store_true",
                    help="run Gemma correction+translation (the full pipeline)")
    ap.add_argument("--mbr", action="store_true",
                    help="opt-in: 2-model n-best MBR consensus before Gemma "
                         "(default: use WhisperX's own transcript — leaner)")
    ap.add_argument("--titu", action="store_true",
                    help="also transcribe with titu (NeMo CTC) and show it "
                         "side-by-side with Whisper — validates titu-large + "
                         "is the independent witness for MBR fusion")
    ap.add_argument("--fuse", action="store_true",
                    help="with --titu: LLM fusion — Gemma reconciles the 3 "
                         "readings (constrained to them) + translates, with a "
                         "verify-against-candidates guard. The SOTA fusion, "
                         "with the anti-hallucination seatbelts.")
    ap.add_argument("--standardize", action="store_true",
                    help="with --fuse: render CLEAN STANDARD Bengali (normalize "
                         "dialectal spelling) instead of faithful-dialectal; "
                         "writes .fused_standard.txt for side-by-side compare")
    ap.add_argument("--align", action="store_true",
                    help="with --titu: validate the cross-segmentation word "
                         "aligner — does it co-locate the age / স্ত্রী / place "
                         "tokens across titu's and Whisper's segmentations? "
                         "(the blocker that must pass before MBR)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--compute-type", default="float16")
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    if not args.audio.exists():
        print(f"no audio at {args.audio}", file=sys.stderr)
        return 1

    try:
        import whisperx
    except ImportError:
        print("whisperx not installed — run:  pip install whisperx",
              file=sys.stderr)
        return 2

    T = Timer()
    mode = (f"MBR({NBEST}-best × {len(MBR_MODELS)} models)" if args.mbr
            else "lean (WhisperX transcript)")
    gem = " → Gemma" if args.gemma else ""
    print(f"PROTOTYPE on {args.audio.name}  |  {mode}{gem}")
    print(f"  backbone={BACKBONE}  align={ALIGN_MODEL}")
    if args.mbr:
        print(f"  mbr_models={MBR_MODELS}")
    print()

    # ── 1. WhisperX backbone: transcribe → align → diarize ──
    with T("load_audio"):
        audio = whisperx.load_audio(str(args.audio))
    with T("whisperx_backbone(transcribe)"):
        # language at load_model is newer-only; passing it to transcribe is
        # supported across versions and forces bn (no language detection on
        # degraded audio, which can misfire).
        m1 = whisperx.load_model(BACKBONE, args.device,
                                 compute_type=args.compute_type)
        res = m1.transcribe(audio, batch_size=args.batch, language=LANG)
    with T("align(wav2vec2)"):
        amodel, ameta = whisperx.load_align_model(
            language_code=LANG, device=args.device, model_name=ALIGN_MODEL)
        res = whisperx.align(res["segments"], amodel, ameta, audio,
                             args.device, return_char_alignments=False)
    with T("diarize"):
        from asr_pipeline.fusion import _hf_token
        try:
            from whisperx.diarize import DiarizationPipeline
        except Exception:
            from whisperx import DiarizationPipeline
        # Current WhisperX uses token=; older uses use_auth_token=. And it
        # defaults to a gated community model — pin to 3.1 which we have access
        # to. Try the kwarg combinations across versions.
        tok = _hf_token()
        if not tok:
            print("  ⚠ no HF token found (_hf_token) — set HF_TOKEN in .env",
                  file=sys.stderr)
        diar = None
        for kw in ({"model_name": DIARIZE_MODEL, "token": tok},
                   {"model_name": DIARIZE_MODEL, "use_auth_token": tok},
                   {"token": tok}, {"use_auth_token": tok}):
            try:
                diar = DiarizationPipeline(device=args.device, **kw)
                break
            except TypeError:
                continue
        if diar is None:
            raise RuntimeError("could not construct DiarizationPipeline")
        # NO speaker-count cap — let community-1 find however many voices are
        # actually present (field interviews often have a 3rd: a family member
        # or bystander). Forcing 2 would merge a real speaker into another and
        # corrupt attribution.
        dsegs = diar(audio)
        res = whisperx.assign_word_speakers(dsegs, res)
    # Use WhisperX's native segments + speaker attribution as-is (don't
    # reinvent alignment/diarization — trust the WhisperX pipeline output).
    segs = res["segments"]
    for s in segs:                       # normalise key for the rest of the code
        s.setdefault("speaker", s.get("speaker", "?"))
    nspk = len({s.get("speaker", "?") for s in segs})
    print(f"  → {len(segs)} segments across {nspk} speakers (WhisperX native)")
    # Free the backbone before any further models load (8 GB card).
    del m1, amodel, diar
    _gpu_flush()

    # ── titu literal witness(es): titu's OWN segmentation, aligned only for
    #    comparison (NOT fed Whisper's VAD spans — that starves titu) ──
    if args.titu:
        from asr_pipeline.ger_guard import ndist
        SR = 16000
        with T("titu_vad(silero)"):
            tchunks = _vad_chunks(audio, SR)
        print(f"  → titu segmentation: {len(tchunks)} Silero-VAD chunks "
              f"(titu's own, not Whisper's)")
        titu_streams = {}
        for tm in TITU_MODELS:
            name = tm.split("/")[-1]
            with T(f"titu:{name}"):
                twords = _titu_transcribe_native(tm, audio, tchunks, SR)
                _gpu_flush()
            titu_streams[name] = twords
            # align-for-scoring: map titu's words into the Whisper segments
            dists = []
            for s in segs:
                txt = _words_in_span(twords, s.get("start", 0), s.get("end", 0))
                s.setdefault("titu", {})[name] = txt
                dists.append(ndist((s.get("text") or "").strip(), txt))
            agree = sum(1 for d in dists if d <= 0.30)
            print(f"  → {name} vs Whisper: {agree}/{len(dists)} agree, "
                  f"mean Δ={sum(dists) / max(len(dists), 1):.2f}")

        # ── SIMPLE MAJORITY-VOTE COMBINE (Whisper + titus) ──
        combined, cflags = majority_combine(segs, titu_streams)
        cp = args.audio.with_suffix(".combined.txt")
        cp.write_text(combined, encoding="utf-8")
        print(f"\n  → majority combine → {cp.name}: {len(cflags)} words had "
              f"NO majority (all 3 readings differ) → flagged")
        for i, cands in cflags[:10]:
            print(f"      no-majority: {cands}")

        # ── ALIGNER VALIDATION: the blocker that must pass before MBR ──
        if args.align:
            from asr_pipeline.ger_guard import numbers
            wstream = _whisper_word_stream(segs)
            wb_mid = _mid_stream(wstream)
            HIGH = ("স্ত্রী", "নবী", "সাঁইত্রিশ", "সাড়", "ত্রিশ", "বয়স")
            for name, twords in titu_streams.items():
                ta_mid = _mid_stream(twords)
                pairs = align_word_streams(ta_mid, wb_mid)
                n_ok = n_dif = n_to = n_wo = 0
                hi = []
                for ai, bj in pairs:
                    wa = ta_mid[ai][0] if ai is not None else ""
                    wbw = wb_mid[bj][0] if bj is not None else ""
                    if wa and wbw:
                        if ndist(wa, wbw) <= 0.34:
                            n_ok += 1
                        else:
                            n_dif += 1
                    elif wa:
                        n_to += 1
                    else:
                        n_wo += 1
                    blob = wa + " " + wbw
                    if (numbers(wa) or numbers(wbw)
                            or any(h in blob for h in HIGH)):
                        t = (ta_mid[ai][1] if ai is not None
                             else wb_mid[bj][1])
                        mark = ("✓same" if (wa and wbw
                                and ndist(wa, wbw) <= 0.34)
                                else ("≠diff" if wa and wbw
                                      else ("titu→∅" if wa else "∅←whisper")))
                        hi.append(f"     [{t:6.1f}s] {mark:11} "
                                  f"titu={wa!r}  whisper={wbw!r}")
                print(f"\n  ── aligner: titu[{name}] ↔ Whisper ──")
                print(f"     {n_ok} agree | {n_dif} differ | "
                      f"{n_to} titu-only | {n_wo} whisper-only")
                print("     high-value tokens (age / স্ত্রী / place / numbers):")
                print("\n".join(hi) if hi else "     (none located)")

    # ── 2. Transcription source per segment ──
    if args.mbr:
        # Opt-in: 2-model n-best MBR consensus (heavier, likely redundant on
        # this strong regional model — see chat). Default path skips this.
        SR = 16000
        spans = [(int(s.get("start", 0) * SR), int(s.get("end", 0) * SR))
                 for s in segs]
        pool = [[] for _ in segs]
        for mid in MBR_MODELS:
            with T(f"nbest:{mid.split('/')[-1]}"):
                for i, cands in enumerate(
                        _nbest_for_model(mid, audio, spans, NBEST,
                                         args.device, SR)):
                    pool[i].extend(cands)
                _gpu_flush()
        rows, agree, disagree = [], 0, 0
        for s, cands in zip(segs, pool):
            if not cands:
                cands = [((s.get("text") or "").strip(), 0.0)]
            consensus, spread = mbr_select(cands)
            agree, disagree = ((agree + 1, disagree) if spread <= 0.30
                               else (agree, disagree + 1))
            rows.append({"start": s.get("start", 0), "end": s.get("end", 0),
                         "speaker": s.get("speaker", "?"),
                         "source": consensus, "spread": spread})
        print(f"  → MBR over {NBEST}-best × {len(MBR_MODELS)} models: "
              f"{agree} agree / {disagree} disagree")
    else:
        # Lean default: WhisperX's own transcript is the source for Gemma.
        rows = [{"start": s.get("start", 0), "end": s.get("end", 0),
                 "speaker": s.get("speaker", "?"),
                 "source": (s.get("text") or "").strip(), "spread": 0.0,
                 "titu": s.get("titu")}
                for s in segs]
        print(f"  → using WhisperX transcript directly ({len(rows)} segments)")

    # ── 4. optional Gemma correction + translation ──
    if args.gemma:
        _gpu_flush()   # backbone freed above
        with T("gemma_clean+translate"):
            _run_gemma(rows)

    # ── 4b. LLM FUSION over the 3 readings (constrained + verified) ──
    if args.fuse:
        _gpu_flush()
        label = "standard" if args.standardize else "faithful-dialectal"
        with T(f"gemma_fuse({label})"):
            _run_fusion_llm(rows, standardize=args.standardize)
        suffix = ".fused_standard.txt" if args.standardize else ".fused.txt"
        fp = args.audio.with_suffix(suffix)
        flines = []
        for r in rows:
            if not r.get("fused_bn"):
                continue
            flines.append(f"[{r['start']:.1f}-{r['end']:.1f}] {r['speaker']}")
            flines.append(f"  bn: {r['fused_bn']}")
            flines.append(f"  en: {r['fused_en']}")
            if r.get("fuse_flags"):
                flines.append(f"  ⚠ {r['fuse_flags']}")
            flines.append("")
        fp.write_text("\n".join(flines), encoding="utf-8")
        print(f"  → wrote fused transcript: {fp.name}")

    # ── summary ──
    total = sum(T.t.values())
    print(f"\n  TOTAL prototype time: {total:.1f}s  (current stack ≈ 222s/5min)")
    print("  stage breakdown:")
    for k, v in T.t.items():
        print(f"    {v:6.1f}s  {k}")

    out = args.audio.with_suffix(".proto.txt")
    lines = []
    for r in rows:
        sp = f"  spread={r['spread']:.2f}" if r["spread"] else ""
        sd = (f"  Δsrc={r['src_dist']:.2f}" if r.get("src_dist") is not None
              else "")
        lines.append(f"[{r['start']:.1f}-{r['end']:.1f}] {r['speaker']}{sp}{sd}")
        lines.append(f"  whisper: {r['source']}")
        if r.get("titu"):
            for name, txt in r["titu"].items():
                short = name.replace("titu_stt_bn_", "")
                lines.append(f"  titu[{short}]: {txt}")
        if r.get("gemma_bn"):
            lines.append(f"  bn : {r['gemma_bn']}")
            lines.append(f"  en : {r['gemma_en']}")
        if r.get("ger"):
            lines.append(f"  ⚠  {r['ger']}")
        lines.append("")
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n  → wrote {out}")

    if args.titu:
        _write_goldtemplate(rows, args.audio.with_suffix(".goldtemplate.tsv"))
    return 0


def _write_goldtemplate(rows, path):
    """Emit a TSV labelling template (one row per segment) for the GOLD SET.
    A human listens to [start,end] and fills the blank gold_* columns. The
    survey-critical columns (gold_numbers, gold_names) are the ones the
    integrity calibration actually needs; gold_text (full transcript) is
    optional, for prose CER. Open in a spreadsheet (import as tab-separated)."""
    import csv

    from asr_pipeline.combine import is_numberish
    models = sorted({m for r in rows if r.get("titu") for m in r["titu"]})
    short = [m.replace("titu_stt_bn_", "titu_") for m in models]
    # auto_numbers = candidate number tokens any system produced — lets you
    # jump straight to the number-bearing segments instead of listening to all.
    header = (["start", "end", "auto_numbers", "whisper"] + short
              + ["gold_numbers", "gold_names", "gold_text", "notes"])
    n_hot = 0
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(header)
        for r in rows:
            titu = r.get("titu") or {}
            texts = [r["source"]] + [titu.get(m, "") for m in models]
            auto = sorted({tok for t in texts for tok in t.split()
                           if is_numberish(tok)})
            n_hot += 1 if auto else 0
            w.writerow([f"{r['start']:.1f}", f"{r['end']:.1f}", " ".join(auto),
                        r["source"]] + [titu.get(m, "") for m in models]
                       + ["", "", "", ""])
    print(f"  → wrote gold template: {path.name}  ({n_hot} segments have "
          f"candidate numbers — label those first, then score_goldset.py)")
    return 0


def _clean_src(s: str) -> str:
    """Mechanical cleanup of the WhisperX transcript before the LLM sees it:
    drop the `<>` unaligned-word markers and collapse whitespace/newlines."""
    import re
    return re.sub(r"\s+", " ", (s or "").replace("<>", " ")).strip()


# Single-witness clean+translate prompt. Models the production fusion prompt's
# FAITHFULNESS scaffolding (domain priors, names/PII, numbers, script, anti-
# invention) — but with the no-fabrication rules turned UP, because there's no
# 2nd witness to ground against. Each rule maps to an observed hallucination:
# নবী→নবী উল্লাহ (names), খাজ→tax (dialect prior), dropped 'wife' (no drop),
# 3-word fragment→full sentence (no completion).
PROMPT_CLEAN_TRANSLATE = """\
You are cleaning ONE Bengali ASR transcript of a recorded survey interview from \
a climate-risk study in coastal Bangladesh (Satkhira / Sunamganj). The audio is \
8 kHz telephone-grade and the speech is COASTAL DIALECT, so the raw ASR has \
garbled spellings, wrong word boundaries and dialectal forms. Produce a clean, \
fluent, FAITHFUL Bengali rendering and a faithful English translation.

FAITHFULNESS (the overriding rule — you have only ONE source and no second \
witness, so never supply anything it does not contain): fix ONLY what is \
clearly an ASR error (spelling, word boundaries, obvious garble) and add \
natural punctuation and sentence breaks. NEVER add, drop, or change content; \
never invent or complete a name, word, phrase, or sentence; never turn a \
fragment into a full sentence. When a span is unintelligible, write \
[অস্পষ্ট] / [unclear] — do NOT guess. When unsure, keep the literal reading.

DIALECT (preserve the meaning, do not standardise it away):
- খাজ/কিষি here in a livelihood context = কাজ work / কৃষি farming — NOT খাজনা \
tax; চাষ cultivation; আড্ডা/আধি/বর্গা sharecropping; দিনমজুরি daily wage.
- household-roster + shocks instrument: খানা = household (NOT থানা); সদস্য = \
member; বন্যা flood, ঘূর্ণিঝড় cyclone, জলোচ্ছ্বাস surge, লবণাক্ততা salinity, \
ফসল মার খাওয়া = crop destroyed.
- বাইশ / তেইশ in a date context = the year 2022 / 2023, not a count.

NAMES/PII: keep a personal name only if clearly heard, else [নাম] / [name]; \
place names verbatim if clear, else [স্থান] / [place]. NEVER guess or extend a \
name from a garbled reading (do NOT turn "নবী" into "নবী উল্লাহ").

NUMBERS (critical, both lines): render exactly, never round or invent — a \
missing one is [অস্পষ্ট] / [unclear]. দেড়=1.5, আড়াই=2.5, সাড়ে X = X.5; \
-শো = ×100 (আড়াইশো=250); হাজার=×1000; লাখ=×100000.

SCRIPT: Bengali script everywhere except genuine English code-switches (NGO, \
bank name, loan), kept in Latin.

Output EXACTLY two lines and nothing else — no notes, no arrows, no \
"old → new", no alternatives in parentheses:
[bn] <clean fluent faithful Bengali; unrecoverable spans → [অস্পষ্ট]>
[en] <natural meaning-faithful English of YOUR [bn]; unrecoverable → [unclear]>

Transcript: {text}"""


PROMPT_FUSE = """\
You are reconciling THREE machine transcriptions of the SAME short Bengali \
audio segment from a climate-survey interview in coastal Bangladesh (8 kHz \
telephone, dialectal). The three systems make DIFFERENT errors; recover the \
single most faithful Bengali transcription from them, then translate it.

READING 1: {r1}
READING 2: {r2}
READING 3: {r3}

RULES (faithfulness is paramount — you have ONLY these three readings, no audio):
- Word by word, choose the most plausible Bengali supported by the readings. \
Where they agree, keep it; where they differ only in the SPELLING of the same \
word, use the cleanest form.
- Use ONLY content that appears in at least one reading. NEVER add a word, \
name, or number that is in none of the three. Do not complete fragments or add \
fluency the readings don't support.
- If the readings DISAGREE on a NUMBER or a NAME, do NOT guess — write \
[অস্পষ্ট] in its place. If they agree on it, keep it exactly.
- Dialect: খাজ/কিষি = work/farming (NOT tax/খাজনা); খানা = household (NOT থানা).

Output EXACTLY two lines, nothing else:
[bn] <single fused faithful Bengali>
[en] <faithful English translation of YOUR [bn]>"""


PROMPT_FUSE_STANDARD = """\
You are reconciling THREE machine transcriptions of the SAME short Bengali \
audio segment from a climate-survey interview in coastal Bangladesh (8 kHz \
telephone, dialectal). Write what was actually said in clean Bengali.

OVERRIDING PRINCIPLE: FAITHFULNESS BEATS FLUENCY. A rough, literal line is far \
better than a smooth, invented one. This is a survey record — a single \
invented word or number corrupts the data. Leave unclear spots rough; never \
paper over them.

READING 1: {r1}
READING 2: {r2}
READING 3: {r3}

DO (safe cleanup):
- For a word you can CONFIDENTLY identify across the readings, write it in \
standard Bengali spelling: টিক→ঠিক, গরি→করি, ইটা→এটা, খাজ→কাজ, খেউ→কেউ. Add \
natural punctuation.
- Where the readings agree, keep that content.

DO NOT (these corrupt the record):
- Do NOT guess at a garbled or uncertain word. If a word/phrase is garbled, or \
the readings disagree on what it even is, KEEP it literally or write \
[অস্পষ্ট] — NEVER replace it with a clean, plausible-sounding word. \
(Real failure to avoid: an unclear phrase "ফাইট্টা বাইট্টা" must NOT become an \
invented count like "পাঁচটি বাচ্চা". Leave it literal or mark [অস্পষ্ট].)
- Do NOT change a word's MEANING while standardising it — normalise FORM only. \
খাজ/কিষি = কাজ/কৃষি (work/farming), NEVER খাজনা (tax). খানা = household, NEVER \
খাবার (food) or থানা (police station).
- Do NOT add, complete, or smooth content the readings don't support.
- NUMBERS and NAMES (spoken as WORDS like সাঁইত্রিশ/ছয় OR as digits): copy \
exactly. If the readings disagree on one, write [অস্পষ্ট]. NEVER introduce a \
number or name that is in none of the three readings.

Output EXACTLY two lines, nothing else:
[bn] <clean but strictly faithful Bengali; unclear → [অস্পষ্ট]>
[en] <faithful English translation of YOUR [bn]>"""


def _run_fusion_llm(rows, standardize=False):
    """LLM fusion: Gemma reconciles the 3 readings (constrained to them) and
    translates, then the verify-against-candidates guard catches fabrication —
    a number in [bn] present in NONE of the readings is abstained to [অস্পষ্ট];
    a number the readings DISAGREE on is flagged (the LLM can't know which is
    right any better than voting can)."""
    import re

    from asr_pipeline import ger_guard
    from asr_pipeline.postprocessor import GGUFProcessor
    proc = GGUFProcessor(
        gguf_repo="unsloth/gemma-3-12b-it-GGUF",
        gguf_file="gemma-3-12b-it-UD-Q4_K_XL.gguf",
        n_ctx=2048, n_gpu_layers=-1, flash_attn=True, kv_quant=True)
    if not proc.load():
        print("  Gemma unavailable — skipping LLM fusion")
        return
    n_fab = n_disagree = 0
    for r in rows:
        cands = [_clean_src(r["source"])]
        cands += [_clean_src(t) for t in (r.get("titu") or {}).values()]
        cands = [c for c in cands if c.strip()]
        if not cands:
            r["fused_bn"] = r["fused_en"] = ""
            continue
        tpl = PROMPT_FUSE_STANDARD if standardize else PROMPT_FUSE
        prompt = tpl.format(
            r1=cands[0], r2=cands[1] if len(cands) > 1 else "(none)",
            r3=cands[2] if len(cands) > 2 else "(none)")
        ans = proc.generate(prompt, temperature=0.0, max_tokens=512)
        mbn, men = re.search(r"\[bn\]\s*(.+)", ans), re.search(r"\[en\]\s*(.+)", ans)
        bn = mbn.group(1).strip() if mbn else cands[0]
        en = men.group(1).strip() if men else ""
        reasons = []
        # number DISAGREEMENT among the readings → flag (LLM can't resolve it)
        numsets = [ger_guard.numbers(c) for c in cands]
        allnums = set().union(*numsets) if numsets else set()
        if allnums and any(numsets[0] != s for s in numsets[1:]):
            reasons.append("number-disagree-REVIEW")
            n_disagree += 1
        # FABRICATED digit-number (in [bn], in NO reading) → abstain
        fab = ger_guard.numbers(bn) - allnums
        if fab:
            bn = ger_guard._blank_fabricated(bn, fab, "[অস্পষ্ট]")
            en = ger_guard._blank_fabricated(en, fab, "[unclear]")
            reasons.append("fabricated-number:" + ",".join(sorted(fab)))
            n_fab += 1
        # FABRICATED number-WORD (spelled-out, e.g. পাঁচটি) — a Bengali number
        # word in [bn] not matched by any reading's number word. Digits-only
        # check above misses these (this is the পাঁচটি-বাচ্চা hole). Flag for
        # review (don't auto-delete — a word may correspond to a digit reading).
        from asr_pipeline.combine import is_numberish
        read_numwords = [w for c in cands for w in c.split() if is_numberish(w)]
        badw = [w for w in bn.split()
                if is_numberish(w) and not ger_guard.numbers(w)
                and not any(ger_guard.ndist(w, rw) <= 0.4 for rw in read_numwords)]
        if badw:
            reasons.append("numberword-not-in-readings:" + ",".join(badw))
            n_fab += 1
        r["fused_bn"], r["fused_en"] = bn, en
        if reasons:
            r["fuse_flags"] = ", ".join(reasons)
    proc.unload()
    print(f"  LLM fusion: {n_disagree} number-disagreements flagged | "
          f"{n_fab} fabricated numbers abstained")


def _run_gemma(rows):
    """Clean + translate each segment with a faithfulness-scaffolded prompt
    (fluent [bn] + [en]), then the number-abstain guard backstops it: a digit
    in [bn]/[en] the source doesn't support → [অস্পষ্ট]/[unclear]. The prompt's
    anti-invention rules — not a translate-only restriction — are what keep
    Gemma from hallucinating; the guard only catches number slips."""
    import re

    from asr_pipeline import ger_guard
    from asr_pipeline.postprocessor import GGUFProcessor
    proc = GGUFProcessor(
        gguf_repo="unsloth/gemma-3-12b-it-GGUF",
        gguf_file="gemma-3-12b-it-UD-Q4_K_XL.gguf",
        n_ctx=2048, n_gpu_layers=-1, flash_attn=True, kv_quant=True)
    if not proc.load():
        print("  Gemma unavailable — skipping clean+translate")
        return
    for r in rows:
        src = _clean_src(r["source"])
        if not src:
            r["gemma_bn"] = r["gemma_en"] = ""
            continue
        ans = proc.generate(PROMPT_CLEAN_TRANSLATE.format(text=src),
                            temperature=0.0, max_tokens=512)
        mbn = re.search(r"\[bn\]\s*(.+)", ans)
        men = re.search(r"\[en\]\s*(.+)", ans)
        bn = mbn.group(1).strip() if mbn else src
        en = men.group(1).strip() if men else ""
        # number-abstain against the source witness (handles word→digit, i.e.
        # only flags a digit when the source itself uses digits).
        v = ger_guard.assess(bn, en, src, [], snap=False)
        r["gemma_bn"], r["gemma_en"] = v.bn, v.en
        # Divergence flag: how far Gemma's [bn] strayed from the source. Small =
        # faithful cleanup; large = a drop (bn shorter) or a fabricated
        # completion (bn longer/different) — the residual prompt-only can't
        # stop. These are the segments a human should check.
        r["src_dist"] = v.min_dist
        reasons = list(v.reasons)
        if v.min_dist > 0.40:
            reasons.append(f"diverged-from-source({v.min_dist:.0%})")
        if reasons:
            r["ger"] = ", ".join(reasons)
    proc.unload()


if __name__ == "__main__":
    sys.exit(main())
