"""
Day-0b (MMS-FA verifier): for each VAD chunk, find the best-matching
GT sentence by running MMS forced alignment (phoneme-level) and
keeping only pairs with high alignment confidence.

This replaces the edit-distance bridge (which kept only 5-6% of
chunks at threshold 0.35) with a phoneme-aware verifier. MMS-FA
is robust to titu's noisy text because it operates on the audio
directly via the universal IPA model.

Workflow per chunk:
  1. Estimate proportional GT-sentence position from time
  2. Try aligning chunk audio with each of N candidate sentences
     (translit-applied + romanized via uroman) in the local window
  3. Keep the highest-scoring candidate above threshold
  4. Label = the original (translit-applied) Bengali sentence
"""
from __future__ import annotations

import argparse
import json
import re
from difflib import SequenceMatcher
from pathlib import Path

import torch
import torchaudio

_RE_SPEAKER_TAG = re.compile(
    r"^(প্র:|উ:|Q:|A:|নোট টেকার:|নোট:|enumerator:|respondent:)\s*",
    re.M | re.I)
_RE_TIMESTAMP_ANNOTATION = re.compile(
    r"^(সময়[ঃ:]|সময়\s*[ঃ:]|time[:\s])", re.I)
_RE_PARENS = re.compile(r"\([^)]{0,200}\)")
_RE_BRACK = re.compile(r"\[[^\]]{0,200}\]")
_RE_WS = re.compile(r"\s+")
_RE_SENT = re.compile(r"[।.?!]+\s*")


def docx_sents(p: Path) -> list[str]:
    import docx
    out = []
    d = docx.Document(str(p))
    for para in d.paragraphs:
        t = para.text or ""
        t = _RE_SPEAKER_TAG.sub("", t)
        t = _RE_PARENS.sub(" ", t)
        t = _RE_BRACK.sub(" ", t)
        t = _RE_WS.sub(" ", t).strip()
        for s in _RE_SENT.split(t):
            s = s.strip()
            if not s:
                continue
            # drop transcriber-inserted timestamp annotations
            if _RE_TIMESTAMP_ANNOTATION.match(s):
                continue
            out.append(s)
    return out


def apply_translit(sent: str, table: dict[str, str]) -> str:
    return " ".join(table.get(w, w) for w in sent.split())


def _normalize_for_mms(romanized: str, dictionary: dict[str, int]) -> str:
    """Filter the romanized text to chars present in MMS_FA dict;
    spaces become '|' (word boundary)."""
    text = romanized.lower().strip()
    text = re.sub(r"\s+", "|", text)
    cleaned = "".join(c for c in text if c in dictionary)
    cleaned = re.sub(r"\|+", "|", cleaned).strip("|")
    return cleaned


def align_score(model, dictionary, waveform, romanized,
                device, blank_id=0) -> float:
    """Return mean per-character alignment confidence (0..1)."""
    norm = _normalize_for_mms(romanized, dictionary)
    if len(norm) < 2:
        return 0.0
    tokens = [dictionary[c] for c in norm if c in dictionary]
    if len(tokens) < 2:
        return 0.0
    with torch.no_grad():
        emission, _ = model(waveform.to(device))
    if emission.shape[1] < len(tokens):
        # not enough frames to align; impossible
        return 0.0
    targets = torch.tensor([tokens], dtype=torch.int32, device=device)
    in_lens = torch.tensor([emission.shape[1]], device=device)
    tgt_lens = torch.tensor([len(tokens)], device=device)
    try:
        _, scores = torchaudio.functional.forced_align(
            emission, targets, in_lens, tgt_lens, blank=blank_id)
    except Exception:
        return 0.0
    # scores: log-probs in [-inf, 0]; convert to confidence in [0,1]
    s = scores[0].float()
    conf = s.exp().mean().item()
    return float(max(0.0, min(1.0, conf)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", required=True)
    ap.add_argument("--titu-dir", type=Path, required=True)
    ap.add_argument("--docx", type=Path, required=True)
    ap.add_argument("--translit-json", type=Path,
                    default=Path("/tmp/translit.json"))
    ap.add_argument("--out-manifest", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--score-threshold", type=float, default=0.20,
                    help="min mean per-char MMS-FA confidence to keep")
    ap.add_argument("--window", type=int, default=6,
                    help="search +/- N sentences around proportional pos")
    ap.add_argument("--limit-chunks", type=int, default=None,
                    help="cap chunks per part (smoke test)")
    args = ap.parse_args()

    table = json.loads(args.translit_json.read_text(encoding="utf-8"))
    sents_raw = docx_sents(args.docx)
    sents_tr = [apply_translit(s, table) for s in sents_raw]

    print(f"GT sentences: {len(sents_tr)}")

    # Romanize via uroman
    import uroman
    u = uroman.Uroman()
    sents_rom = [u.romanize_string(s, lcode="ben") for s in sents_tr]
    print("Romanization sample:")
    for i in (0, len(sents_rom) // 2, len(sents_rom) - 1):
        print(f"  [{i}] BN:  {sents_tr[i][:90]}")
        print(f"      ROM: {sents_rom[i][:90]}")

    # Load MMS_FA
    print("\nLoading torchaudio MMS_FA bundle ...")
    bundle = torchaudio.pipelines.MMS_FA
    model = bundle.get_model().to(args.device)
    model.eval()
    dictionary = bundle.get_dict()
    sample_rate = bundle.sample_rate
    print(f"  MMS_FA loaded: {len(dictionary)} tokens, sr={sample_rate}")

    # Chunks
    d = args.titu_dir / args.stem
    chunks_files = sorted(d.glob("*.chunks.jsonl"))
    if not chunks_files:
        raise SystemExit(f"no *.chunks.jsonl in {d}")

    args.out_manifest.parent.mkdir(parents=True, exist_ok=True)
    n_total = n_kept = 0
    score_hist = [0] * 10
    last_acc_j = -1  # mild monotonic preference; not strict

    with args.out_manifest.open("w", encoding="utf-8") as fh:
        for cf in chunks_files:
            partstem = cf.name[: -len(".chunks.jsonl")]
            wavdir = d / "_chunkwav" / partstem
            chunks = [json.loads(ln) for ln in
                      cf.read_text(encoding="utf-8").splitlines()
                      if ln.strip()]
            chunks.sort(key=lambda c: c["idx"])
            if args.limit_chunks is not None:
                chunks = chunks[: args.limit_chunks]
            total_audio = max((c.get("end") or 0.0) for c in chunks) \
                          if chunks else 0.0
            for n, ch in enumerate(chunks):
                n_total += 1
                wav_path = wavdir / f"{ch['idx']:05d}.wav"
                if not wav_path.exists():
                    continue
                waveform, sr = torchaudio.load(str(wav_path))
                if sr != sample_rate:
                    waveform = torchaudio.functional.resample(
                        waveform, sr, sample_rate)
                # collapse to mono if needed
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)
                # find candidate sentence range by proportional position
                prop = (ch.get("start") or 0.0) / max(total_audio, 1e-6)
                j_center = int(prop * len(sents_rom))
                # mild monotonic nudge: don't go below last accepted-2
                lo = max(0, max(last_acc_j - 2, j_center - args.window))
                hi = min(len(sents_rom), j_center + args.window)
                if hi <= lo:
                    continue
                best = (-1.0, -1)
                for j in range(lo, hi):
                    if not sents_rom[j]:
                        continue
                    sc = align_score(model, dictionary, waveform,
                                     sents_rom[j], args.device)
                    if sc > best[0]:
                        best = (sc, j)
                score_hist[min(9, int(best[0] * 10))] += 1
                if best[0] >= args.score_threshold and best[1] >= 0:
                    fh.write(json.dumps({
                        "idx": ch["idx"],
                        "partstem": partstem,
                        "wav": str(wav_path),
                        "gt_text": sents_tr[best[1]],
                        "gt_text_rom": sents_rom[best[1]],
                        "titu_text": (ch.get("text") or "").strip(),
                        "score": round(best[0], 3),
                        "start": ch.get("start"),
                        "end": ch.get("end"),
                    }, ensure_ascii=False) + "\n")
                    fh.flush()
                    n_kept += 1
                    last_acc_j = best[1]
                if (n + 1) % 50 == 0:
                    print(f"  {partstem} {n+1}/{len(chunks)}  "
                          f"kept-so-far={n_kept}")

    rate = 100 * n_kept / max(n_total, 1)
    print(f"\nchunks total: {n_total}  kept: {n_kept} "
          f"({rate:.1f}%)  threshold={args.score_threshold}")
    print("MMS-FA score histogram:")
    for i, c in enumerate(score_hist):
        bar = "#" * min(60, c // 5)
        print(f"  [{i/10:.1f}-{(i+1)/10:.1f}] {c:4d}  {bar}")
    print(f"-> {args.out_manifest}")


if __name__ == "__main__":
    main()
