"""WhisperX-style word-level alignment + per-word speaker
attribution + regroup into clean single-speaker segments.

Best-practice pipeline for accurate diarization:
  1. titu CTC produces chunks with text but only chunk-level
     timestamps.
  2. MMS_FA (torchaudio Massively Multilingual Speech forced
     aligner, supports Bengali via ISO 639-3 'ben') refines each
     chunk's text against the audio waveform → WORD-LEVEL
     timestamps (~20 ms accuracy).
  3. For each word, look up which pyannote speaker turn its
     timestamp falls in. This produces per-word speaker
     attribution.
  4. Walk through words in time order, group adjacent
     same-speaker words into a NEW segment. The output segments
     have exactly one speaker each — chunks that straddled a
     handoff get cleanly split at the word where the speaker
     changed.

Output: <stem-dir>/<partstem>.aligned_segments.jsonl with rows:
  {
    "segment_id": N,
    "start_s": float,
    "end_s": float,
    "speaker": "SPEAKER_xx",       // raw pyannote label
    "text": "...",
    "words": [
       {"word": "...", "start": s, "end": s, "speaker": "..."},
       ...
    ]
  }

assemble_transcript.py reads this when present (instead of
chunks.jsonl + diar.jsonl), and applies the ENUMERATOR/RESPONDENT
mapping to the new segments' speakers.

Usage:
  uv run python scripts/word_align_and_split.py \\
      --stem 15_Sunamganj_CA-MMI_URBAN_Male_15_0602_1635 \\
      --audio-folder "for testing (Noureini)"
"""
from __future__ import annotations

# Silence torchaudio/pyannote/speechbrain warnings BEFORE imports
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import argparse
import json
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parents[1]
DEFAULT_TITU_ROOT = REPO / "outputs" / "real_eval_titu_excellent"
DEFAULT_AUDIO_FOLDER = Path("for testing (Noureini)")


def to_wav(src: Path, dst: Path) -> None:
    """ffmpeg decode to mono 16 kHz PCM WAV — identical timeline to
    titu's chunk timestamps. pyannote/MMS_FA both need 16 kHz mono."""
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(src),
         "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
         str(dst)],
        check=True,
    )


def load_audio_part(audio_path: Path) -> tuple:
    """Decode .m4a → 16kHz mono float waveform tensor."""
    import torchaudio
    if audio_path.suffix.lower() != ".wav":
        tmp = Path(tempfile.mkdtemp()) / "audio.wav"
        to_wav(audio_path, tmp)
        wav_path = tmp
    else:
        wav_path = audio_path
    waveform, sr = torchaudio.load(str(wav_path))
    if sr != 16000:
        import torchaudio.functional as F
        waveform = F.resample(waveform, sr, 16000)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform, 16000


# ---- MMS_FA forced alignment per chunk -----------------------------

def align_chunk_words(
    model, tokenizer, aligner, dictionary,
    chunk_audio,        # 1D tensor at 16 kHz
    chunk_text: str,
    chunk_offset_s: float,
    device,
) -> list[dict]:
    """Run MMS_FA on one chunk. Returns a list of
    {word, start, end} dicts with GLOBAL second timestamps."""
    import torch
    # Normalize/strip — MMS_FA expects lowercase tokens for some
    # languages; for Bengali script it's mostly pass-through.
    words = chunk_text.strip().split()
    if not words:
        return []

    # Tokenize each word into the model's dictionary indices
    try:
        token_ids_per_word = [tokenizer(w) for w in words]
    except Exception:
        # Tokenizer may not accept some chars; skip this chunk
        return []
    # Flatten + remember word boundaries
    flat_tokens: list[int] = []
    word_token_spans: list[tuple[int, int]] = []
    for ids in token_ids_per_word:
        if not ids:
            word_token_spans.append((len(flat_tokens), len(flat_tokens)))
            continue
        start_i = len(flat_tokens)
        flat_tokens.extend(ids)
        word_token_spans.append((start_i, len(flat_tokens)))
    if not flat_tokens:
        return []

    with torch.inference_mode():
        emissions, _ = model(chunk_audio.to(device).unsqueeze(0))
    # emissions: (1, T, V); aligner expects (T, V)
    try:
        token_spans = aligner(emissions[0],
                               torch.tensor([flat_tokens],
                                             device=device))
    except Exception:
        return []
    # token_spans is list[list[TokenSpan]] — per batch element.
    # Take batch 0.
    spans = token_spans[0]

    # Time per emission frame (from MMS_FA model stride)
    n_frames = emissions.shape[1]
    n_samples = chunk_audio.shape[0]
    sec_per_frame = (n_samples / 16000.0) / max(1, n_frames)

    out: list[dict] = []
    for w_idx, (ts, te) in enumerate(word_token_spans):
        if ts == te:
            continue
        word_spans = spans[ts:te]
        if not word_spans:
            continue
        w_start = word_spans[0].start * sec_per_frame
        w_end = word_spans[-1].end * sec_per_frame
        out.append({
            "word": words[w_idx],
            "start": round(chunk_offset_s + w_start, 3),
            "end": round(chunk_offset_s + w_end, 3),
        })
    return out


# ---- per-word speaker assignment + regroup -------------------------

def assign_speakers_to_words(words: list[dict],
                              turns: list[dict]) -> list[dict]:
    """For each word, find pyannote speaker whose turn covers the
    word's midpoint. Falls back to the speaker with max overlap
    if no turn covers midpoint."""
    for w in words:
        mid = (w["start"] + w["end"]) / 2.0
        # Try midpoint-in-turn first (fast and clean)
        for t in turns:
            if t["start"] <= mid <= t["end"]:
                w["speaker"] = t["speaker"]
                break
        else:
            # Fallback: max overlap
            ov = defaultdict(float)
            for t in turns:
                o = min(w["end"], t["end"]) - max(w["start"],
                                                    t["start"])
                if o > 0:
                    ov[t["speaker"]] += o
            w["speaker"] = (max(ov.items(), key=lambda x: x[1])[0]
                             if ov else "UNKNOWN")
    return words


def regroup_words_into_segments(words: list[dict]
                                  ) -> list[dict]:
    """Walk words in time order; start a new segment whenever the
    speaker changes."""
    if not words:
        return []
    segments: list[dict] = []
    cur_speaker = words[0]["speaker"]
    cur_words: list[dict] = [words[0]]
    seg_id = 0
    for w in words[1:]:
        if w["speaker"] == cur_speaker:
            cur_words.append(w)
        else:
            segments.append({
                "segment_id": seg_id,
                "start_s": cur_words[0]["start"],
                "end_s": cur_words[-1]["end"],
                "speaker": cur_speaker,
                "text": " ".join(w["word"] for w in cur_words),
                "words": cur_words,
            })
            seg_id += 1
            cur_speaker = w["speaker"]
            cur_words = [w]
    segments.append({
        "segment_id": seg_id,
        "start_s": cur_words[0]["start"],
        "end_s": cur_words[-1]["end"],
        "speaker": cur_speaker,
        "text": " ".join(w["word"] for w in cur_words),
        "words": cur_words,
    })
    return segments


# ---- main per-part driver ------------------------------------------

def process_part(audio_path: Path, chunks_path: Path,
                  diar_path: Path, out_path: Path,
                  device: str = "cuda") -> None:
    import torch
    import torchaudio

    # Load chunks
    chunks = []
    for ln in chunks_path.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if ln:
            chunks.append(json.loads(ln))
    chunks.sort(key=lambda r: r["start"])

    # Load diar turns
    turns = []
    for ln in diar_path.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if ln:
            turns.append(json.loads(ln))

    # Load MMS_FA model
    print(f"  loading torchaudio MMS_FA bundle ...", flush=True)
    bundle = torchaudio.pipelines.MMS_FA
    dev = torch.device(device if torch.cuda.is_available()
                        else "cpu")
    model = bundle.get_model().to(dev).eval()
    tokenizer = bundle.get_tokenizer()
    aligner = bundle.get_aligner()
    dictionary = bundle.get_dict()

    # Load audio (full part)
    print(f"  loading audio: {audio_path.name}", flush=True)
    waveform, sr = load_audio_part(audio_path)
    # waveform: (1, N); strip channel
    waveform = waveform.squeeze(0)

    # Align each chunk → words with global timestamps
    print(f"  aligning {len(chunks)} chunks ...", flush=True)
    all_words: list[dict] = []
    n_aligned = 0
    n_skipped = 0
    for c in chunks:
        s, e = float(c["start"]), float(c["end"])
        s_sample = max(0, int(s * 16000))
        e_sample = min(waveform.shape[0], int(e * 16000))
        if e_sample - s_sample < 1600:  # <100ms — skip
            n_skipped += 1
            continue
        slice_audio = waveform[s_sample:e_sample]
        words = align_chunk_words(
            model, tokenizer, aligner, dictionary,
            slice_audio,
            c.get("text", "") or "",
            chunk_offset_s=s,
            device=dev,
        )
        if words:
            all_words.extend(words)
            n_aligned += 1
        else:
            n_skipped += 1
    print(f"    aligned {n_aligned} chunks, "
          f"skipped {n_skipped}, "
          f"{len(all_words)} words total", flush=True)

    # Assign per-word speaker
    all_words = assign_speakers_to_words(all_words, turns)

    # Regroup into clean single-speaker segments
    segments = regroup_words_into_segments(all_words)
    print(f"    -> {len(segments)} clean speaker-pure segments",
          flush=True)

    # Write
    with out_path.open("w", encoding="utf-8") as fh:
        for seg in segments:
            fh.write(json.dumps(seg, ensure_ascii=False) + "\n")
    print(f"    -> {out_path}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", required=True)
    ap.add_argument("--audio-folder", type=Path,
                    default=DEFAULT_AUDIO_FOLDER)
    ap.add_argument("--titu-root", type=Path,
                    default=DEFAULT_TITU_ROOT)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    stem_dir = args.titu_root / args.stem
    audio_files = sorted(
        p for p in args.audio_folder.iterdir()
        if p.suffix.lower() == ".m4a"
        and p.name.startswith(args.stem))
    if not audio_files:
        sys.exit(f"no .m4a for {args.stem}")
    print(f"audio parts: {[p.name for p in audio_files]}",
          flush=True)

    for audio_path in audio_files:
        partstem = audio_path.stem
        chunks_path = stem_dir / f"{partstem}.chunks.jsonl"
        diar_path = stem_dir / f"{partstem}.diar.jsonl"
        if not chunks_path.exists():
            print(f"  skip {partstem}: no chunks.jsonl", flush=True)
            continue
        if not diar_path.exists():
            print(f"  skip {partstem}: no diar.jsonl "
                  f"(run diarize_parts.py first)", flush=True)
            continue
        out_path = stem_dir / f"{partstem}.aligned_segments.jsonl"
        if out_path.exists():
            print(f"\n[{partstem}] aligned_segments already exists, "
                  f"skipping (delete to redo)", flush=True)
            continue
        print(f"\n[{partstem}] processing", flush=True)
        process_part(audio_path, chunks_path, diar_path, out_path,
                      device=args.device)

    print("\ndone.")


if __name__ == "__main__":
    main()
