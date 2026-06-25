"""
Long-form MMS forced alignment: align FULL audio ↔ FULL GT text,
then segment by GT sentence boundaries.

This is what WhisperX / OpenAI Whisper data pipeline / the Swahili
paper actually do. Replaces the per-VAD-chunk verifier that mis-paired
sentences at coarse proportional positions.

Memory plan: audio in 5-min non-overlapping windows; per-window the
proportional GT-text portion is aligned independently. Boundary
sentences may be slightly off, mitigated by dropping the first and
last sentence of each window's emission span.

Output (same schema as align_mms_fa.py manifest):
  {idx, partstem, wav, gt_text, score, start, end}
where `wav` is the extracted utterance audio slice saved under
<out-dir>/_uttwav/<partstem>/<idx:05d>.wav.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import torch
import torchaudio

_RE_SPEAKER_TAG = re.compile(
    r"^(প্র:|উ:|Q:|A:|নোট টেকার:|নোট:|enumerator:|respondent:)\s*",
    re.M | re.I)
_RE_TIMESTAMP = re.compile(r"^(সময়[ঃ:]|time[:\s])", re.I)
_RE_PARENS = re.compile(r"\([^)]{0,200}\)")
_RE_BRACK = re.compile(r"\[[^\]]{0,200}\]")
_RE_WS = re.compile(r"\s+")
_RE_SENT = re.compile(r"[।.?!]+\s*")
_RE_MULTI_BAR = re.compile(r"\|+")


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
            if not s or _RE_TIMESTAMP.match(s):
                continue
            out.append(s)
    return out


def apply_translit(sent: str, table: dict[str, str]) -> str:
    return " ".join(table.get(w, w) for w in sent.split())


def normalize_for_mms(rom: str, dictionary: dict) -> str:
    """Lowercase, filter to chars present in MMS_FA dict (Latin
    lowercase + apostrophe + *). MMS_FA's dict has NO word-boundary
    token — alignment is purely character-level.

    Critically excludes the blank token char ('-' at index 0). uroman
    occasionally emits '-' from compound graphemes; allowing it in
    targets crashes forced_align (which requires blank-free targets).
    """
    text = rom.lower().strip()
    # blank char must be excluded from targets
    blank_chars = {c for c, i in dictionary.items() if i == 0}
    text = "".join(c for c in text
                    if c in dictionary and c not in blank_chars)
    return text


def align_window(model, dictionary, waveform, normalized,
                 device, blank_id=0):
    """Return per-char (start_frame, end_frame, score) tuples,
    relative to this window. Uses torchaudio.functional.forced_align
    + merge_tokens — the standard CTC alignment pattern."""
    if not normalized:
        return None
    tokens = [dictionary[c] for c in normalized]
    if len(tokens) < 2:
        return None
    with torch.no_grad():
        emission, _ = model(waveform.to(device))
    if emission.shape[1] < len(tokens):
        return None
    targets = torch.tensor([tokens], dtype=torch.int32, device=device)
    in_lens = torch.tensor([emission.shape[1]], device=device)
    tgt_lens = torch.tensor([len(tokens)], device=device)
    try:
        alignments, scores = torchaudio.functional.forced_align(
            emission, targets, in_lens, tgt_lens, blank=blank_id)
    except Exception as e:
        print(f"    forced_align failed: {type(e).__name__}: "
              f"{str(e)[:200]}", flush=True)
        return None
    # alignments: (1, T), scores: (1, T) — per-frame.
    # merge_tokens collapses consecutive-same to per-token spans.
    try:
        spans = torchaudio.functional.merge_tokens(
            alignments[0], scores[0].exp())
    except Exception as e:
        print(f"    merge_tokens failed: {type(e).__name__}: "
              f"{str(e)[:200]}", flush=True)
        return None
    # spans is a list of TokenSpan; spans[k] = (token, start, end, score)
    # for the k-th NON-BLANK token in temporal order. Our tokens list
    # has the same length and order as the non-blank token list.
    if len(spans) != len(tokens):
        print(f"    span/token mismatch: {len(spans)} vs {len(tokens)}",
              flush=True)
        return None
    return [(int(s.start), int(s.end), float(s.score)) for s in spans]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", required=True)
    ap.add_argument("--audio-folder", type=Path,
                    default=Path("for testing (Noureini)"))
    ap.add_argument("--docx", type=Path, required=True)
    _REPO = Path(__file__).resolve().parents[1]
    ap.add_argument("--translit-json", type=Path,
                    default=_REPO / "data" / "translit.json")
    ap.add_argument("--out-dir", type=Path,
                    default=_REPO / "data" / "w2v_train_longform")
    ap.add_argument("--window-s", type=float, default=300.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--score-threshold", type=float, default=0.30)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    utt_dir = args.out_dir / "_uttwav" / args.stem
    utt_dir.mkdir(parents=True, exist_ok=True)

    table = json.loads(args.translit_json.read_text(encoding="utf-8"))
    sents_raw = docx_sents(args.docx)
    sents_tr = [apply_translit(s, table) for s in sents_raw]
    print(f"GT sentences: {len(sents_tr)}")

    import uroman
    u = uroman.Uroman()
    sents_rom = [u.romanize_string(s, lcode="ben") for s in sents_tr]

    # Load MMS
    bundle = torchaudio.pipelines.MMS_FA
    model = bundle.get_model().to(args.device)
    model.eval()
    dictionary = bundle.get_dict()
    sr = bundle.sample_rate
    print(f"MMS_FA: {len(dictionary)} tokens, sr={sr}")

    # Normalize per sentence (Latin chars only, no separators),
    # concatenate directly (MMS-FA is character-level).
    parts: list[str] = []
    sent_norm_ranges: list[tuple[int, int]] = []
    cur_len = 0
    for s_rom in sents_rom:
        n = normalize_for_mms(s_rom, dictionary)
        if not n:
            sent_norm_ranges.append((-1, -1))
            continue
        sent_norm_ranges.append((cur_len, cur_len + len(n)))
        parts.append(n)
        cur_len += len(n)
    concat = "".join(parts)
    print(f"normalized concat length: {len(concat)} chars  "
          f"(non-empty sentences: "
          f"{sum(1 for r in sent_norm_ranges if r[0] >= 0)})")

    # Find the audio file(s) for this stem (may have _1 etc.)
    audio_files = sorted(
        p for p in args.audio_folder.iterdir()
        if p.suffix.lower() == ".m4a" and p.name.startswith(args.stem))
    if not audio_files:
        raise SystemExit(f"no audio for {args.stem}")

    # We'll align EACH part separately, then assign GT sentences
    # proportionally across all parts (cumulative duration)
    print(f"audio parts: {[p.name for p in audio_files]}")

    # First, load each part to get its duration
    part_durations = []
    part_waveforms = []
    for ap_ in audio_files:
        wav, file_sr = torchaudio.load(str(ap_))
        if file_sr != sr:
            wav = torchaudio.functional.resample(wav, file_sr, sr)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        part_durations.append(wav.shape[1] / sr)
        part_waveforms.append(wav)
        print(f"  {ap_.name}: {wav.shape[1]/sr:.1f}s")
    total_dur = sum(part_durations)

    # Concat all parts into one timeline (for proportional text mapping)
    # but keep part offsets so we can save audio slices from the right
    # part. Global time -> (part_idx, time_in_part).
    cum_offsets = [0.0]
    for d in part_durations:
        cum_offsets.append(cum_offsets[-1] + d)

    # Char positions per audio window
    num_windows = max(1, math.ceil(total_dur / args.window_s))
    print(f"windows: {num_windows} of {args.window_s}s each "
          f"(total {total_dur:.1f}s)")

    # For each window: get audio, run alignment with proportional text
    char_to_global_frame: dict[int, int] = {}
    for w in range(num_windows):
        ws = w * args.window_s
        we = min(ws + args.window_s, total_dur)
        if we - ws < 5.0:
            continue
        # extract audio slice across parts
        slices = []
        for part_idx, pdur in enumerate(part_durations):
            p_start = cum_offsets[part_idx]
            p_end = cum_offsets[part_idx + 1]
            ov_start = max(ws, p_start)
            ov_end = min(we, p_end)
            if ov_end > ov_start:
                local_s = int((ov_start - p_start) * sr)
                local_e = int((ov_end - p_start) * sr)
                slices.append(
                    part_waveforms[part_idx][:, local_s:local_e])
        if not slices:
            continue
        waveform = torch.cat(slices, dim=1)
        # proportional text range
        cs = int(w * len(concat) / num_windows)
        ce = int((w + 1) * len(concat) / num_windows)
        win_text = concat[cs:ce]
        spans = align_window(
            model, dictionary, waveform, win_text, args.device)
        if spans is None:
            print(f"  window {w}: align failed")
            continue
        avg = sum(s[2] for s in spans) / len(spans)
        # win_text position k -> spans[k] = (start_frame, end_frame, score)
        # convert to global time: add window start in frames
        win_offset_frames = ws * 50.0  # MMS @ 50Hz
        for k in range(len(win_text)):
            s_frame, e_frame, _sc = spans[k]
            # store midpoint of the char's frame span as its global frame
            mid = (s_frame + e_frame) / 2.0
            global_frame = mid + win_offset_frames
            char_to_global_frame[cs + k] = global_frame
        print(f"  window {w}: {ws:.0f}-{we:.0f}s align ok "
              f"avg_score={avg:.2f} chars={len(spans)}")

    # Build manifest from sent_norm_ranges + char_to_global_frame
    manifest_path = args.out_dir / f"{args.stem}.longform.jsonl"
    n_kept = 0
    with manifest_path.open("w", encoding="utf-8") as fh:
        for sent_idx, (s_start, s_end) in enumerate(sent_norm_ranges):
            if s_start < 0 or s_end - s_start < 4:
                continue
            # Find first and last char in this sentence range with
            # a mapped frame
            frames_in_sent = [
                char_to_global_frame[k]
                for k in range(s_start, s_end)
                if k in char_to_global_frame
            ]
            if len(frames_in_sent) < 2:
                continue
            start_frame = min(frames_in_sent)
            end_frame = max(frames_in_sent)
            start_s = start_frame / 50.0
            end_s = end_frame / 50.0 + 0.1  # tiny tail pad
            if end_s - start_s < 0.3 or end_s - start_s > 30.0:
                continue
            # Which part is start_s in?
            part_idx = 0
            while (part_idx < len(part_durations)
                   and cum_offsets[part_idx + 1] <= start_s):
                part_idx += 1
            if part_idx >= len(part_durations):
                continue
            local_start = start_s - cum_offsets[part_idx]
            local_end = min(end_s - cum_offsets[part_idx],
                             part_durations[part_idx])
            if local_end - local_start < 0.3:
                continue
            # extract slice
            ws_lo = int(local_start * sr)
            ws_hi = int(local_end * sr)
            slice_wav = part_waveforms[part_idx][:, ws_lo:ws_hi]
            partstem = audio_files[part_idx].stem
            out_wav = utt_dir / f"{sent_idx:05d}.wav"
            torchaudio.save(str(out_wav), slice_wav, sr)
            fh.write(json.dumps({
                "idx": sent_idx,
                "partstem": partstem,
                "wav": str(out_wav),
                "gt_text": sents_tr[sent_idx],
                "start": round(start_s, 3),
                "end": round(end_s, 3),
            }, ensure_ascii=False) + "\n")
            n_kept += 1
    print(f"\nutterances written: {n_kept} -> {manifest_path}")


if __name__ == "__main__":
    main()
