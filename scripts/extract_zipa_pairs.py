"""Run ZIPA on every utterance in a long-form-aligned manifest to
produce (real-ZIPA-IPA, GT-text) training pairs for the P2G model.

This is the domain-calibration ingredient: real ZIPA output on our
own coastal Bangladeshi audio, paired with the long-form-aligned GT
text. Mixed into the synthetic training set at a small fraction
(~5%), it anchors the model to ZIPA's empirical error distribution
on our domain — far better than randomly-noised Epitran IPA.

Input:  data/w2v_train_longform/<stem>.longform.jsonl
        (rows: idx, wav, gt_text, start, end, partstem)

Output: data/p2g_synth/real_pairs_<stem>.jsonl
        (rows: ipa, text)  — same schema as gen_synthetic_p2g.py

Resumable: skips utts already in the output file.

Usage:
  uv run python scripts/extract_zipa_pairs.py \\
      --manifest data/w2v_train_longform/07_*.longform.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def load_zipa():
    """Reuse zipa_chunks.py's loader pattern."""
    import numpy as np
    import onnxruntime as ort
    import torch
    import torchaudio.compliance.kaldi as kaldi
    from huggingface_hub import snapshot_download

    providers = ort.get_available_providers()
    print(f"  ONNX providers: {providers}", flush=True)
    if "CUDAExecutionProvider" not in providers:
        print("  WARN: CUDA not available — running on CPU (slow)",
              flush=True)

    print("  loading anyspeech/zipa-large-crctc-ns-800k ...", flush=True)
    zipa_dir = Path(snapshot_download(
        repo_id="anyspeech/zipa-large-crctc-ns-800k",
        ignore_patterns=["*.bin", "*.pt", "*.safetensors"],
    ))
    onnx_files = list(zipa_dir.rglob("*.onnx"))
    onnx_path = str(onnx_files[0])
    for p in onnx_files:
        if "fp16" not in p.name.lower() and "int8" not in p.name.lower():
            onnx_path = str(p)
            break
    tokens_path = str(next(zipa_dir.rglob("tokens.txt")))
    vocab = {}
    with open(tokens_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                vocab[int(parts[1])] = parts[0]

    use_providers = [
        ("CUDAExecutionProvider", {"device_id": 0}),
        "CPUExecutionProvider",
    ]
    sess = ort.InferenceSession(onnx_path, providers=use_providers)
    print(f"  ZIPA ready: {len(vocab)} phonemes, "
          f"{sess.get_providers()[0]}", flush=True)

    def predict(wav, sr=16000):
        if len(wav) == 0:
            return ""
        if sr != 16000:
            import librosa
            wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
        audio_t = torch.from_numpy(wav).float().unsqueeze(0)
        feats = kaldi.fbank(
            audio_t, sample_frequency=16000, num_mel_bins=80,
            frame_length=25.0, frame_shift=10.0, dither=0.0,
            snip_edges=False,
        )
        feat_batch = feats.unsqueeze(0).numpy()
        feat_lens = np.array([feats.shape[0]], dtype=np.int64)
        log_probs = sess.run(
            None, {"x": feat_batch, "x_lens": feat_lens})[0][0]
        ids = log_probs.argmax(-1)
        toks, prev = [], -1
        for i in ids:
            if i != prev and i != 0:
                toks.append(vocab.get(int(i), ""))
            prev = i
        return " ".join(t for t in toks if t)
    return predict


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True,
                    help="long-form manifest (data/w2v_train_longform/"
                         "<stem>.longform.jsonl)")
    ap.add_argument("--out", type=Path, default=None,
                    help="output JSONL (default: "
                         "data/p2g_synth/real_pairs_<stem>.jsonl)")
    ap.add_argument("--limit", type=int, default=None,
                    help="only process first N utts (smoke test)")
    args = ap.parse_args()

    if not args.manifest.exists():
        sys.exit(f"missing manifest: {args.manifest}")

    # Derive output path
    if args.out is None:
        stem = args.manifest.stem.replace(".longform", "")
        args.out = REPO / "data" / "p2g_synth" / f"real_pairs_{stem}.jsonl"
    args.out.parent.mkdir(parents=True, exist_ok=True)

    # Load manifest
    rows = []
    for ln in args.manifest.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        rows.append(json.loads(ln))
    rows.sort(key=lambda r: r["idx"])
    if args.limit:
        rows = rows[: args.limit]
    print(f"manifest: {len(rows)} utts")

    # Resume: skip already-processed
    done_idxs: set[int] = set()
    if args.out.exists():
        for ln in args.out.read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if ln:
                try:
                    done_idxs.add(json.loads(ln)["_idx"])
                except (KeyError, json.JSONDecodeError):
                    pass
    todo = [r for r in rows if r["idx"] not in done_idxs]
    print(f"  {len(done_idxs)} already done, {len(todo)} to process")

    if not todo:
        print("nothing to do.")
        return

    predict = load_zipa()

    import soundfile as sf
    t0 = time.time()
    n_written = 0
    n_skipped = 0
    with args.out.open("a", encoding="utf-8") as fh:
        for n, r in enumerate(todo, 1):
            wav_path = r["wav"]
            if not Path(wav_path).exists():
                n_skipped += 1
                continue
            try:
                wav, sr = sf.read(wav_path, dtype="float32")
                if wav.ndim > 1:
                    wav = wav.mean(axis=1)
                ipa = predict(wav, sr)
            except Exception as e:
                print(f"  ERR idx {r['idx']}: {type(e).__name__}: "
                      f"{str(e)[:120]}", flush=True)
                n_skipped += 1
                continue
            if not ipa.strip():
                n_skipped += 1
                continue
            fh.write(json.dumps({
                "_idx": r["idx"],
                "ipa": ipa,
                "text": r["gt_text"],
                "source": "real_zipa",
            }, ensure_ascii=False) + "\n")
            fh.flush()
            n_written += 1
            if n % 50 == 0 or n == len(todo):
                rate = n / max(time.time() - t0, 1e-6)
                eta = (len(todo) - n) / rate if rate > 0 else 0
                print(f"  {n}/{len(todo)} done  "
                      f"({rate:.1f} utt/s, eta {eta:.0f}s)  "
                      f"written={n_written}  skipped={n_skipped}",
                      flush=True)

    print(f"\nwrote {n_written} real pairs -> {args.out}")
    if n_skipped:
        print(f"  skipped {n_skipped} utts (missing wav or empty IPA)")


if __name__ == "__main__":
    main()
