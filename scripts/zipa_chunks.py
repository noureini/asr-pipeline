"""
Run ZIPA (universal IPA phone recognizer, anyspeech/zipa-large-crctc-
ns-800k) over the SAME VAD chunks dump_vad_chunks.py produced, so
we can inspect phoneme-level recognition on the same audio titu /
Omnilingual saw. Output (per part):

    <out-dir>/<stem>/<partstem>.zipa.jsonl
      {idx, start, end, ipa}

Reads manifests written by scripts/dump_vad_chunks.py. RESUMABLE.

  uv run python scripts/zipa_chunks.py --stem <STEM> \
      --out-dir outputs/real_eval_titu_excellent
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def load_zipa():
    """Load ZIPA non-interactively (the original loader prompts on
    CPU-only; we just fail loud if CUDA missing — keep it simple)."""
    import numpy as np
    import onnxruntime as ort
    import torch
    import torchaudio.compliance.kaldi as kaldi
    from huggingface_hub import snapshot_download

    providers = ort.get_available_providers()
    print(f"  ONNX providers: {providers}", flush=True)
    if "CUDAExecutionProvider" not in providers:
        print("  WARN: CUDA not in providers — running on CPU (slow).",
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
            onnx_path = str(p); break
    tokens_path = str(next(zipa_dir.rglob("tokens.txt")))
    vocab = {}
    with open(tokens_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                vocab[int(parts[1])] = parts[0]

    use_providers = [
        ('CUDAExecutionProvider', {'device_id': 0}),
        'CPUExecutionProvider',
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
    ap.add_argument("--stem", required=True)
    ap.add_argument("--out-dir", type=Path,
                    default=Path("outputs/real_eval_titu_excellent"))
    ap.add_argument("--limit", type=int, default=None,
                    help="first N chunks PER PART (validation slice)")
    args = ap.parse_args()

    import soundfile as sf

    d = args.out_dir / args.stem
    mans = sorted(d.glob("*.manifest.jsonl"))
    if not mans:
        sys.exit(f"no *.manifest.jsonl in {d} — run dump_vad_chunks.py")

    predict = load_zipa()

    for man in mans:
        partstem = man.name[: -len(".manifest.jsonl")]
        rows = []
        for ln in man.read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if ln:
                rows.append(json.loads(ln))
        rows.sort(key=lambda r: r["idx"])
        if args.limit is not None:
            rows = rows[: args.limit]

        ckpt = d / f"{partstem}.zipa.jsonl"
        done = set()
        if ckpt.exists():
            for ln in ckpt.read_text(encoding="utf-8").splitlines():
                ln = ln.strip()
                if ln:
                    done.add(json.loads(ln)["idx"])
        todo = [r for r in rows if r["idx"] not in done]
        print(f"  {partstem}: {len(rows)} chunks ({len(done)} done) "
              f"-> {len(todo)} to ZIPA", flush=True)

        with ckpt.open("a", encoding="utf-8") as fh:
            t0 = time.time()
            for n, r in enumerate(todo, 1):
                wav, sr = sf.read(r["wav"], dtype="float32")
                if wav.ndim > 1:
                    wav = wav.mean(axis=1)
                ipa = predict(wav, sr)
                fh.write(json.dumps(
                    {"idx": r["idx"], "start": r["start"],
                     "end": r["end"], "ipa": ipa},
                    ensure_ascii=False) + "\n")
                fh.flush()
                if n % 25 == 0 or n == len(todo):
                    rate = n / max(time.time() - t0, 1e-6)
                    print(f"    {n}/{len(todo)} ({rate:.1f}/s)",
                          flush=True)
        print(f"  -> {ckpt}", flush=True)


if __name__ == "__main__":
    main()
