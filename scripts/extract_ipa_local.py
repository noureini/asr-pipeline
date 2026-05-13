"""
Local IPA dataset extraction. Designed for your laptop (RTX 3060 6GB).

Downloads each dataset fully (cached in ~/.cache/huggingface), then iterates
locally — much faster than streaming. Saves IPA tokens + Bengali (+ English
where available) per sample.

Resume-safe. If you Ctrl+C, just rerun — skips already-extracted samples.

Usage:
  export HF_TOKEN=hf_xxxxxxxxxxxxx
  uv run python scripts/extract_ipa_local.py \
    --output-dir ./lora_data_ipa \
    --max-hours-sknahin 4
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

logger = logging.getLogger("extract_ipa_local")


# ─── Quality filter ─────────────────────────────────────────────────────

def is_valid_bengali(text: str, min_ratio: float = 0.5) -> bool:
    if not text:
        return False
    bn = sum(1 for c in text if 0x0980 <= ord(c) <= 0x09FF)
    letters = sum(1 for c in text if c.isalpha())
    return letters > 0 and bn / letters >= min_ratio


# ─── ZIPA loading ───────────────────────────────────────────────────────

def load_zipa():
    import torch
    import torchaudio.compliance.kaldi as kaldi
    import onnxruntime as ort
    from huggingface_hub import snapshot_download

    # Verify GPU available
    providers = ort.get_available_providers()
    logger.info(f"ONNX providers: {providers}")
    if "CUDAExecutionProvider" not in providers:
        logger.warning("CUDA not in providers — will run on CPU (slow). "
                       "If you have a GPU, reinstall onnxruntime-gpu:")
        logger.warning("  uv pip uninstall onnxruntime onnxruntime-gpu")
        logger.warning("  uv pip install onnxruntime-gpu")
        proceed = input("Continue on CPU anyway? [y/N]: ").strip().lower()
        if proceed != "y":
            raise RuntimeError("Aborted by user — fix GPU setup first")

    logger.info("Loading ZIPA (cached after first run)...")
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
        ('CUDAExecutionProvider', {'device_id': 0}),
        'CPUExecutionProvider',
    ]
    session = ort.InferenceSession(onnx_path, providers=use_providers)
    logger.info(f"  ✓ ZIPA: {len(vocab)} phonemes, provider {session.get_providers()[0]}")

    def predict(wav, sr=16000):
        if sr != 16000:
            import librosa
            wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
        if len(wav) == 0:
            return []
        audio_t = torch.from_numpy(wav).float().unsqueeze(0)
        feats = kaldi.fbank(
            audio_t, sample_frequency=16000, num_mel_bins=80,
            frame_length=25.0, frame_shift=10.0, dither=0.0, snip_edges=False,
        )
        feat_batch = feats.unsqueeze(0).numpy()
        feat_lens = np.array([feats.shape[0]], dtype=np.int64)
        log_probs = session.run(None, {"x": feat_batch, "x_lens": feat_lens})[0][0]
        ids = log_probs.argmax(-1)
        tokens, prev = [], -1
        for i in ids:
            if i != prev and i != 0:
                tokens.append(vocab.get(int(i), ""))
            prev = i
        return [t for t in tokens if t]

    return session, vocab, predict


# ─── Generic extraction loop ────────────────────────────────────────────

def extract_to_jsonl(predict_fn, items, src_name, out_path,
                    text_key="transcription",
                    subsource="", english_lookup=None,
                    cap=-1, max_seconds=None,
                    min_s=1.0, max_s=20.0,
                    indexable=True):
    """Extract IPA per item. Saves resume-safe JSONL."""
    done_ids = set()
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)["id"])
                except Exception:
                    continue
        logger.info(f"  Resuming: {len(done_ids)} already done")

    n = len(done_ids)
    n_new = 0
    n_filtered = 0
    n_errors = 0
    t0 = time.time()
    deadline = t0 + max_seconds if max_seconds else None

    if indexable:
        iterator = ((i, items[i]) for i in range(len(items)))
    else:
        iterator = enumerate(items)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "a", encoding="utf-8") as f:
        for idx, item in iterator:
            if cap > 0 and n >= cap:
                logger.info(f"  Cap reached: {cap}")
                break
            if deadline and time.time() > deadline:
                logger.info(f"  Time budget reached")
                break
            sid = f"{subsource}_{idx}" if subsource else str(idx)
            if sid in done_ids:
                continue
            try:
                wav = np.array(item["audio"]["array"], dtype=np.float32)
                sr = item["audio"]["sampling_rate"]
                text = (item.get(text_key) or "").strip()
                dur = len(wav) / sr
                if not text or dur < min_s or dur > max_s:
                    n_filtered += 1
                    continue
                if not is_valid_bengali(text):
                    n_filtered += 1
                    continue
                ipa_tokens = predict_fn(wav, sr)
                if not ipa_tokens:
                    n_filtered += 1
                    continue

                english = ""
                if english_lookup is not None:
                    try:
                        english = english_lookup.get(int(item.get("id", -1)), "")
                    except Exception:
                        english = ""

                f.write(json.dumps({
                    "src": src_name,
                    "subsource": subsource,
                    "id": sid,
                    "ipa": " ".join(ipa_tokens),
                    "bengali": text,
                    "english": english,
                }, ensure_ascii=False) + "\n")
                n += 1
                n_new += 1

                if n_new % 100 == 0:
                    rate = n_new / max(time.time() - t0, 1e-6)
                    elapsed = (time.time() - t0) / 60
                    logger.info(
                        f"  [{src_name}/{subsource}] {n} done "
                        f"({rate:.1f}/s, {elapsed:.0f}min)"
                    )
                    f.flush()
            except Exception:
                n_errors += 1
                continue

    elapsed = (time.time() - t0) / 60
    logger.info(
        f"  ✓ {src_name}/{subsource}: {n_new} new ({n} total), "
        f"{n_filtered} filtered, {n_errors} errors, {elapsed:.1f}min"
    )
    return n


# ─── Source loaders ─────────────────────────────────────────────────────

def extract_fleurs(predict_fn, output_dir, token):
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download

    logger.info("=" * 60)
    logger.info("FLEURS bn_in (with parallel English)")
    logger.info("=" * 60)

    # Build English lookup
    en_idx = {}
    for split in ["train", "validation"]:
        for path in [f"data/en_us/{split}.tsv",
                     f"data/en_us/audio/{split}.tsv"]:
            try:
                tsv = hf_hub_download("google/fleurs", path,
                                      repo_type="dataset", token=token)
                with open(tsv, encoding="utf-8") as f:
                    for line in f:
                        parts = line.rstrip("\n").split("\t")
                        if len(parts) >= 4:
                            try:
                                en_idx[int(parts[0])] = parts[3].strip()
                            except Exception:
                                continue
                break
            except Exception:
                continue
    logger.info(f"  English lookup: {len(en_idx)} entries")

    for split in ["train", "validation"]:
        logger.info(f"\n--- FLEURS {split} (downloading + iterating) ---")
        ds = load_dataset(
            "google/fleurs", "bn_in", split=split,
            streaming=False, trust_remote_code=True, token=token,
        )
        logger.info(f"  ✓ Downloaded {len(ds)} samples")
        out_path = output_dir / f"fleurs_{split}_ipa.jsonl"
        extract_to_jsonl(
            predict_fn, ds, "fleurs", out_path,
            text_key="transcription",
            subsource=split, english_lookup=en_idx,
            indexable=True,
        )


def extract_bengali_ai_speech(predict_fn, output_dir, token, cap=4000):
    from datasets import load_dataset
    logger.info("=" * 60)
    logger.info(f"arif11/Bengali_AI_Speech (cap {cap})")
    logger.info("=" * 60)

    ds = load_dataset(
        "arif11/Bengali_AI_Speech", split="train",
        streaming=False, token=token,
    )
    logger.info(f"  ✓ Downloaded {len(ds)} samples")
    out_path = output_dir / "bengali_ai_speech_ipa.jsonl"
    extract_to_jsonl(
        predict_fn, ds, "bengali_ai_speech", out_path,
        text_key="transcription", cap=cap, indexable=True,
    )


def extract_banspeech(predict_fn, output_dir, token, cap_per_domain=500):
    from datasets import load_dataset
    logger.info("=" * 60)
    logger.info(f"SUST-CSE-Speech/banspeech (13 domains, cap {cap_per_domain} each)")
    logger.info("=" * 60)

    DOMAINS = [
        "audio_books", "biography", "celebrity_interview", "class_lecture",
        "documentary", "drama_series", "kid_cartoon", "kid_voice", "medicine",
        "parliament_speech", "political_talkshow", "sports", "television_news",
    ]

    for domain in DOMAINS:
        logger.info(f"\n--- banspeech / {domain} ---")
        try:
            ds = load_dataset(
                "SUST-CSE-Speech/banspeech", split=domain,
                streaming=False, token=token,
            )
            logger.info(f"  ✓ Downloaded {len(ds)} samples")
            out_path = output_dir / f"banspeech_{domain}_ipa.jsonl"
            extract_to_jsonl(
                predict_fn, ds, "banspeech", out_path,
                text_key="transcription",
                subsource=domain, cap=cap_per_domain,
                indexable=True,
            )
        except Exception as e:
            logger.warning(f"  ⚠ {domain} failed: {str(e)[:200]}")


def extract_sknahin(predict_fn, output_dir, token,
                    cap_per_corpus=1500, max_hours=4.0,
                    wer_max=1.0, wps_min=0.5, wps_max=8.0):
    from datasets import load_dataset
    logger.info("=" * 60)
    logger.info(f"SKNahin (9 corpora STREAMING, cap {cap_per_corpus} each, "
                f"{max_hours}h budget)")
    logger.info("=" * 60)

    CORPORA = [
        ("flerus", 3010), ("kathbath", 4590), ("gali", 10000),
        ("indictts", 12800), ("openslr", 199000), ("shrutilipi", 246000),
        ("madasr", 372000), ("commonvoice", 964000), ("ucla", 1920000),
    ]

    seconds_per_corpus = (max_hours * 3600) / len(CORPORA)
    logger.info(f"  Time per corpus: {seconds_per_corpus/60:.0f} min")

    for split_name, available in CORPORA:
        logger.info(f"\n--- SKNahin / {split_name} ---")
        try:
            ds = load_dataset(
                "SKNahin/open-large-bengali-asr-data",
                split=split_name, streaming=True, token=token,
            )
            ds = ds.filter(lambda ex: (
                ex.get("wer") is not None
                and ex["wer"] < wer_max
                and ex.get("wps", 5) >= wps_min
                and ex.get("wps", 5) <= wps_max
            ))
            out_path = output_dir / f"sknahin_{split_name}_ipa.jsonl"
            extract_to_jsonl(
                predict_fn, ds, "sknahin", out_path,
                text_key="transcription",
                subsource=split_name,
                cap=min(cap_per_corpus, available),
                max_seconds=seconds_per_corpus,
                indexable=False,
            )
        except Exception as e:
            logger.warning(f"  ⚠ {split_name} failed: {str(e)[:200]}")


# ─── Combine + split ────────────────────────────────────────────────────

def combine_all(output_dir):
    import glob
    logger.info("=" * 60)
    logger.info("COMBINING all IPA jsonl files")
    logger.info("=" * 60)

    combined = output_dir / "lora_dataset_full_ipa.jsonl"
    all_files = sorted(glob.glob(str(output_dir / "*_ipa.jsonl")))
    all_files = [f for f in all_files if "_full_" not in f
                 and "_train_" not in f and "_val_" not in f]

    n_per_file = {}
    total = 0
    with open(combined, "w", encoding="utf-8") as out:
        for src in all_files:
            n = 0
            with open(src, encoding="utf-8") as f:
                for line in f:
                    out.write(line)
                    n += 1
            n_per_file[Path(src).stem] = n
            total += n

    logger.info(f"\n  Total: {total}")
    for k, v in sorted(n_per_file.items(), key=lambda x: -x[1]):
        logger.info(f"  {k:<45} {v:>6}")
    logger.info(f"\n  ✓ Combined → {combined}")
    return combined


def split_train_val(combined, eval_per_source=100, seed=42):
    import random
    logger.info("=" * 60)
    logger.info("SPLITTING train/val (stratified)")
    logger.info("=" * 60)

    train_path = combined.parent / "lora_dataset_full_ipa_train.jsonl"
    val_path = combined.parent / "lora_dataset_full_ipa_val.jsonl"

    groups = defaultdict(list)
    with open(combined, encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line)
                key = (d.get("src", "?"), d.get("subsource", ""))
                groups[key].append(line)
            except Exception:
                continue

    logger.info(f"  Found {len(groups)} (src, subsource) groups")

    rng = random.Random(seed)
    train_lines, val_lines = [], []
    for key, lines in groups.items():
        rng.shuffle(lines)
        n_eval = min(eval_per_source, max(1, len(lines) // 5))
        val_lines.extend(lines[:n_eval])
        train_lines.extend(lines[n_eval:])

    rng.shuffle(train_lines)
    rng.shuffle(val_lines)

    with open(train_path, "w", encoding="utf-8") as f:
        f.writelines(train_lines)
    with open(val_path, "w", encoding="utf-8") as f:
        f.writelines(val_lines)

    logger.info(f"\n  train: {len(train_lines):>6}  → {train_path}")
    logger.info(f"  val:   {len(val_lines):>6}  → {val_path}")

    val_groups = defaultdict(int)
    for line in val_lines:
        d = json.loads(line)
        val_groups[(d.get("src", "?"), d.get("subsource", ""))] += 1
    logger.info(f"\n  Val composition:")
    for k in sorted(val_groups.keys(), key=lambda x: -val_groups[x]):
        logger.info(f"    {str(k):<45} {val_groups[k]:>4}")


# ─── Main ────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, default=Path("./lora_data_ipa"))
    p.add_argument("--sources", nargs="+",
                   default=["fleurs", "bengali_ai_speech", "banspeech", "sknahin"],
                   choices=["fleurs", "bengali_ai_speech", "banspeech", "sknahin"])
    p.add_argument("--cap-bengali-ai-speech", type=int, default=4000)
    p.add_argument("--cap-banspeech-domain", type=int, default=500)
    p.add_argument("--cap-sknahin-corpus", type=int, default=1500)
    p.add_argument("--max-hours-sknahin", type=float, default=4.0)
    p.add_argument("--skip-combine", action="store_true")
    p.add_argument("--eval-per-source", type=int, default=100)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Try .env file if HF_TOKEN not already set
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        # Look for .env in common locations
        env_paths = [
            Path(".env"),
            Path(".env.local"),
            Path(__file__).parent.parent / ".env",   # repo root
            Path.home() / ".env",
        ]
        # Common HF token env var names (any of these)
        TOKEN_KEYS = {
            "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN",
            "HF_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGINGFACE_API_TOKEN",
            "HF_API_TOKEN", "HF_ACCESS_TOKEN",
        }
        for env_path in env_paths:
            if env_path.exists():
                logger.info(f"Loading env from {env_path.resolve()}")
                found_keys = []
                with open(env_path, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#") or "=" not in line:
                            continue
                        key, _, val = line.partition("=")
                        key, val = key.strip(), val.strip().strip('"').strip("'")
                        found_keys.append(key)
                        if key in TOKEN_KEYS or key.upper() in TOKEN_KEYS:
                            token = val
                            os.environ["HF_TOKEN"] = val
                            os.environ["HUGGING_FACE_HUB_TOKEN"] = val
                            logger.info(f"  ✓ Found token via {key}")
                            break
                if not token:
                    logger.info(f"  Keys in this .env: {found_keys}")
                if token:
                    break

    if token:
        logger.info(f"HF token loaded (first 8 chars): {token[:8]}...")
    else:
        logger.warning("No HF_TOKEN found. Bengali_AI_Speech / banspeech / "
                       "SKNahin may fail.")
        logger.warning("Either: export HF_TOKEN=hf_..., or add HF_TOKEN=hf_... "
                       "to .env in repo root")

    # Load ZIPA once
    _, _, predict_fn = load_zipa()

    # Run requested sources
    if "fleurs" in args.sources:
        extract_fleurs(predict_fn, args.output_dir, token)
    if "bengali_ai_speech" in args.sources:
        extract_bengali_ai_speech(predict_fn, args.output_dir, token,
                                  cap=args.cap_bengali_ai_speech)
    if "banspeech" in args.sources:
        extract_banspeech(predict_fn, args.output_dir, token,
                          cap_per_domain=args.cap_banspeech_domain)
    if "sknahin" in args.sources:
        extract_sknahin(predict_fn, args.output_dir, token,
                        cap_per_corpus=args.cap_sknahin_corpus,
                        max_hours=args.max_hours_sknahin)

    # Combine + split
    if not args.skip_combine:
        combined = combine_all(args.output_dir)
        split_train_val(combined, eval_per_source=args.eval_per_source)

    logger.info("\n" + "=" * 60)
    logger.info("ALL DONE")
    logger.info("=" * 60)
    logger.info(f"Output dir: {args.output_dir}")


if __name__ == "__main__":
    main()
