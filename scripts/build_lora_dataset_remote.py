"""
Self-contained dataset extraction for Bengali ASR LoRA training.

Designed for remote compute (98GB VRAM box) — single Python invocation,
fully resumable, time-budgeted, no manual uploads.

Pipeline per audio sample:
  audio → ZIPA (CTC, GPU) → IPA tokens → FST → noisy Bengali ──┐
  ground truth Bengali ────────────────────────────────────────┴─→ JSONL

Sources (in priority order):
  1. arif11/Bengali_AI_Speech     (32K conversational, parquet, ungated)
  2. SUST-CSE-Speech/banspeech    (8K multi-domain: audiobook/news/drama/etc.)
  3. SKNahin/open-large-bengali-asr-data
                                  (3.73M aggregator: CV+UCLA+OpenSLR+MADASR
                                   +Shrutilipi+kathbath+indictts+gali+fleurs)
  4. google/fleurs (bn_in)        (~3K formal news, parallel with English)

All data is filtered by:
  - Bengali script ratio ≥ 0.5
  - Duration 1-20s
  - For SKNahin: WER < 1.0 AND 0.5 ≤ WPS ≤ 8.0

Usage:
  uv run python scripts/build_lora_dataset_remote.py \
    --output-dir ./lora_data \
    --max-hours 24 \
    --sources fleurs bengali_ai_speech banspeech sknahin
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Iterator

import numpy as np

logger = logging.getLogger("build_lora_dataset_remote")


# ─── ZIPA (frozen IPA model) ─────────────────────────────────────────────

def load_zipa(prefer_fp32: bool = True):
    """Returns (session, vocab_dict, predict_fn)."""
    import torch
    import torchaudio.compliance.kaldi as kaldi
    import onnxruntime as ort
    from huggingface_hub import snapshot_download

    logger.info("Loading ZIPA model (cached after first run)...")
    zipa_dir = Path(snapshot_download(
        repo_id="anyspeech/zipa-large-crctc-ns-800k",
        ignore_patterns=["*.bin", "*.pt", "*.safetensors"],
    ))
    onnx_files = list(zipa_dir.rglob("*.onnx"))
    onnx_path = str(onnx_files[0])
    if prefer_fp32:
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

    providers = ["CPUExecutionProvider"]
    try:
        if ort.get_device() == "GPU":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    except Exception:
        pass
    session = ort.InferenceSession(onnx_path, providers=providers)
    logger.info(f"  ZIPA: {len(vocab)} phonemes, providers={session.get_providers()}")

    def predict(wav: np.ndarray, sr: int = 16000) -> list[str]:
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


# ─── Quality filters ─────────────────────────────────────────────────────

def is_valid_bengali(text: str, min_ratio: float = 0.5) -> bool:
    if not text:
        return False
    bn = sum(1 for c in text if 0x0980 <= ord(c) <= 0x09FF)
    letters = sum(1 for c in text if c.isalpha())
    return letters > 0 and bn / letters >= min_ratio


# ─── Source loaders ─────────────────────────────────────────────────────

def load_fleurs(token: str | None) -> tuple[Iterator, str, dict[str, str]]:
    """FLEURS bn_in train + validation, with English from en_us joined by id."""
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download

    # English index from TSV (small, fast)
    en_idx: dict[int, str] = {}
    for split in ["train", "validation"]:
        for path in [f"data/en_us/{split}.tsv", f"data/en_us/audio/{split}.tsv"]:
            try:
                tsv = hf_hub_download("google/fleurs", path, repo_type="dataset")
                with open(tsv, encoding="utf-8") as f:
                    for line in f:
                        parts = line.rstrip("\n").split("\t")
                        if len(parts) >= 4:
                            try:
                                en_idx[int(parts[0])] = parts[3].strip()
                            except (ValueError, IndexError):
                                continue
                logger.info(f"  en_us {split}: {len(en_idx)} indexed")
                break
            except Exception:
                continue

    def gen():
        for split in ["train", "validation"]:
            ds = load_dataset("google/fleurs", "bn_in", split=split,
                              streaming=True, trust_remote_code=True)
            for item in ds:
                try:
                    sid = int(item["id"])
                    yield {
                        "src": "fleurs",
                        "id": f"{split}_{sid}",
                        "audio": item["audio"],
                        "transcription": item["transcription"].strip(),
                        "english": en_idx.get(sid, ""),
                    }
                except Exception:
                    continue

    return gen(), "transcription", {"english_index": "via en_us TSV"}


def load_bengali_ai_speech(token: str | None) -> tuple[Iterator, str, dict[str, str]]:
    from datasets import load_dataset
    ds = load_dataset("arif11/Bengali_AI_Speech", split="train",
                      streaming=True, token=token)

    def gen():
        for idx, item in enumerate(ds):
            yield {
                "src": "bengali_ai_speech",
                "id": str(idx),
                "audio": item["audio"],
                "transcription": (item.get("transcription") or "").strip(),
                "english": "",
            }

    return gen(), "transcription", {}


def load_banspeech(token: str | None) -> tuple[Iterator, str, dict[str, str]]:
    from datasets import load_dataset
    ds = load_dataset("SUST-CSE-Speech/banspeech", split="train",
                      streaming=True, token=token)

    def gen():
        for idx, item in enumerate(ds):
            yield {
                "src": "banspeech",
                "id": str(idx),
                "audio": item["audio"],
                "transcription": (item.get("transcription") or "").strip(),
                "english": "",
            }

    return gen(), "transcription", {}


def load_sknahin(token: str | None,
                 wer_max: float = 1.0,
                 wps_min: float = 0.5,
                 wps_max: float = 8.0) -> tuple[Iterator, str, dict[str, str]]:
    """3.73M aggregated Bengali ASR with quality metadata."""
    from datasets import load_dataset
    ds = load_dataset("SKNahin/open-large-bengali-asr-data", split="train",
                      streaming=True, token=token)

    def gen():
        for idx, item in enumerate(ds):
            wer = item.get("wer")
            wps = item.get("wps", 5.0)
            if wer is None or wer >= wer_max:
                continue
            if wps < wps_min or wps > wps_max:
                continue
            yield {
                "src": "sknahin",
                "id": str(idx),
                "audio": item["audio"],
                "transcription": (item.get("transcription") or "").strip(),
                "english": "",
            }

    return gen(), "transcription", {
        "filter": f"wer<{wer_max} AND {wps_min}≤wps≤{wps_max}",
    }


SOURCES = {
    "fleurs": load_fleurs,
    "bengali_ai_speech": load_bengali_ai_speech,
    "banspeech": load_banspeech,
    "sknahin": load_sknahin,
}


# ─── Extraction loop (resume-safe + time-budget) ─────────────────────────

def extract_source(stream: Iterator, src_name: str, out_path: Path,
                   predict_fn, ipa_to_bengali_fn,
                   min_s: float = 1.0, max_s: float = 20.0,
                   cap: int = -1, max_seconds: float | None = None,
                   output_format: str = "fst"):
    """Stream → ZIPA → JSONL. Resume-safe by id.

    output_format:
      "fst" (default) — save FST output as `noisy` field
      "ipa"           — save space-separated IPA tokens as `ipa` field
      "both"          — save both `ipa` and `noisy` fields
    """
    done_ids: set[str] = set()
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

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "a", encoding="utf-8") as f:
        for item in stream:
            if cap > 0 and n >= cap:
                logger.info(f"  Cap reached: {cap}")
                break
            if deadline and time.time() > deadline:
                logger.info(f"  Time budget reached: {max_seconds/3600:.1f}h")
                break

            sid = item["id"]
            if sid in done_ids:
                continue

            try:
                wav = np.array(item["audio"]["array"], dtype=np.float32)
                sr = item["audio"]["sampling_rate"]
                text = item["transcription"]
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

                ipa_text = " ".join(ipa_tokens)

                if output_format == "ipa":
                    record = {
                        "src": item["src"],
                        "id": sid,
                        "ipa": ipa_text,
                        "bengali": item["transcription"],
                        "english": item.get("english", ""),
                    }
                elif output_format == "both":
                    noisy = ipa_to_bengali_fn(ipa_tokens)
                    record = {
                        "src": item["src"],
                        "id": sid,
                        "ipa": ipa_text,
                        "noisy": noisy,
                        "bengali": item["transcription"],
                        "english": item.get("english", ""),
                    }
                else:  # "fst" (default — backward compatible)
                    noisy = ipa_to_bengali_fn(ipa_tokens)
                    if not noisy.strip():
                        n_filtered += 1
                        continue
                    record = {
                        "src": item["src"],
                        "id": sid,
                        "noisy": noisy,
                        "bengali": item["transcription"],
                        "english": item.get("english", ""),
                    }

                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                n += 1
                n_new += 1

                if n_new % 100 == 0:
                    rate = n_new / max(time.time() - t0, 1e-6)
                    elapsed = (time.time() - t0) / 60
                    logger.info(
                        f"  [{src_name}] {n} done "
                        f"({rate:.1f}/s, {elapsed:.0f}min)"
                    )
                    f.flush()
            except Exception as e:
                n_errors += 1
                if n_errors % 100 == 0:
                    logger.warning(f"  Error count: {n_errors} (last: {e})")
                continue

    elapsed = (time.time() - t0) / 60
    logger.info(
        f"  ✓ {src_name}: {n_new} new ({n} total), "
        f"{n_filtered} filtered, {n_errors} errors, {elapsed:.1f}min"
    )
    return n


def combine_sources(out_dir: Path, combined_path: Path):
    """Concatenate all per-source JSONL into one training file."""
    n_per_src: dict[str, int] = {}
    with open(combined_path, "w", encoding="utf-8") as out:
        for jsonl in sorted(out_dir.glob("*.jsonl")):
            if jsonl.name == combined_path.name:
                continue
            n = 0
            with open(jsonl, encoding="utf-8") as f:
                for line in f:
                    out.write(line)
                    n += 1
            n_per_src[jsonl.stem] = n
    return n_per_src


# ─── CLI ────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, default=Path("./lora_data"))
    p.add_argument("--sources", nargs="+",
                   default=["fleurs", "bengali_ai_speech", "banspeech", "sknahin"],
                   choices=list(SOURCES.keys()))
    p.add_argument("--max-hours", type=float, default=None,
                   help="Total wall-clock budget across all sources (default: unbounded)")
    p.add_argument("--per-source-cap", type=int, default=-1,
                   help="Per-source sample cap (default -1 = no cap)")
    p.add_argument("--sknahin-wer-max", type=float, default=1.0)
    p.add_argument("--sknahin-wps-min", type=float, default=0.5)
    p.add_argument("--sknahin-wps-max", type=float, default=8.0)
    p.add_argument("--combined-name", default="lora_dataset_full.jsonl")
    p.add_argument("--output-format", choices=["fst", "ipa", "both"],
                   default="fst",
                   help="What to save per sample: "
                        "'fst' = Bengali script via FST (current default), "
                        "'ipa' = space-separated IPA tokens (for IPA→LLM training), "
                        "'both' = save both")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Make local FST module importable.
    # transliteration_p2g.py imports `from asr_pipeline.phoneme_index import PhonemeIndex`,
    # so we need src/ on path (not src/asr_pipeline/) for the package import to work.
    repo_root = Path(__file__).parent.parent
    sys.path.insert(0, str(repo_root / "src"))
    from asr_pipeline.transliteration_p2g import ipa_to_bengali_script

    # Load ZIPA once
    _, _, predict_fn = load_zipa()

    # Optional: pass HF token if available
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        logger.info("Using HF token from environment")
    else:
        logger.warning("No HF_TOKEN in env; ungated repos will work")

    # Time budget split equally among sources (rough)
    per_source_seconds = None
    if args.max_hours is not None:
        per_source_seconds = (args.max_hours * 3600) / max(len(args.sources), 1)
        logger.info(f"Per-source time budget: "
                    f"{per_source_seconds/3600:.1f}h ({len(args.sources)} sources)")

    for src_name in args.sources:
        logger.info("=" * 60)
        logger.info(f"SOURCE: {src_name}")
        logger.info("=" * 60)

        loader = SOURCES[src_name]
        try:
            if src_name == "sknahin":
                stream, _, meta = loader(token,
                                         wer_max=args.sknahin_wer_max,
                                         wps_min=args.sknahin_wps_min,
                                         wps_max=args.sknahin_wps_max)
            else:
                stream, _, meta = loader(token)
            for k, v in meta.items():
                logger.info(f"  meta.{k}: {v}")

            # Filename reflects the output format
            suffix = "_ipa" if args.output_format == "ipa" else ""
            out_path = args.output_dir / f"{src_name}{suffix}.jsonl"
            extract_source(
                stream, src_name=src_name, out_path=out_path,
                predict_fn=predict_fn,
                ipa_to_bengali_fn=ipa_to_bengali_script,
                cap=args.per_source_cap,
                max_seconds=per_source_seconds,
                output_format=args.output_format,
            )
        except Exception as e:
            logger.error(f"  Source {src_name} FAILED: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Combine
    logger.info("=" * 60)
    logger.info("COMBINING SOURCES")
    logger.info("=" * 60)
    combined_path = args.output_dir / args.combined_name
    n_per_src = combine_sources(args.output_dir, combined_path)
    logger.info(f"Combined → {combined_path}")
    for k, v in n_per_src.items():
        logger.info(f"  {k:>30}: {v:>8}")
    logger.info(f"  {'TOTAL':>30}: {sum(n_per_src.values()):>8}")


if __name__ == "__main__":
    main()
