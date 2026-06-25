"""
Decode titu BPE-CTC log-probs with pyctcdecode + Bengali KenLM
(arijitx/wav2vec2-xls-r-300m-bengali  ->  5gram.bin + unigrams).

Reads the SAME persisted chunk WAVs the rest of the pipeline uses
(dumped by scripts/dump_vad_chunks.py), gets titu's per-frame
log-probs via NeMo Hypothesis.alignments, runs pyctcdecode with
the 5-gram word LM + 2.2M-unigram lexicon, writes per-chunk
decoded text to <partstem>.titu_lm.jsonl alongside the greedy
.chunks.jsonl. RESUMABLE.

Honest scope: KenLM is Bengali Wikipedia / news-style. Dialect-
suppression is the known risk; the side-by-side prints dialectal
spot-check chunks so we can SEE if dialect was pushed to standard.

  /tmp/titu-venv/bin/python scripts/titu_kenlm_decode.py \
      --stem <STEM> --out-dir outputs/real_eval_titu_excellent \
      [--limit N] [--beam-width 100] [--alpha 0.5 --beta 1.5]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

KENLM_BIN = "/tmp/bn_kenlm/language_model/5gram.bin"
UNIGRAMS = "/tmp/bn_kenlm/language_model/unigrams.txt"
# GT-bootstrapped alternative (in-domain, 8.3k words from 8 GT docx):
GT_UNIGRAMS = "/tmp/gt_lm/gt_unigrams.txt"


def _read_jsonl(p: Path) -> dict:
    m = {}
    if p.exists():
        for ln in p.read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if ln:
                r = json.loads(ln)
                m[r["idx"]] = r
    return m


def _labels_from_titu(model) -> list[str]:
    """Build pyctcdecode label list (length 1025): SP pieces with
    ▁ -> leading-space convention, plus blank '' at the last
    index (titu's blank id)."""
    vocab = list(model.tokenizer.vocab)  # 1024
    labels = []
    for p in vocab:
        labels.append((" " + p[1:]) if p.startswith("▁") else p)
    labels.append("")  # blank at idx 1024 (matches alignments shape)
    return labels


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", required=True)
    ap.add_argument("--out-dir", type=Path,
                    default=Path("outputs/real_eval_titu_excellent"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--beam-width", type=int, default=100)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--beta", type=float, default=1.5)
    ap.add_argument("--kenlm-path", default=KENLM_BIN,
                    help="path to KenLM .bin/.arpa, or 'none' "
                         "to skip LM (lexicon-filter only)")
    ap.add_argument("--unigrams-path", default=UNIGRAMS,
                    help="path to unigrams .txt (one word/line)")
    ap.add_argument("--out-suffix", default="titu_lm",
                    help="output file suffix for *.<suf>.jsonl")
    args = ap.parse_args()

    use_lm = args.kenlm_path.lower() != "none"
    if use_lm and not Path(args.kenlm_path).exists():
        sys.exit(f"KenLM not at {args.kenlm_path}")
    if not Path(args.unigrams_path).exists():
        sys.exit(f"unigrams not at {args.unigrams_path}")
    d = args.out_dir / args.stem
    mans = sorted(d.glob("*.manifest.jsonl"))
    if not mans:
        sys.exit(f"no *.manifest.jsonl in {d}")

    import nemo.collections.asr as nemo_asr
    from pyctcdecode import build_ctcdecoder

    print("loading titu …", flush=True)
    model = nemo_asr.models.ASRModel.from_pretrained(
        "hishab/titu_stt_bn_fastconformer")

    print("building labels …", flush=True)
    labels = _labels_from_titu(model)
    print(f"  {len(labels)} labels (1024 SP + 1 blank)", flush=True)

    print(f"loading unigrams from {args.unigrams_path} …", flush=True)
    with open(args.unigrams_path, encoding="utf-8") as f:
        unigrams = [w.strip() for w in f if w.strip()]
    print(f"  {len(unigrams):,} unigrams", flush=True)
    if use_lm:
        print(f"loading KenLM {args.kenlm_path} …", flush=True)
    decoder = build_ctcdecoder(
        labels=labels,
        kenlm_model_path=args.kenlm_path if use_lm else None,
        unigrams=unigrams,
        alpha=args.alpha, beta=args.beta,
    )
    print(f"  decoder ready (LM={use_lm} alpha={args.alpha} "
          f"beta={args.beta} beam={args.beam_width})", flush=True)

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

        ckpt = d / f"{partstem}.{args.out_suffix}.jsonl"
        done = _read_jsonl(ckpt)
        todo = [r for r in rows if r["idx"] not in done]
        print(f"  {partstem}: {len(rows)} chunks "
              f"({len(done)} done) -> {len(todo)} to decode",
              flush=True)
        t0 = time.time()
        with ckpt.open("a", encoding="utf-8") as fh:
            for n, r in enumerate(todo, 1):
                wav = r["wav"]
                try:
                    out = model.transcribe(
                        [wav], batch_size=1,
                        return_hypotheses=True, verbose=False)
                    h = out[0] if isinstance(out, list) else out
                    logits = h.alignments.cpu().numpy()  # (T, 1025)
                    text = decoder.decode(
                        logits, beam_width=args.beam_width).strip()
                except Exception as e:
                    text = ""
                    print(f"    WARN idx {r['idx']}: {e}", flush=True)
                fh.write(json.dumps(
                    {"idx": r["idx"], "start": r.get("start"),
                     "end": r.get("end"), "text": text},
                    ensure_ascii=False) + "\n")
                fh.flush()
                if n % 5 == 0 or n == len(todo):
                    rate = n / max(time.time() - t0, 1e-6)
                    print(f"    {n}/{len(todo)} ({rate:.1f}/s) "
                          f"\"{text[:60]}\"", flush=True)
        print(f"  -> {ckpt}", flush=True)
    print("done.")


if __name__ == "__main__":
    main()
