"""Decode existing ZIPA IPA outputs with the trained P2G model and
score against the long-form GT manifest.

Pipeline:
  1. Load P2G checkpoint (data/p2g/ckpt/best.pt)
  2. Read existing ZIPA chunk output:
       outputs/real_eval_titu_excellent/<stem>/*.zipa.jsonl
     (each row: {idx, start, end, ipa})
  3. Greedy-decode each IPA → Bengali text via P2G
  4. Match each chunk to overlapping utterances in the long-form
     manifest by time overlap; concatenate chunks per utt
  5. Compute WER / CER vs gt_text per utt; report distribution

Output:
  data/p2g/eval_<stem>/p2g_hyps.jsonl    — per-chunk P2G output
  data/p2g/eval_<stem>/per_utt.tsv       — per-utt CER/WER with ref/hyp
  data/p2g/eval_<stem>/summary.txt       — distribution + headline

Usage:
  uv run python scripts/decode_p2g.py \\
      --stem 15_Sunamganj_CA-MMI_URBAN_Male_15_0602_1635
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[1]


# ---- minimal vocab class (same as train_p2g) ------------------------

class Vocab:
    def __init__(self, itos: list[str], specials: list[str]):
        self.itos = itos
        self.specials = specials
        self.stoi = {t: i for i, t in enumerate(itos)}
        self.pad_id = self.stoi["<pad>"]
        self.bos_id = self.stoi["<bos>"]
        self.eos_id = self.stoi["<eos>"]
        self.unk_id = self.stoi["<unk>"]

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, toks) -> list[int]:
        return [self.stoi.get(t, self.unk_id) for t in toks]

    def decode(self, ids: list[int]) -> str:
        out = []
        for i in ids:
            if i in (self.pad_id, self.bos_id, self.eos_id):
                continue
            out.append(self.itos[i] if 0 <= i < len(self.itos) else "?")
        return "".join(out)


# ---- model (same architecture as train_p2g) -------------------------

class P2GModel(nn.Module):
    def __init__(self, src_vocab: int, tgt_vocab: int,
                 d_model: int = 256, nhead: int = 4,
                 enc_layers: int = 3, dec_layers: int = 3,
                 dim_ff: int = 512, max_len: int = 512,
                 dropout: float = 0.1, pad_src: int = 0,
                 pad_tgt: int = 0):
        super().__init__()
        self.d_model = d_model
        self.pad_src = pad_src
        self.pad_tgt = pad_tgt
        self.src_emb = nn.Embedding(src_vocab, d_model,
                                     padding_idx=pad_src)
        self.tgt_emb = nn.Embedding(tgt_vocab, d_model,
                                     padding_idx=pad_tgt)
        self.src_pos = nn.Embedding(max_len, d_model)
        self.tgt_pos = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)
        self.transformer = nn.Transformer(
            d_model=d_model, nhead=nhead,
            num_encoder_layers=enc_layers,
            num_decoder_layers=dec_layers,
            dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True,
            activation="gelu", norm_first=True,
        )
        self.out = nn.Linear(d_model, tgt_vocab)

    def _add_pos(self, x, pos_emb):
        T = x.shape[1]
        return x + pos_emb(torch.arange(T, device=x.device)).unsqueeze(0)

    @torch.no_grad()
    def greedy(self, src: torch.Tensor, bos: int, eos: int,
               max_len: int = 200) -> list[list[int]]:
        self.eval()
        B = src.shape[0]
        dev = src.device
        src_pad = src == self.pad_src
        s = self._add_pos(
            self.src_emb(src) * math.sqrt(self.d_model), self.src_pos)
        mem = self.transformer.encoder(s,
                                         src_key_padding_mask=src_pad)
        ys = torch.full((B, 1), bos, dtype=torch.long, device=dev)
        done = torch.zeros(B, dtype=torch.bool, device=dev)
        for _ in range(max_len):
            T = ys.shape[1]
            causal = nn.Transformer.generate_square_subsequent_mask(
                T, device=dev)
            t = self._add_pos(
                self.tgt_emb(ys) * math.sqrt(self.d_model),
                self.tgt_pos)
            h = self.transformer.decoder(
                t, mem, tgt_mask=causal,
                memory_key_padding_mask=src_pad)
            nxt = self.out(h[:, -1, :]).argmax(dim=-1)
            ys = torch.cat([ys, nxt.unsqueeze(1)], dim=1)
            done = done | (nxt == eos)
            if done.all():
                break
        out = []
        for i in range(B):
            seq = ys[i, 1:].tolist()
            if eos in seq:
                seq = seq[: seq.index(eos)]
            out.append(seq)
        return out


# ---- WER / CER ------------------------------------------------------

def edit_distance(a, b):
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1,
                            prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


_PUNCT = ",.?!|:;-—()[]{}\"'`/\\“”’‘"

def norm(s: str) -> str:
    s = s.lower()
    for p in _PUNCT:
        s = s.replace(p, " ")
    return " ".join(s.split())


def wer_cer(ref: str, hyp: str) -> tuple[float, float]:
    r = norm(ref).split()
    h = norm(hyp).split()
    rd = edit_distance(r, h) / max(1, len(r))
    rc = list(norm(ref).replace(" ", ""))
    hc = list(norm(hyp).replace(" ", ""))
    cd = edit_distance(rc, hc) / max(1, len(rc))
    return rd, cd


# ---- main -----------------------------------------------------------

def _percentiles(xs: list[float]) -> dict:
    xs = sorted(xs)
    n = len(xs)
    pick = lambda q: xs[min(n - 1, max(0, int(q * (n - 1))))]
    return {
        "p10": pick(0.1), "p25": pick(0.25), "p50": pick(0.5),
        "p75": pick(0.75), "p90": pick(0.9),
        "mean": sum(xs) / max(1, n), "n": n,
        "under_0.3": sum(1 for x in xs if x < 0.3),
        "over_0.8": sum(1 for x in xs if x > 0.8),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", required=True)
    ap.add_argument("--ckpt", type=Path,
                    default=REPO / "data" / "p2g" / "ckpt" / "best.pt")
    ap.add_argument("--zipa-root", type=Path,
                    default=REPO / "outputs" / "real_eval_titu_excellent")
    ap.add_argument("--gt-manifest", type=Path, default=None,
                    help="long-form GT manifest "
                         "(default: data/w2v_train_longform/<stem>"
                         ".longform.jsonl)")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="default: data/p2g/eval_<stem>")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if not args.ckpt.exists():
        sys.exit(f"missing checkpoint: {args.ckpt}")

    if args.gt_manifest is None:
        args.gt_manifest = (REPO / "data" / "w2v_train_longform"
                            / f"{args.stem}.longform.jsonl")
    if args.out_dir is None:
        args.out_dir = REPO / "data" / "p2g" / f"eval_{args.stem}"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Load checkpoint
    print(f"loading checkpoint {args.ckpt} ...")
    device = torch.device(args.device)
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    ipa_vocab = Vocab(ck["ipa_vocab"], ck["specials"])
    txt_vocab = Vocab(ck["txt_vocab"], ck["specials"])
    cfg = ck.get("config", {})
    model = P2GModel(
        src_vocab=len(ipa_vocab),
        tgt_vocab=len(txt_vocab),
        d_model=cfg.get("d_model", 256),
        pad_src=ipa_vocab.pad_id,
        pad_tgt=txt_vocab.pad_id,
    ).to(device).eval()
    model.load_state_dict(ck["model"])
    print(f"  ckpt epoch={ck.get('epoch')}  "
          f"val_loss={ck.get('val_loss'):.3f}")

    # Read all ZIPA chunks for this stem
    stem_dir = args.zipa_root / args.stem
    zipa_files = sorted(stem_dir.glob("*.zipa.jsonl"))
    if not zipa_files:
        sys.exit(f"no *.zipa.jsonl in {stem_dir}")
    chunks: list[dict] = []
    for f in zipa_files:
        partstem = f.name[: -len(".zipa.jsonl")]
        for ln in f.read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if not ln:
                continue
            r = json.loads(ln)
            r["partstem"] = partstem
            chunks.append(r)
    print(f"loaded {len(chunks)} ZIPA chunks "
          f"from {len(zipa_files)} files")

    # Batch decode
    print("decoding ...")
    t0 = time.time()
    hyps_path = args.out_dir / "p2g_hyps.jsonl"
    decoded: dict[tuple, str] = {}
    with hyps_path.open("w", encoding="utf-8") as fh:
        for start in range(0, len(chunks), args.batch_size):
            batch = chunks[start: start + args.batch_size]
            srcs = [ipa_vocab.encode(c["ipa"].split()) for c in batch]
            srcs = [s for s in srcs if s]
            if not srcs:
                continue
            max_s = max(len(s) for s in srcs)
            src_t = torch.full((len(srcs), max_s),
                                ipa_vocab.pad_id, dtype=torch.long,
                                device=device)
            for i, s in enumerate(srcs):
                src_t[i, : len(s)] = torch.tensor(s,
                                                    dtype=torch.long,
                                                    device=device)
            try:
                out_ids = model.greedy(
                    src_t, txt_vocab.bos_id, txt_vocab.eos_id,
                    max_len=300)
            except Exception as e:
                print(f"  batch {start} failed: {e}")
                continue
            for c, ids in zip(batch, out_ids):
                text = txt_vocab.decode(ids)
                decoded[(c["partstem"], c["idx"])] = text
                fh.write(json.dumps({
                    "partstem": c["partstem"],
                    "idx": c["idx"],
                    "start": c.get("start"),
                    "end": c.get("end"),
                    "p2g_text": text,
                }, ensure_ascii=False) + "\n")
            if (start // args.batch_size) % 5 == 0:
                el = time.time() - t0
                rate = (start + len(batch)) / max(el, 1e-6)
                print(f"  {start + len(batch)}/{len(chunks)} chunks "
                      f"({rate:.1f}/s)", flush=True)
    print(f"decoded {len(decoded)} chunks in {time.time()-t0:.0f}s")

    # Score: for each GT utt, gather P2G chunks whose [start,end]
    # overlap the utt's [start,end], concatenate by chunk order.
    print(f"\nloading GT manifest {args.gt_manifest} ...")
    gt = []
    for ln in args.gt_manifest.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        gt.append(json.loads(ln))
    print(f"  {len(gt)} GT utts")

    # Build per-partstem sorted-by-start chunk list with their P2G text
    by_part: dict[str, list[dict]] = {}
    for c in chunks:
        if (c["partstem"], c["idx"]) not in decoded:
            continue
        by_part.setdefault(c["partstem"], []).append({
            "start": c.get("start", 0.0),
            "end": c.get("end", 0.0),
            "text": decoded[(c["partstem"], c["idx"])],
            "idx": c["idx"],
        })
    for v in by_part.values():
        v.sort(key=lambda x: x["start"])

    rows = []
    wers, cers = [], []
    for u in gt:
        partstem = u.get("partstem")
        chs = by_part.get(partstem, [])
        # collect chunks whose time range overlaps u's [start, end]
        ut0, ut1 = u["start"], u["end"]
        hits = [c for c in chs
                if (c["end"] > ut0 and c["start"] < ut1)]
        hyp = " ".join(c["text"] for c in hits).strip()
        ref = u["gt_text"]
        w, ce = wer_cer(ref, hyp)
        rows.append((u["idx"], ut0, ref, hyp, w, ce))
        wers.append(w)
        cers.append(ce)

    # Write per-utt tsv
    tsv = args.out_dir / "per_utt.tsv"
    with tsv.open("w", encoding="utf-8") as fh:
        fh.write("idx\tstart\twer\tcer\tref\thyp\n")
        for idx, s, ref, hyp, w, ce in rows:
            fh.write(f"{idx}\t{s:.1f}\t{w:.3f}\t{ce:.3f}\t"
                     f"{ref[:200]}\t{hyp[:200]}\n")

    wp = _percentiles(wers)
    cp = _percentiles(cers)
    summary = (
        f"P2G eval on {args.stem}\n"
        f"  ckpt: epoch={ck.get('epoch')} val_loss="
        f"{ck.get('val_loss'):.3f}\n"
        f"  chunks decoded: {len(decoded)}  utts scored: {len(rows)}\n\n"
        f"  WER       p10={wp['p10']:.2f}  p25={wp['p25']:.2f}  "
        f"p50={wp['p50']:.2f}  p75={wp['p75']:.2f}  "
        f"p90={wp['p90']:.2f}  mean={wp['mean']:.2f}\n"
        f"  CER       p10={cp['p10']:.2f}  p25={cp['p25']:.2f}  "
        f"p50={cp['p50']:.2f}  p75={cp['p75']:.2f}  "
        f"p90={cp['p90']:.2f}  mean={cp['mean']:.2f}\n"
        f"  good (CER<0.3): {cp['under_0.3']}/{cp['n']}\n"
        f"  bad  (CER>0.8): {cp['over_0.8']}/{cp['n']}\n"
    )
    (args.out_dir / "summary.txt").write_text(summary, encoding="utf-8")
    print("\n" + summary)
    print(f"per-utt details -> {tsv}")


if __name__ == "__main__":
    main()
