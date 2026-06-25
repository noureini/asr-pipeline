"""Per-utterance P2G decode + eval.

Reads a per-utt jsonl produced by extract_zipa_pairs.py (one row per
GT utterance: {_idx, ipa, text}), runs the trained P2G model on each
IPA, and computes WER/CER against the per-row text. 1-to-1 alignment,
no chunk-to-utt mismatch.

Usage:
  uv run python scripts/decode_p2g_perutt.py \\
      --pairs data/p2g/zipa_15.jsonl \\
      --out-dir data/p2g/eval_perutt_15
"""
from __future__ import annotations
import argparse, json, math, sys, time
from pathlib import Path
import torch, torch.nn as nn

REPO = Path(__file__).resolve().parents[1]


class Vocab:
    def __init__(self, itos, specials):
        self.itos = itos
        self.specials = specials
        self.stoi = {t: i for i, t in enumerate(itos)}
        self.pad_id = self.stoi["<pad>"]
        self.bos_id = self.stoi["<bos>"]
        self.eos_id = self.stoi["<eos>"]
        self.unk_id = self.stoi["<unk>"]

    def encode(self, toks):
        return [self.stoi.get(t, self.unk_id) for t in toks]

    def decode(self, ids):
        out = []
        for i in ids:
            if i in (self.pad_id, self.bos_id, self.eos_id):
                continue
            out.append(self.itos[i] if 0 <= i < len(self.itos) else "?")
        return "".join(out)

    def __len__(self):
        return len(self.itos)


class P2GModel(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, d_model=256, nhead=4,
                 enc_layers=3, dec_layers=3, dim_ff=512, max_len=512,
                 dropout=0.1, pad_src=0, pad_tgt=0):
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
            dim_feedforward=dim_ff, dropout=dropout, batch_first=True,
            activation="gelu", norm_first=True)
        self.out = nn.Linear(d_model, tgt_vocab)

    def _add_pos(self, x, pos_emb):
        T = x.shape[1]
        return x + pos_emb(torch.arange(T, device=x.device)).unsqueeze(0)

    @torch.no_grad()
    def greedy(self, src, bos, eos, max_len=300):
        self.eval()
        B = src.shape[0]
        dev = src.device
        src_pad = src == self.pad_src
        s = self._add_pos(self.src_emb(src) * math.sqrt(self.d_model),
                          self.src_pos)
        mem = self.transformer.encoder(s, src_key_padding_mask=src_pad)
        ys = torch.full((B, 1), bos, dtype=torch.long, device=dev)
        done = torch.zeros(B, dtype=torch.bool, device=dev)
        for _ in range(max_len):
            T = ys.shape[1]
            causal = nn.Transformer.generate_square_subsequent_mask(
                T, device=dev)
            t = self._add_pos(
                self.tgt_emb(ys) * math.sqrt(self.d_model), self.tgt_pos)
            h = self.transformer.decoder(t, mem, tgt_mask=causal,
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


def edit_distance(a, b):
    if not a: return len(b)
    if not b: return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j-1] + 1,
                           prev[j-1] + (ca != cb)))
        prev = cur
    return prev[-1]


_PUNCT = ",.?!|:;-—()[]{}\"'`/\\“”’‘।"

def norm(s):
    s = s.lower()
    for p in _PUNCT:
        s = s.replace(p, " ")
    return " ".join(s.split())


def wer_cer(ref, hyp):
    r = norm(ref).split()
    h = norm(hyp).split()
    wr = edit_distance(r, h) / max(1, len(r))
    rc = list(norm(ref).replace(" ", ""))
    hc = list(norm(hyp).replace(" ", ""))
    cr = edit_distance(rc, hc) / max(1, len(rc))
    return wr, cr


def percentiles(xs):
    xs = sorted(xs)
    n = len(xs)
    if n == 0:
        return {}
    pick = lambda q: xs[min(n - 1, int(q * (n - 1)))]
    return {
        "p10": pick(0.1), "p25": pick(0.25), "p50": pick(0.5),
        "p75": pick(0.75), "p90": pick(0.9),
        "mean": sum(xs) / n, "n": n,
        "under_0.3": sum(1 for x in xs if x < 0.3),
        "under_0.5": sum(1 for x in xs if x < 0.5),
        "over_0.8": sum(1 for x in xs if x > 0.8),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path,
                    default=REPO / "data" / "p2g" / "ckpt" / "best.pt")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    ipa_v = Vocab(ck["ipa_vocab"], ck["specials"])
    txt_v = Vocab(ck["txt_vocab"], ck["specials"])
    cfg = ck.get("config", {})
    model = P2GModel(
        src_vocab=len(ipa_v), tgt_vocab=len(txt_v),
        d_model=cfg.get("d_model", 256),
        pad_src=ipa_v.pad_id, pad_tgt=txt_v.pad_id,
    ).to(device).eval()
    model.load_state_dict(ck["model"])
    print(f"ckpt ep={ck.get('epoch')} val_loss={ck.get('val_loss'):.3f}")

    rows = [json.loads(l) for l in
            args.pairs.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"loaded {len(rows)} per-utt rows from {args.pairs}")

    t0 = time.time()
    results = []
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start: start + args.batch_size]
        srcs = [ipa_v.encode(r["ipa"].split()) for r in batch]
        # filter empty after encoding
        keep = [(r, s) for r, s in zip(batch, srcs) if s]
        if not keep:
            continue
        batch_keep = [r for r, _ in keep]
        srcs = [s for _, s in keep]
        max_s = max(len(s) for s in srcs)
        src_t = torch.full((len(srcs), max_s), ipa_v.pad_id,
                           dtype=torch.long, device=device)
        for i, s in enumerate(srcs):
            src_t[i, : len(s)] = torch.tensor(s, dtype=torch.long,
                                              device=device)
        out_ids = model.greedy(src_t, txt_v.bos_id, txt_v.eos_id,
                                max_len=300)
        for r, ids in zip(batch_keep, out_ids):
            hyp = txt_v.decode(ids)
            ref = r["text"]
            w, ce = wer_cer(ref, hyp)
            results.append({
                "idx": r.get("_idx", r.get("idx", -1)),
                "ref": ref, "hyp": hyp, "wer": w, "cer": ce,
            })
        if (start // args.batch_size) % 10 == 0:
            print(f"  {start + len(batch)}/{len(rows)} "
                  f"({(start+len(batch))/(time.time()-t0+1e-6):.1f}/s)",
                  flush=True)

    # Write per-utt
    tsv = args.out_dir / "per_utt.tsv"
    with tsv.open("w", encoding="utf-8") as fh:
        fh.write("idx\twer\tcer\tref\thyp\n")
        for r in results:
            fh.write(f"{r['idx']}\t{r['wer']:.3f}\t{r['cer']:.3f}\t"
                     f"{r['ref'][:200]}\t{r['hyp'][:200]}\n")

    wp = percentiles([r["wer"] for r in results])
    cp = percentiles([r["cer"] for r in results])
    summary = (
        f"P2G per-utt eval ({args.pairs.name})\n"
        f"  ckpt ep={ck.get('epoch')} val_loss="
        f"{ck.get('val_loss'):.3f}\n"
        f"  utts scored: {len(results)}\n\n"
        f"  WER  p10={wp['p10']:.2f} p25={wp['p25']:.2f} "
        f"p50={wp['p50']:.2f} p75={wp['p75']:.2f} "
        f"p90={wp['p90']:.2f} mean={wp['mean']:.2f}\n"
        f"  CER  p10={cp['p10']:.2f} p25={cp['p25']:.2f} "
        f"p50={cp['p50']:.2f} p75={cp['p75']:.2f} "
        f"p90={cp['p90']:.2f} mean={cp['mean']:.2f}\n"
        f"  good (CER<0.3): {cp['under_0.3']}/{cp['n']}  "
        f"({100*cp['under_0.3']/cp['n']:.0f}%)\n"
        f"  ok   (CER<0.5): {cp['under_0.5']}/{cp['n']}  "
        f"({100*cp['under_0.5']/cp['n']:.0f}%)\n"
        f"  bad  (CER>0.8): {cp['over_0.8']}/{cp['n']}  "
        f"({100*cp['over_0.8']/cp['n']:.0f}%)\n"
    )
    (args.out_dir / "summary.txt").write_text(summary, encoding="utf-8")
    print("\n" + summary)
    print(f"per-utt -> {tsv}")


if __name__ == "__main__":
    main()
