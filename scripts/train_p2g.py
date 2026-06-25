"""Train a tiny Transformer encoder-decoder P2G model from scratch.

Input  : IPA token sequence (space-separated ZIPA-style tokens)
Output : Bengali+English character sequence

Architecture: ~10M params
  - d_model=256, nhead=4
  - 3 encoder layers, 3 decoder layers
  - FFN dim 512, dropout 0.1
  - Learned positional embeddings (max 512)
  - Sinusoidal could substitute but learned is fine at this size

Training:
  - CE loss with label smoothing 0.1
  - AdamW lr 3e-4 + linear warmup 1000 steps + cosine decay
  - fp16 autocast on CUDA
  - Greedy decode 8 val samples each epoch (quick eyeball)
  - Save best (lowest val loss) + latest checkpoints

Run:
  uv run python scripts/train_p2g.py --epochs 20 --batch-size 32
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data" / "p2g"


# ----------- vocab --------------------------------------------------

class Vocab:
    def __init__(self, items: list[str], specials: list[str]):
        self.specials = specials
        self.itos = specials + [t for t in items if t not in specials]
        self.stoi = {t: i for i, t in enumerate(self.itos)}
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


def load_vocabs() -> tuple[Vocab, Vocab]:
    v = json.loads((DATA / "vocab.json").read_text(encoding="utf-8"))
    ipa = Vocab(v["ipa_vocab"], v["specials"])
    txt = Vocab(v["text_vocab"], v["specials"])
    return ipa, txt


# ----------- dataset ------------------------------------------------

class P2GDataset(Dataset):
    def __init__(self, path: Path, ipa: Vocab, txt: Vocab,
                 max_src: int = 320, max_tgt: int = 200):
        self.items = []
        for ln in path.read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if not ln:
                continue
            r = json.loads(ln)
            src_toks = r["ipa"].split()
            if not (1 <= len(src_toks) <= max_src):
                continue
            tgt_chars = list(r["text"])
            if not (1 <= len(tgt_chars) <= max_tgt):
                continue
            src = ipa.encode(src_toks)
            tgt = [txt.bos_id] + txt.encode(tgt_chars) + [txt.eos_id]
            self.items.append((src, tgt, r["text"], r["ipa"]))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def make_collate(pad_src: int, pad_tgt: int):
    def collate(batch):
        srcs, tgts, _, _ = zip(*batch)
        max_s = max(len(s) for s in srcs)
        max_t = max(len(t) for t in tgts)
        src = torch.full((len(batch), max_s), pad_src, dtype=torch.long)
        tgt = torch.full((len(batch), max_t), pad_tgt, dtype=torch.long)
        for i, (s, t) in enumerate(zip(srcs, tgts)):
            src[i, : len(s)] = torch.tensor(s, dtype=torch.long)
            tgt[i, : len(t)] = torch.tensor(t, dtype=torch.long)
        return src, tgt
    return collate


# ----------- model --------------------------------------------------

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
        # tie? skip — small vocab
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _add_pos(self, x: torch.Tensor, pos_emb: nn.Embedding):
        T = x.shape[1]
        pos = torch.arange(T, device=x.device)
        return x + pos_emb(pos).unsqueeze(0)

    def forward(self, src: torch.Tensor, tgt_in: torch.Tensor):
        src_pad = src == self.pad_src
        tgt_pad = tgt_in == self.pad_tgt
        T = tgt_in.shape[1]
        causal = nn.Transformer.generate_square_subsequent_mask(
            T, device=src.device)
        s = self.dropout(self._add_pos(
            self.src_emb(src) * math.sqrt(self.d_model), self.src_pos))
        t = self.dropout(self._add_pos(
            self.tgt_emb(tgt_in) * math.sqrt(self.d_model),
            self.tgt_pos))
        h = self.transformer(
            s, t,
            tgt_mask=causal,
            src_key_padding_mask=src_pad,
            tgt_key_padding_mask=tgt_pad,
            memory_key_padding_mask=src_pad,
        )
        return self.out(h)

    @torch.no_grad()
    def greedy(self, src: torch.Tensor, bos: int, eos: int,
               max_len: int = 200) -> list[list[int]]:
        self.eval()
        B = src.shape[0]
        dev = src.device
        src_pad = src == self.pad_src
        s = self.dropout(self._add_pos(
            self.src_emb(src) * math.sqrt(self.d_model), self.src_pos))
        mem = self.transformer.encoder(s,
                                         src_key_padding_mask=src_pad)
        ys = torch.full((B, 1), bos, dtype=torch.long, device=dev)
        done = torch.zeros(B, dtype=torch.bool, device=dev)
        for _ in range(max_len):
            T = ys.shape[1]
            causal = nn.Transformer.generate_square_subsequent_mask(
                T, device=dev)
            t = self.dropout(self._add_pos(
                self.tgt_emb(ys) * math.sqrt(self.d_model),
                self.tgt_pos))
            h = self.transformer.decoder(
                t, mem, tgt_mask=causal,
                memory_key_padding_mask=src_pad)
            logits = self.out(h[:, -1, :])
            nxt = logits.argmax(dim=-1)
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


# ----------- training -----------------------------------------------

def lr_schedule(step: int, warmup: int, total: int,
                peak: float) -> float:
    if step < warmup:
        return peak * step / max(1, warmup)
    prog = (step - warmup) / max(1, total - warmup)
    return peak * 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--label-smooth", type=float, default=0.1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, default=DATA / "ckpt")
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--max-train", type=int, default=None,
                    help="limit training set size (smoke testing)")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    ipa_vocab, txt_vocab = load_vocabs()
    print(f"vocab — IPA: {len(ipa_vocab)}  text: {len(txt_vocab)}")

    ds_tr = P2GDataset(DATA / "train.jsonl", ipa_vocab, txt_vocab)
    ds_va = P2GDataset(DATA / "val.jsonl", ipa_vocab, txt_vocab)
    if args.max_train:
        ds_tr.items = ds_tr.items[: args.max_train]
    print(f"train: {len(ds_tr)}  val: {len(ds_va)}")

    coll = make_collate(ipa_vocab.pad_id, txt_vocab.pad_id)
    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True,
                        collate_fn=coll, num_workers=args.num_workers,
                        drop_last=True, pin_memory=True)
    dl_va = DataLoader(ds_va, batch_size=args.batch_size,
                        shuffle=False, collate_fn=coll,
                        num_workers=args.num_workers, pin_memory=True)

    model = P2GModel(
        src_vocab=len(ipa_vocab), tgt_vocab=len(txt_vocab),
        pad_src=ipa_vocab.pad_id, pad_tgt=txt_vocab.pad_id,
    ).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_par/1e6:.2f}M")

    total_steps = args.epochs * len(dl_tr)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                              betas=(0.9, 0.98), eps=1e-9,
                              weight_decay=0.01)
    scaler = torch.amp.GradScaler(args.device)
    step = 0
    best_va = float("inf")
    t0 = time.time()

    eyeball_indices = random.sample(range(len(ds_va)),
                                      min(8, len(ds_va)))

    for ep in range(args.epochs):
        model.train()
        losses = []
        for batch in dl_tr:
            src, tgt = (b.to(device, non_blocking=True) for b in batch)
            tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]
            lr = lr_schedule(step, args.warmup, total_steps, args.lr)
            for pg in opt.param_groups:
                pg["lr"] = lr
            opt.zero_grad()
            with torch.amp.autocast(args.device, dtype=torch.float16):
                logits = model(src, tgt_in)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    tgt_out.reshape(-1),
                    ignore_index=txt_vocab.pad_id,
                    label_smoothing=args.label_smooth,
                )
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            losses.append(loss.item())
            step += 1
            if step % 200 == 0:
                el = time.time() - t0
                avg = sum(losses[-200:]) / min(200, len(losses))
                print(f"  ep{ep+1} step{step:5d}  "
                      f"loss={loss.item():.3f}  avg200={avg:.3f}  "
                      f"lr={lr:.2e}  elapsed={el:.0f}s", flush=True)

        # validation
        model.eval()
        va_losses = []
        with torch.no_grad():
            for batch in dl_va:
                src, tgt = (b.to(device, non_blocking=True)
                            for b in batch)
                tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]
                with torch.amp.autocast(args.device,
                                          dtype=torch.float16):
                    logits = model(src, tgt_in)
                    vloss = F.cross_entropy(
                        logits.reshape(-1, logits.shape[-1]),
                        tgt_out.reshape(-1),
                        ignore_index=txt_vocab.pad_id,
                    )
                va_losses.append(vloss.item())
        va_avg = sum(va_losses) / max(1, len(va_losses))
        tr_avg = sum(losses) / max(1, len(losses))
        print(f"=== ep {ep+1}/{args.epochs}  "
              f"train_loss={tr_avg:.3f}  val_loss={va_avg:.3f}  "
              f"elapsed={time.time()-t0:.0f}s ===", flush=True)

        # eyeball: greedy decode 8 val samples
        print("  eyeball (greedy on 8 val):")
        eye_src = []
        eye_ref = []
        for i in eyeball_indices:
            src_ids, _tgt_ids, text, _ipa = ds_va.items[i]
            eye_src.append(src_ids)
            eye_ref.append(text)
        max_s = max(len(s) for s in eye_src)
        src_pad = torch.full((len(eye_src), max_s),
                              ipa_vocab.pad_id, dtype=torch.long,
                              device=device)
        for i, s in enumerate(eye_src):
            src_pad[i, : len(s)] = torch.tensor(s, dtype=torch.long,
                                                 device=device)
        hyps = model.greedy(src_pad, txt_vocab.bos_id,
                             txt_vocab.eos_id, max_len=200)
        for ref, hyp_ids in zip(eye_ref, hyps):
            hyp = txt_vocab.decode(hyp_ids)
            print(f"    REF: {ref[:90]}")
            print(f"    HYP: {hyp[:90]}")
            print()

        # save
        ckpt = {
            "model": model.state_dict(),
            "ipa_vocab": ipa_vocab.itos,
            "txt_vocab": txt_vocab.itos,
            "specials": ipa_vocab.specials,
            "epoch": ep + 1,
            "val_loss": va_avg,
            "train_loss": tr_avg,
            "config": {
                "d_model": model.d_model,
                "src_vocab": len(ipa_vocab),
                "tgt_vocab": len(txt_vocab),
            },
        }
        torch.save(ckpt, args.out_dir / "latest.pt")
        if va_avg < best_va:
            best_va = va_avg
            torch.save(ckpt, args.out_dir / "best.pt")
            print(f"  ** new best val_loss={va_avg:.3f} ** "
                  f"saved -> {args.out_dir/'best.pt'}")

    print(f"\ndone. best val_loss={best_va:.3f}")


if __name__ == "__main__":
    main()
