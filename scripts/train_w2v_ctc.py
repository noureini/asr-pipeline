"""
Day 1 — train a tiny bilingual CTC head on top of the frozen
omniASR_W2V_300M encoder.

First-principles, no usine à gaz:
  * Encoder: frozen (1024-dim features, 50 Hz output rate)
  * Head:    Linear(1024 -> vocab_size+1) — bilingual char vocab
             built from train GT (Bengali + Latin code-switch chars)
  * Loss:    CTC
  * Data:    long-form-aligned utterances from align_mms_longform.py

Runs on 8 GB GPU. ~30-60 min for several epochs on ~1k utterances.
Saves a checkpoint that decode_w2v_ctc.py loads.

Usage (defaults point at <repo>/data/, which survives WSL reboots):
  uv run python scripts/train_w2v_ctc.py
or override:
  uv run python scripts/train_w2v_ctc.py \\
    --train-manifest <repo>/data/w2v_train_longform/<stem>.longform.jsonl \\
    --out-dir <repo>/data/w2v_ckpt
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


# ---------- vocab ----------------------------------------------------

def build_vocab(texts: list[str]) -> tuple[dict, list]:
    """Bilingual char vocab from text. Index 0 reserved for CTC blank."""
    chars = set()
    for t in texts:
        chars.update(t)
    # Strip control chars / very low Unicode (just in case)
    chars = {c for c in chars if ord(c) >= 0x20 or c == "\n"}
    chars = sorted(chars)
    # 0 = blank ; 1..N = chars
    char2id = {"<blank>": 0}
    for c in chars:
        char2id[c] = len(char2id)
    id2char = ["<blank>"] + chars
    return char2id, id2char


def encode_text(text: str, char2id: dict) -> list[int]:
    return [char2id[c] for c in text if c in char2id]


# ---------- dataset --------------------------------------------------

class UttDataset(torch.utils.data.Dataset):
    def __init__(self, manifest_path: Path, char2id: dict,
                 sr_target: int = 16000,
                 min_dur: float = 0.4, max_dur: float = 18.0):
        self.items = []
        with manifest_path.open(encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                r = json.loads(ln)
                dur = float(r.get("end") or 0.0) - float(r.get("start") or 0.0)
                if dur < min_dur or dur > max_dur:
                    continue
                if not Path(r["wav"]).exists():
                    continue
                tgt = encode_text(r["gt_text"], char2id)
                if len(tgt) < 2:
                    continue
                self.items.append({"wav": r["wav"],
                                    "tgt": tgt,
                                    "gt_text": r["gt_text"]})
        self.sr = sr_target

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        r = self.items[i]
        wav, sr = torchaudio.load(r["wav"])
        if sr != self.sr:
            wav = torchaudio.functional.resample(wav, sr, self.sr)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        return wav.squeeze(0), torch.tensor(r["tgt"], dtype=torch.long)


def collate(batch):
    wavs, tgts = zip(*batch)
    wav_lens = torch.tensor([w.shape[0] for w in wavs], dtype=torch.long)
    max_wav = max(w.shape[0] for w in wavs)
    wav_pad = torch.zeros(len(wavs), max_wav, dtype=wavs[0].dtype)
    for i, w in enumerate(wavs):
        wav_pad[i, :w.shape[0]] = w
    tgt_lens = torch.tensor([t.shape[0] for t in tgts], dtype=torch.long)
    tgt_concat = torch.cat(tgts)  # CTC accepts concatenated targets
    return wav_pad, wav_lens, tgt_concat, tgt_lens


# ---------- training -------------------------------------------------

def main():
    _REPO = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--train-manifest", type=Path,
        default=_REPO / "data" / "w2v_train_longform"
        / "07_Shatkhira_CA-MMI_URBAN_Male_07_0202_1716.longform.jsonl")
    ap.add_argument("--out-dir", type=Path,
                    default=_REPO / "data" / "w2v_ckpt")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit-utts", type=int, default=None)
    ap.add_argument("--unfreeze-top", type=int, default=2,
                    help="number of top transformer layers to fine-tune; "
                         "0 = pure frozen encoder (CTC collapse risk)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Build vocab from train manifest
    texts = []
    with args.train_manifest.open(encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                texts.append(json.loads(ln)["gt_text"])
    char2id, id2char = build_vocab(texts)
    V = len(char2id)
    print(f"vocab size: {V} (includes <blank>)")
    print(f"first 40 chars: {[id2char[i] for i in range(min(40, V))]}")

    # Dataset
    ds = UttDataset(args.train_manifest, char2id)
    if args.limit_utts:
        ds.items = ds.items[: args.limit_utts]
    print(f"train utterances after filter: {len(ds)}")
    if len(ds) == 0:
        raise SystemExit("no utterances after filtering")
    dl = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate, num_workers=2, drop_last=False)

    # Encoder: freeze everything EXCEPT top `unfreeze_top_n` transformer
    # layers. Pure frozen-encoder + linear head collapses to the CTC
    # "all-blank" trivial minimum on this data — verified via diag_head.
    # Matching the wav2vec2 (2020) recipe: fine-tune top layers w/ a
    # smaller LR than the head.
    from fairseq2.models.hub import load_model
    from fairseq2.nn.batch_layout import BatchLayout
    from fairseq2.models.transformer.attention_bias import (
        AttentionBiasCache)
    device = torch.device(args.device)
    print("loading omniASR_W2V_300M ...")
    encoder = load_model("omniASR_W2V_300M", device=device,
                          dtype=torch.float32)
    encoder.eval()  # keeps dropout off; we set requires_grad below
    for p in encoder.parameters():
        p.requires_grad = False
    # Unfreeze top N transformer layers
    layers = encoder.encoder.layers  # ModuleList of 24
    unfreeze_top_n = args.unfreeze_top
    trainable_enc_params = []
    if unfreeze_top_n > 0:
        for layer in list(layers)[-unfreeze_top_n:]:
            for p in layer.parameters():
                p.requires_grad = True
                trainable_enc_params.append(p)
    n_enc = sum(p.numel() for p in encoder.parameters())
    n_enc_trn = sum(p.numel() for p in trainable_enc_params)
    print(f"  encoder params total: {n_enc/1e6:.1f}M  "
          f"trainable (top {unfreeze_top_n} layers): "
          f"{n_enc_trn/1e6:.1f}M")

    # CTC head — bias-init blank logit low so we don't start in the
    # trivial all-blank basin.
    head = nn.Linear(1024, V).to(device)
    with torch.no_grad():
        head.bias.zero_()
        head.bias[0] = -3.0  # blank starts disadvantaged
    n_head = sum(p.numel() for p in head.parameters())
    print(f"  CTC head params: {n_head/1000:.1f}K  (Linear(1024, {V}))  "
          f"bias[blank]=-3.0")

    # Two param groups: head at full LR, encoder layers at LR/30.
    param_groups = [{"params": head.parameters(), "lr": args.lr}]
    if trainable_enc_params:
        param_groups.append(
            {"params": trainable_enc_params, "lr": args.lr / 30.0})
    opt = torch.optim.AdamW(param_groups)
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)

    # Encoder feature extraction — bypasses both the SSL contrastive
    # head AND the masker. We can't use `encoder.extract_features()`
    # because it routes through `run_frontend()` which unconditionally
    # invokes the Wav2Vec2Masker (and asserts mask is non-None) — that
    # crashes on short utts (`min_num_spans` > generatable spans).
    # Instead: call frontend.extract_features (conv features) ->
    # frontend.process_features(masker=None) (proj + pos-enc + norm,
    # no masking) -> encoder (transformer). All under no_grad.
    frontend = encoder.encoder_frontend
    transformer = encoder.encoder

    def get_features(wav_batch, wav_lens):
        # Frontend (conv) + bottom (24-N) transformer layers: no_grad.
        # Top N layers: grad-on so unfrozen params get updated.
        layout = BatchLayout(wav_batch.shape, seq_lens=wav_lens.tolist())
        with torch.no_grad():
            seqs, slay, _raw = frontend.extract_features(
                wav_batch.to(device), layout)
            seqs, _mask = frontend.process_features(
                seqs, slay, masker=None)
        if unfreeze_top_n <= 0:
            with torch.no_grad():
                feats = transformer(seqs, slay)
        else:
            # Run the frozen prefix under no_grad, then the trainable
            # suffix WITH grad. fairseq2's StandardTransformerEncoder
            # applies layers in sequence with a final layer_norm — we
            # reproduce that here so the top-N layers receive gradient.
            n_total = len(transformer.layers)
            split = n_total - unfreeze_top_n
            attn_cache = AttentionBiasCache()
            with torch.no_grad():
                for layer in list(transformer.layers)[:split]:
                    seqs = layer(seqs, slay, attn_cache)
            for layer in list(transformer.layers)[split:]:
                seqs = layer(seqs, slay, attn_cache)
            if transformer.layer_norm is not None:
                seqs = transformer.layer_norm(seqs)
            feats = seqs
        feat_lens = torch.tensor(
            slay.seq_lens, dtype=torch.long, device=device)
        return feats, feat_lens

    # Training loop
    t0 = time.time()
    step = 0
    for ep in range(args.epochs):
        head.train()
        losses = []
        for batch in dl:
            wavs, wav_lens, tgts, tgt_lens = batch
            try:
                wavs = wavs.to(device)
                tgts = tgts.to(device)
                feats, feat_lens = get_features(wavs, wav_lens)
            except torch.cuda.OutOfMemoryError as e:
                print(f"  OOM at batch (max_wav={wavs.shape[1]}); "
                      f"skipping. {str(e)[:120]}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            except Exception as e:
                print(f"  encoder forward failed: {type(e).__name__}: "
                      f"{str(e)[:200]}")
                continue
            logits = head(feats)  # (B, T, V)
            log_probs = F.log_softmax(logits, dim=-1)
            # CTC expects (T, B, V)
            log_probs = log_probs.transpose(0, 1)
            loss = ctc(log_probs, tgts, feat_lens.cpu(), tgt_lens)
            if not torch.isfinite(loss):
                continue
            opt.zero_grad()
            loss.backward()
            trainable = list(head.parameters()) + trainable_enc_params
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            losses.append(loss.item())
            step += 1
            if step % 20 == 0:
                el = time.time() - t0
                print(f"  ep{ep+1} step{step:4d}  loss={loss.item():.3f}  "
                      f"avg={sum(losses[-20:])/min(20,len(losses)):.3f}  "
                      f"elapsed={el:.0f}s")
        avg = sum(losses) / max(1, len(losses))
        print(f"=== epoch {ep+1}/{args.epochs}  avg_loss={avg:.3f}  "
              f"elapsed={time.time()-t0:.0f}s ===")
        # save checkpoint
        # Save head + unfrozen top-N transformer layer weights (so
        # decode_w2v_ctc.py can reconstruct the exact forward pass).
        enc_top_state = {}
        if unfreeze_top_n > 0:
            n_total = len(encoder.encoder.layers)
            for i in range(n_total - unfreeze_top_n, n_total):
                layer = encoder.encoder.layers[i]
                enc_top_state[i] = {
                    k: v.detach().cpu()
                    for k, v in layer.state_dict().items()}
        ckpt = {
            "head_state": head.state_dict(),
            "enc_top_state": enc_top_state,
            "unfreeze_top_n": unfreeze_top_n,
            "char2id": char2id,
            "id2char": id2char,
            "epoch": ep + 1,
            "avg_loss": avg,
        }
        torch.save(ckpt, args.out_dir / f"head_ep{ep+1}.pt")
        torch.save(ckpt, args.out_dir / "head_latest.pt")
        print(f"  saved -> {args.out_dir / f'head_ep{ep+1}.pt'}")

    print(f"\ndone. final ckpt: {args.out_dir / 'head_latest.pt'}")


if __name__ == "__main__":
    main()
