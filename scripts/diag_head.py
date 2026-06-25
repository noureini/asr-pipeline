"""Quick diagnostic: load latest checkpoint, decode 8 train utterances
greedily, and report what the head predicts vs the GT.

Distinguishes a real code bug from a frozen-encoder ceiling:
  - If head outputs are nearly all <blank>: head is stuck at the
    CTC trivial minimum. Solutions: higher LR, MLP head, unfreeze
    top layers, or CPT.
  - If head outputs garbage characters with no relation to GT:
    likely a vocab / label-encoding bug.
  - If outputs partially track GT: training is working but slow;
    keep going / bump LR.
"""
from __future__ import annotations
import json, sys, random
from pathlib import Path
import torch
import torch.nn.functional as F
import torchaudio
from fairseq2.models.hub import load_model
from fairseq2.nn.batch_layout import BatchLayout

REPO = Path(__file__).resolve().parents[1]
CKPT = REPO / "data" / "w2v_ckpt" / "head_latest.pt"
MANIFEST = (REPO / "data" / "w2v_train_longform"
            / "07_Shatkhira_CA-MMI_URBAN_Male_07_0202_1716.longform.jsonl")

device = torch.device("cuda")
ckpt = torch.load(CKPT, map_location=device, weights_only=False)
id2char = ckpt["id2char"]
V = len(id2char)
print(f"vocab size: {V}")

print("loading encoder (frozen) ...")
enc = load_model("omniASR_W2V_300M", device=device,
                 dtype=torch.float32).eval()
for p in enc.parameters():
    p.requires_grad = False
frontend = enc.encoder_frontend
transformer = enc.encoder

import torch.nn as nn
head = nn.Linear(1024, V).to(device)
head.load_state_dict(ckpt["head_state"])
head.eval()

# Sample 8 utts spread across the manifest
rows = [json.loads(l) for l in MANIFEST.read_text(encoding="utf-8").splitlines() if l.strip()]
random.seed(0)
picks = random.sample(rows, 8)

def greedy(log_probs_TBV, blank=0):
    """Per-batch greedy CTC: argmax per frame, collapse repeats, drop blank."""
    ids = log_probs_TBV.argmax(dim=-1)  # (T, B)
    outs = []
    for b in range(ids.shape[1]):
        seq = ids[:, b].tolist()
        decoded, prev = [], -1
        for i in seq:
            if i != prev and i != blank:
                decoded.append(i)
            prev = i
        outs.append("".join(id2char[i] for i in decoded))
    return outs

for r in picks:
    wav, sr = torchaudio.load(r["wav"])
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    wav = wav.to(device)  # (1, S)
    layout = BatchLayout(wav.shape, seq_lens=[wav.shape[1]])
    with torch.no_grad():
        seqs, slay, _ = frontend.extract_features(wav, layout)
        seqs, _ = frontend.process_features(seqs, slay, masker=None)
        feats = transformer(seqs, slay)
        logits = head(feats)
        log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)  # (T, B, V)
    # frame-level argmax distribution
    ids = log_probs.argmax(dim=-1).squeeze(1)  # (T,)
    counts: dict[int, int] = {}
    for i in ids.tolist():
        counts[i] = counts.get(i, 0) + 1
    top = sorted(counts.items(), key=lambda x: -x[1])[:5]
    top_str = ", ".join(f"{id2char[i]!r}:{c}" for i, c in top)
    blank_frac = counts.get(0, 0) / ids.numel()
    decoded = greedy(log_probs)[0]
    print(f"\nGT:  {r['gt_text'][:90]}")
    print(f"HYP: {decoded[:90] or '(empty)'}")
    print(f"  frames={ids.numel()}  blank_frac={blank_frac:.2f}  "
          f"top5={top_str}")
