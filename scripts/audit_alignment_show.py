"""Show Whisper hyp + GT + audio stats side-by-side for 5 random utts.
Diagnoses WHICH of these is true:

  A. Alignment is broken — Whisper transcribes the audio fine, but
     hyp doesn't match GT (different content)
  B. Whisper fails — Whisper returns empty/garbage on dialect, so
     we can't trust the overlap=0 verdict
  C. Audio is silent/corrupted — wav files don't contain audible
     speech at all
"""
from __future__ import annotations
import json, random, sys
from pathlib import Path
import torch, torchaudio, numpy as np

REPO = Path(__file__).resolve().parents[1]
MANIFEST = (REPO / "data" / "w2v_train_longform"
            / "15_Sunamganj_CA-MMI_URBAN_Male_15_0602_1635"
            ".longform.jsonl")

rows = [json.loads(l) for l in
        MANIFEST.read_text(encoding="utf-8").splitlines() if l.strip()]
random.seed(7)
picks = random.sample([r for r in rows if Path(r["wav"]).exists()], 5)

print("loading whisper-small ...")
from transformers import WhisperForConditionalGeneration, WhisperProcessor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
proc = WhisperProcessor.from_pretrained("openai/whisper-small")
mdl = WhisperForConditionalGeneration.from_pretrained(
    "openai/whisper-small",
    torch_dtype=torch.float16 if device.type == "cuda" else torch.float32
).to(device).eval()
forced = proc.get_decoder_prompt_ids(language="bn", task="transcribe")
print("loaded.\n")

for r in picks:
    wav, sr = torchaudio.load(r["wav"])
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    arr = wav.squeeze(0).numpy()
    rms = float(np.sqrt(np.mean(arr ** 2)))
    peak = float(np.max(np.abs(arr)))
    dur = len(arr) / 16000.0

    inp = proc(arr, sampling_rate=16000,
                return_tensors="pt").input_features.to(
        device, dtype=torch.float16 if device.type == "cuda" else torch.float32)
    with torch.no_grad():
        ids = mdl.generate(inp, forced_decoder_ids=forced,
                            max_new_tokens=200, num_beams=1)
    hyp = proc.batch_decode(ids, skip_special_tokens=True)[0]

    print(f"--- idx={r['idx']}  ({r['start']:.1f}-{r['end']:.1f}s, "
          f"manifest dur={r['end']-r['start']:.1f}s, "
          f"actual dur={dur:.1f}s) ---")
    print(f"  audio    : rms={rms:.4f}  peak={peak:.3f}  "
          f"{'SILENT' if rms < 0.001 else 'has signal'}")
    print(f"  GT       : {r['gt_text'][:140]}")
    print(f"  whisper  : {hyp[:140]}")
    print()
