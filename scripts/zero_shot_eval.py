"""Zero-shot inference shootout: test 4 Bengali-aware ASR candidates
on a 100-utt subset of the held-out Sunamganj manifest, NO fine-tuning.

Goal: pick the right base model to fine-tune by measuring which one
already understands our coastal Bangladeshi dialect best.

Candidates:
  1. ai4bharat/indic-conformer-600m-multilingual   (NeMo)
  2. tugstugi/bengaliai-regional-asr_whisper-medium  (HF Whisper)
  3. arijitx/wav2vec2-xls-r-300m-bengali — greedy CTC
  4. arijitx/wav2vec2-xls-r-300m-bengali — 5-gram LM beam search

Reference handling:
  Our manifest's gt_text has 38 Bengali admin terms transliterated to
  Latin (NGO, NID, MP, project, ...). All ASR outputs are Bengali
  script. We REVERSE the transliteration on the reference for a fair
  Bengali↔Bengali comparison. Reports both WER and CER.

Output:
  data/zero_shot_eval/results.tsv  — full per-utterance hypotheses
  data/zero_shot_eval/summary.txt  — final WER/CER table
"""
from __future__ import annotations

import json
import random
import time
import traceback
from pathlib import Path

import torch
import torchaudio

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
TEST_MANIFEST = (DATA / "w2v_train_longform"
                 / "15_Sunamganj_CA-MMI_URBAN_Male_15_0602_1635"
                 ".longform.jsonl")
TRANSLIT_JSON = DATA / "translit.json"
OUT_DIR = DATA / "zero_shot_eval"
N_UTTS = 100
SR = 16000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------- ref/hyp normalization ------------------------------------

def load_reverse_translit() -> dict[str, str]:
    """Latin -> Bengali, reversing translit.json so we can de-Latinize
    the reference back to pure Bengali script for fair scoring."""
    fwd = json.loads(TRANSLIT_JSON.read_text(encoding="utf-8"))
    return {lat: bn for bn, lat in fwd.items()}


def to_bengali_ref(text: str, rev: dict[str, str]) -> str:
    """Reverse-transliterate Latin words in the GT back to Bengali
    script. Tokens not in the table are left as-is (rare proper
    nouns)."""
    return " ".join(rev.get(w, w) for w in text.split())


_PUNCT = ",.?!|:;-—()[]{}\"'`/\\"

def norm(s: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace.
    Bengali script unaffected by lowercase."""
    s = s.lower()
    for p in _PUNCT:
        s = s.replace(p, " ")
    return " ".join(s.split())


# ---------- WER/CER (jiwer-style, no external dep) -------------------

def edit_distance(a: list, b: list) -> int:
    """Standard Levenshtein."""
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(
                cur[j - 1] + 1,         # insert
                prev[j] + 1,            # delete
                prev[j - 1] + cost)     # substitute
        prev = cur
    return prev[m]


def wer(ref: str, hyp: str) -> tuple[float, int, int]:
    r = ref.split()
    h = hyp.split()
    d = edit_distance(r, h)
    return (d / max(1, len(r))), d, len(r)


def cer(ref: str, hyp: str) -> tuple[float, int, int]:
    r = list(ref.replace(" ", ""))
    h = list(hyp.replace(" ", ""))
    d = edit_distance(r, h)
    return (d / max(1, len(r))), d, len(r)


# ---------- pick 100 utts evenly across the timeline -----------------

def pick_utts(n: int) -> list[dict]:
    rows = [json.loads(l) for l in
            TEST_MANIFEST.read_text(encoding="utf-8").splitlines()
            if l.strip()]
    rows = [r for r in rows
            if Path(r["wav"]).exists()
            and 1.0 <= (r["end"] - r["start"]) <= 15.0
            and len(r["gt_text"]) >= 8]
    rows.sort(key=lambda r: r["start"])
    # uniform stride across the timeline
    if len(rows) <= n:
        return rows
    idx = [int(i * len(rows) / n) for i in range(n)]
    return [rows[i] for i in idx]


# ---------- audio loader (cached) ------------------------------------

def load_audio(path: str) -> torch.Tensor:
    wav, sr = torchaudio.load(path)
    if sr != SR:
        wav = torchaudio.functional.resample(wav, sr, SR)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    return wav.squeeze(0)  # (S,)


# ---------- candidate runners ----------------------------------------

def run_indic_conformer(utts: list[dict]) -> list[str]:
    """ai4bharat/indic-conformer-600m-multilingual (NeMo)."""
    try:
        import nemo.collections.asr as nemo_asr  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            f"NeMo not installed (`uv pip install "
            f"'nemo_toolkit[asr]'`). Skipping. ({e!s:.80})")
    from nemo.collections.asr.models import ASRModel
    m = ASRModel.from_pretrained(
        "ai4bharat/indic-conformer-600m-multilingual").to(device).eval()
    hyps = []
    for r in utts:
        try:
            res = m.transcribe([r["wav"]], language_id="bn",
                                verbose=False)
            txt = res[0] if isinstance(res, list) else str(res)
            if isinstance(txt, (list, tuple)):
                txt = txt[0] if txt else ""
            hyps.append(str(txt))
        except Exception as e:
            hyps.append(f"[ERR: {type(e).__name__}]")
    del m
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return hyps


def run_whisper_bn_regional(utts: list[dict]) -> list[str]:
    """bengaliAI/tugstugi_bengaliai-regional-asr_whisper-medium.
    Trained on ben10: 10 Bangla regional dialects (incl. Bangladeshi).
    Reported ~34% WER on the Bengali-Loop regional eval split."""
    from transformers import (WhisperForConditionalGeneration,
                              WhisperProcessor)
    name = ("bengaliAI/tugstugi_bengaliai-regional-asr_whisper-medium")
    proc = WhisperProcessor.from_pretrained(name)
    m = WhisperForConditionalGeneration.from_pretrained(
        name, torch_dtype=torch.float16).to(device).eval()
    forced_ids = proc.get_decoder_prompt_ids(
        language="bn", task="transcribe")
    hyps = []
    for r in utts:
        try:
            wav = load_audio(r["wav"]).numpy()
            inp = proc(wav, sampling_rate=SR,
                        return_tensors="pt").input_features.to(
                device, dtype=torch.float16)
            with torch.no_grad():
                ids = m.generate(inp, forced_decoder_ids=forced_ids,
                                  max_new_tokens=200,
                                  num_beams=1)
            txt = proc.batch_decode(ids,
                                      skip_special_tokens=True)[0]
            hyps.append(txt)
        except Exception as e:
            hyps.append(f"[ERR: {type(e).__name__}]")
    del m
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return hyps


def run_arijitx_greedy(utts: list[dict]) -> list[str]:
    """arijitx/wav2vec2-xls-r-300m-bengali — argmax CTC."""
    from transformers import (Wav2Vec2ForCTC, Wav2Vec2Processor)
    name = "arijitx/wav2vec2-xls-r-300m-bengali"
    proc = Wav2Vec2Processor.from_pretrained(name)
    m = Wav2Vec2ForCTC.from_pretrained(name).to(device).eval()
    hyps = []
    for r in utts:
        try:
            wav = load_audio(r["wav"])
            inp = proc(wav.numpy(), sampling_rate=SR,
                        return_tensors="pt").input_values.to(device)
            with torch.no_grad():
                logits = m(inp).logits
            ids = logits.argmax(dim=-1)
            txt = proc.batch_decode(ids)[0]
            hyps.append(txt)
        except Exception as e:
            hyps.append(f"[ERR: {type(e).__name__}]")
    del m
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return hyps


def run_arijitx_lm(utts: list[dict]) -> list[str]:
    """arijitx/wav2vec2-xls-r-300m-bengali + bundled 5-gram LM.

    The HF model card ships with a `language_model/` subdir that
    Wav2Vec2ProcessorWithLM auto-loads (pyctcdecode + KenLM)."""
    try:
        from pyctcdecode import build_ctcdecoder  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            f"pyctcdecode not installed (`uv pip install pyctcdecode "
            f"kenlm`). Skipping. ({e!s:.80})")
    from transformers import (Wav2Vec2ForCTC,
                              Wav2Vec2ProcessorWithLM)
    name = "arijitx/wav2vec2-xls-r-300m-bengali"
    proc = Wav2Vec2ProcessorWithLM.from_pretrained(name)
    m = Wav2Vec2ForCTC.from_pretrained(name).to(device).eval()
    hyps = []
    for r in utts:
        try:
            wav = load_audio(r["wav"])
            inp = proc(wav.numpy(), sampling_rate=SR,
                        return_tensors="pt").input_values.to(device)
            with torch.no_grad():
                logits = m(inp).logits.cpu().numpy()
            # batch_decode_with_lm
            txt = proc.batch_decode(logits).text[0]
            hyps.append(txt)
        except Exception as e:
            hyps.append(f"[ERR: {type(e).__name__}]")
    del m
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return hyps


def run_whisper_large_v3(utts: list[dict]) -> list[str]:
    """openai/whisper-large-v3 — multilingual baseline, supports bn."""
    from transformers import (WhisperForConditionalGeneration,
                              WhisperProcessor)
    name = "openai/whisper-large-v3"
    proc = WhisperProcessor.from_pretrained(name)
    m = WhisperForConditionalGeneration.from_pretrained(
        name, torch_dtype=torch.float16).to(device).eval()
    forced_ids = proc.get_decoder_prompt_ids(
        language="bn", task="transcribe")
    hyps = []
    for r in utts:
        try:
            wav = load_audio(r["wav"]).numpy()
            inp = proc(wav, sampling_rate=SR,
                        return_tensors="pt").input_features.to(
                device, dtype=torch.float16)
            with torch.no_grad():
                ids = m.generate(inp, forced_decoder_ids=forced_ids,
                                  max_new_tokens=200, num_beams=1)
            txt = proc.batch_decode(ids,
                                      skip_special_tokens=True)[0]
            hyps.append(txt)
        except Exception as e:
            hyps.append(f"[ERR: {type(e).__name__}]")
    del m
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return hyps


def run_whisper_small_bangla_regional(utts: list[dict]) -> list[str]:
    """ashrafulparan/whisper-small-bangla — tagged Bengali regional
    dialects, small footprint (~244M, fast inference)."""
    from transformers import (WhisperForConditionalGeneration,
                              WhisperProcessor)
    name = "ashrafulparan/whisper-small-bangla"
    proc = WhisperProcessor.from_pretrained(name)
    m = WhisperForConditionalGeneration.from_pretrained(
        name, torch_dtype=torch.float16).to(device).eval()
    forced_ids = proc.get_decoder_prompt_ids(
        language="bn", task="transcribe")
    hyps = []
    for r in utts:
        try:
            wav = load_audio(r["wav"]).numpy()
            inp = proc(wav, sampling_rate=SR,
                        return_tensors="pt").input_features.to(
                device, dtype=torch.float16)
            with torch.no_grad():
                ids = m.generate(inp, forced_decoder_ids=forced_ids,
                                  max_new_tokens=200, num_beams=1)
            txt = proc.batch_decode(ids,
                                      skip_special_tokens=True)[0]
            hyps.append(txt)
        except Exception as e:
            hyps.append(f"[ERR: {type(e).__name__}]")
    del m
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return hyps


CANDIDATES = [
    ("whisper_bn_regional",    run_whisper_bn_regional),
    ("whisper_small_bangla",   run_whisper_small_bangla_regional),
    ("whisper_large_v3",       run_whisper_large_v3),
    # arijitx already failed at 93% CER on read-speech-only training;
    # not retesting unless inspection of results.tsv proves a bug.
]


# ---------- main -----------------------------------------------------

def main() -> None:
    if not TEST_MANIFEST.exists():
        raise SystemExit(
            f"missing test manifest: {TEST_MANIFEST}\n"
            f"run scripts/align_mms_longform.py for stem 15 first")

    utts = pick_utts(N_UTTS)
    print(f"selected {len(utts)} test utterances "
          f"({utts[0]['start']:.0f}s .. {utts[-1]['start']:.0f}s)")

    rev = load_reverse_translit()
    refs = [norm(to_bengali_ref(r["gt_text"], rev)) for r in utts]

    results: dict[str, dict] = {}
    tsv_lines = [
        "model\tidx\tstart_s\twer\tcer\tref\thyp"]

    for name, fn in CANDIDATES:
        print(f"\n=== {name} ===")
        t0 = time.time()
        try:
            hyps = fn(utts)
        except Exception as e:
            print(f"  SKIPPED: {e}")
            traceback.print_exc()
            results[name] = {"skipped": True, "reason": str(e)}
            continue
        wer_num = wer_den = 0
        cer_num = cer_den = 0
        err_count = 0
        for r, ref, hyp in zip(utts, refs, hyps):
            if hyp.startswith("[ERR"):
                err_count += 1
                continue
            h = norm(hyp)
            _, wn, wd = wer(ref, h)
            _, cn, cd = cer(ref, h)
            wer_num += wn
            wer_den += wd
            cer_num += cn
            cer_den += cd
            tsv_lines.append(
                f"{name}\t{r['idx']}\t{r['start']:.1f}\t"
                f"{wn/max(1,wd):.3f}\t{cn/max(1,cd):.3f}\t"
                f"{ref[:140]}\t{h[:140]}")
        W = wer_num / max(1, wer_den)
        C = cer_num / max(1, cer_den)
        el = time.time() - t0
        print(f"  WER = {W*100:.2f}%   CER = {C*100:.2f}%   "
              f"errs = {err_count}/{len(utts)}   elapsed = {el:.0f}s")
        results[name] = {
            "wer": W, "cer": C, "err": err_count,
            "n": len(utts), "sec": el}

    # Persist
    (OUT_DIR / "results.tsv").write_text(
        "\n".join(tsv_lines), encoding="utf-8")

    # Summary
    summary = ["model                    WER       CER     err   sec"]
    for name, _ in CANDIDATES:
        r = results.get(name, {})
        if r.get("skipped"):
            summary.append(f"{name:24s} SKIPPED  ({r['reason'][:50]})")
        else:
            summary.append(
                f"{name:24s} {r['wer']*100:6.2f}%   "
                f"{r['cer']*100:5.2f}%   {r['err']:3d}   "
                f"{r['sec']:4.0f}s")
    text = "\n".join(summary)
    (OUT_DIR / "summary.txt").write_text(text, encoding="utf-8")
    print("\n" + "=" * 60)
    print(text)
    print(f"\nfull per-utt details -> {OUT_DIR/'results.tsv'}")


if __name__ == "__main__":
    main()
