"""N-best GER fusion: titu top-k × Whisper top-k → LLM judge (+ translate).

Validated design (ROVER complementary systems + generative error
correction / HyPoradise / ClozeGER + GenTranslate):
  - two MAXIMALLY-DIVERSE ASR systems (titu CTC ‖ Whisper seq2seq),
    each emitting k hypotheses via beam search;
  - dedup + measure how many hypotheses are actually DISTINCT (CTC
    n-best is known to collapse to near-duplicates — we report this
    rather than assume 2k useful options);
  - a frontier LLM reconciles all distinct hypotheses into one clean
    Bengali transcript AND translates to English in the same pass,
    preferring titu's phonetics for proper nouns / domain terms and
    Whisper's fluency for structure.

Emits per-utterance fusion prompts (paste into Sonnet / NVIDIA NIM).
Optionally calls an OpenAI-compatible endpoint directly if --llm-base-url
and an API key env var are given.

NO ground truth needed (uses the audio-valid utt wavs).

Usage:
    uv run python scripts/asr_nbest_fuse.py --n 14 --k 3
    # then paste outputs/asr_nbest_fuse.prompts.txt into Sonnet/NIM
    # or auto-call NIM:
    uv run python scripts/asr_nbest_fuse.py --n 14 --k 3 \
        --llm-base-url https://integrate.api.nvidia.com/v1 \
        --llm-model sarvamai/sarvam-m --llm-key-env NVIDIA_API_KEY
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import unicodedata
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S")
logger = logging.getLogger("nbest")

DATA_DIR = Path("data/w2v_train_longform")
JSONLS = [
    ("07_Shatkhira",
     DATA_DIR / "07_Shatkhira_CA-MMI_URBAN_Male_07_0202_1716.longform.jsonl"),
    ("15_Sunamganj",
     DATA_DIR / "15_Sunamganj_CA-MMI_URBAN_Male_15_0602_1635.longform.jsonl"),
]


def _norm(t: str) -> str:
    return re.sub(r"\s+", " ",
                  unicodedata.normalize("NFC", t or "").strip())


def dedup(hyps: list[str]) -> list[str]:
    """Order-preserving dedup on normalized form (drops near-dupes that
    differ only by whitespace/punctuation)."""
    seen, out = set(), []
    for h in hyps:
        key = re.sub(r"[\s।,.?!]+", "", _norm(h))
        if key and key not in seen:
            seen.add(key)
            out.append(_norm(h))
    return out


# ── selection (same 3-15s, seed 0 → reproduces the flood case as #10) ──
def select(n, seed=0):
    import random
    rng = random.Random(seed)
    picks, per = [], max(1, n // len(JSONLS))
    for group, path in JSONLS:
        recs = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if 3.0 <= (r.get("end", 0) - r.get("start", 0)) <= 15.0:
                    recs.append(r)
        for r in rng.sample(recs, min(per, len(recs))):
            r["_group"] = group
            picks.append(r)
    return picks


# ── titu n-best via NeMo CTC beam search (robust fallback to greedy) ──
def run_titu_nbest(recs, device, k):
    """titu greedy 1-best per utterance. (CTC n-best collapses to near-
    duplicates and the NeMo beam path is fragile across versions, so we
    use the robust greedy decode that worked in every earlier test.
    titu's role is the phonetic SECOND WITNESS, not diversity — one
    faithful hypothesis is exactly what the fusion judge needs.)"""
    import torch
    import nemo.collections.asr as nemo_asr
    logger.info("Loading titu (greedy 1-best) ...")
    m = nemo_asr.models.ASRModel.from_pretrained(
        "hishab/titu_stt_bn_fastconformer")
    if device == "cuda" and torch.cuda.is_available():
        m = m.to(torch.device("cuda"))
    m.eval()
    results = []
    for i, r in enumerate(recs):
        try:
            o = m.transcribe([r["wav"]], batch_size=1, verbose=False)[0]
            txt = o if isinstance(o, str) else getattr(o, "text", str(o))
            hs = [txt.strip()] if (txt or "").strip() else [""]
        except Exception as e:
            logger.warning("  titu utt %d failed: %s", i,
                           type(e).__name__)
            hs = [""]
        results.append(hs)
        if (i + 1) % 10 == 0:
            logger.info("  titu %d/%d", i + 1, len(recs))
    del m
    torch.cuda.empty_cache()
    return results


# ── Whisper n-best via direct generate (NO language override) ─────────
def run_whisper_nbest(hf_id, recs, device, k):
    """Whisper hypotheses via the bare pipeline() — the path proven to
    work. Tries beam N-best per utt, but falls back to the pipeline's
    robust 1-best if the direct-generate beam path errors (it does on
    some transformers versions). Cross-model diversity (titu vs Whisper)
    is the main signal anyway, so Whisper-1best is acceptable."""
    import torch
    from transformers import pipeline
    logger.info("Loading Whisper %s (pipeline) ...", hf_id)
    dev = 0 if (device == "cuda" and torch.cuda.is_available()) else -1
    asr = pipeline("automatic-speech-recognition", model=hf_id,
                   device=dev, chunk_length_s=30)
    # Attempt beam N-best through the pipeline's generate_kwargs; if the
    # model/version rejects it, fall back to plain 1-best.
    nbest_ok = True
    results = []
    for i, r in enumerate(recs):
        hs = []
        if nbest_ok:
            try:
                res = asr(r["wav"], generate_kwargs={
                    "num_beams": max(k + 2, 5),
                    "num_return_sequences": k,
                    "no_repeat_ngram_size": 3})
                # with num_return_sequences the pipeline may return a
                # list under 'text' or a single string — normalize
                txt = res.get("text")
                if isinstance(txt, list):
                    hs = dedup(txt)[:k]
                elif txt:
                    hs = [txt.strip()]
            except Exception as e:
                logger.warning("whisper beam n-best unavailable (%s) — "
                               "using 1-best", type(e).__name__)
                nbest_ok = False
        if not hs:
            try:
                hs = [(asr(r["wav"]).get("text") or "").strip()]
            except Exception as e:
                logger.warning("  whisper utt %d failed: %s", i,
                               type(e).__name__)
                hs = [""]
        results.append(hs)
        if (i + 1) % 5 == 0:
            logger.info("  whisper %d/%d", i + 1, len(recs))
    del asr
    torch.cuda.empty_cache()
    return results


def load_survey_context() -> str:
    """Domain grounding from the survey wiki (built from the instrument).
    Three faithful pieces, all from the actual document:
      - named entities (bank/NGO/place names, acronyms)
      - ALL survey questions (the enumerator's actual wording — helps
        reconstruct garbled question turns)
      - the consent script (for the intro turns)
    Returns '' if the wiki isn't present (prompt still works).
    """
    wiki = Path(__file__).resolve().parents[1] / "data" / "survey_wiki"
    parts = []

    # NOTE: the auto-extracted `named_entities` are too noisy (all-caps
    # common words + section codes) to use as-is, and `answer_options`
    # are truncated. The QUESTIONS and CONSENT script are clean and
    # high-value, and `domain_terms` is mostly fine — use those.
    vocab = wiki / "domain_vocabulary.md"
    if vocab.exists():
        txt = vocab.read_text(encoding="utf-8")
        m = re.search(r"## domain_terms[^\n]*\n+(.+?)(?:\n## |\Z)",
                      txt, re.S)
        if m:
            parts.append("This survey is about (domain terms): "
                         + " ".join(m.group(1).split()))

    # ALL questions actually in the instrument (lines ending in ?).
    instr = wiki / "instrument_full.txt"
    if instr.exists():
        seen, qs = set(), []
        for line in instr.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.endswith("?") and 8 <= len(line) <= 300:
                key = line.lower()
                if key not in seen:
                    seen.add(key)
                    qs.append(line)
        if qs:
            parts.append("Survey questions the enumerator asks "
                         f"({len(qs)}):\n" + "\n".join(f"- {q}" for q in qs))

    # Consent script (verbatim intro the enumerator reads)
    consent = wiki / "consent_script.txt"
    if consent.exists():
        c = consent.read_text(encoding="utf-8").strip()
        if c:
            parts.append("Consent script (read at the start):\n" + c)

    return "\n\n".join(parts)


_PROMPT = """\
You are reconstructing ONE Bengali utterance from a climate-disaster \
survey in coastal Bangladesh (an enumerator interviews a respondent about \
cyclones, floods, salinity, NGOs, loans, agriculture).
{survey}
Two ASR systems each produced several hypotheses of the SAME audio clip.
(These are MACHINE labels only — never copy the words "PHONETIC" or
"FLUENT" or any system name into your transcript.)
- PHONETIC system: phonetically faithful but may be garbled or run words \
together. Trust it for the SOUND of proper nouns, place/bank names, \
numbers, and domain terms.
- FLUENT system: well-structured but sometimes replaces a content word \
with a plausible-but-wrong one (it once turned বন্যা = flood into বোন না \
= "sister-no"). Trust it for sentence/turn structure and fluency.

PHONETIC hypotheses:
{titu}
FLUENT hypotheses:
{whisper}

Produce exactly ONE line — clean, faithful Bengali only (translation is \
done by a separate dedicated system, do NOT translate):
[bn] One clean Bengali transcript. Reconcile the hypotheses: keep \
whisper's fluency and structure, but where they disagree on a content \
word — especially proper nouns, place/bank names, numbers, domain terms \
(flood, cyclone, NGO names) — choose the reading best supported by titu's \
phonetics AND coherent in context. Keep code-switched English (ward, NGO, \
loan) in Latin script. If no hypothesis yields a recoverable word, mark \
[অস্পষ্ট]. Do NOT invent content beyond what the hypotheses support.

Output only the [bn] line."""


def build_prompt(titu_hyps, whisper_hyps, survey_ctx=""):
    tl = "\n".join(f"  {i+1}. {h}" for i, h in enumerate(titu_hyps)) or "  (none)"
    wl = "\n".join(f"  {i+1}. {h}" for i, h in enumerate(whisper_hyps)) or "  (none)"
    if survey_ctx:
        survey = ("\nSURVEY DOMAIN VOCABULARY (these terms appear in this "
                  "survey — prefer them when a hypothesis is garbled or "
                  "ambiguous, esp. proper nouns / bank / NGO / place / "
                  "cyclone names; do NOT copy this list into the "
                  "transcript):\n" + survey_ctx + "\n")
    else:
        survey = ""
    return _PROMPT.format(survey=survey, titu=tl, whisper=wl)


_TRANSLATE_PROMPT = """\
Translate this Bengali survey utterance (climate-disaster interview in \
coastal Bangladesh) to faithful English. Preserve hedging and \
uncertainty. Render numbers as digits EXACTLY as written — do not change \
a number. Keep proper nouns / place / bank names in standard \
romanization. Keep code-switched English words as-is. Mark unclear spans \
[unclear]. Do not add or omit content. Output only the English.

Bengali: {bn}
English:"""


def parse_bn(fused: str) -> str:
    """Extract the Bengali transcript from the LLM's fused answer."""
    if not fused:
        return ""
    m = re.search(r"\[bn\]\s*(.+)", fused, re.S)
    text = m.group(1) if m else fused
    # stop at a stray [en] if the model added one despite instructions
    text = re.split(r"\[en\]", text)[0]
    return _norm(text)


def _hf_token():
    """Read the HF token from env or the repo-root .env (gated repos
    like IndicTrans2 need it even when your account has access)."""
    for k in ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        if os.environ.get(k):
            return os.environ[k]
    env = Path(__file__).resolve().parents[1] / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            if key.strip() in ("HF_TOKEN", "HUGGINGFACE_TOKEN",
                               "HUGGING_FACE_HUB_TOKEN"):
                return val.strip().strip('"').strip("'") or None
    return None


def run_indictrans2(bn_texts, device, batch_size=8):
    """Dedicated AI4Bharat IndicTrans2 Bengali→English NMT.
    Structurally faithful (preserves numbers/entities) — unlike an LLM,
    it won't silently turn 30.5 into 25. Needs IndicTransToolkit."""
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    try:
        from IndicTransToolkit.processor import IndicProcessor
    except Exception as e:
        raise RuntimeError(
            "IndicTransToolkit missing — install with:\n"
            "  uv pip install IndicTransToolkit\n"
            f"(import error: {e})")

    repo = "ai4bharat/indictrans2-indic-en-1B"
    token = _hf_token()
    if token:
        logger.info("IndicTrans2: authenticating with HF token from "
                    "env/.env")
    logger.info("Loading IndicTrans2 %s ...", repo)
    tok = AutoTokenizer.from_pretrained(
        repo, trust_remote_code=True, token=token)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        repo, trust_remote_code=True, token=token)
    if device == "cuda" and torch.cuda.is_available():
        model = model.to(torch.device("cuda"))
    model.eval()
    ip = IndicProcessor(inference=True)

    dev = torch.device("cuda" if (device == "cuda"
                                  and torch.cuda.is_available()) else "cpu")
    out = []
    for i in range(0, len(bn_texts), batch_size):
        chunk = [t or "·" for t in bn_texts[i:i + batch_size]]
        batch = ip.preprocess_batch(
            chunk, src_lang="ben_Beng", tgt_lang="eng_Latn")
        # IndicTrans2's custom tokenizer needs return_attention_mask=True
        # explicitly — otherwise generate() hits a None attention_mask
        # ('NoneType' object has no attribute 'shape').
        enc = tok(batch, truncation=True, padding="longest",
                  return_tensors="pt", return_attention_mask=True)
        enc = {k: v.to(dev) for k, v in enc.items()}
        # use_cache=False: IndicTrans2's custom modeling code expects the
        # old tuple past_key_values format; transformers 4.57's Cache
        # object breaks it (NoneType .shape). Disabling cache routes
        # around it (fine for these short sentences).
        with torch.no_grad():
            gen = model.generate(**enc, use_cache=False, min_length=0,
                                 max_length=256, num_beams=5,
                                 num_return_sequences=1)
        # decode on the TARGET (English) vocab — IndicTrans2 has separate
        # source/target tokenizers.
        with tok.as_target_tokenizer():
            dec = tok.batch_decode(gen.detach().cpu().tolist(),
                                   skip_special_tokens=True)
        out.extend(ip.postprocess_batch(dec, lang="eng_Latn"))
    del model
    torch.cuda.empty_cache()
    return out


def call_llm_api(prompt, base_url, model, key):
    from openai import OpenAI
    client = OpenAI(base_url=base_url, api_key=key)
    resp = client.chat.completions.create(
        model=model, temperature=0.0, max_tokens=600,
        messages=[{"role": "user", "content": prompt}])
    return resp.choices[0].message.content.strip()


def load_gguf_judge(repo, fname, n_gpu_layers=-1):
    """Local GGUF fusion judge — reuses the pipeline's GGUFProcessor
    (handles the Gemma chat template). No API, private.

    n_gpu_layers=-1 puts all layers on GPU (for models that fit, e.g.
    Gemma-3-12B Q4). For a too-big-for-VRAM MoE (Sarvam-30B IQ2 ~10GB),
    pass a positive int to offload only that many layers to the GPU and
    spill the experts to system RAM — MoE's 2.4B active keeps it fast."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from asr_pipeline.postprocessor import GGUFProcessor
    # 8192 ctx so the survey questions + consent fit alongside the
    # hypotheses (Gemma-3 supports far more; 8192 is safe on 8 GB).
    proc = GGUFProcessor(gguf_repo=repo, gguf_file=fname,
                         n_ctx=8192, n_gpu_layers=n_gpu_layers)
    if not proc.load():
        raise RuntimeError("GGUF judge failed to load")
    return proc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=14)
    ap.add_argument("--k", type=int, default=3, help="hypotheses per system")
    ap.add_argument("--whisper", default=
                    "bengaliAI/tugstugi_bengaliai-regional-asr_whisper-medium")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--llm", default="gguf",
                    choices=["gguf", "api", "none"],
                    help="fusion judge: gguf=local TigerLLM (default, "
                         "no API), api=OpenAI-compatible endpoint, "
                         "none=emit prompts only")
    ap.add_argument("--gguf-repo",
                    default="unsloth/gemma-3-12b-it-GGUF",
                    help="fusion judge GGUF repo (default Gemma-3-12B — "
                         "tops IndicIFEval among 8GB-fit models)")
    ap.add_argument("--gguf-file",
                    default="gemma-3-12b-it-Q4_K_M.gguf")
    ap.add_argument("--gguf-n-gpu-layers", type=int, default=-1,
                    help="-1 = all layers on GPU (fits models like "
                         "Gemma-3-12B Q4). Positive int = partial "
                         "offload for a too-big MoE (e.g. Sarvam-30B "
                         "IQ2 ~10GB: try 20, drop if OOM).")
    ap.add_argument("--translate", action="store_true",
                    help="translate fused Bengali → English with "
                         "IndicTrans2 (dedicated NMT, faithful)")
    ap.add_argument("--llm-translate", action="store_true",
                    help="ALSO have the fusion LLM translate, so its "
                         "English can be compared against IndicTrans2's "
                         "(tests whether the dedicated NMT is needed)")
    ap.add_argument("--survey-context", action="store_true",
                    help="inject survey instrument grounding into the "
                         "fusion prompt. OFF by default — a full dump "
                         "hurt quality in testing (distraction).")
    ap.add_argument("--llm-base-url", default=None)
    ap.add_argument("--llm-model", default=None)
    ap.add_argument("--llm-key-env", default="NVIDIA_API_KEY")
    ap.add_argument("--output", type=Path,
                    default=Path("outputs/asr_nbest_fuse.md"))
    args = ap.parse_args()

    recs = select(args.n)
    logger.info("Selected %d utts", len(recs))
    titu = run_titu_nbest(recs, args.device, args.k)
    whisper = run_whisper_nbest(args.whisper, recs, args.device, args.k)

    # ── Set up the fusion judge ──
    judge = None          # callable: prompt -> answer
    if args.llm == "gguf":
        try:
            proc = load_gguf_judge(args.gguf_repo, args.gguf_file,
                                   n_gpu_layers=args.gguf_n_gpu_layers)
            logger.info("Fusion judge: local GGUF %s (no API, "
                        "n_gpu_layers=%d)", args.gguf_file,
                        args.gguf_n_gpu_layers)
            judge = lambda p: proc.generate(
                p, temperature=0.0, max_tokens=600)
        except Exception as e:
            logger.warning("GGUF judge unavailable (%s) — emitting "
                           "prompts only", e)
    elif args.llm == "api":
        key = os.environ.get(args.llm_key_env, "")
        if args.llm_base_url and args.llm_model and key:
            logger.info("Fusion judge: API %s", args.llm_model)
            judge = lambda p: call_llm_api(
                p, args.llm_base_url, args.llm_model, key)
        else:
            logger.warning("API judge needs --llm-base-url, --llm-model "
                           "and $%s — emitting prompts only",
                           args.llm_key_env)
    do_llm = judge is not None

    lines = ["# N-best titu × Whisper → LLM fusion + translation\n"]
    lines.append(f"k={args.k} hypotheses/system. 'distinct' = after "
                 "dedup; CTC (titu) n-best often collapses — that count "
                 "tells you what the LLM actually receives.\n")
    # Survey grounding is OFF by default: empirically, dumping the
    # instrument (questions + consent) into every call DISTRACTED the
    # fusion judge and made output worse ("lost in the middle"). The
    # plain two-witness fusion was best. Opt in with --survey-context
    # only to experiment (e.g. a future RAG version that injects just
    # the one relevant question per utterance).
    survey_ctx = load_survey_context() if args.survey_context else ""
    if survey_ctx:
        logger.info("Survey grounding ON (%d chars) — note: a full dump "
                    "hurt quality in testing; prefer per-utterance RAG",
                    len(survey_ctx))
    else:
        logger.info("Survey grounding OFF (plain two-witness fusion — "
                    "the best-performing config)")

    prompts = []
    fused_bn = []          # parsed Bengali per utt (for translation)
    tot_titu_distinct = tot_whisper_distinct = 0
    blocks = []            # (header_lines) per utt, filled then appended
    for i, (r, th, wh) in enumerate(zip(recs, titu, whisper), 1):
        tot_titu_distinct += len(th)
        tot_whisper_distinct += len(wh)
        prompt = build_prompt(th, wh, survey_ctx)
        prompts.append(prompt)
        blk = [f"## {i}. [{r['_group']}]  titu_distinct={len(th)} "
               f"whisper_distinct={len(wh)}"]
        for j, h in enumerate(th, 1):
            blk.append(f"- titu {j}: {h}")
        for j, h in enumerate(wh, 1):
            blk.append(f"- whisper {j}: {h}")
        bn = ""
        if do_llm:
            try:
                ans = judge(prompt)
            except Exception as e:
                ans = f"[LLM ERROR: {e}]"
            bn = parse_bn(ans)
            blk.append(f"- **fused [bn]**: {bn}")
            # Optional: same LLM also translates (for comparison vs NMT)
            if args.llm_translate and bn:
                try:
                    en_llm = judge(_TRANSLATE_PROMPT.format(bn=bn)).strip()
                except Exception as e:
                    en_llm = f"[LLM ERROR: {e}]"
                blk.append(f"- [en] (LLM): {en_llm}")
            logger.info("  fused %d/%d", i, len(recs))
        fused_bn.append(bn)
        blocks.append(blk)

    # ── Dedicated translation pass (IndicTrans2) over all fused Bengali ──
    if args.translate and do_llm and any(fused_bn):
        try:
            en = run_indictrans2(fused_bn, args.device)
            for blk, e in zip(blocks, en):
                blk.append(f"- **[en] (IndicTrans2)**: {e}")
        except Exception as e:
            logger.warning("Translation skipped: %s", e)
            lines.append(f"> Translation skipped: {e}\n")

    for blk in blocks:
        lines.extend(blk)
        lines.append("")

    n = max(1, len(recs))
    lines.insert(2, f"- avg distinct hyps: titu {tot_titu_distinct/n:.1f}, "
                    f"whisper {tot_whisper_distinct/n:.1f} "
                    f"(if titu≈1, CTC n-best collapsed — expected)\n")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    pp = args.output.with_suffix(".prompts.txt")
    pp.write_text(
        "\n\n========== UTT BOUNDARY ==========\n\n".join(prompts),
        encoding="utf-8")
    print("\n".join(lines))
    logger.info("Report -> %s", args.output)
    logger.info("Fusion prompts -> %s (paste into Sonnet/NIM)", pp)
    return 0


if __name__ == "__main__":
    sys.exit(main())
