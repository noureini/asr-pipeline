"""Best-quality Qwen3.5-9B cleanup of titu ASR + multi-aspect
LLM-as-judge evaluation. No GT used.

Cleanup upgrades over single-pass:
  (1) Few-shot prompt — 3 worked examples showing the exact failure
      modes we want fixed (broken word boundaries, code-switch
      preservation, dialect preservation). Per arXiv 2407.16370 this
      is the highest-ROI improvement for LLM correction tasks.
  (2) Cross-chunk continuity context — previous chunk's cleaned
      text (last 200 chars) passed as continuity reference for
      proper-noun and register consistency.

Evaluation: multi-aspect rubric judge instead of single A/B vote.
Per pair we get verdicts on:
  - fluency      (which reads as more grammatical Bengali?)
  - faithfulness (which preserves input content without adding?)
  - hallucination (does B add words not in A's content?)
  - code_switch  (is English code-switch preserved if present?)
  - dialect      (are colloquial forms preserved if present?)

Outputs:
  data/p2g/cleanup_<stem>/cleaned.jsonl   per-chunk cleaned output
  data/p2g/cleanup_<stem>/judge.jsonl     per-chunk multi-aspect verdicts
  data/p2g/cleanup_<stem>/summary.txt     headline numbers per aspect

Usage:
  uv run python scripts/cleanup_titu.py \\
      --stem 15_Sunamganj_CA-MMI_URBAN_Male_15_0602_1635
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_GGUF_REPO = "unsloth/Qwen3.5-9B-GGUF"
DEFAULT_GGUF_FILE = "Qwen3.5-9B-UD-Q4_K_XL.gguf"

# Anthropic model IDs (update as new versions release)
CLAUDE_MODELS = {
    "claude-opus":    "claude-opus-4-5",
    "claude-sonnet":  "claude-sonnet-4-5",
    "claude-haiku":   "claude-haiku-4-5",
}


# ---- cleanup prompt (few-shot + cross-chunk context) ---------------
# Examples deliberately avoid listing specific English words (that
# caused the prior hallucination bug). Each demonstrates one principle.

CLEANUP_PREAMBLE = (
    "You are reconstructing the most plausible Bengali utterance "
    "from a noisy ASR transcript of coastal Bangladesh household "
    "survey audio. You have access to BOTH the lexical ASR guess "
    "(HYPOTHESIS) and the IPA phoneme stream (PHONEMES).\n\n"
    "DOMAIN: Survey speakers in coastal Bangladesh discuss family "
    "(family, children, parents), agriculture (চাষ, ফসল, ধান, "
    "শাকসবজি, jomi/জমি, krishi/কৃষি), livelihood (mach/মাছ, "
    "majur/মজুর, ricksha), poverty/wealth, water (পানি, বন্যা, "
    "ghurnijhar/ঘূর্ণিঝড়), administrative units (ward, NGO, "
    "ইউনিয়ন, পৌরসভা, পাড়া, গ্রাম), money/loans (টাকা, ঋণ, "
    "loan), age (numbers in colloquial Bengali). They are "
    "typically adults, often farmers or laborers, sometimes "
    "elderly. Code-switch English words: ward, number, NGO, "
    "project, vaccine, hospital, family, interview, record.\n\n"
    "YOUR TASK:\n"
    "Use Bengali knowledge + the survey domain + PHONEMES "
    "(when titu fails) to reconstruct the most likely sentence "
    "that was spoken. Be willing to recover meaning from broken "
    "ASR — but stay anchored to what the input actually "
    "supports.\n\n"
    "ALLOWED reconstructions:\n"
    "- Recover a broken word from context. Example: 'সা ত্র "
    "বছর' from an adult speaker → 'সাত্তর বছর' (70 years).\n"
    "- Recognize English code-switch from PHONEMES. Example: "
    "PHONEMES /w a r d  n a m b a r/ → 'ward number' even if "
    "HYPOTHESIS is garbled.\n"
    "- Join split words (অ রেক → recognize as 'অডিও রেকর্ড' if "
    "context is consent recording).\n"
    "- Fix dropped vowels (আুন → আছেন).\n"
    "- Add light punctuation (commas, periods, question marks).\n\n"
    "FORBIDDEN behaviors (these are real failure modes — do NOT "
    "do these):\n"
    "- Do NOT introduce topics absent from input — no newspaper "
    "names (প্রথম আলো etc.), no fictional content, no unrelated "
    "events.\n"
    "- Do NOT translate to English; do NOT romanize Bengali.\n"
    "- Do NOT standardize coastal/colloquial dialect to formal "
    "Bengali (নাই stays নাই; করতেছি stays করতেছি).\n"
    "- Do NOT output thinking tags, OUTPUT: prefixes, code "
    "blocks, programming words, or explanations — output is ONLY "
    "the reconstructed sentence on one line.\n\n"
    "CRITICAL RULE — IPA grounding for English substitution:\n"
    "You may ONLY substitute an English word for a Bengali fragment "
    "if the PHONEMES clearly spell out that English word "
    "phonemically. Check before substituting:\n"
    "  - /w a r d/ unambiguously spells 'ward'         → substitute OK\n"
    "  - /n a m b a r/ unambiguously spells 'number'   → substitute OK\n"
    "  - /v æ k s i n/ or /v a k s i n/ spells 'vaccine' → OK\n"
    "  - /ɛ n d͡ʒ i o/ spells 'NGO' (letter by letter) → OK\n"
    "  - /p o ɾ u/ does NOT unambiguously spell any English word "
    "→ keep Bengali (likely 'পৌরা' or 'পুরো')\n"
    "  - /s o l/ or /b a r k e/ alone is NOT enough to inject "
    "English → keep Bengali\n"
    "When the IPA is short, fragmentary, or could equally be "
    "Bengali phonemes, ALWAYS prefer the Bengali interpretation. "
    "Do NOT introduce English unless the IPA evidence is "
    "unambiguous and the word is one of the standard survey "
    "code-switch tokens (ward, number, NGO, project, vaccine, "
    "hospital, family, audio, record, interview).\n\n"
    "EXAMPLES:\n\n"
    "Example 1 — minimal word-boundary fix:\n"
    "HYPOTHESIS: আমি কাজ ক রি না\n"
    "PHONEMES:   ▁ a m i ▁ k a d͡ʒ ▁ k o r i ▁ n a ▁\n"
    "OUTPUT:     আমি কাজ করি না\n\n"
    "Example 2 — recover broken word with domain context:\n"
    "HYPOTHESIS: আমার বয়স সা ত্র বছর\n"
    "PHONEMES:   ▁ a m a r ▁ b ɔ j o s ▁ s a t̪ t̪ o r ▁ b o t͡ ɕʰ o r ▁\n"
    "OUTPUT:     আমার বয়স সাত্তর বছর\n"
    "(PHONEMES /s a t̪ t̪ o r/ matches 'সাত্তর'=70 — plausible "
    "age)\n\n"
    "Example 3 — recognize English code-switch from PHONEMES:\n"
    "HYPOTHESIS: এক ন্বার এই গ্রামের নাম কি\n"
    "PHONEMES:   ▁ e k ▁ n a m b a r ▁ e i ▁ g r a m e r ▁ n a m ▁ k i ▁\n"
    "OUTPUT:     এক number এই গ্রামের নাম কি\n"
    "(PHONEMES /n a m b a r/ is English 'number'; HYPOTHESIS "
    "'ন্বার' was a broken transliteration)\n\n"
    "Example 4 — leave broken when input does NOT support a fix:\n"
    "HYPOTHESIS: আমি বলমানে কৃ খজ ফসলের আমার টাচ্চা\n"
    "PHONEMES:   ▁ a m i ▁ b o l m a n e ▁ k r i s i ▁ k h ɔ d͡ʒ ▁ "
    "p h o s ɔ l e r ▁ a m a r ▁ b a t̪ t͡ɕa ▁\n"
    "OUTPUT:     আমি বলমানে কৃষি কাজ ফসলের আমার বাচ্চা\n"
    "(PHONEMES support 'কৃষি কাজ' and 'বাচ্চা' — use them. But "
    "'বলমানে' stays as-is — neither titu nor IPA gives a clear "
    "alternative.)\n\n"
)


def _load_survey_vocab() -> str:
    """Read survey wiki vocabulary if present, format for prompt."""
    vfile = (Path(__file__).resolve().parents[1]
             / "data" / "survey_wiki" / "domain_vocabulary.md")
    if not vfile.exists():
        return ""
    # Skip the markdown header lines, keep just the term lists
    lines = []
    for ln in vfile.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        if ln.startswith("Extracted from"):
            continue
        lines.append(ln)
    return " | ".join(lines)[:1500]  # cap to keep prompt compact


_SURVEY_VOCAB = _load_survey_vocab()


def cleanup_prompt(raw: str, ipa: str, prev_cleaned: str) -> str:
    ctx = (prev_cleaned[-200:] if prev_cleaned else "").strip()
    parts = [CLEANUP_PREAMBLE]
    if _SURVEY_VOCAB:
        parts.append(
            "SURVEY DOMAIN VOCABULARY (these terms are valid "
            "survey-domain words — when the input is ambiguous, "
            "prefer reconstructions that land on these):\n"
            f"{_SURVEY_VOCAB}\n\n")
    if ctx:
        parts.append(
            f"PREVIOUS UTTERANCE (continuity reference only — "
            f"do NOT repeat or extend it):\n{ctx}\n\n")
    parts.append("NOW RECONSTRUCT THIS UTTERANCE:\n")
    parts.append(f"HYPOTHESIS: {raw}\n")
    parts.append(
        f"PHONEMES:   {ipa[:600] if ipa else '(unavailable)'}\n")
    parts.append("OUTPUT: ")
    return render_qwen_prompt("".join(parts))


def claude_cleanup_call(client, model: str, raw: str, ipa: str,
                          prev_cleaned: str, max_tokens: int = 512
                          ) -> str:
    """Single cleanup call via Anthropic API. Claude follows
    instructions reliably so we use system + user pattern instead of
    the Qwen-style ChatML continuation."""
    ctx = (prev_cleaned[-200:] if prev_cleaned else "").strip()
    user_msg = (
        (f"PREVIOUS UTTERANCE (continuity reference only — do NOT "
         f"repeat or extend it):\n{ctx}\n\n" if ctx else "")
        + "Reconstruct THIS utterance:\n"
        + f"HYPOTHESIS: {raw}\n"
        + f"PHONEMES:   {ipa[:600] if ipa else '(unavailable)'}\n\n"
        + "Output ONLY the corrected single-line Bengali sentence. "
        "No explanation, no preamble, no quotation marks, no code "
        "blocks."
    )
    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=0.2,
        system=CLEANUP_PREAMBLE,
        messages=[{"role": "user", "content": user_msg}],
    )
    # Extract text from response
    parts = []
    for block in resp.content:
        if hasattr(block, "text"):
            parts.append(block.text)
    return "".join(parts).strip()




# ---- multi-aspect judge prompt -------------------------------------

JUDGE_PREAMBLE = (
    "You are evaluating two Bengali ASR transcripts of the same "
    "audio segment. Judge them on five aspects and output strict "
    "JSON.\n\n"
    "Aspects (decide independently for each):\n"
    "  fluency       — which reads as more grammatical, natural "
    "Bengali? Answer 'A', 'B', or 'TIE'.\n"
    "  faithfulness  — which preserves the input acoustic content "
    "without adding new words or phrases? Answer 'A', 'B', or 'TIE'.\n"
    "  hallucination_B — does Transcript B contain words/phrases "
    "that don't appear to come from A's content (i.e. fabricated)? "
    "Answer 'yes' or 'no'.\n"
    "  code_switch   — if Transcript A contains English words in "
    "Latin script, does B preserve them verbatim? Answer 'yes', "
    "'no', or 'NA' (if A has no English).\n"
    "  dialect       — if A has colloquial/dialect Bengali, does B "
    "preserve those forms (vs standardizing)? Answer 'yes', 'no', "
    "or 'NA' (if not applicable).\n\n"
    "RULES:\n"
    "  - Do NOT prefer the longer transcript on length alone.\n"
    "  - Judge on Bengali quality, not on which has more English.\n"
    "  - Output ONLY the JSON object. No explanation. No code block.\n"
)


def judge_prompt(a: str, b: str) -> str:
    msg = (JUDGE_PREAMBLE
           + f"\nTranscript A:\n{a}\n\n"
           + f"Transcript B:\n{b}\n\n"
           + "JSON output:\n")
    return render_qwen_prompt(msg)


# ---- Qwen3.5 manual ChatML template (force-close thinking) ---------

def render_qwen_prompt(msg: str) -> str:
    return (
        "<|im_start|>user\n" + msg + "<|im_end|>\n"
        "<|im_start|>assistant\n"
        "<think>\n\n</think>\n\n"
    )


# ---- output artifact stripping --------------------------------------

import re as _re

# Patterns observed leaking from TigerLLM/Gemma2/Qwen outputs
_THOUGHT_BLOCK = _re.compile(r"<thought>.*?</thought>", _re.DOTALL)
_THOUGHT_OPEN = _re.compile(r"<thought>.*", _re.DOTALL)
_THOUGHT_CLOSE = _re.compile(r".*?</thought>\s*", _re.DOTALL)
_THINK_BLOCK = _re.compile(r"<think>.*?</think>", _re.DOTALL)
# Match standalone </think> or <think> anywhere (Tiger sometimes
# emits them mid- or end-of-output without a matching opener)
_THINK_TAG = _re.compile(r"</?think\s*>", _re.IGNORECASE)
_SPECIAL_TOKENS = _re.compile(
    r"<\|file_separator\|>|<\|im_end\|>|<\|im_start\|>"
    r"|<\|endoftext\|>|<bos>|<eos>|<s>|</s>")
_CODE_FENCE = _re.compile(
    r"^```(?:\w+)?\s*\n(.*?)\n```\s*$", _re.DOTALL)
# Match leading OUTPUT: with optional </ or > variants we've seen
_OUTPUT_PREFIX = _re.compile(
    r"^\s*</?\s*(OUTPUT|Output|output)\s*[:：]?\s*",
    _re.MULTILINE)
# Latin words allowed mid-Bengali (code-switch + audio/survey vocab).
# Anything Latin NOT in this set, mid-Bengali, is likely a model
# leak ("Deserialize", "pouches", etc.) and gets dropped.
_ALLOWED_LATIN = {
    "ward", "wards", "number", "numbers", "ngo", "ngos",
    "project", "projects", "vaccine", "vaccines",
    "hospital", "hospitals", "family", "families",
    "interview", "interviews", "audio", "record", "records",
    "mp", "nid", "id", "ok", "okay", "yes", "no",
    "bank", "bus", "loan", "phone",
}
# Find Latin-script word (run of Latin letters). We'll inspect each
# and drop ones not in the allowlist.
_LATIN_WORD = _re.compile(r"[A-Za-z]+")
# Devanagari (Hindi/etc.) — should NEVER appear in Bengali output.
# We've seen Tiger/Gemma occasionally leak जाने, अगर, etc.
_DEVANAGARI = _re.compile(r"[ऀ-ॿ]+")
# Other Indic scripts that shouldn't appear in Bengali output
_OTHER_INDIC = _re.compile(r"[਀-૿଀-୿"
                              r"஀-௿ఀ-౿"
                              r"ಀ-೿ഀ-ൿ]+")
_LEADING_QUOTES = _re.compile(r'^[\'"`“”‘’]+|[\'"`“”‘’]+$')


def _drop_unknown_latin(text: str) -> str:
    """Drop Latin-script words not in the survey-domain allowlist.
    These are usually model leaks (Deserialize, pouches, etc.)."""
    def repl(m: _re.Match) -> str:
        w = m.group(0)
        return w if w.lower() in _ALLOWED_LATIN else ""
    out = _LATIN_WORD.sub(repl, text)
    # Collapse the resulting double-spaces
    return _re.sub(r"\s{2,}", " ", out).strip()


def strip_artifacts(text: str) -> str:
    """Remove all known leak patterns from an LLM cleanup response.
    Returns a single-line Bengali sentence (best effort).
    """
    if not text:
        return ""
    s = text

    # 1. Drop <think>/<thought> reasoning blocks (paired + dangling)
    s = _THINK_BLOCK.sub("", s)
    s = _THOUGHT_BLOCK.sub("", s)
    s = _THOUGHT_CLOSE.sub("", s)
    s = _THOUGHT_OPEN.sub("", s)
    # Strip any remaining standalone <think>/</think> tags
    s = _THINK_TAG.sub("", s)

    # 2. Drop special control tokens
    s = _SPECIAL_TOKENS.sub("", s)

    # 3. Unwrap ```...``` code fences
    m = _CODE_FENCE.match(s.strip())
    if m:
        s = m.group(1)
    elif s.strip().startswith("```"):
        # Partial fence — drop first line, drop trailing ```
        s = s.strip().split("\n", 1)[-1]
        if "```" in s:
            s = s.rsplit("```", 1)[0]

    # 4. Drop any leading "OUTPUT:" prefix lines
    s = _OUTPUT_PREFIX.sub("", s)

    # 5. Collapse multiple blank lines, take the LAST non-empty line
    #    (models sometimes write commentary then the answer last)
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    if not lines:
        return ""
    # If multiple lines, pick the last that contains Bengali script.
    # If none has Bengali, return the last non-empty.
    def has_bengali(ln: str) -> bool:
        return any("ঀ" <= c <= "৿" for c in ln)
    bn_lines = [ln for ln in lines if has_bengali(ln)]
    s = bn_lines[-1] if bn_lines else lines[-1]

    # 6. Strip leading/trailing quote marks and stray whitespace
    s = _LEADING_QUOTES.sub("", s).strip()
    # Strip stray leading punctuation (':', '-', '—') from leaked
    # OUTPUT prefix variants like ": দুটো আনা"
    s = _re.sub(r"^[\s:：\-—•\.]+", "", s).strip()

    # 7. Drop unexpected Latin words (model leakage). Allowed:
    #    survey-domain code-switch vocabulary only.
    s = _drop_unknown_latin(s)
    # 8. Drop any Devanagari / other-Indic script leaks. These should
    #    NEVER appear in Bengali output.
    s = _DEVANAGARI.sub("", s)
    s = _OTHER_INDIC.sub("", s)
    s = _re.sub(r"\s{2,}", " ", s).strip()
    # 9. Trim end-of-output repetition (model occasionally appends a
    #    duplicate clause).  Look for "...X. X." or "...X, X." with
    #    X >= 15 chars and identical.
    s = _trim_tail_repetition(s)
    return s


def _trim_tail_repetition(s: str, min_len: int = 15) -> str:
    """If the string ends with a near-duplicate of an earlier clause
    of length >= min_len, drop the duplicate tail."""
    if not s:
        return s
    # Split on , . ? ! । and look for the longest tail that matches
    # something earlier
    parts = _re.split(r"([,।\.?!])\s*", s)
    if len(parts) < 4:
        return s
    # Reassemble clauses with their punctuation
    clauses = []
    i = 0
    while i < len(parts):
        text = parts[i].strip()
        punct = parts[i + 1] if i + 1 < len(parts) else ""
        if text:
            clauses.append(text + punct)
        i += 2
    if len(clauses) < 2:
        return s
    # Check if last clause's normalized form matches any earlier clause
    last = clauses[-1].rstrip(",.?!। ").strip().lower()
    if len(last) < min_len:
        return s
    for prev in clauses[:-1]:
        prev_norm = prev.rstrip(",.?!। ").strip().lower()
        if prev_norm == last:
            return " ".join(clauses[:-1]).strip()
    return s


# ---- main -----------------------------------------------------------

def parse_judge_json(text: str) -> dict:
    """Pull the first {...} block out of the response and parse."""
    text = text.strip()
    # Find outermost JSON object
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return {}
    blob = text[start:end + 1]
    try:
        obj = json.loads(blob)
    except json.JSONDecodeError:
        # try truncating to first '}'
        try:
            obj = json.loads(blob[:blob.find("}") + 1])
        except json.JSONDecodeError:
            return {}
    # Normalize values
    out: dict = {}
    for k in ("fluency", "faithfulness"):
        v = str(obj.get(k, "")).strip().upper()
        if v in ("A", "B", "TIE"):
            out[k] = v
    v = str(obj.get("hallucination_B", "")).strip().lower()
    if v in ("yes", "no"):
        out["hallucination_B"] = v
    for k in ("code_switch", "dialect"):
        v = str(obj.get(k, "")).strip().lower()
        if v in ("yes", "no", "na"):
            out[k] = v if v != "na" else "NA"
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", required=True)
    ap.add_argument("--titu-root", type=Path,
                    default=REPO / "outputs"
                            / "real_eval_titu_excellent")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--backend",
                    choices=["tiger", "claude-opus", "claude-sonnet",
                             "claude-haiku"],
                    default="tiger",
                    help="LLM backend. 'tiger' = local TigerLLM/Qwen "
                         "GGUF (free, lower quality). 'claude-*' = "
                         "Anthropic API via your configured server "
                         "(production quality, ~$10-50/interview).")
    ap.add_argument("--claude-model", default=None,
                    help="override default Claude model name "
                         "(e.g. claude-opus-4-5-20250605)")
    ap.add_argument("--gguf-repo", default=DEFAULT_GGUF_REPO,
                    help=f"HF repo for GGUF (default: "
                         f"{DEFAULT_GGUF_REPO})")
    ap.add_argument("--gguf-file", default=DEFAULT_GGUF_FILE,
                    help=f"GGUF filename "
                         f"(default: {DEFAULT_GGUF_FILE})")
    ap.add_argument("--titu-suffix", default=".titu_lm.jsonl",
                    help="titu file suffix to load (e.g. "
                         "'.titu_lm.jsonl' = LM-rescored, "
                         "'.chunks.jsonl' = raw titu — more chunks)")
    ap.add_argument("--judge", action="store_true",
                    help="run the multi-aspect LLM-as-judge "
                         "evaluation pass after cleanup. OFF by "
                         "default (production mode). Only enable for "
                         "methodology validation — adds 3-4 hours "
                         "of GPU compute per interview on TigerLLM "
                         "to re-confirm quality numbers we already "
                         "measured on the 30-chunk smoke test.")
    ap.add_argument("--max-tokens-cleanup", type=int, default=512)
    ap.add_argument("--max-tokens-judge", type=int, default=200)
    ap.add_argument("--n-gpu-layers", type=int, default=-1)
    ap.add_argument("--n-ctx", type=int, default=4096)
    args = ap.parse_args()

    # Default out-dir tags by model short-name to avoid mixing runs
    model_tag = args.gguf_file.split(".")[0].lower().replace("-", "_")
    if args.out_dir is None:
        args.out_dir = (REPO / "data" / "p2g"
                        / f"cleanup_{args.stem}__{model_tag}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  model: {args.gguf_repo}/{args.gguf_file}", flush=True)
    print(f"  out dir: {args.out_dir}", flush=True)

    # Load titu chunks
    stem_dir = args.titu_root / args.stem
    titu_files = sorted(stem_dir.glob(f"*{args.titu_suffix}"))
    if not titu_files:
        sys.exit(f"no *{args.titu_suffix} in {stem_dir}")
    chunks: list[dict] = []
    for f in titu_files:
        partstem = f.name[: -len(args.titu_suffix)]
        for ln in f.read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if not ln:
                continue
            r = json.loads(ln)
            r["partstem"] = partstem
            chunks.append(r)
    chunks.sort(key=lambda c: (c["partstem"], c.get("start", 0.0),
                                c["idx"]))

    # Load matching ZIPA outputs (per-chunk IPA) — keyed by
    # (partstem, idx). Same chunk indexing as titu.
    ipa_map: dict[tuple[str, int], str] = {}
    zipa_files = sorted(stem_dir.glob("*.zipa.jsonl"))
    for f in zipa_files:
        partstem = f.name[: -len(".zipa.jsonl")]
        for ln in f.read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if not ln:
                continue
            r = json.loads(ln)
            ipa_map[(partstem, r["idx"])] = r.get("ipa", "")
    print(f"  loaded ZIPA for {len(ipa_map)} chunks "
          f"from {len(zipa_files)} files", flush=True)
    if args.limit:
        chunks = chunks[: args.limit]
    print(f"loaded {len(chunks)} titu chunks from "
          f"{len(titu_files)} files", flush=True)

    # Resume — load already-cleaned chunks
    cleaned_path = args.out_dir / "cleaned.jsonl"
    cleaned_map: dict[tuple[str, int], str] = {}
    if cleaned_path.exists():
        for ln in cleaned_path.read_text(
                encoding="utf-8").splitlines():
            ln = ln.strip()
            if not ln:
                continue
            try:
                r = json.loads(ln)
                cleaned_map[(r["partstem"], r["idx"])] = r["cleaned"]
            except (KeyError, json.JSONDecodeError):
                continue
    todo_clean = [c for c in chunks
                  if (c["partstem"], c["idx"]) not in cleaned_map]
    print(f"  {len(cleaned_map)} chunks already cleaned, "
          f"{len(todo_clean)} to process", flush=True)

    # Load backend
    llm = None
    claude_client = None
    claude_model_id = None
    if args.backend == "tiger":
        from huggingface_hub import hf_hub_download
        from llama_cpp import Llama
        print(f"loading {args.gguf_file} ...", flush=True)
        model_path = hf_hub_download(repo_id=args.gguf_repo,
                                       filename=args.gguf_file)
        llm = Llama(model_path=model_path, n_ctx=args.n_ctx,
                    n_gpu_layers=args.n_gpu_layers, verbose=False)
        print("  model loaded\n", flush=True)
    else:
        try:
            import anthropic
        except ImportError:
            sys.exit("pip install anthropic    # required for "
                     "Claude backends")
        claude_model_id = (args.claude_model
                           or CLAUDE_MODELS[args.backend])
        claude_client = anthropic.Anthropic()  # uses env config
        print(f"using Anthropic backend  model={claude_model_id}\n",
              flush=True)

    # --- CLEANUP PASS (sequential, uses previous cleaned as context) -
    t0 = time.time()
    # Build per-partstem ordering so cross-chunk context flows by time
    chunk_order = sorted(
        chunks,
        key=lambda c: (c["partstem"], c.get("start", 0.0), c["idx"]))
    # For context lookup as we process: prev_cleaned per partstem
    prev_cleaned_by_part: dict[str, str] = {}
    # Pre-populate from resumed state in time order
    for c in chunk_order:
        key = (c["partstem"], c["idx"])
        if key in cleaned_map:
            prev_cleaned_by_part[c["partstem"]] = cleaned_map[key]
    if todo_clean:
        print("CLEANUP PASS — few-shot + cross-chunk context\n",
              flush=True)
    n_done = 0
    with cleaned_path.open("a", encoding="utf-8") as fh:
        for c in chunk_order:
            key = (c["partstem"], c["idx"])
            if key in cleaned_map:
                continue
            raw = (c.get("text") or "").strip()
            if not raw:
                cleaned = ""
            else:
                prev = prev_cleaned_by_part.get(c["partstem"], "")
                ipa = ipa_map.get(key, "")
                try:
                    if llm is not None:
                        # Local Tiger / Qwen GGUF backend
                        out = llm.create_completion(
                            prompt=cleanup_prompt(raw, ipa, prev),
                            max_tokens=args.max_tokens_cleanup,
                            temperature=0.2, top_p=0.9, top_k=40,
                            repeat_penalty=1.05,
                            stop=["<|im_end|>", "<|endoftext|>",
                                  "\nINPUT:", "\nOUTPUT:"],
                        )
                        cleaned = out["choices"][0]["text"]
                    else:
                        # Claude (Anthropic) backend
                        cleaned = claude_cleanup_call(
                            claude_client, claude_model_id,
                            raw, ipa, prev,
                            max_tokens=args.max_tokens_cleanup,
                        )
                    cleaned = strip_artifacts(cleaned)
                    # Safety fallback: if cleanup produced empty or
                    # severely-truncated output, fall back to raw to
                    # avoid losing the chunk entirely. Threshold:
                    # cleaned must be at least 30% the length of raw.
                    if (not cleaned
                            or len(cleaned) < max(8, 0.3 * len(raw))):
                        cleaned = raw
                except Exception as e:
                    print(f"  ERR cleanup {key}: "
                          f"{type(e).__name__}: {e}", flush=True)
                    cleaned = raw
            row = {
                "partstem": c["partstem"], "idx": c["idx"],
                "start": c.get("start"), "end": c.get("end"),
                "raw": raw, "cleaned": cleaned,
            }
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            cleaned_map[key] = cleaned
            prev_cleaned_by_part[c["partstem"]] = cleaned
            n_done += 1
            if n_done % 20 == 0 or n_done == len(todo_clean):
                el = time.time() - t0
                rate = n_done / max(el, 1e-6)
                eta = (len(todo_clean) - n_done) / rate if rate else 0
                print(f"  cleanup {n_done}/{len(todo_clean)} "
                      f"({rate:.1f}/s, eta {eta:.0f}s)", flush=True)

    # --- JUDGE PASS (multi-aspect) ----------------------------------
    # Production default: skip the LLM-as-judge pass entirely. It
    # was useful for calibrating quality on the 30-chunk smoke
    # test but is wasteful on full-interview runs — the cleaned
    # output is what we ship, and the per-aspect numbers we
    # measured already characterize the pipeline. Enable with
    # --judge for methodology validation runs.
    if not args.judge:
        print(f"\nskipping LLM-as-judge pass (use --judge to enable; "
              f"production mode does not need it)", flush=True)
        print(f"cleaned JSONL  -> {cleaned_path}")
        return
    # Also skip for Claude — judging via Claude itself is circular.
    if args.backend != "tiger":
        print(f"\nbackend={args.backend}: skipping LLM-judge pass "
              f"(Claude output trusted; manual eyeball recommended)",
              flush=True)
        print(f"\ncleaned JSONL  -> {cleaned_path}")
        return
    judge_path = args.out_dir / "judge.jsonl"
    judge_rows: list[dict] = []
    judge_done: set[tuple[str, int]] = set()
    if judge_path.exists():
        for ln in judge_path.read_text(
                encoding="utf-8").splitlines():
            ln = ln.strip()
            if not ln:
                continue
            try:
                r = json.loads(ln)
                judge_done.add((r["partstem"], r["idx"]))
                judge_rows.append(r)
            except (KeyError, json.JSONDecodeError):
                continue
    todo_judge = [c for c in chunks
                  if (c["partstem"], c["idx"]) not in judge_done
                  and (c["partstem"], c["idx"]) in cleaned_map]
    print(f"\nJUDGE PASS — multi-aspect rubric  "
          f"({len(judge_done)} done, "
          f"{len(todo_judge)} to process)\n", flush=True)
    rng = random.Random(42)
    t1 = time.time()
    n_judge = 0
    with judge_path.open("a", encoding="utf-8") as fh:
        for c in todo_judge:
            key = (c["partstem"], c["idx"])
            raw = (c.get("text") or "").strip()
            cleaned = cleaned_map.get(key, "").strip()
            if not raw or not cleaned or raw == cleaned:
                # nothing to judge meaningfully
                continue
            # Randomize A/B order to avoid position bias
            if rng.random() < 0.5:
                a_text, b_text = raw, cleaned
                a_is = "raw"
            else:
                a_text, b_text = cleaned, raw
                a_is = "cleaned"
            try:
                out = llm.create_completion(
                    prompt=judge_prompt(a_text, b_text),
                    max_tokens=args.max_tokens_judge,
                    temperature=0.1, top_p=0.9, top_k=20,
                    repeat_penalty=1.0,
                    stop=["<|im_end|>"],
                )
                vote_raw = out["choices"][0]["text"]
            except Exception as e:
                vote_raw = f"ERR:{type(e).__name__}"
            verdict = parse_judge_json(vote_raw)
            # Map A/B in the verdict to raw/cleaned
            def ab_to_side(v: str) -> str:
                if v == "A":
                    return a_is
                if v == "B":
                    return "cleaned" if a_is == "raw" else "raw"
                return "tie"
            row = {
                "partstem": c["partstem"], "idx": c["idx"],
                "start": c.get("start"), "end": c.get("end"),
                "raw": raw, "cleaned": cleaned,
                "a_is": a_is, "vote_raw_text": vote_raw[:300],
                "fluency": ab_to_side(verdict.get("fluency", "")),
                "faithfulness": ab_to_side(
                    verdict.get("faithfulness", "")),
                "hallucination_B": verdict.get(
                    "hallucination_B", ""),
                "code_switch": verdict.get("code_switch", ""),
                "dialect": verdict.get("dialect", ""),
            }
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            judge_rows.append(row)
            n_judge += 1
            if n_judge % 20 == 0 or n_judge == len(todo_judge):
                el = time.time() - t1
                rate = n_judge / max(el, 1e-6)
                eta = (len(todo_judge) - n_judge) / rate if rate else 0
                print(f"  judge {n_judge}/{len(todo_judge)} "
                      f"({rate:.1f}/s, eta {eta:.0f}s)", flush=True)

    # --- TALLY ------------------------------------------------------
    def tally_aspect(key: str, vals: tuple) -> dict:
        d = {v: 0 for v in vals}
        for r in judge_rows:
            v = r.get(key, "")
            if v in d:
                d[v] += 1
        return d

    flu = tally_aspect("fluency", ("cleaned", "raw", "tie"))
    fai = tally_aspect("faithfulness", ("cleaned", "raw", "tie"))
    hal = tally_aspect("hallucination_B", ("yes", "no"))
    cs  = tally_aspect("code_switch", ("yes", "no", "NA"))
    dia = tally_aspect("dialect", ("yes", "no", "NA"))
    n = len(judge_rows)
    pct = lambda x: f"{100*x/max(1,n):.0f}%"

    summary = (
        f"Qwen3.5-9B cleanup of titu — {args.stem}\n"
        f"  total titu chunks: {len(chunks)}\n"
        f"  chunks cleaned   : {len(cleaned_map)}\n"
        f"  chunks judged    : {n}\n\n"
        f"MULTI-ASPECT LLM-AS-JUDGE (A/B order randomized per pair)\n\n"
        f"  FLUENCY (more grammatical Bengali?)\n"
        f"    cleaned wins : {flu['cleaned']}/{n}  "
        f"({pct(flu['cleaned'])})\n"
        f"    raw wins     : {flu['raw']}/{n}  ({pct(flu['raw'])})\n"
        f"    tie          : {flu['tie']}/{n}  ({pct(flu['tie'])})\n\n"
        f"  FAITHFULNESS (preserves input content, no additions?)\n"
        f"    cleaned wins : {fai['cleaned']}/{n}  "
        f"({pct(fai['cleaned'])})\n"
        f"    raw wins     : {fai['raw']}/{n}  ({pct(fai['raw'])})\n"
        f"    tie          : {fai['tie']}/{n}  ({pct(fai['tie'])})\n\n"
        f"  HALLUCINATION in cleaned output?\n"
        f"    yes (bad)    : {hal['yes']}/{n}  ({pct(hal['yes'])})\n"
        f"    no (good)    : {hal['no']}/{n}  ({pct(hal['no'])})\n\n"
        f"  CODE-SWITCH preservation (when applicable)\n"
        f"    yes (good)   : {cs['yes']}/{n}  ({pct(cs['yes'])})\n"
        f"    no  (bad)    : {cs['no']}/{n}  ({pct(cs['no'])})\n"
        f"    NA           : {cs['NA']}/{n}  ({pct(cs['NA'])})\n\n"
        f"  DIALECT preservation (when applicable)\n"
        f"    yes (good)   : {dia['yes']}/{n}  ({pct(dia['yes'])})\n"
        f"    no  (bad)    : {dia['no']}/{n}  ({pct(dia['no'])})\n"
        f"    NA           : {dia['NA']}/{n}  ({pct(dia['NA'])})\n"
    )
    (args.out_dir / "summary.txt").write_text(
        summary, encoding="utf-8")
    print("\n" + summary)
    print(f"cleaned JSONL  -> {cleaned_path}")
    print(f"judge JSONL    -> {judge_path}")
    print(f"\nFor stratified human eyeball:")
    print(f"  uv run python scripts/eyeball_cleanup_nogt.py")


if __name__ == "__main__":
    main()
