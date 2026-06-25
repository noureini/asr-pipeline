"""Model-agnostic Bengali capability smoke test for any GGUF LLM.

Three tests:
  1. Generate a Bengali sentence about a coastal farmer
  2. Correct a garbled Bengali ASR line (must preserve dialect forms
     নাই / করতেছি / তাইনে AND code-switched English NGO / ward / project)
  3. Reconcile a real (titu_text, zipa_ipa) pair from our outputs
     into a single coherent Bengali+English transcript

Defaults to BanglaLLM/bangla-llama-7b-instruct-v0.1-GGUF (~4.2 GB
Q4_K_M, fits 8 GB GPU, Bengali-native instruction-tuned Llama-2-7B).
Override with --model-repo / --model-file to test other candidates.

Usage:
  # Default — BanglaLLM-7B
  uv run python scripts/test_bengali_llm.py

  # Try a different candidate
  uv run python scripts/test_bengali_llm.py \\
    --model-repo KillerShoaib/llama-3-8b-bangla-GGUF-Q4_K_M \\
    --model-file llama-3-8b-bangla-Q4_K_M.gguf
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

DEFAULT_REPO = "BanglaLLM/bangla-llama-7b-instruct-v0.1-GGUF"
DEFAULT_FILE = None  # auto-pick Q4_K_M


def find_gguf(repo_id: str, prefer: str = "Q4_K_M") -> str:
    """List files in the repo and pick the preferred quant."""
    from huggingface_hub import HfApi
    api = HfApi()
    files = api.list_repo_files(repo_id)
    ggufs = [f for f in files if f.endswith(".gguf")]
    if not ggufs:
        raise RuntimeError(f"no .gguf files in {repo_id}")
    pref = [f for f in ggufs if prefer in f]
    if pref:
        return sorted(pref, key=len)[0]
    return sorted(ggufs, key=len)[0]


def load_real_titu_ipa() -> tuple[str, str, int]:
    """Pull one real (titu, ZIPA) pair from existing outputs for test 3."""
    titu_path = (REPO / "outputs/real_eval_titu_excellent"
                 / "15_Sunamganj_CA-MMI_URBAN_Male_15_0602_1635"
                 / "15_Sunamganj_CA-MMI_URBAN_Male_15_0602_1635"
                    ".titu_lm.jsonl")
    zipa_path = (REPO / "outputs/real_eval_titu_excellent"
                 / "15_Sunamganj_CA-MMI_URBAN_Male_15_0602_1635"
                 / "15_Sunamganj_CA-MMI_URBAN_Male_15_0602_1635"
                    ".zipa.jsonl")
    if not titu_path.exists() or not zipa_path.exists():
        return ("ক্ষুদ্র পৌরসা ঠিক আছে এই এলাকাতে বেশ ভাগ গরিবের",
                "▁ k u d u p o ɾ u ʂ ɔ m a ː t ̪ i k a ː s e", -1)
    titu_rows = {}
    for ln in titu_path.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if ln:
            r = json.loads(ln)
            titu_rows[r["idx"]] = r["text"]
    zipa_rows = {}
    for ln in zipa_path.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if ln:
            r = json.loads(ln)
            zipa_rows[r["idx"]] = r["ipa"]
    common = sorted(set(titu_rows) & set(zipa_rows))
    if not common:
        return (titu_rows.popitem()[1] if titu_rows else "",
                zipa_rows.popitem()[1] if zipa_rows else "", -1)
    idx = common[0]
    return titu_rows[idx], zipa_rows[idx][:400], idx


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-repo", default=DEFAULT_REPO)
    ap.add_argument("--model-file", default=DEFAULT_FILE)
    ap.add_argument("--n-gpu-layers", type=int, default=-1,
                    help="-1 = all on GPU; 0 = CPU only")
    ap.add_argument("--n-ctx", type=int, default=4096)
    ap.add_argument("--max-tokens", type=int, default=2048,
                    help="non-think: 1024 plenty; --think: 6000+ "
                         "recommended (Qwen3.5 reasoning is verbose)")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--think", action="store_true",
                    help="enable Qwen3 thinking mode (chain-of-thought "
                         "wrapped in <think>...</think> before answer)")
    args = ap.parse_args()

    try:
        from llama_cpp import Llama
    except Exception as e:
        sys.exit(f"missing llama-cpp-python: {e}")

    from huggingface_hub import hf_hub_download

    # Resolve GGUF filename if not specified
    file_name = args.model_file
    if file_name is None:
        print(f"auto-picking Q4_K_M GGUF from {args.model_repo} ...",
              flush=True)
        try:
            file_name = find_gguf(args.model_repo, prefer="Q4_K_M")
            print(f"  picked: {file_name}")
        except Exception as e:
            sys.exit(f"could not auto-pick GGUF: {e}\n"
                     f"specify --model-file explicitly")

    print(f"downloading {file_name} from {args.model_repo} ...",
          flush=True)
    path = hf_hub_download(repo_id=args.model_repo, filename=file_name)
    print(f"  -> {path}")

    print("loading model ...", flush=True)
    llm = Llama(
        model_path=path,
        n_ctx=args.n_ctx,
        n_gpu_layers=args.n_gpu_layers,
        verbose=False,
    )
    print(f"  model loaded\n")

    # ----- gather test 3 inputs -----
    titu_text, ipa_text, real_idx = load_real_titu_ipa()
    if real_idx >= 0:
        print(f"  using real (titu, ZIPA) from stem 15 idx {real_idx}\n")

    # English instructions per Qwen3 model card: "in multilingual
    # contexts, write instructions in English". For Bengali OUTPUT we
    # pin the constraint twice (positive + negative) and add an
    # explicit code-switch-preservation clause — the documented
    # pattern that prevents English drift.
    garbled = ("ক্ষুদ্র পৌরসা ঠিক আছে এই এলাকাতে বেশ ভাগ গরিবের "
               "ward number এক")
    bn_tests = [
        ("Test 1 — generate Bengali",
         "Write ONE Bengali sentence (in Bangla Unicode script) "
         "about a farmer in coastal Bangladesh discussing flooding.\n"
         "Output requirements:\n"
         "  - Must be valid Bengali (Bangla script, Unicode)\n"
         "  - Do NOT translate to English; do NOT romanize\n"
         "  - Output only the sentence, no preamble, no explanation",
         ""),

        ("Test 2 — fix garbled Bengali, preserve dialect/code-switch",
         "Correct this garbled Bengali ASR transcript into one "
         "coherent Bengali sentence.\n"
         "STRICT rules:\n"
         "  - Output must be valid Bengali (Bangla script, Unicode)\n"
         "  - Do NOT translate to English; do NOT romanize\n"
         "  - Preserve dialect forms (নাই, করতেছি, তাইনে) — do not "
         "standardize\n"
         "  - Preserve English code-switch tokens (ward, number, "
         "NGO, project, vaccine) VERBATIM in Latin script\n"
         "  - Output only the corrected sentence, no explanation",
         garbled),

        ("Test 3 — reconcile titu + IPA (the real task)",
         "You are correcting a Bengali speech transcript from a "
         "household survey in coastal Bangladesh.\n\n"
         "Two noisy signals are given for the SAME audio:\n"
         "  HYPOTHESIS  — Bengali ASR guess (often wrong, "
         "especially on word boundaries, endings, rare words, and "
         "English code-switch terms)\n"
         "  PHONEMES    — IPA phoneme transcription = exact "
         "sound-by-sound representation of what was said. Read "
         "this like a pronunciation guide.\n\n"
         "===== HOW TO READ IPA FOR BENGALI =====\n"
         "Word boundary: ▁  (each ▁ separates one spoken word)\n"
         "Vowels:  a/ɔ=অ/আ, e=এ, i=ই, o=ও, u=উ, ː=long vowel\n"
         "Consonants:\n"
         "  k=ক  kʰ=খ  g=গ  gʰ=ঘ\n"
         "  t̪=ত  tʰ=থ  d̪=দ  d̪ʰ=ধ      (dental, written with ̪)\n"
         "  ʈ=ট  ʈʰ=ঠ  ɖ=ড  ɖʰ=ঢ        (retroflex)\n"
         "  p=প  pʰ=ফ  b=ব  bʰ=ভ\n"
         "  t͡ʃ=চ  d͡ʒ=জ  s=স  ʃ=শ  ʂ=ষ  h=হ\n"
         "  m=ম  n̪=ন  ŋ=ঙ  r=র  ɾ=র  l=ল  j=য়\n"
         "Aspiration: ʰ after a consonant = aspirated (add ্হ)\n\n"
         "===== CODE-SWITCHED ENGLISH IN IPA =====\n"
         "English words in survey speech appear in IPA too. "
         "Recognize them by their English-like phonetic shape:\n"
         "  /w a r d/        → ward       (NOT ওয়ার্ড or any Bengali word)\n"
         "  /n a m b a r/    → number     (English, not নাম্বার)\n"
         "  /ɛ n d͡ʒ i o/    → NGO        (each letter spelled out)\n"
         "  /h o s p i t a l/ → hospital   (English)\n"
         "  /v æ k s i n/    → vaccine    (English)\n"
         "When PHONEMES match an English shape, output the English "
         "word in Latin script — do NOT transliterate to Bengali.\n\n"
         "===== STRATEGY =====\n"
         "1. Walk PHONEMES word by word (split on ▁)\n"
         "2. For each phoneme group, identify the most likely word:\n"
         "   — does it phonetically match an English code-switch term?\n"
         "   — otherwise, map to the Bengali word with the matching sound\n"
         "3. Compare to HYPOTHESIS:\n"
         "   — where they AGREE, keep HYPOTHESIS's word\n"
         "   — where they DISAGREE, trust PHONEMES (use the IPA-"
         "derived word)\n"
         "4. Preserve coastal-Bangladesh dialect forms if PHONEMES "
         "supports them (নাই, করতেছি, তাইনে)\n\n"
         "===== EXAMPLE =====\n"
         "  HYPOTHESIS: আমি ওয়ার ছাত্র এখানে\n"
         "  PHONEMES:   ▁ a m i ▁ w a r d ▁ n a m b a r ▁ k ɔ t̪ ▁\n"
         "  Walking PHONEMES:\n"
         "    ▁ a m i  → 'ami' → আমি  (agrees with HYP)\n"
         "    ▁ w a r d → English 'ward' (HYP said 'ওয়ার' — close "
         "but wrong, IPA wins)\n"
         "    ▁ n a m b a r → English 'number' (HYP had 'ছাত্র' — "
         "totally wrong, IPA wins)\n"
         "    ▁ k ɔ t̪ → 'kot' → কত (HYP had 'এখানে' — wrong)\n"
         "  OUTPUT: আমি ward number কত\n\n"
         "===== STRICT OUTPUT RULES =====\n"
         "- One single line\n"
         "- Bengali words in Bangla Unicode script\n"
         "- English code-switch words verbatim in Latin script\n"
         "- Do NOT include IPA, explanations, rationales, or "
         "'OUTPUT:' prefix\n"
         "- Just the corrected transcript line",
         f"HYPOTHESIS: {titu_text}\nPHONEMES:   {ipa_text}\nOUTPUT:"),
    ]

    # BanglaLLM-7B: Bangla-Alpaca format with Bengali preamble (verified
    # from the Bangla-Alpaca-Orca dataset card). Llama-3 / Qwen3 variants
    # use ChatML — auto-switch on filename.
    fname_lower = file_name.lower()
    if any(tag in fname_lower for tag in
            ("llama-3", "qwen3", "gemma", "tigerllm", "mistral",
             "qwen2.5")):
        chat_mode = "chat"
    else:
        chat_mode = "bn_alpaca"
    print(f"  prompt mode: {chat_mode}\n")

    # Bengali preambles from the actual training dataset
    PREAMBLE_NO_INPUT = (
        "এখানে একটি নির্দেশনা দেওয়া হলো, যা একটি কাজ সম্পন্ন করার "
        "উপায় বর্ণনা করে। একটি উত্তর লিখুন যা অনুরোধটি সঠিকভাবে "
        "পূরণ করে।")
    PREAMBLE_WITH_INPUT = (
        "এখানে একটি নির্দেশনা দেওয়া হলো, যা একটি কাজ সম্পন্ন করার "
        "উপায় বর্ণনা করে, এবং এর সাথে একটি ইনপুট দেওয়া হলো যা "
        "আরও প্রেক্ষাপট প্রদান করে। একটি উত্তর লিখুন যা অনুরোধটি "
        "সঠিকভাবে পূরণ করে।")

    def bn_alpaca(instruction: str, inp: str) -> str:
        if inp.strip():
            return (f"{PREAMBLE_WITH_INPUT}\n\n"
                    f"### Instruction: {instruction}\n\n"
                    f"### Input: {inp}\n\n"
                    f"### Response:")
        else:
            return (f"{PREAMBLE_NO_INPUT}\n\n"
                    f"### Instruction: {instruction}\n\n"
                    f"### Response:")

    stop_seqs = ["\n### Instruction:", "\n### Input:", "\n### Response:",
                 "</s>", "\n\n\n"]

    for name, instruction, inp in bn_tests:
        print("=" * 76)
        print(name)
        print("-" * 76)
        print("INSTRUCTION (bn):")
        print(instruction[:200] + ("..." if len(instruction) > 200 else ""))
        if inp:
            print(f"\nINPUT: {inp[:200]}")
        try:
            if chat_mode == "bn_alpaca":
                full_prompt = bn_alpaca(instruction, inp)
                out = llm.create_completion(
                    prompt=full_prompt,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_p=0.9,
                    repeat_penalty=1.1,
                    stop=stop_seqs,
                )
                text = out["choices"][0]["text"].strip()
            else:
                # Manually render Qwen3.5 chat template — avoids the
                # transformers/torch CUDA ABI issue. The template is
                # just ChatML with an optional `<think>\n` prefix
                # for enable_thinking, exactly as documented in the
                # Qwen3.5 model card.
                msg = instruction + (f"\n\n{inp}" if inp else "")
                rendered = (
                    "<|im_start|>user\n" + msg + "<|im_end|>\n"
                    "<|im_start|>assistant\n"
                )
                if args.think:
                    # Open thinking block — model continues reasoning
                    rendered += "<think>\n"
                else:
                    # Force-close thinking immediately. This is what
                    # Qwen3.5's chat template does when
                    # enable_thinking=False — without this, the model
                    # is free to spontaneously emit <think> on complex
                    # tasks and we run out of token budget.
                    rendered += "<think>\n\n</think>\n\n"
                # Per Qwen docs: thinking sampling = temp=0.6,
                # top_p=0.95, top_k=20, min_p=0. NO greedy.
                if args.think:
                    # Qwen-prescribed sampling for thinking mode.
                    temp, top_p, top_k = 0.6, 0.95, 20
                else:
                    temp, top_p, top_k = args.temperature, 0.9, 40
                max_tok = args.max_tokens  # honor user value verbatim
                out = llm.create_completion(
                    prompt=rendered,
                    max_tokens=max_tok,
                    temperature=temp,
                    top_p=top_p,
                    top_k=top_k,
                    repeat_penalty=1.05,
                    stop=["<|im_end|>", "<|endoftext|>"],
                )
                raw = out["choices"][0]["text"]
                if args.think and "</think>" in raw:
                    parts = raw.split("</think>", 1)
                    think_block = (parts[0]
                                    .replace("<think>", "")
                                    .strip())
                    text = parts[1].strip()
                    print("\nTHINKING:")
                    print(think_block[:800] +
                          ("..." if len(think_block) > 800 else ""))
                else:
                    text = raw.strip()
        except Exception as e:
            text = f"[ERR: {type(e).__name__}: {e}]"
        print("\nOUTPUT:")
        print(text)
        print()

    print("=" * 76)
    print("Eyeball checklist:")
    print("  (1) Test 1: did it produce coherent Bengali script?")
    print("      (Not Hindi/Devanagari, not English, real Bengali words)")
    print("  (2) Test 2: did it KEEP dialect (নাই) and English "
          "(ward/number)?")
    print("  (3) Test 3: did it produce a plausible reconciliation?")
    print()
    print(f"Model tested: {args.model_repo}/{file_name}")


if __name__ == "__main__":
    main()
