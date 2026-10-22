"""Two-witness fusion + translation (production stage).

The validated post-ASR architecture (ROVER complementary-systems +
generative error correction + GenTranslate):

  CTC witness (phonetically faithful) ‖ Whisper witness (seq2seq, fluent)
     → Gemma-3-12B fusion judge   (reconcile the two witnesses into one clean
                                    native line AND translate to English [en]
                                    AND label speaker roles, in ONE call)

  Whisper witness uses CT2/faster-whisper (batched N-best, ~1.9x faster) with
  an HF transformers fallback.

The witnesses, prompt, output tag and script rules come from the language
profile (languages/<code>.yaml; Bengali: titu ‖ regional Whisper). The CTC
witness is the pipeline's primary engine, so its per-segment text arrives as
AlignedSegment.raw_text. This stage runs Whisper as the SECOND witness on
each segment's audio slice, fuses, translates and labels roles.

Fully local, fully open. No frontier API. Empirically beats single-ASR +
LLM-correction: recovers semantic near-misses (flood vs "sister-no"),
fixes proper nouns from the phonetic witness (Pubali Bank, SSC), and
fills unclear spans with real words instead of guessing.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

from asr_pipeline.models import AlignedSegment, ProcessedSegment
from asr_pipeline.profiles import REPO_ROOT, LanguageProfile, load_profile

logger = logging.getLogger("asr_pipeline")


# Neutral system labels (never "titu"/"whisper" — those leak into the
# Bengali output as তিতু etc.).
# Negative constraints ("never X", "do not Y") are phrased as POSITIVE
# directives ("do Z") on purpose: negation is the instruction class LLMs
# follow least reliably, and 4-bit quantization (this is a Q4_K_XL GGUF)
# degrades instruction-following the most — the two compound. The
# deterministic GER + foreign-script guards still backstop the highest-stakes
# rules (numbers, script) regardless of how well the model follows the text.
# ONE negation is kept deliberately — "not a copy of either" — because it is
# a load-bearing anti-elaboration brake: positivizing it to "combine their
# strengths" let Gemma confabulate a fabricated (but plausible) sentence on a
# near-empty "okay okay" segment at temp=1.0. The "don't just copy" framing
# is what keeps it grounded, so it stays.
_FUSION_PROMPT = """\
You reconstruct ONE utterance from a climate-disaster household survey in \
coastal Bangladesh. The speaker uses colloquial coastal Bengali \
(Barishal/Noakhali/Chittagong belt): expect ক→খ shifts and forms like \
হিয়া (=সেটা/ওটা). The inputs carry internal labels — treat them as \
scaffolding and output only the spoken words.

You get two noisy machine readings of the SAME audio, both ordinary \
Bengali text. Produce a better reconciliation, not a copy of either:
- LITERAL = a CTC transcriber: acoustically literal, weak language model, \
so it preserves how an unusual word actually SOUNDED rather than \
normalizing it. May be garbled or run-together where unsure. Trust it for \
names, numbers, rare/dialect words.
- FLUENT = a seq2seq transcriber: well-formed grammar and structure, but \
its strong language model can swap a content word for a plausible WRONG \
one. May list candidates (a)/(b)/(c); pick the one whose form best \
matches LITERAL given context.
Conflict rule (procedure, in order): (1) take FLUENT's sentence \
segmentation and clause boundaries as the skeleton — your [bn] must have \
sentence breaks even when LITERAL is one long unbroken run; (2) clause by \
clause, take the content word from LITERAL where the two differ — UNLESS \
the FLUENT candidates (a)/(b)/(c) agree with each other AND LITERAL is \
garbled/unintelligible at that span, in which case take the FLUENT \
consensus (several FLUENT readings agreeing outweigh a single garbled \
LITERAL); (3) keep FLUENT's structure while taking LITERAL's word choices \
where LITERAL is the clearer reading; keep every content word grounded in \
at least one witness.

DOMAIN PRIORS (household-roster + shocks instrument; prefer these when the \
reading is ambiguous):
- খানা = household (transcribe থানা-like sounds as খানা here); সদস্য = \
member; বর্গা/আধি = sharecropping; দিনমজুরি = daily wage; কৃষি/চাষ = farming
- shocks: বন্যা flood, ঘূর্ণিঝড় cyclone, জলোচ্ছ্বাস surge, লবণাক্ততা \
salinity, ফসল মার খাওয়া = crop destroyed
- in a date context, read বাইশ/তেইশ as a year (2022/23)

NAMES/PII: spoken personal names → [নাম]/[name]; place names verbatim if \
clear, else [স্থান]/[place]. For a garbled name, output the placeholder \
[নাম]/[name].

NUMBERS (critical, BOTH lines): render each number exactly as spoken, \
digit for digit; mark any number you cannot make out as [অস্পষ্ট]/\
[unclear]. Parse by structure: দেড়=1.5, আড়াই=2.5, সাড়ে X = X.5; -শো = \
×100 (আড়াইশো=250); হাজার=×1000; লাখ=×100000.

SCRIPT: write every word in Bengali script, keeping only genuine English \
code-switches (NGO, loan, bank name) in Latin. If unsure whether a word is \
English, write it in Bengali. For an unclear Bengali word, keep it in \
Bengali script or mark [অস্পষ্ট].

ROLES: label each speaker turn from WHAT IS SAID. ENUMERATOR = the \
interviewer: asks the survey questions, reads the consent/recording \
script, gives instructions, probes. RESPONDENT = the interviewee: answers, \
describing their own family, livelihood, age, floods, crops, money, land. \
Most utterances are one turn. When the utterance holds a question AND its \
answer, write two turns in spoken order, splitting only where the speaker \
changes and keeping every word in exactly one turn.

Output a strippable <think> block (1–2 short sentences, no more), then \
one labelled turn per speaker, each turn exactly three lines:
<think> who is speaking + what's asked/answered (and where the turn \
changes, if it does); any content word you overrode and why — keep it to \
1–2 short sentences </think>
@ENUMERATOR or @RESPONDENT
[bn] clean coherent Bengali reconciling both readings; unrecoverable → \
[অস্পষ্ট]
[en] natural meaning-faithful English of YOUR [bn]; render it entirely in \
English, keeping only genuine code-switches (place/proper names); \
unrecoverable → [unclear]
On the [bn] and [en] lines, write ONLY the final reconciled text as plain \
prose; put any reasoning, arrows, or alternatives in the reasoning block \
above.

LITERAL: $literal
FLUENT (candidate readings): $fluent"""


_ROLES = ("ENUMERATOR", "RESPONDENT")


def load_prompts(prompts_file: Optional[str]) -> dict:
    """Load the fusion prompt template from a YAML file, falling back to
    the built-in default. Production: edit the YAML, not the code."""
    prompts = {"fusion_prompt": _FUSION_PROMPT}
    if not prompts_file:
        return prompts
    p = Path(prompts_file)
    if not p.is_absolute():
        # resolve relative to repo root (two levels up from this file)
        p = Path(__file__).resolve().parents[2] / prompts_file
    if not p.exists():
        logger.warning("  prompts file not found (%s) — using built-in "
                       "defaults", p)
        return prompts
    try:
        import yaml
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        for key in ("fusion_prompt",):
            if isinstance(data.get(key), str) and data[key].strip():
                prompts[key] = data[key]
        logger.info("  prompts loaded from %s", p)
    except Exception as ex:
        logger.warning("  failed to read prompts file (%s) — using "
                       "built-in defaults", type(ex).__name__)
    return prompts


def _strip_think(text: str) -> str:
    """Return the post-reasoning body: drop everything up to and including
    </think> so a stray "[bn]"/"@ROLE" inside the reasoning can't match. A
    <think> opened but never closed (token cap) drops to empty — its tail
    can't contain the real output lines."""
    if "</think>" in text:
        return text.split("</think>", 1)[1]
    if "<think>" in text:
        return text.split("<think>", 1)[0]
    return text


def _parse_bn_en(text: str, tag: str = "bn") -> tuple[str, str]:
    """Pull the native-language line ([bn], [ne] … per the profile's `tag`)
    and the [en] line from one turn's text (already <think>- and
    @ROLE-delimited by the caller; strip any stray <think> too)."""
    if not text:
        return "", ""
    text = _strip_think(text)
    bn = re.search(rf"\[{re.escape(tag)}\]\s*(.+?)(?=\n*\[en\]|\Z)",
                   text, re.S)
    en = re.search(r"\[en\]\s*(.+)", text, re.S)
    bn_t = re.sub(r"\s+", " ", bn.group(1)).strip() if bn else ""
    en_t = re.sub(r"\s+", " ", en.group(1)).strip() if en else ""
    return bn_t, en_t


# Speaker-turn marker the fusion emits before each turn's [bn]/[en].
_ROLE_MARK = re.compile(r"@\s*(ENUMERATOR|RESPONDENT)\b", re.I)


def _parse_fusion(text: str, tag: str = "bn") -> tuple[str, list[tuple]]:
    """Parse a fusion reply into (think, turns).

    `turns` is a list of (role, bn, en) — ONE per speaker turn. The prompt
    tags each turn with @ENUMERATOR / @RESPONDENT, so a segment that
    captured BOTH a question and its answer is split into two role-labeled
    turns (the "one timestamp = one speaker" assumption was wrong). A
    single-speaker segment yields one turn; if the model emits no @ROLE
    marker, role is None and the caller assigns a default. Delimiter-based
    (not JSON) so Bengali text needs no escaping — the old split format
    put text inside JSON and broke ~48% of the time.

    `think` is the reasoning block ('' if skipped); it is the audit trail
    for BOTH the reconciliation and the role/split decision.
    """
    if not text:
        return "", []
    m = re.search(r"<think>(.*?)</think>", text, re.S)
    think = re.sub(r"\s+", " ", m.group(1)).strip() if m else ""
    body = _strip_think(text)
    marks = list(_ROLE_MARK.finditer(body))
    turns: list[tuple] = []
    if marks:
        for j, mk in enumerate(marks):
            end = marks[j + 1].start() if j + 1 < len(marks) else len(body)
            bn, en = _parse_bn_en(body[mk.end():end], tag)
            if bn or en:
                turns.append((mk.group(1).upper(), bn, en))
    else:                                  # no @ROLE → one untagged turn
        bn, en = _parse_bn_en(body, tag)
        if bn or en:
            turns.append((None, bn, en))
    return think, turns


def _split_span(start: float, end: float,
                bns: list[str]) -> list[tuple[float, float]]:
    """Divide [start, end] across N turns proportionally to each turn's
    Bengali length (more text ≈ spoke longer). Exact endpoints at the
    boundaries; a single turn keeps the whole span."""
    if len(bns) <= 1:
        return [(start, end)]
    lens = [max(1, len((b or "").strip())) for b in bns]
    total = sum(lens)
    dur = max(0.0, (end or 0.0) - (start or 0.0))
    spans, t = [], start
    for L in lens[:-1]:
        nxt = t + dur * (L / total)
        spans.append((t, nxt))
        t = nxt
    spans.append((t, end))
    return spans


def _split_sentences(text: str) -> list[str]:
    """Split into sentences on Bengali danda (।), ?, ! and Latin '.' —
    keeping the delimiter. Used by the role stage so the model can name a
    split POINT by sentence index (it never re-emits the text). Always
    returns at least one item for non-empty input."""
    t = (text or "").strip()
    if not t:
        return []
    parts = re.split(r"(?<=[।?!.])\s+", t)
    return [p.strip() for p in parts if p.strip()] or [t]


def _strip_think_instruction(prompt: str) -> str:
    """Derive the think-OFF prompt from the think-ON one, for the matched
    ablation arm — so think-off is a one-flag change on the same audio,
    not a hand-edited prompt. Drops the <think> example line and softens
    the preamble to "Output exactly two lines:". Robustness is enforced by
    test_fusion_think_off_has_no_think_tag (no '<think>' may survive, and
    placeholders must still fill)."""
    # soften the preamble first (its own <think> goes with it)
    out = re.sub(r"a strippable <think> block[^\n]*?then ", "", prompt)
    # then drop the standalone <think> ... </think> example line
    out = re.sub(r"(?m)^<think>.*?</think>[ \t]*\n?", "", out, flags=re.S)
    return out


class FusionTranslator:
    """Whisper (2nd witness) → Gemma-3 fusion + translation (one call).

    Models load lazily and sequentially (load → use → unload) so the
    whole chain fits on a single 8 GB GPU.
    """

    def __init__(
        self,
        profile: Optional[LanguageProfile] = None,
        device: str = "cuda",
        gguf_repo: str = "unsloth/gemma-3-12b-it-GGUF",
        gguf_file: str = "gemma-3-12b-it-UD-Q4_K_XL.gguf",
        gguf_revision: Optional[str] = None,
        gguf_n_gpu_layers: int = -1,
        # 0 = OFF. Must stay 0 for the in-process gemma backend: prompt-lookup
        # speculative decoding crashes llama-cpp here ("could not broadcast
        # ... into shape (0,)") and silently drops fusion to the heuristic
        # fallback. Matches FusionConfig's default; do not raise without testing.
        prompt_lookup_tokens: int = 0,
        whisper_batch_size: int = 8,
        whisper_nbest: int = 3,
        temperature: float = 1.0,
        top_p: float = 0.95,
        top_k: int = 64,
        min_p: float = 0.0,
        repeat_penalty: float = 1.0,
        seed: int = 1234,
        n_ctx: int = 2048,
        flash_attn: bool = True,
        kv_quant: bool = True,
        fusion_max_tokens: int = 1024,
        fusion_think: bool = True,
        ger_guard: bool = True,
        ger_snap: bool = False,
        ger_snap_tau: float = 0.72,
        fusion_backend: str = "inprocess",
        n_parallel: int = 3,
        server_port: int = 8081,
    ) -> None:
        # Language-specific parts: Whisper witness, prompt, tag, scripts.
        self._profile = profile or load_profile("ben")
        ww = self._profile.witness_whisper
        self._tag = self._profile.tag
        self._whisper_model = ww.model
        self._whisper_revision = ww.revision
        self._whisper_language = ww.language
        # CT2 copy of the Whisper witness, relative to the repo root.
        self._whisper_ct2_path = (
            str(ww.ct2_path if Path(ww.ct2_path).is_absolute()
                else REPO_ROOT / ww.ct2_path) if ww.ct2_path else "")
        # GER guard: which scripts may appear, what replaces unclear text.
        self._script_rules = {
            "scripts": tuple(self._profile.scripts),
            "unclear_marker": self._profile.unclear_marker,
        }
        self._device = device
        self._gguf_repo = gguf_repo
        self._gguf_file = gguf_file
        self._gguf_revision = gguf_revision
        self._gguf_n_gpu_layers = gguf_n_gpu_layers
        self._prompt_lookup_tokens = prompt_lookup_tokens
        self._whisper_batch_size = whisper_batch_size
        self._whisper_nbest = max(1, whisper_nbest)
        self._fusion_backend = fusion_backend
        self._n_parallel = n_parallel
        self._server_port = server_port
        # Generation hyperparameters (Gemma-team config + fixed seed →
        # reproducible). Production runs greedy (temperature 0).
        self._temperature = temperature
        self._top_p = top_p
        self._top_k = top_k
        self._min_p = min_p
        self._repeat_penalty = repeat_penalty
        self._seed = seed
        self._n_ctx = n_ctx
        self._flash_attn = flash_attn
        self._kv_quant = kv_quant
        self._fusion_max_tokens = fusion_max_tokens
        self._ger_enabled = ger_guard
        self._ger_snap = ger_snap
        self._ger_snap_tau = ger_snap_tau
        # fusion stat accumulators (also reset per-interview in process();
        # init here so _run_fusion is safe to call standalone in scripts).
        self._reset_fusion_stats()
        # Judge prompt from the profile (operational config — editable YAML).
        # No silent fallback: the built-in prompt is Bengali-only.
        prompt_path = self._profile.prompt_path
        if not prompt_path.exists():
            raise FileNotFoundError(
                f"Prompt file for '{self._profile.code}' not found: "
                f"{prompt_path} (set prompt_file in "
                f"languages/{self._profile.code}.yaml)")
        prompts = load_prompts(str(prompt_path))
        self._fusion_prompt = prompts["fusion_prompt"]
        # Ablation: think-on (default) vs the matched think-off arm. Derived
        # from the same prompt so the only difference is the <think> block.
        self._fusion_think_on = fusion_think
        if not fusion_think:
            self._fusion_prompt = _strip_think_instruction(self._fusion_prompt)
            logger.info("  fusion: think-OFF arm (no <think> block)")

    # ── second witness: Whisper-regional, BATCHED across chunks ───────
    #    Intra-stage parallelism: many chunks per GPU forward pass via
    #    processor → model.generate (NO HF pipeline). The pipeline's
    #    generator/dataset path spins up a DataLoader with PINNED host
    #    memory; on this 15GB-RAM box cudaHostAlloc fails with a *driver*
    #    "out of memory" even with ~7GB VRAM free. Manual batching has no
    #    DataLoader and turns ~30 min one-at-a-time into a few minutes.
    def _run_whisper(self, aligned: list[AlignedSegment],
                     wav_path: Path) -> list[list[str]]:
        """Second witness. Prefer CT2/faster-whisper (batched, ~1.9x faster,
        A/B-validated) when the local CT2 conversion exists; otherwise fall
        back to the HF transformers path."""
        from pathlib import Path as _P
        if self._whisper_ct2_path and _P(self._whisper_ct2_path).exists():
            try:
                return self._run_whisper_ct2(aligned, wav_path)
            except Exception as ex:
                logger.warning("  CT2 Whisper failed (%s) — falling back to "
                               "HF transformers", type(ex).__name__)
        return self._run_whisper_hf(aligned, wav_path)

    def _run_whisper_ct2(self, aligned: list[AlignedSegment],
                         wav_path: Path) -> list[list[str]]:
        """Batched CT2/faster-whisper N-best over the shared-spine spans.
        whisperX-canonical: stack features into one StorageView per batch and
        halve the batch on CUDA OOM. Sampled N-best (matches HF diversity)."""
        import gc
        import ctranslate2
        import numpy as np
        import torch
        import torchaudio
        from faster_whisper import WhisperModel
        from faster_whisper.tokenizer import Tokenizer

        nbest = self._whisper_nbest
        logger.info("  Fusion 1/3: Whisper second witness (CT2 "
                    f"{self._whisper_ct2_path}) — N-best={nbest}")
        use_cuda = (self._device == "cuda" and torch.cuda.is_available())
        if use_cuda:
            gc.collect(); torch.cuda.empty_cache()
        wm = WhisperModel(
            self._whisper_ct2_path,
            device="cuda" if use_cuda else "cpu",
            compute_type="int8_float16" if use_cuda else "int8")
        fe = wm.feature_extractor
        tok = Tokenizer(wm.hf_tokenizer, multilingual=True,
                        task="transcribe", language=self._whisper_language)
        prompt = list(tok.sot_sequence) + [tok.no_timestamps]
        # Seed CT2's RNG so the sampled N-best is reproducible run-to-run
        # (matches the Gemma fixed-seed → fully deterministic deliverable).
        ctranslate2.set_random_seed(self._seed)
        n_frames = 3000  # Whisper encoder is fixed at 30 s

        wav, sr = torchaudio.load(str(wav_path))
        if wav.size(0) > 1:
            wav = wav.mean(0, keepdim=True)
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000); sr = 16000
        wav = wav[0].numpy().astype(np.float32)

        valid_idx, feats = [], []
        for i, seg in enumerate(aligned):
            s = max(0, int(seg.start_s * sr)); e = min(len(wav), int(seg.end_s * sr))
            if e - s < int(0.2 * sr):
                continue
            f = np.asarray(fe(wav[s:e]), dtype=np.float32)
            f = (np.pad(f, ((0, 0), (0, n_frames - f.shape[-1])))
                 if f.shape[-1] < n_frames else f[:, :n_frames])
            valid_idx.append(i); feats.append(f)

        out: list[list[str]] = [[""] for _ in range(len(aligned))]
        items = list(zip(valid_idx, feats))

        def run_batch(batch):
            if not batch:
                return
            try:
                sv = ctranslate2.StorageView.from_array(
                    np.ascontiguousarray(np.stack([f for _, f in batch])))
                res = wm.model.generate(
                    sv, [prompt] * len(batch),
                    num_hypotheses=nbest, beam_size=1,
                    sampling_temperature=0.5 if nbest > 1 else 1.0,
                    sampling_topk=0, return_scores=False)
                for (oi, _), r in zip(batch, res):
                    hyps, seen = [], set()
                    for ids in r.sequences_ids:
                        txt = tok.decode(ids).strip()
                        if txt and txt not in seen:
                            seen.add(txt); hyps.append(txt)
                    out[oi] = hyps or [""]
            except RuntimeError as ex:
                if "out of memory" in str(ex).lower() and len(batch) > 1:
                    if use_cuda:
                        torch.cuda.empty_cache()
                    mid = len(batch) // 2
                    run_batch(batch[:mid]); run_batch(batch[mid:])
                else:
                    raise

        bs = max(1, self._whisper_batch_size)
        for b in range(0, len(items), bs):
            run_batch(items[b:b + bs])

        n_ok = sum(1 for g in out if any(g))
        logger.info("    whisper(CT2) done: %d/%d segments produced text",
                    n_ok, len(aligned))
        del wm
        if use_cuda:
            torch.cuda.empty_cache()
        return out

    def _run_whisper_hf(self, aligned: list[AlignedSegment],
                        wav_path: Path) -> list[list[str]]:
        import torch
        import torchaudio
        from transformers import (AutoProcessor,
                                  AutoModelForSpeechSeq2Seq)

        nbest = self._whisper_nbest
        logger.info("  Fusion 1/3: Whisper second witness (HF) "
                    f"({self._whisper_model}) — N-best={nbest}")
        use_cuda = (self._device == "cuda" and torch.cuda.is_available())
        dev = "cuda" if use_cuda else "cpu"
        dtype = torch.float16 if use_cuda else torch.float32
        if use_cuda:
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            free_b, total_b = torch.cuda.mem_get_info()
            logger.info("    GPU free: %.2f / %.2f GB",
                        free_b / 1e9, total_b / 1e9)

        processor = AutoProcessor.from_pretrained(
            self._whisper_model, revision=self._whisper_revision)
        # SDPA attention (PyTorch-native flash-attention path): ~1.5-2x
        # faster + lower memory than eager, no flash-attn build needed.
        # The "insanely fast whisper" win that applies to our exact
        # regional model. Fall back to eager if unsupported.
        try:
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                self._whisper_model, revision=self._whisper_revision,
                torch_dtype=dtype, attn_implementation="sdpa")
        except Exception:
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                self._whisper_model, revision=self._whisper_revision,
                torch_dtype=dtype)
        model.to(dev).eval()

        wav, sr = torchaudio.load(str(wav_path))
        if wav.size(0) > 1:
            wav = wav.mean(0, keepdim=True)
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
            sr = 16000

        # Collect valid segment slices, remembering their indices.
        valid_idx, arrays = [], []
        for i, seg in enumerate(aligned):
            s = max(0, int(seg.start_s * sr))
            e = min(wav.shape[1], int(seg.end_s * sr))
            if e - s >= int(0.2 * sr):
                valid_idx.append(i)
                arrays.append(wav[0, s:e].contiguous().numpy())

        def _transcribe(batch: list) -> list[list[str]]:
            """Return up to `nbest` hypotheses per input (beam search)."""
            feats = processor(
                batch, sampling_rate=sr, return_tensors="pt",
                padding=True).input_features.to(dev, dtype)
            with torch.no_grad():
                if nbest > 1:
                    # SEEDED sampling for diverse-yet-reproducible N-best.
                    # Plain beams converge to one text (no diversity);
                    # diverse-beam was removed from transformers. Sampling
                    # gives alternates exactly where the ASR is uncertain
                    # (ward vs world); the fixed seed keeps it
                    # deterministic across runs (no new instability).
                    torch.manual_seed(self._seed)
                    gen = model.generate(
                        feats, do_sample=True, temperature=0.5,
                        top_p=0.9, num_return_sequences=nbest,
                        language=self._whisper_language,
                        task="transcribe", max_new_tokens=256)
                else:
                    gen = model.generate(
                        feats, language=self._whisper_language,
                        task="transcribe", max_new_tokens=256)
            flat = processor.batch_decode(gen, skip_special_tokens=True)
            # generate() returns sequences grouped by input then beam:
            # [in0_b0, in0_b1, ..., in1_b0, ...]. Regroup + dedupe.
            groups: list[list[str]] = []
            for i in range(len(batch)):
                hyps = [flat[i * nbest + j].strip()
                        for j in range(nbest)]
                seen, uniq = set(), []
                for h in hyps:
                    if h and h not in seen:
                        seen.add(h)
                        uniq.append(h)
                groups.append(uniq or [""])
            return groups

        out: list[list[str]] = [[""] for _ in range(len(aligned))]
        bs = max(1, self._whisper_batch_size)
        done = 0
        for b in range(0, len(arrays), bs):
            batch = arrays[b:b + bs]
            try:
                groups = _transcribe(batch)
            except Exception as ex:
                # A normal torch OOM here is recoverable (unlike the
                # DataLoader driver error): clear and retry per-item.
                logger.warning("  whisper batch @%d failed (%s) — "
                               "per-item retry", b, type(ex).__name__)
                if use_cuda:
                    torch.cuda.empty_cache()
                groups = []
                for a in batch:
                    try:
                        groups.append(_transcribe([a])[0])
                    except Exception:
                        groups.append([""])
            for k, g in enumerate(groups):
                out[valid_idx[b + k]] = g
            done += len(batch)
            if done % 40 == 0 or done >= len(arrays):
                logger.info("    whisper %d/%d (batch x%d, N-best %d)",
                            done, len(arrays), bs, nbest)

        n_ok = sum(1 for g in out if any(g))
        logger.info("    whisper done: %d/%d segments produced text",
                    n_ok, len(aligned))
        del model, processor
        if use_cuda:
            torch.cuda.empty_cache()
        return out

    # ── fusion judge: Gemma-3 reconciles the two witnesses AND
    #    translates in the same call (one model, fully local) ──────────
    @staticmethod
    def _fmt_fluent(whyps: list[str]) -> str:
        """Format Whisper's N-best candidate readings for the prompt."""
        whyps = [h for h in whyps if h]
        if not whyps:
            return "(none)"
        if len(whyps) == 1:
            return whyps[0]
        return "\n".join(f"  ({chr(97 + j)}) {h}"
                         for j, h in enumerate(whyps))

    def _finish_segment(self, i: int, ans: str, meta: dict, literal: str,
                        whyps: list[str], error: str = "") -> list[tuple]:
        """Parse one fusion reply into role-labelled turns, record it, and
        run the GER guard. A reply with no usable turn (error, <think>
        overrun) falls back to one FLUENT-raw turn so the segment never
        vanishes. Returns the final [(role, bn, en), …]."""
        wt = whyps[0] if whyps else ""
        think, turns = _parse_fusion(ans, self._tag) if not error else ("", [])
        self._record_fusion(i, think, turns, meta, wt, error=error)
        if not any((t[1] or "").strip() for t in turns):
            turns = [(None, wt, "")]
        turns = self._apply_ger(i, turns, literal, whyps)
        self._fusion_turns[i] = turns
        return turns

    def _run_fusion(self, aligned: list[AlignedSegment],
                    ctc_texts: list[str],
                    whisper_nbest: list[list[str]]
                    ) -> tuple[list[str], list[str]]:
        from asr_pipeline.postprocessor import GGUFProcessor

        logger.info("  Fusion 2/2: Gemma-3 fusion + translation "
                    f"({self._gguf_file})")
        # Aggressively release any VRAM PyTorch (CTC witness/Whisper) still holds
        # before llama.cpp loads Gemma — otherwise the reserved-but-unused
        # torch pool shrinks what's physically free for the 7GB model +
        # KV, forcing PCIe paging (slow decode). Then LOG the actual free
        # VRAM so paging vs. real output-length is no longer a guess.
        try:
            import gc
            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                free_b, total_b = torch.cuda.mem_get_info()
                logger.info("  GPU free before Gemma load: %.2f / %.2f GB",
                            free_b / 1e9, total_b / 1e9)
        except Exception:
            pass
        from string import Template
        tpl = Template(self._fusion_prompt)
        proc = GGUFProcessor(
            gguf_repo=self._gguf_repo, gguf_file=self._gguf_file,
            gguf_revision=self._gguf_revision, n_ctx=self._n_ctx,
            n_gpu_layers=self._gguf_n_gpu_layers,
            prompt_lookup_tokens=self._prompt_lookup_tokens,
            flash_attn=self._flash_attn, kv_quant=self._kv_quant)
        if not proc.load():
            logger.warning("  fusion judge unavailable — falling back to "
                           "CTC witness text unchanged (no translation)")
            return list(ctc_texts), [""] * len(ctc_texts)
        bn_out, en_out = [], []
        for i, (tt, whyps) in enumerate(zip(ctc_texts, whisper_nbest)):
            tt = (tt or "").strip()
            whyps = [h.strip() for h in (whyps or []) if h and h.strip()]
            wt = whyps[0] if whyps else ""   # top whisper hypothesis
            if not tt and not wt:
                bn_out.append("")
                en_out.append("")
                continue
            if not wt:                     # only one witness → pass through
                bn_out.append(tt)
                en_out.append("")
                continue
            prompt = tpl.safe_substitute(literal=tt or "(none)",
                                         fluent=self._fmt_fluent(whyps))
            try:
                # role-labelled [bn] + [en] turns + reasoning in one call;
                # meta carries the truncation signal (finish_reason) and
                # output token count.
                ans, meta = proc.generate(
                    prompt, temperature=self._temperature,
                    top_p=self._top_p, top_k=self._top_k,
                    min_p=self._min_p, repeat_penalty=self._repeat_penalty,
                    seed=self._seed, max_tokens=self._fusion_max_tokens,
                    return_meta=True)
                turns = self._finish_segment(i, ans, meta, tt, whyps)
            except Exception as ex:
                logger.warning("  fusion seg %d failed: %s", i,
                               type(ex).__name__)
                turns = self._finish_segment(i, "", {}, tt, whyps,
                                             error=type(ex).__name__)
            bn_out.append(" ".join(t[1] for t in turns if t[1]))
            en_out.append(" ".join(t[2] for t in turns if t[2]))
            if (i + 1) % 25 == 0:
                logger.info("    fused+translated %d/%d", i + 1,
                            len(ctc_texts))
        proc.unload()
        self._log_ger_summary(len(ctc_texts))
        return bn_out, en_out

    # NOTE: translation is done by Gemma in the fusion call (the [en] line) —
    # there is no separate NMT stage. A former IndicTrans2 _run_translation was
    # removed (it was never called; Gemma [en] was A/B-validated as faithful).

    # ── server-backed (parallel / continuous-batching) fusion + roles ─
    def _run_fusion_server(self, server, aligned, ctc_texts, whisper_nbest):
        from string import Template
        tpl = Template(self._fusion_prompt)
        n = len(ctc_texts)
        bn_out, en_out = [""] * n, [""] * n
        idxs, prompts = [], []
        for i, (tt, whyps) in enumerate(zip(ctc_texts, whisper_nbest)):
            tt = (tt or "").strip()
            whyps = [h.strip() for h in (whyps or []) if h and h.strip()]
            wt = whyps[0] if whyps else ""
            if not tt and not wt:
                continue
            if not wt:
                bn_out[i] = tt
                continue
            idxs.append(i)
            prompts.append(tpl.safe_substitute(
                literal=tt or "(none)", fluent=self._fmt_fluent(whyps)))
        logger.info("  Fusion 2/2 (server, %d slots): %d segments",
                    self._n_parallel, len(prompts))
        answers = server.generate_batch(
            prompts, max_tokens=self._fusion_max_tokens,
            temperature=self._temperature, top_p=self._top_p)
        for j, ans in zip(idxs, answers):
            # server path has no per-request finish_reason → no truncation
            # signal, but the empty→FLUENT coverage save still applies.
            whyps = [h.strip() for h in (whisper_nbest[j] or [])
                     if h and h.strip()]
            turns = self._finish_segment(j, ans, {}, ctc_texts[j], whyps)
            bn_out[j] = " ".join(t[1] for t in turns if t[1])
            en_out[j] = " ".join(t[2] for t in turns if t[2])
        self._log_ger_summary(n)
        return bn_out, en_out

    def _resolve_gguf(self) -> Optional[str]:
        """Local path to the fusion GGUF (for the server), offline-safe."""
        try:
            from huggingface_hub import hf_hub_download
            return hf_hub_download(repo_id=self._gguf_repo,
                                   filename=self._gguf_file,
                                   revision=self._gguf_revision,
                                   local_files_only=True)
        except Exception:
            return None

    # ── confidence flag: should a human review this segment? ──────────
    @staticmethod
    def _review_flag(literal: str, whisper: str, bn: str, en: str,
                     unclear_marker: str = "[অস্পষ্ট]") -> tuple[bool, str]:
        """Decide if a segment needs targeted human QA. Reasons:
        - the two witnesses disagree strongly (fusion had to choose)
        - a number is present (numbers are the faithfulness risk)
        - the fusion marked text unclear / produced empty output
        Returns (needs_review, reason).

        Witness disagreement is NOT flagged here — the GER guard does it at
        grapheme level over the full N-best (script-aware), so flagging it
        again with char-level difflib here would double-count the same
        finding in the review reason."""
        reasons = []
        # number present anywhere (digits of any script) → verify
        if re.search(r"\d", f"{bn} {en}"):
            reasons.append("number")
        # explicit unclear markers (incl. the GER number-abstain marker)
        if unclear_marker in bn or "[unclear]" in en.lower():
            reasons.append("unclear")
        # empty / near-empty output on non-empty input
        t, w = (literal or "").strip(), (whisper or "").strip()
        if (t or w) and len(en.strip()) < 3:
            reasons.append("empty-output")
        return (bool(reasons), ", ".join(reasons))

    def _reset_fusion_stats(self) -> None:
        """Per-interview accumulators for the fusion pass (read by
        process() to log coverage and write the reasoning sidecar)."""
        self._fusion_think: dict[int, str] = {}
        self._fusion_turns: dict[int, list] = {}  # i → [(role, bn, en), …]
        self._fusion_fallback: set[int] = set()   # LLM empty → FLUENT raw
        self._fusion_truncated: set[int] = set()  # hit the token cap
        self._fusion_en_empty: set[int] = set()   # bn ok but no English
        self._fusion_tokens: list[int] = []       # output-token sizes
        # GER guard: per-segment reasons, snapped set, and the (min_dist,
        # witness_agree) metrics that calibrate snap_tau.
        self._ger_reasons: dict[int, list] = {}
        self._ger_snapped: set[int] = set()
        self._ger_metrics: dict[int, tuple] = {}

    def _apply_ger(self, i: int, turns: list[tuple], literal: str,
                   whyps: list[str]) -> list[tuple]:
        """Constrain one fused segment against its witness pool: closest-to-
        N-best snap (when enabled + calibrated), fabricated-number and
        foreign-script abstain, and agreement flags for the review worklist.
        The witnesses cover the WHOLE segment, so distances and flags are
        measured on all turns joined; the abstain rewrites are then applied
        turn by turn so a question/answer split survives the guard. Records
        the distance metrics for calibration. No-op when disabled."""
        bn = " ".join(t[1] for t in turns if t[1])
        if not self._ger_enabled or not bn.strip():
            return turns
        from asr_pipeline import ger_guard
        en = " ".join(t[2] for t in turns if t[2])
        v = ger_guard.assess(
            bn, en, literal, whyps,
            snap=self._ger_snap, snap_tau=self._ger_snap_tau,
            **self._script_rules)
        # Always record the distance metrics — this is the calibration data
        # for setting snap_tau from the slice's distribution.
        self._ger_metrics[i] = (v.min_dist, v.witness_agree)
        if v.reasons:
            self._ger_reasons[i] = v.reasons
        if v.snapped:
            # the snapped witness is one undivided reading → one turn
            self._ger_snapped.add(i)
            return [(turns[0][0], v.bn, v.en)]
        if (v.bn, v.en) == (bn, en):
            return turns
        guarded = []
        for role, tbn, ten in turns:
            vt = ger_guard.assess(tbn, ten, literal, whyps, snap=False,
                                  **self._script_rules)
            guarded.append((role, vt.bn, vt.en))
        return guarded

    def _log_ger_summary(self, n: int) -> None:
        """One-line GER-guard summary for the run (next to fusion coverage).
        In observe-mode (snap off) reports would-snap counts + the min_dist
        distribution so snap_tau can be calibrated from real numbers."""
        if not self._ger_enabled:
            return

        def _count(prefix: str) -> int:
            return sum(1 for rs in self._ger_reasons.values()
                       if any(r.startswith(prefix) for r in rs))

        n_abstain = _count("number-abstain")
        n_dis = _count("witness-disagree")
        n_dev = _count("deviated")
        dists = sorted(m[0] for m in self._ger_metrics.values())
        if self._ger_snap:
            logger.info(
                "  GER guard [snap ON, tau=%.2f]: %d/%d snapped | "
                "number-abstain %d | witness-disagree %d | deviated %d",
                self._ger_snap_tau, len(self._ger_snapped), n, n_abstain,
                n_dis, n_dev)
        else:
            would = _count("would-snap")
            p = (lambda q: dists[min(len(dists) - 1, int(q * len(dists)))]
                 ) if dists else (lambda q: 0.0)
            logger.info(
                "  GER guard [OBSERVE, snap off]: would-snap %d/%d @tau=%.2f | "
                "number-abstain %d | witness-disagree %d | deviated %d | "
                "min_dist p50=%.2f p90=%.2f max=%.2f — calibrate snap_tau "
                "(scripts/ger_calibrate.py) before enabling snap",
                would, n, self._ger_snap_tau, n_abstain, n_dis, n_dev,
                p(0.5), p(0.9), dists[-1] if dists else 0.0)

    def _record_fusion(self, i: int, think: str, turns: list,
                       meta: dict, wt: str, error: str = "") -> None:
        self._fusion_think[i] = think
        self._fusion_turns[i] = turns
        ct = (meta or {}).get("completion_tokens")
        if isinstance(ct, int):
            self._fusion_tokens.append(ct)
        if (meta or {}).get("finish_reason") == "length":
            self._fusion_truncated.add(i)
        has_bn = any((t[1] or "").strip() for t in turns)
        if error or not has_bn:
            # coverage save: no usable Bengali came back from the LLM
            self._fusion_fallback.add(i)
        elif any((t[1] or "").strip() and not (t[2] or "").strip()
                 for t in turns):
            self._fusion_en_empty.add(i)

    def _log_fusion_coverage(self, aligned: list) -> None:
        """Coverage + output-token distribution for this interview. The
        coverage hole the empty-on-truncation behaviour would create lives
        in these numbers, so they are logged every run (and belong next to
        WER per arm in the paper — WER-on-output hides dropped segments)."""
        n = len(aligned)
        fb, tr = self._fusion_fallback, self._fusion_truncated
        en_e = self._fusion_en_empty
        logger.info(
            "  fusion coverage: %d segs | bn-fallback %d (%.1f%%) | "
            "truncated %d | en-empty %d",
            n, len(fb), 100.0 * len(fb) / max(n, 1), len(tr), len(en_e))
        toks = sorted(self._fusion_tokens)
        if toks:
            cap = self._fusion_max_tokens
            p = lambda q: toks[min(len(toks) - 1, int(q * len(toks)))]
            logger.info(
                "  fusion output tokens: p50=%d p90=%d p99=%d max=%d "
                "(cap=%d%s)", p(0.50), p(0.90), p(0.99), toks[-1], cap,
                " — RAISE CAP" if toks[-1] >= cap else "")

        def _ts(sec: float) -> str:
            sec = int(sec or 0)
            return f"{sec // 3600:02d}:{sec % 3600 // 60:02d}:{sec % 60:02d}"
        for i in sorted(fb | tr):
            seg = aligned[i] if i < len(aligned) else None
            t = _ts(getattr(seg, "start_s", 0)) if seg else "??:??:??"
            why = ("fallback" if i in fb else "") + (
                "+truncated" if i in tr else "")
            logger.warning("    fusion %s @ %s (seg %d)", why.strip("+"),
                           t, i)

    # ── orchestration ─────────────────────────────────────────────────
    def process(self, aligned: list[AlignedSegment],
                wav_path: Path) -> list[ProcessedSegment]:
        if not aligned:
            return []
        self._reset_fusion_stats()
        ctc_texts = [s.raw_text for s in aligned]
        # Whisper N-best: list[list[str]] — up to N candidate readings/seg.
        whisper_nbest = self._run_whisper(aligned, wav_path)

        # Backend: "server" = llama-server continuous batching (~4.5x on
        # the N-best workload); falls back to in-process if unavailable.
        server = None
        if self._fusion_backend == "server":
            from asr_pipeline.llama_server import LlamaServer
            model_path = self._resolve_gguf()
            if model_path:
                self._free_vram_for_server()
                server = LlamaServer(
                    model_path, n_parallel=self._n_parallel,
                    n_ctx_per_slot=self._n_ctx, flash_attn=self._flash_attn,
                    kv_quant=self._kv_quant, port=self._server_port)
                if not server.start():
                    logger.warning("  server unavailable — in-process "
                                   "fallback")
                    server.stop()
                    server = None

        # Gemma reconciles the CTC witness (phonetic) + Whisper's N-best (fluent
        # candidates) and translates (one model, fully local).
        if server is not None:
            fused_bn, english = self._run_fusion_server(
                server, aligned, ctc_texts, whisper_nbest)
        else:
            fused_bn, english = self._run_fusion(
                aligned, ctc_texts, whisper_nbest)

        # One ProcessedSegment per role-labelled turn. The fusion call reads
        # WHAT IS SAID and tags each turn @ENUMERATOR / @RESPONDENT, splitting
        # a segment that holds both a question and its answer — no voice
        # clustering. The decision is in the <think> trace (.reasoning.txt).
        if server is not None:
            server.stop()
        processed = []
        sid = 0
        # An untagged turn (fallback / pass-through) continues the previous
        # speaker; the interview opens with the enumerator (consent script).
        last_role = "ENUMERATOR"
        for i, (seg, bn0, en0, whyps) in enumerate(
                zip(aligned, fused_bn, english, whisper_nbest)):
            wt = whyps[0] if whyps else ""   # top whisper hyp for QA flag
            # empty → one FLUENT-raw fallback turn so the (hard) segment
            # never vanishes (coverage stays 100%).
            turns = self._fusion_turns.get(i) or [(None, bn0 or wt, en0)]
            spans = _split_span(seg.start_s, seg.end_s,
                                [t[1] for t in turns])
            fellback = i in self._fusion_fallback
            truncated = i in self._fusion_truncated
            think = self._fusion_think.get(i, "")
            for k, ((role, bn, en), (s0, s1)) in enumerate(zip(turns, spans)):
                if role not in _ROLES:
                    role = last_role
                last_role = role
                review, reason = self._review_flag(
                    seg.raw_text, wt, bn, en, self._profile.unclear_marker)
                if fellback or truncated:
                    review = True
                    tag = ("fusion-fallback" if fellback
                           else "fusion-truncated")
                    reason = f"{reason}, {tag}" if reason else tag
                # GER guard reasons (snap / number-mismatch / agreement) —
                # recorded per fusion segment i, surfaced on its first turn.
                ger = self._ger_reasons.get(i) if k == 0 else None
                if ger:
                    review = True
                    extra = ", ".join(ger)
                    reason = f"{reason}, {extra}" if reason else extra
                if len(turns) > 1:
                    reason = (f"{reason}, split-turn" if reason
                              else "split-turn")
                processed.append(ProcessedSegment(
                    segment_id=sid,
                    start_s=s0,
                    end_s=s1,
                    speaker_id=role,
                    language=seg.language,
                    # the CTC raw text belongs to the whole segment → show it once,
                    # on the first turn only.
                    raw_text=seg.raw_text if k == 0 else "",
                    corrected_text=bn,               # fused Bengali
                    english_translation=en,
                    refined_translation=en,
                    confidence=seg.confidence,
                    needs_review=review,
                    review_reason=reason,
                    reasoning=think,
                    fusion_fallback=fellback,
                    # raw witnesses for the snap audit. FLUENT is the FULL
                    # Whisper N-best (a)/(b)/(c) — not top-1 — so the sort can
                    # tell "fusion ignored an available correct candidate"
                    # (Tier-1 bug) from "the word was never captured" (Tier-3).
                    witness_literal=(seg.raw_text or "") if k == 0 else "",
                    witness_fluent=(self._fmt_fluent(whyps)) if k == 0 else "",
                    ger_min_dist=(self._ger_metrics.get(i, (None, None))[0]
                                  if k == 0 else None),
                    ger_witness_agree=(self._ger_metrics.get(i, (None, None))[1]
                                       if k == 0 else None),
                ))
                sid += 1
        n_flag = sum(1 for s in processed if s.needs_review)
        logger.info("  review flags: %d/%d segments need human QA",
                    n_flag, len(processed))
        self._log_fusion_coverage(aligned)
        return processed

    @staticmethod
    def _free_vram_for_server() -> None:
        """Release any PyTorch VRAM (CTC witness/Whisper) before the server
        loads the model, so the GPU is clear for the slots."""
        try:
            import gc
            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception:
            pass
