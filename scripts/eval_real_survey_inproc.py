"""
In-process real-survey validation — THE production gate, fast path.

Same pre-registered metric as scripts/eval_real_survey.py, but:
  * loads Omnilingual ONCE (no per-part subprocess),
  * runs ONLY preprocessing (VAD chunking) + Omnilingual ASR
    in-process — NO pyannote diarization, NO MMS forced alignment,
    NO LLM post-processing (all irrelevant to a document-level
    CER/WER score, and the cause of the multi-hour stalls / hangs
    in the subprocess-per-part harness),
  * is RESUMABLE: per-part chunk transcriptions are checkpointed to
    JSONL, so a crash / network drop / restart continues instead of
    re-transcribing hours of audio,
  * prints the device + GPU memory so CUDA use is verifiable.

Scoring (docx_text / strip_speakers / base_norm / num_norm / cer /
wer / find_assets) is copied VERBATIM from scripts/eval_real_survey.py
so numbers are directly comparable to that production gate.

Default config = omni_test_1b.yaml (Omnilingual LLM 1B, post-proc &
alignment OFF) → isolated ASR number for the chosen model. This
script bypasses ASRPipeline entirely, so its diarization settings
are never touched.

Usage:
  uv run python scripts/eval_real_survey_inproc.py \
      --stem "07_Shatkhira_CA-MMI_URBAN_Male_07_0202_1716"
  # resume after an interruption — just run the same command again.

Output:
  outputs/real_eval_inproc/<stem>/<part>.chunks.jsonl  (resumable)
  results/real_survey_inproc.json                       (scores)
"""
from __future__ import annotations

import argparse
import html
import json
import re
import sys
import tempfile
import time
import unicodedata
import zipfile
from collections import deque
from pathlib import Path

DEFAULT_FOLDER = Path("for testing (Noureini)")

# ── Cross-segment context (Layer 1 rolling prior + Layer 2 entity
# memory). Cheap, single-task-preserving discourse context so
# per-segment correct/translate isn't blind to what came before. ──
_ENT_RE = re.compile(r"\b[A-Z][A-Za-z][A-Za-z.&'-]{0,24}\b")
_CTX_PREV = 2          # Layer 1: previous corrected/translated turns
_CTX_PREV_CHARS = 200  # truncate each prior turn
_CTX_ENT_CAP = 25      # Layer 2: max distinct entities remembered
_CTX_ENT_CHARS = 300   # cap entity-list length in the prompt


def _harvest_entities(text: str, entities: dict) -> None:
    """Accumulate distinct proper-noun-ish Latin tokens (cyclone/org/
    place names the corrector romanised) as a compact running memory.
    Insertion-ordered dict, capped — never grows the prompt unbounded."""
    for m in _ENT_RE.findall(text or ""):
        if m not in entities and len(entities) < _CTX_ENT_CAP:
            entities[m] = None


def _dyn_block(prev: deque, entities: dict) -> str:
    """Build the read-only dynamic-context block (Layers 1+2)."""
    parts: list[str] = []
    if prev:
        parts.append("Preceding transcript (context ONLY — do NOT "
                      "translate, repeat, or edit this):")
        parts += [p[:_CTX_PREV_CHARS] for p in prev if p]
        parts.append("")
    if entities:
        ents = ", ".join(list(entities))[:_CTX_ENT_CHARS]
        parts.append("Known names/entities mentioned so far (use to "
                      "disambiguate; do NOT force them in if absent): "
                      + ents)
        parts.append("")
    return ("\n".join(parts) + "\n") if parts else ""


def _rebuild_ctx(done: dict) -> tuple[deque, dict]:
    """On resume, rebuild rolling prior + entity memory from the
    already-done outputs (in idx order) so context stays consistent."""
    prev: deque = deque(maxlen=_CTX_PREV)
    entities: dict = {}
    for i in sorted(done):
        t = (done[i] or "").strip()
        if t:
            _harvest_entities(t, entities)
            prev.append(t)
    return prev, entities


_LOOP_MIN_RUN = 4  # >=4 identical consecutive units => degenerate loop


def _deloop(text: str, min_run: int = _LOOP_MIN_RUN) -> tuple[str, bool]:
    """Collapse a degenerate repetition loop — a unit of ANY length
    (one token, a short n-gram, OR a whole sentence) repeated >=
    min_run times in a row. Catches BOTH failure modes seen here:
      * ASR input loop:  "হয় হয় হয় ...x30"
      * LLM output loop:  the translator repeating a full sentence
                          ~15x in its own generation.

    A degenerate loop is a *recognition/decoding artifact*, not
    speech, so collapsing it is repair — consistent with intelligent
    verbatim, NOT a verbatim violation. Stays faithful: keep ONE
    occurrence of the repeated unit and append [unclear] (flag the
    unrecognisable span; never silently delete content or fabricate).

    Apply to BOTH the ASR input AND the LLM output of the corrector
    and translator (LLMs loop on hard input too).

    Returns (cleaned, looped). `looped` True => a loop was collapsed.
    O(n^2) but chunks are short — negligible.
    """
    toks = (text or "").split()
    n = len(toks)
    if n < min_run:
        return text, False
    out: list[str] = []
    i = 0
    looped = False
    while i < n:
        best_p, best_reps = 0, 1
        max_p = (n - i) // min_run  # need >= min_run reps to qualify
        for p in range(1, max_p + 1):
            unit = toks[i:i + p]
            reps, j = 1, i + p
            while j + p <= n and toks[j:j + p] == unit:
                reps += 1
                j += p
            if reps >= min_run and reps > best_reps:
                best_p, best_reps = p, reps
        if best_p:
            out.extend(toks[i:i + best_p])
            out.append("[unclear]")
            looped = True
            i += best_p * best_reps
        else:
            out.append(toks[i])
            i += 1
    cleaned = re.sub(r"(?:\[unclear\]\s*){2,}", "[unclear] ",
                     " ".join(out)).strip()
    return cleaned, looped

_BN = {"০": "0", "১": "1", "২": "2", "৩": "3", "৪": "4",
       "৫": "5", "৬": "6", "৭": "7", "৮": "8", "৯": "9"}
_WN = {"একশ": "100", "একশো": "100", "এক হাজার": "1000", "একহাজার": "1000"}


# ─── text utilities — VERBATIM from scripts/eval_real_survey.py ──────────

def docx_text(path: Path) -> str:
    z = zipfile.ZipFile(path)
    xml = z.read("word/document.xml").decode("utf-8", "ignore")
    xml = xml.replace("</w:p>", "\n")
    xml = re.sub(r"<[^>]+>", "", xml)
    return html.unescape(xml)


_SPK = re.compile(r"(প্র|উ)\s*[:ঃ]\s*")


def strip_speakers(text: str) -> str:
    return _SPK.sub(" ", text)


def base_norm(s: str) -> str:
    s = unicodedata.normalize("NFC", s).strip()
    s = re.sub(r"[।,.!?;:\"'()\[\]{}—–\-‌‍]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def num_norm(s: str) -> str:
    for w, d in _WN.items():
        s = s.replace(w, d)
    for b, e in _BN.items():
        s = s.replace(b, e)
    return re.sub(r"\s+", " ", s).strip()


try:
    import editdistance as _ed

    def _lev(ref, hyp) -> int:
        # C-backed Levenshtein (editdistance.eval works on str OR
        # list[str]) — identical result to the pure-Python DP below,
        # but ~100-1000x faster. Essential for document-level CER
        # where ref/hyp are tens of thousands of characters.
        return int(_ed.eval(ref, hyp))

except ImportError:  # pragma: no cover - pure-Python fallback

    def _lev(ref, hyp) -> int:
        n, m = len(ref), len(hyp)
        d = list(range(m + 1))
        for i in range(1, n + 1):
            p = d[0]
            d[0] = i
            for j in range(1, m + 1):
                c = d[j]
                cost = 0 if ref[i - 1] == hyp[j - 1] else 1
                d[j] = min(d[j] + 1, d[j - 1] + 1, p + cost)
                p = c
        return d[m]


def cer(ref: str, hyp: str) -> float:
    return _lev(list(ref), list(hyp)) / max(len(ref), 1)


def wer(ref: str, hyp: str) -> float:
    r, h = ref.split(), hyp.split()
    return _lev(r, h) / max(len(r), 1)


def find_assets(folder: Path, stem: str):
    files = list(folder.iterdir())
    audio = sorted(
        f for f in files
        if f.suffix.lower() == ".m4a" and f.name.startswith(stem)
    )
    docx = None
    for f in files:
        n = f.name.lower()
        if (f.suffix.lower() == ".docx"
                and n.startswith(stem.lower())
                and "ranscription" in n):
            docx = f
            break
    return audio, docx


# ─── resumable per-part transcription ───────────────────────────────────

def transcribe_part(engine, preproc_cfg, audio_path: Path,
                     ckpt: Path, omni_lang: str,
                     limit: int | None = None) -> str:
    """VAD-chunk one audio part and transcribe each chunk in-process.

    Per-chunk results are appended to `ckpt` (JSONL: {idx,start,end,
    text}). Chunks whose idx is already in `ckpt` are skipped, so a
    re-run resumes instead of re-doing expensive ASR.
    """
    from asr_pipeline.preprocessor import AudioPreprocessor

    done: dict[int, dict] = {}
    if ckpt.exists():
        for line in ckpt.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                done[rec["idx"]] = rec
            except Exception:
                pass

    with tempfile.TemporaryDirectory(prefix="rs_inproc_") as tmp:
        pp = AudioPreprocessor(preproc_cfg, Path(tmp))
        _, chunks, _ = pp.preprocess(audio_path)
        total = len(chunks)
        print(f"      {total} VAD chunks "
              f"({len(done)} already done, resuming)", flush=True)

        with ckpt.open("a", encoding="utf-8") as fh:
            for k, ch in enumerate(chunks):
                if limit is not None and k >= limit:
                    print(f"      --limit-chunks {limit} reached; "
                          f"stopping ({len(done)} windows cached). "
                          f"Re-run with a higher/no limit to continue.",
                          flush=True)
                    break
                if k in done:
                    continue
                ct0 = time.time()
                segs = engine.transcribe_chunk(ch, omni_lang)
                txt = " ".join(s.text for s in segs).strip()
                rec = {"idx": k,
                       "start": round(ch.start_s, 2),
                       "end": round(ch.end_s, 2),
                       "text": txt}
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
                done[k] = rec
                if k % 10 == 0 or k == total - 1:
                    print(f"        chunk {k + 1}/{total} "
                          f"({time.time() - ct0:.1f}s) "
                          f"\"{txt[:40]}\"", flush=True)

    # Reassemble in chunk order.
    return " ".join(done[i]["text"]
                     for i in sorted(done) if done[i]["text"]).strip()


def read_part_text(ckpt: Path) -> str:
    """Rebuild a part's hypothesis from its JSONL checkpoint only —
    no audio, no preprocessing, no model. Used by --score-only."""
    if not ckpt.exists():
        print(f"ERROR: --score-only but checkpoint missing: {ckpt}\n"
              f"       Run without --score-only first to transcribe.")
        sys.exit(1)
    recs: dict[int, dict] = {}
    for line in ckpt.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
            recs[r["idx"]] = r
        except Exception:
            pass
    return " ".join(recs[i]["text"] for i in sorted(recs)
                    if recs[i]["text"]).strip()


def read_part_records(ckpt: Path) -> dict:
    """idx -> raw chunk text, from a .chunks.jsonl checkpoint."""
    recs: dict[int, str] = {}
    if not ckpt.exists():
        print(f"ERROR: missing checkpoint for correction: {ckpt}")
        sys.exit(1)
    for line in ckpt.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
            recs[r["idx"]] = r.get("text", "")
        except Exception:
            pass
    return recs


def correct_part(qc, raw_records: dict, ckpt: Path,
                  xseg: bool = True) -> str:
    """Run the Qwen corrector chunk-by-chunk, RESUMABLE.

    Each corrected chunk is appended to `ckpt` (JSONL: {idx,text});
    already-corrected idx are skipped, so an interrupted run resumes
    instead of re-running the (slow) LLM over the whole interview.
    """
    done: dict[int, str] = {}
    if ckpt.exists():
        for line in ckpt.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                done[r["idx"]] = r.get("text", "")
            except Exception:
                pass

    order = sorted(raw_records)
    total = len(order)
    prev, entities = _rebuild_ctx(done)
    print(f"      {total} chunks to correct "
          f"({len(done)} already done, resuming) "
          f"[ctx: {len(entities)} entities]", flush=True)
    with ckpt.open("a", encoding="utf-8") as fh:
        for n, idx in enumerate(order):
            if idx in done:
                continue
            src = (raw_records[idx] or "").strip()
            if not src:
                fixed = src
            else:
                src, looped = _deloop(src)
                if looped:
                    # Degenerate ASR loop: deterministic collapse is
                    # safer/faster than asking the LLM. Skip LLM.
                    fixed = src
                else:
                    if xseg:
                        qc.set_context(_dyn_block(prev, entities))
                    fixed = qc.correct([src])[0]
                    fixed, _ = _deloop(fixed)  # LLM may loop too
            rec = {"idx": idx, "text": fixed}
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            done[idx] = fixed
            if xseg and fixed:
                _harvest_entities(fixed, entities)
                prev.append(fixed)
            if n % 10 == 0 or n == total - 1:
                print(f"        corrected {n + 1}/{total} "
                      f"\"{fixed[:40]}\"", flush=True)

    return " ".join(done[i] for i in sorted(done)
                     if done[i]).strip()


def read_jsonl_map(path: Path) -> dict:
    """idx -> text from a {idx,text} JSONL; {} if file missing."""
    m: dict[int, str] = {}
    if not path.exists():
        return m
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
            m[r["idx"]] = r.get("text", "")
        except Exception:
            pass
    return m


def translate_part(qt, src_records: dict, ckpt: Path,
                    xseg: bool = True) -> str:
    """Translate each chunk Bengali->English, RESUMABLE (mirrors
    correct_part). Empty source -> empty translation."""
    done: dict[int, str] = {}
    if ckpt.exists():
        for line in ckpt.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                done[r["idx"]] = r.get("text", "")
            except Exception:
                pass
    order = sorted(src_records)
    total = len(order)
    prev, entities = _rebuild_ctx(done)
    print(f"      {total} chunks to translate "
          f"({len(done)} already done, resuming) "
          f"[ctx: {len(entities)} entities]", flush=True)
    with ckpt.open("a", encoding="utf-8") as fh:
        for n, idx in enumerate(order):
            if idx in done:
                continue
            src = (src_records[idx] or "").strip()
            if not src:
                en = ""
            else:
                src, _ = _deloop(src)  # safety: never translate a loop
                if xseg:
                    qt.set_context(_dyn_block(prev, entities))
                en = qt.translate_batch([src])[0]
                en, _ = _deloop(en)  # translator loops its own output
            fh.write(json.dumps({"idx": idx, "text": en},
                                ensure_ascii=False) + "\n")
            fh.flush()
            done[idx] = en
            if xseg and en:
                _harvest_entities(en, entities)
                prev.append(en)
            if n % 10 == 0 or n == total - 1:
                print(f"        translated {n + 1}/{total} "
                      f"\"{en[:40]}\"", flush=True)
    return " ".join(done[i] for i in sorted(done)
                     if done[i]).strip()


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--folder", type=Path, default=DEFAULT_FOLDER)
    p.add_argument("--stem", required=True,
                   help="Interview filename stem WITHOUT extension")
    p.add_argument("--config", type=Path, default=Path("omni_test_1b.yaml"),
                   help="Config for preprocessing + omnilingual model "
                        "card (default omni_test_1b.yaml = LLM 1B)")
    p.add_argument("--model-card", default=None,
                   help="Override cfg.omnilingual.model_card without "
                        "editing the yaml (single-variable A/B), e.g. "
                        "omniASR_LLM_Unlimited_1B_v2")
    p.add_argument("--chunk-max-s", type=float, default=None,
                   help="Override preprocessing.chunking.max_duration_s. "
                        "B-lite long-form: use ~120 with the Unlimited "
                        "card so each ~2-min window (cut at VAD silence, "
                        "never inside speech) is decoded by Omnilingual's "
                        "internal 15s/M=1 cross-segment conditioning. "
                        "Engine max_audio_length_s auto-raised to match.")
    p.add_argument("--limit-chunks", type=int, default=None,
                   help="Stop ASR after N windows PER PART (slice "
                        "validation, e.g. --limit-chunks 3 ~= first 6 "
                        "min at --chunk-max-s 120). Resumable: raise/drop "
                        "it and re-run to continue.")
    p.add_argument("--omni-lang", default="ben_Beng")
    p.add_argument("--device", default="cuda")
    p.add_argument("--out-dir", type=Path,
                   default=Path("outputs/real_eval_inproc"))
    p.add_argument("--results", type=Path,
                   default=Path("results/real_survey_inproc.json"))
    p.add_argument("--score-only", action="store_true",
                   help="Skip ASR + preprocessing entirely; rebuild the "
                        "hypothesis from cached per-part JSONL "
                        "checkpoints and just (re)score. Instant — use "
                        "after a completed run or to re-score with "
                        "updated metrics.")
    p.add_argument("--dump", action="store_true",
                   help="Also write hypothesis.txt + ground_truth_raw.txt "
                        "+ ground_truth_stripped.txt to the out dir for "
                        "manual side-by-side inspection.")
    p.add_argument("--correct", action="store_true",
                   help="After ASR, run the Qwen GGUF corrector over the "
                        "cached chunks (RESUMABLE) and also score "
                        "corrected-vs-GT. Reuses cached ASR — no "
                        "re-transcribe. Pairs well with --score-only.")
    p.add_argument("--standardize", action="store_true",
                   help="Normalize dialect toward standard Bengali "
                        "(NOT intelligent-verbatim; editorial). DEFAULT "
                        "PRESERVES the speaker's register/dialect "
                        "verbatim (intelligent-verbatim).")
    p.add_argument("--context",
                   default="Transcript from a household survey in "
                           "coastal Bangladesh (Satkhira district); an "
                           "enumerator interviews a respondent. Speech "
                           "is colloquial regional Bengali.",
                   help="Free-text domain/provenance line injected into "
                        "BOTH the corrector and translator prompts. "
                        "Disambiguates proper nouns the model already "
                        "knows (Amphan/Bulbul/Aila) with no glossary. "
                        "Set \"\" to disable.")
    p.add_argument("--no-xseg", action="store_true",
                   help="Disable cross-segment context (Layer 1 rolling "
                        "prior + Layer 2 entity memory). Use for a "
                        "controlled A/B vs the default (xseg ON).")
    p.add_argument("--gguf-repo",
                   default="unsloth/Qwen3.5-9B-GGUF",
                   help="HF repo for the corrector GGUF.")
    p.add_argument("--gguf-file",
                   default="Qwen3.5-9B-UD-Q4_K_XL.gguf",
                   help="GGUF filename within --gguf-repo (case-sensitive). "
                        "UD-Q4_K_XL = Unsloth-dynamic, ~lossless, 6GB. "
                        "OOM? try Qwen3.5-9B-UD-Q3_K_XL.gguf (5.1GB).")
    p.add_argument("--gguf-path", default=None,
                   help="Local GGUF file (overrides --gguf-repo/-file).")
    p.add_argument("--translate", action="store_true",
                   help="After ASR (and correction if --correct), run the "
                        "Qwen GGUF translator Bengali->English over the "
                        "cached chunks (RESUMABLE) and write "
                        "translation.txt. Translates corrected text if "
                        "available, else raw.")
    args = p.parse_args()

    if not args.folder.is_dir():
        print(f"ERROR: folder not found: {args.folder}")
        sys.exit(1)
    audio, docx = find_assets(args.folder, args.stem)
    if not audio:
        print(f"ERROR: no .m4a parts for stem '{args.stem}' in "
              f"{args.folder}")
        sys.exit(1)
    if docx is None:
        print(f"ERROR: no *ranscription*.docx for '{args.stem}'")
        sys.exit(1)
    out = args.out_dir / args.stem
    out.mkdir(parents=True, exist_ok=True)

    print(f"Interview : {args.stem}")
    print(f"Audio     : {len(audio)} part(s) — "
          f"{', '.join(a.name for a in audio)}")
    print(f"GT docx   : {docx.name}")
    print(f"Config    : {args.config}\n")

    # Config is light (yaml/pydantic) — needed for the model_card label
    # in both modes.
    from asr_pipeline.config import load_config

    cfg = load_config(args.config)
    if args.model_card:
        print(f"  model_card override: {cfg.omnilingual.model_card} "
              f"-> {args.model_card}")
        cfg.omnilingual.model_card = args.model_card
    if args.chunk_max_s is not None:
        print(f"  chunk max_duration_s override: "
              f"{cfg.preprocessing.chunking.max_duration_s} "
              f"-> {args.chunk_max_s} (B-lite long-form window)")
        cfg.preprocessing.chunking.max_duration_s = args.chunk_max_s

    if args.score_only:
        print("SCORE-ONLY: rebuilding hypothesis from cached JSONL "
              "checkpoints (no ASR, no preprocessing, no model load).")
        actual_device, gpu_mb, load_s = "(score-only)", 0.0, 0.0
        t_run = time.time()
        raw_parts = []
        for i, a in enumerate(audio, 1):
            ckpt = out / f"{a.stem}.chunks.jsonl"
            print(f"  [{i}/{len(audio)}] {a.name}  <- {ckpt.name}",
                  flush=True)
            raw_parts.append(read_part_text(ckpt))
        raw_hyp = " ".join(raw_parts).strip()
    else:
        import torch

        from asr_pipeline.engines.omnilingual_engine import (
            OmnilingualEngine,
        )

        # Avoid silent chunk drops: engine skips chunks longer than
        # max_audio_length_s. Make it exceed the chunker's max.
        chunk_max = cfg.preprocessing.chunking.max_duration_s
        cfg.omnilingual.max_audio_length_s = max(
            cfg.omnilingual.max_audio_length_s, chunk_max + 5)

        print(f"torch {torch.__version__} | cuda available: "
              f"{torch.cuda.is_available()}")
        print(f"Loading Omnilingual [{cfg.omnilingual.model_card}] on "
              f"{args.device} ...", flush=True)
        t0 = time.time()
        engine = OmnilingualEngine(cfg.omnilingual, device=args.device)
        engine.load()
        load_s = time.time() - t0
        pipe = engine._pipeline  # noqa: SLF001 — diagnostic
        actual_device = str(getattr(pipe, "device", "unknown"))
        gpu_mb = 0.0
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            gpu_mb = torch.cuda.memory_allocated() / (1024 * 1024)
        print(f"  loaded in {load_s:.1f}s | pipeline.device = "
              f"{actual_device} | GPU mem {gpu_mb:.0f} MiB")
        if actual_device.startswith("cpu"):
            print("  WARNING: model on CPU — extremely slow.")

        t_run = time.time()
        raw_parts = []
        for i, a in enumerate(audio, 1):
            print(f"  [{i}/{len(audio)}] {a.name}", flush=True)
            ckpt = out / f"{a.stem}.chunks.jsonl"
            raw_parts.append(
                transcribe_part(engine, cfg.preprocessing, a, ckpt,
                                args.omni_lang, args.limit_chunks))
        engine.unload()
        raw_hyp = " ".join(raw_parts).strip()

    gt_raw = docx_text(docx)
    gt = strip_speakers(gt_raw)

    # ── Optional: Qwen GGUF corrector over cached chunks (resumable) ──
    corrected_hyp = None
    if args.correct:
        from asr_pipeline.postprocessor import QwenCorrector

        qcfg = cfg.postprocessing.qwen_corrector
        qcfg.backend = "gguf"
        qcfg.gguf_path = args.gguf_path
        qcfg.gguf_repo = None if args.gguf_path else args.gguf_repo
        qcfg.gguf_file = None if args.gguf_path else args.gguf_file
        qcfg.preserve_register = not args.standardize
        qcfg.context = args.context
        reg = "standardize" if args.standardize else "preserve-dialect"
        src = args.gguf_path or f"{args.gguf_repo}/{args.gguf_file}"
        print(f"\nCorrector: loading Qwen GGUF [{src}] "
              f"register={reg} ...", flush=True)
        qc = QwenCorrector(qcfg)
        if not qc.load():
            print("ERROR: corrector failed to load (llama-cpp-python "
                  "missing, or GGUF unavailable). NOT writing a "
                  "misleading pass-through 'corrected' result. Fix the "
                  "GGUF/llama-cpp install, then re-run (cached ASR is "
                  "reused; correction is resumable).")
            sys.exit(2)
        c_parts = []
        for i, a in enumerate(audio, 1):
            print(f"  [{i}/{len(audio)}] correcting {a.name}", flush=True)
            recs = read_part_records(out / f"{a.stem}.chunks.jsonl")
            c_ckpt = out / f"{a.stem}.corrected.jsonl"
            c_parts.append(correct_part(qc, recs, c_ckpt,
                                        not args.no_xseg))
        qc.unload()
        corrected_hyp = " ".join(c_parts).strip()

    # ── Optional: Qwen GGUF translator (Bengali -> English), resumable ──
    translation_hyp = None
    if args.translate:
        from asr_pipeline.postprocessor import QwenTranslator

        tcfg = cfg.postprocessing.qwen_translator
        tcfg.backend = "gguf"
        tcfg.gguf_path = args.gguf_path
        tcfg.gguf_repo = None if args.gguf_path else args.gguf_repo
        tcfg.gguf_file = None if args.gguf_path else args.gguf_file
        tcfg.context = args.context
        tsrc = args.gguf_path or f"{args.gguf_repo}/{args.gguf_file}"
        print(f"\nTranslator: loading Qwen GGUF [{tsrc}] ...", flush=True)
        qt = QwenTranslator(tcfg)
        if not qt.load():
            print("ERROR: translator failed to load (llama-cpp-python "
                  "missing or GGUF unavailable). Skipping translation; "
                  "re-run when fixed (cached ASR/correction reused).")
        else:
            t_parts = []
            for i, a in enumerate(audio, 1):
                print(f"  [{i}/{len(audio)}] translating {a.name}",
                      flush=True)
                src_map = (
                    read_jsonl_map(out / f"{a.stem}.corrected.jsonl")
                    or read_part_records(
                        out / f"{a.stem}.chunks.jsonl"))
                t_ckpt = out / f"{a.stem}.translated.jsonl"
                t_parts.append(translate_part(qt, src_map, t_ckpt,
                                              not args.no_xseg))
            qt.unload()
            translation_hyp = " ".join(t_parts).strip()

    if args.dump:
        (out / "hypothesis.txt").write_text(raw_hyp, encoding="utf-8")
        (out / "ground_truth_raw.txt").write_text(
            gt_raw, encoding="utf-8")
        (out / "ground_truth_stripped.txt").write_text(
            gt, encoding="utf-8")
        if corrected_hyp is not None:
            (out / "corrected.txt").write_text(
                corrected_hyp, encoding="utf-8")
        if translation_hyp is not None:
            (out / "translation.txt").write_text(
                translation_hyp, encoding="utf-8")
        print(f"  DUMPED -> {out}/")
        print(f"    hypothesis.txt            {len(raw_hyp):>7} chars "
              f"({len(raw_hyp.split()):>6} words)")
        if corrected_hyp is not None:
            print(f"    corrected.txt             "
                  f"{len(corrected_hyp):>7} chars "
                  f"({len(corrected_hyp.split()):>6} words)")
        if translation_hyp is not None:
            print(f"    translation.txt           "
                  f"{len(translation_hyp):>7} chars "
                  f"({len(translation_hyp.split()):>6} words)")
        print(f"    ground_truth_raw.txt      {len(gt_raw):>7} chars "
              f"({len(gt_raw.split()):>6} words)")
        print(f"    ground_truth_stripped.txt {len(gt):>7} chars "
              f"({len(gt.split()):>6} words)")

    rows = []

    def score(ref_raw, hyp_raw, label):
        for mtag, fn in (("M1", base_norm),
                         ("M2", lambda s: num_norm(base_norm(s)))):
            r, h = fn(ref_raw), fn(hyp_raw)
            c, w = cer(r, h) * 100, wer(r, h) * 100
            print(f"  {label:<18} {mtag}  CER={c:6.2f}%  WER={w:6.2f}%  "
                  f"(ref {len(r)} ch / hyp {len(h)} ch)")
            rows.append({"system": label, "metric": mtag,
                         "cer": c, "wer": w,
                         "ref_chars": len(r), "hyp_chars": len(h)})

    run_min = (time.time() - t_run) / 60
    print(f"\n{'=' * 64}")
    print(f"Real survey eval (in-process) — {args.stem}")
    print(f"  model {cfg.omnilingual.model_card} | device "
          f"{actual_device} | GPU {gpu_mb:.0f} MiB | run {run_min:.1f} min")
    print(f"{'=' * 64}")
    score(gt, raw_hyp, "raw Omnilingual")
    if corrected_hyp is not None:
        score(gt, corrected_hyp, "Qwen-corrected")

    args.results.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"stem": args.stem, "config": str(args.config),
               "model_card": cfg.omnilingual.model_card,
               "device": actual_device, "gpu_mem_mb": gpu_mb,
               "n_audio_parts": len(audio), "rows": rows,
               "run_min": run_min},
              open(args.results, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"\n  saved -> {args.results}")
    print("  NOTE: real field audio, document-level, full transcript "
          "— this is the production-gate number, not FLEURS.")


if __name__ == "__main__":
    main()
