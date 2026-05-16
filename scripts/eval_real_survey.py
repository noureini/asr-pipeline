"""
Real-survey validation — THE production gate.

Runs the LOCAL pipeline on a real ~hour-long Bangladesh survey
interview and computes document-level CER + WER against the
human ground-truth transcription (.docx).

This is NOT FLEURS clean read-speech. This is the only measurement
that tells you whether the pipeline is production-ready for the
national survey.

Pre-registered metric (decided BEFORE running, per agreement):
  - Speaker scope: FULL transcript (both প্র enumerator + উ respondent).
  - Both audio parts (….m4a + …_1.m4a) transcribed and concatenated
    (partial audio vs full-interview GT would inflate WER).
  - Speaker markers stripped from BOTH sides (প্র:/উ: in GT; the
    pipeline's JSON text is already header-free).
  - Normalization: M1 = NFC + strip punctuation/ZWJ + collapse ws.
    M2 = M1 + Bengali-numeral/number-word normalization.
  - Document-level CER (char Levenshtein) and WER (whitespace-token).
  - Reported for raw-Omnilingual-vs-GT (the ASR gate) AND
    corrected-vs-GT (if a corrector ran).

Default config = omni_test.yaml (Omnilingual forced, post-processing
OFF) → isolates the ASR number. Pass --config default.yaml later to
measure the Qwen correction delta.

Usage:
  uv run python scripts/eval_real_survey.py \
      --stem "06_Shatkhira_CA-MMI_RURAL_Male_06_0202_1150"
  # re-score without re-running the (hours-long) pipeline:
  uv run python scripts/eval_real_survey.py --stem "..." --skip-run

Honest runtime note: one interview is ~1.5 h of audio. On a 6 GB
laptop this is an hours-long, likely overnight run. The 98 GB box
is where the full set belongs.
"""
from __future__ import annotations

import argparse
import html
import json
import re
import subprocess
import sys
import time
import unicodedata
import zipfile
from pathlib import Path

DEFAULT_FOLDER = Path("for testing (Noureini)")

_BN = {"০": "0", "১": "1", "২": "2", "৩": "3", "৪": "4",
       "৫": "5", "৬": "6", "৭": "7", "৮": "8", "৯": "9"}
_WN = {"একশ": "100", "একশো": "100", "এক হাজার": "1000", "একহাজার": "1000"}


# ─── text utilities ──────────────────────────────────────────────────────

def docx_text(path: Path) -> str:
    """Extract plain text from a .docx (paragraph breaks preserved)."""
    z = zipfile.ZipFile(path)
    xml = z.read("word/document.xml").decode("utf-8", "ignore")
    xml = xml.replace("</w:p>", "\n")
    xml = re.sub(r"<[^>]+>", "", xml)
    return html.unescape(xml)


# Speaker-turn markers in the GT: প্র (প্রশ্ন/enumerator), উ (উত্তর/respondent)
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


def _lev(ref: list, hyp: list) -> int:
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


# ─── pipeline I/O ────────────────────────────────────────────────────────

def find_assets(folder: Path, stem: str):
    """Return (sorted audio parts, transcription .docx) for an interview."""
    files = list(folder.iterdir())
    audio = sorted(
        f for f in files
        if f.suffix.lower() == ".m4a" and f.name.startswith(stem)
    )
    # '.' (0x2E) < '_' (0x5F): base '<stem>.m4a' sorts before '<stem>_1.m4a'
    docx = None
    for f in files:
        n = f.name.lower()
        if (f.suffix.lower() == ".docx"
                and n.startswith(stem.lower())
                and "ranscription" in n):
            docx = f
            break
    return audio, docx


def hyp_from_json(json_path: Path) -> tuple[str, str]:
    """Concatenate (raw_text, corrected_text) over all segments."""
    try:
        d = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return "", ""
    raw, corr = [], []
    for s in d.get("segments", []) or []:
        rt = (s.get("raw_text") or "").strip()
        ct = (s.get("corrected_text") or s.get("raw_text") or "").strip()
        if rt:
            raw.append(rt)
        if ct:
            corr.append(ct)
    return " ".join(raw), " ".join(corr)


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--folder", type=Path, default=DEFAULT_FOLDER)
    p.add_argument("--stem", required=True,
                   help="Interview filename stem WITHOUT extension, e.g. "
                        "06_Shatkhira_CA-MMI_RURAL_Male_06_0202_1150")
    p.add_argument("--config", default="omni_test.yaml",
                   help="Pipeline config. Default omni_test.yaml "
                        "(Omnilingual, post-proc OFF) = isolated ASR. "
                        "Use default.yaml to include Qwen correction.")
    p.add_argument("--out-dir", type=Path, default=Path("outputs/real_eval"))
    p.add_argument("--results", type=Path,
                   default=Path("results/real_survey_eval.json"))
    p.add_argument("--skip-run", action="store_true",
                   help="Score existing JSON only (no re-run)")
    args = p.parse_args()

    if not args.folder.is_dir():
        print(f"ERROR: folder not found: {args.folder}")
        sys.exit(1)
    audio, docx = find_assets(args.folder, args.stem)
    if not audio:
        print(f"ERROR: no .m4a parts found for stem '{args.stem}' in "
              f"{args.folder}")
        sys.exit(1)
    if docx is None:
        print(f"ERROR: no *ranscription*.docx found for '{args.stem}'")
        sys.exit(1)
    out = args.out_dir / args.stem
    out.mkdir(parents=True, exist_ok=True)

    print(f"Interview : {args.stem}")
    print(f"Audio     : {len(audio)} part(s) — "
          f"{', '.join(a.name for a in audio)}")
    print(f"GT docx   : {docx.name}")
    print(f"Config    : {args.config}\n")

    # ── Run pipeline on each audio part ──────────────────────────────
    t0 = time.time()
    part_jsons = []
    for i, a in enumerate(audio, 1):
        j = out / f"{a.stem}.json"
        part_jsons.append(j)
        if args.skip_run:
            continue
        print(f"  [{i}/{len(audio)}] transcribing {a.name} "
              f"(LONG — ~hour of audio) ...", flush=True)
        r = subprocess.run(
            ["uv", "run", "asr-pipeline", "transcribe", str(a),
             "-l", "ben", "-c", args.config, "-f", "json", "-o", str(out)],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            print(f"      pipeline failed (rc={r.returncode}):\n"
                  f"{r.stderr[-600:]}")
            sys.exit(1)

    # ── Build hypothesis (concatenate parts in order) ────────────────
    raw_parts, corr_parts = [], []
    for j in part_jsons:
        if not j.exists():
            print(f"ERROR: expected JSON missing: {j}")
            sys.exit(1)
        rt, ct = hyp_from_json(j)
        raw_parts.append(rt)
        corr_parts.append(ct)
    raw_hyp = " ".join(raw_parts).strip()
    corr_hyp = " ".join(corr_parts).strip()

    # ── Ground truth ─────────────────────────────────────────────────
    gt = strip_speakers(docx_text(docx))

    # ── Score (document-level, pre-registered) ───────────────────────
    def score(ref_raw, hyp_raw, label):
        for mtag, fn in (("M1", lambda s: base_norm(s)),
                         ("M2", lambda s: num_norm(base_norm(s)))):
            r, h = fn(ref_raw), fn(hyp_raw)
            print(f"  {label:<22} {mtag}  "
                  f"CER={cer(r, h)*100:6.2f}%  WER={wer(r, h)*100:6.2f}%  "
                  f"(ref {len(r)} ch / hyp {len(h)} ch)")
            rows.append({"system": label, "metric": mtag,
                         "cer": cer(r, h) * 100, "wer": wer(r, h) * 100,
                         "ref_chars": len(r), "hyp_chars": len(h)})

    rows = []
    print(f"\n{'=' * 64}")
    print(f"Real survey eval — {args.stem}")
    print(f"{'=' * 64}")
    score(gt, raw_hyp, "raw Omnilingual")
    if corr_hyp and corr_hyp != raw_hyp:
        score(gt, corr_hyp, "Qwen-corrected")
    else:
        print("  (corrected == raw — no corrector ran with this config)")

    elapsed = (time.time() - t0) / 60
    print(f"\n  elapsed: {elapsed:.1f} min")
    print("  NOTE: real field audio, document-level, full transcript.")
    print("  This is the production-gate number — not FLEURS.")

    args.results.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"stem": args.stem, "config": args.config,
               "n_audio_parts": len(audio), "rows": rows,
               "elapsed_min": elapsed},
              open(args.results, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"\n  saved -> {args.results}")


if __name__ == "__main__":
    main()
