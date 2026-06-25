"""Stage 0 smoke test — distinctive-feature decodability on REAL field audio.

Tests claim C1 of the White-Box Recognizer plan using ONLY assets already
on disk — no new model or data downloads:

  Encoder + aligner : torchaudio.pipelines.MMS_FA (cached — the pipeline's
                      stage-3b aligner). It is an MMS wav2vec2 encoder with
                      a CTC alignment head; extract_features() exposes all
                      transformer layers for probing.
  Data              : data/w2v_train_longform/ GT-aligned field utterances
                      07_Shatkhira (1,047) + 15_Sunamganj (1,294) —
                      8 kHz-origin narrowband coastal-Bangladesh audio.

CAVEAT (honest): MMS_FA's encoder is CTC-FINE-TUNED for alignment, not a
pristine pretrained-only SSL model. Probing it is mildly OPTIMISTIC vs the
plan's frozen mms-300m/mms-1b (fine-tuning sharpens phonetic information
in mid/late layers). Treat a GREEN here as "promising, replicate on the
pristine encoder"; treat a RED here as strong evidence against C1 (if even
a phonetically fine-tuned encoder can't expose a feature linearly, the
pristine one won't).

Pipeline:
  1. gt_text -> epitran ben-Beng G2P -> panphon segmentation
     (code-switched Latin words aligned but unlabeled; single Bengali
      digits -> number words; multi-digit utterances skipped)
  2. MMS_FA forced alignment at PHONE granularity (each phone romanized,
     aligned as its own "word" -> per-phone frame spans)
  3. extract_features() -> mid-phone frame embeddings, ~6 layers
  4. One class-balanced logistic regression per panphon feature
  5. Cells: in-domain (random 80/20) AND cross-interview
     (train 07_Shatkhira -> test 15_Sunamganj: speaker- AND
      district-disjoint, on real field audio)
  6. Shuffled-label control (chance floor ~0.50)

Usage:
    uv pip install epitran panphon
    uv run python scripts/stage0_probe.py --max-utts 400   # quick pass
    uv run python scripts/stage0_probe.py                  # full 2,341
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from bn_g2p_patch import detect_inherent, patch_segs  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("stage0")

DATA_DIR = Path("data/w2v_train_longform")
JSONLS = [
    ("07_Shatkhira",
     DATA_DIR / "07_Shatkhira_CA-MMI_URBAN_Male_07_0202_1716.longform.jsonl"),
    ("15_Sunamganj",
     DATA_DIR / "15_Sunamganj_CA-MMI_URBAN_Male_15_0602_1635.longform.jsonl"),
]

# Single Bengali digits -> number words (multi-digit utts are skipped).
_BN_DIGIT_WORD = {
    "০": "শূন্য", "১": "এক", "২": "দুই", "৩": "তিন", "৪": "চার",
    "৫": "পাঁচ", "৬": "ছয়", "৭": "সাত", "৮": "আট", "৯": "নয়",
}

# IPA -> roman (MMS_FA vocab is a-z). Char-level; diacritics dropped,
# aspiration keeps an 'h' (kʰ -> "kh").
_IPA2ROMAN = {
    "ʈ": "t", "ɖ": "d", "ɽ": "r", "ʃ": "s", "ʂ": "s", "ɕ": "s",
    "ŋ": "n", "ɲ": "n", "ɳ": "n", "ɡ": "g", "ɣ": "g",
    "ɔ": "o", "æ": "a", "ɛ": "e", "ə": "a", "ɪ": "i", "ʊ": "u",
    "j": "y", "ʋ": "w", "β": "b", "ɸ": "f", "ʒ": "j",
    "ʰ": "h", "ʱ": "h", "x": "k", "q": "k",
    **{c: c for c in "abcdefghiklmnoprstuvwyz"},
}


def ipa_to_roman(seg: str) -> str:
    out = []
    for ch in unicodedata.normalize("NFD", seg):
        if unicodedata.category(ch) == "Mn":
            continue
        if ch in ("ː", "ˑ", "̯", "͡", "‿"):
            continue
        out.append(_IPA2ROMAN.get(ch, ""))
    return re.sub(r"[^a-z]", "", "".join(out))


_BENGALI = re.compile(r"[ঀ-৿]")
_LATIN = re.compile(r"^[A-Za-z]+$")
_TOKEN_CLEAN = re.compile(r"[^A-Za-zঀ-৿০-৯]")


def utt_to_units(text: str, epi, ft, g2p_cache: dict,
                 inherent: str = "", patch_mode: str = "none"):
    """Tokenize gt_text into alignment units.

    Returns list of (roman_string, phone_or_None) — Latin words align
    as one unit with no phone label; Bengali words expand to per-phone
    units. Returns None if the utterance should be skipped.

    patch_mode applies the inherent-vowel (schwa) deletion patch to
    epitran's output: "final" = word-final rule only (safe);
    "final-medial" = also the condition-blind medial rule (NEEDS
    LINGUIST REVIEW); "none" = raw epitran (baseline).
    """
    units: list[tuple[str, str | None]] = []
    for raw_tok in text.split():
        tok = _TOKEN_CLEAN.sub("", raw_tok)
        if not tok:
            continue
        if re.search(r"[০-৯0-9]", tok):
            if tok in _BN_DIGIT_WORD:
                tok = _BN_DIGIT_WORD[tok]
            else:
                return None                      # multi-digit -> skip utt
        if _LATIN.match(tok):
            roman = re.sub(r"[^a-z]", "", tok.lower())
            if roman:
                units.append((roman, None))      # align, no label
            continue
        if not _BENGALI.search(tok):
            continue
        if tok not in g2p_cache:
            try:
                segs = ft.ipa_segs(epi.transliterate(tok))
                if segs and patch_mode != "none":
                    segs, _ = patch_segs(
                        tok, segs, inherent,
                        medial=(patch_mode == "final-medial"))
                g2p_cache[tok] = segs
            except Exception:
                g2p_cache[tok] = []
        segs = g2p_cache[tok]
        if not segs:
            return None                          # G2P failure -> skip utt
        for ph in segs:
            roman = ipa_to_roman(ph)
            if roman:
                units.append((roman, ph))
    n_phones = sum(1 for _, p in units if p is not None)
    return units if n_phones >= 3 else None


def pick_layers(n_layers: int, spec: str) -> list[int]:
    """Resolve --layers. 'auto' -> ~6 evenly spaced (0-indexed)."""
    if spec != "auto":
        return [int(x) - 1 for x in spec.split(",")]
    ks = sorted({max(0, round(n_layers * k / 6) - 1) for k in range(1, 7)})
    return ks


def build_dataset(max_utts: int, device: str, layer_spec: str,
                  encoder: str = "mms300m", patch_mode: str = "none"):
    import torch
    import torchaudio
    import epitran
    import panphon

    ft = panphon.FeatureTable()
    epi = epitran.Epitran("ben-Beng")
    g2p_cache: dict = {}

    inherent = ""
    if patch_mode != "none":
        inherent = detect_inherent(epi)
        logger.info("G2P patch mode=%s | inherent-vowel symbol "
                    "(empirical): %r", patch_mode, inherent)
        if patch_mode == "final-medial":
            logger.warning("Medial schwa deletion is CONDITION-BLIND "
                           "(no morphology available) — results need "
                           "linguist review before any conclusions.")

    logger.info("Loading MMS_FA (cached) — used for ALIGNMENT ...")
    bundle = torchaudio.pipelines.MMS_FA
    fa_model = bundle.get_model(with_star=True).to(device).eval()
    tokenizer = bundle.get_tokenizer()
    aligner = bundle.get_aligner()

    if encoder == "mms300m":
        # The decisive probe target: pristine pretrained-only SSL
        # encoder (claim C1 as written). ~1.2 GB one-time download.
        logger.info("Loading facebook/mms-300m (pristine SSL — the "
                    "probe target) ...")
        from transformers import Wav2Vec2Model
        ssl = Wav2Vec2Model.from_pretrained(
            "facebook/mms-300m").to(device).eval()

        def get_feats(wav_d):
            # mms-300m expects zero-mean/unit-var input; hidden_states[0]
            # is the conv-frontend output — drop it so indices are
            # transformer layers, matching MMS_FA's extract_features.
            wn = (wav_d - wav_d.mean()) / (wav_d.std() + 1e-7)
            with torch.inference_mode():
                out = ssl(wn, output_hidden_states=True)
            return list(out.hidden_states[1:])
    elif encoder == "mmsfa":
        # Comparison target: CTC-fine-tuned alignment encoder (mildly
        # optimistic vs pristine SSL — see docstring caveat).
        extractor = getattr(fa_model, "extract_features", None)
        if extractor is None and hasattr(fa_model, "model"):
            extractor = fa_model.model.extract_features
        if extractor is None:
            raise RuntimeError(
                "MMS_FA model exposes no extract_features(); cannot probe.")

        def get_feats(wav_d):
            with torch.inference_mode():
                feats, _ = extractor(wav_d)
            return feats
    else:
        raise ValueError(f"unknown encoder '{encoder}'")

    layers: list[int] | None = None
    Xm: dict[int, list] = {}     # mid-frame embedding per phone
    Xp: dict[int, list] = {}     # mean-over-span ("pooled") per phone
    Y: list[list[int]] = []
    G: list[str] = []
    phone_counter: Counter = Counter()
    stats = Counter()

    for group, jsonl_path in JSONLS:
        n_group = 0
        with open(jsonl_path, encoding="utf-8") as f:
            records = [json.loads(line) for line in f]
        if max_utts:
            per = max_utts // len(JSONLS)
            step = max(1, len(records) // per)
            records = records[::step][:per]
        logger.info("%s: processing %d utterances", group, len(records))

        for rec in records:
            units = utt_to_units(rec["gt_text"], epi, ft, g2p_cache,
                                 inherent=inherent, patch_mode=patch_mode)
            if units is None:
                stats["skip_text"] += 1
                continue
            try:
                wav, sr = torchaudio.load(rec["wav"])
            except Exception:
                stats["skip_audio"] += 1
                continue
            if wav.size(0) > 1:
                wav = wav.mean(0, keepdim=True)
            if sr != 16000:
                wav = torchaudio.functional.resample(wav, sr, 16000)
            dur = wav.shape[1] / 16000
            if dur < 0.4 or dur > 30:
                stats["skip_dur"] += 1
                continue

            romans = [r for r, _ in units]
            wav_d = wav.to(device)
            try:
                with torch.inference_mode():
                    emission, _ = fa_model(wav_d)
                    spans = aligner(emission[0], tokenizer(romans))
                feats = get_feats(wav_d)          # list[L] of [1, T, D]
            except Exception:
                stats["skip_align"] += 1
                continue

            if layers is None:
                layers = pick_layers(len(feats), layer_spec)
                Xm = {ly: [] for ly in layers}
                Xp = {ly: [] for ly in layers}
                logger.info("Encoder has %d layers; probing layers %s "
                            "(1-indexed: %s)",
                            len(feats), layers,
                            [ly + 1 for ly in layers])

            T = feats[layers[0]].shape[1]
            for (_, phone), sp in zip(units, spans):
                if phone is None or not sp:
                    continue
                s = min(int(sp[0].start), T - 1)
                e = max(s + 1, min(int(sp[-1].end), T))
                mid = min((s + e) // 2, T - 1)
                fseg = ft.word_fts(phone)
                if not fseg:
                    continue
                for ly in layers:
                    Xm[ly].append(
                        feats[ly][0, mid].float().cpu().numpy())
                    Xp[ly].append(
                        feats[ly][0, s:e].mean(0).float().cpu().numpy())
                Y.append(fseg[0].numeric())
                G.append(group)
                phone_counter[phone] += 1

            n_group += 1
            stats["used"] += 1
            if n_group % 100 == 0:
                logger.info("  %s: %d utts, %d phones total",
                            group, n_group, len(Y))

    logger.info("Done. utts used=%d | skipped: text=%d audio=%d dur=%d "
                "align=%d | phones=%d (%d types)",
                stats["used"], stats["skip_text"], stats["skip_audio"],
                stats["skip_dur"], stats["skip_align"],
                len(Y), len(phone_counter))
    # Persist the G2P cache exactly as used, so pre/post-patch probe
    # runs diff cleanly at the label level.
    cache_path = Path("outputs") / \
        f"g2p_cache_used.{encoder}.{patch_mode}.json"
    cache_path.parent.mkdir(exist_ok=True)
    cache_path.write_text(
        json.dumps(g2p_cache, ensure_ascii=False), encoding="utf-8")
    logger.info("G2P cache (as used) -> %s", cache_path)

    Xm = {ly: np.stack(v) for ly, v in Xm.items()}
    Xp = {ly: np.stack(v) for ly, v in Xp.items()}
    return Xm, Xp, np.array(Y), np.array(G), ft.names, phone_counter, layers


def run_probes(Xm, Xp, Y, G, feat_names, layers, device="cuda",
               min_minority=0.03, seed=0):
    """Batched per-feature logistic probes on GPU (torch LBFGS).

    Mathematically equivalent to independent class-balanced L2 logistic
    regressions per feature: BCE over a Linear(D, F) decomposes per
    output unit, per-feature masks zero excluded samples in the loss.
    One deliberate difference from the earlier sklearn runs:
    standardization uses all training rows (per-feature masking happens
    in the loss, not the scaler). ESTIMATOR CHANGE — diff probe runs
    only against baselines produced by this same estimator.
    """
    import torch
    from sklearn.metrics import roc_auc_score, balanced_accuracy_score

    rng = np.random.RandomState(seed)
    n = len(Y)
    idx = rng.permutation(n)
    tr_r, te_r = idx[: int(0.8 * n)], idx[int(0.8 * n):]
    tr_x = np.where(G == "07_Shatkhira")[0]
    te_x = np.where(G == "15_Sunamganj")[0]

    # Feature keep-list + per-feature support stats (same gates as before)
    meta, keep = [], []
    for fi, fname in enumerate(feat_names):
        yv = Y[:, fi]
        mask = yv != 0
        trr = tr_r[mask[tr_r]]
        ter = te_r[mask[te_r]]
        if len(trr) < 100 or len(ter) < 40:
            continue
        frac = float((yv[trr] == 1).mean())
        if min(frac, 1 - frac) < min_minority:
            continue
        tex = te_x[mask[te_x]]
        meta.append(dict(feature=fname,
                         n_test=len(ter), pos_frac=frac,
                         n_test_x=len(tex),
                         pos_frac_x=(float((yv[tex] == 1).mean())
                                     if len(tex) else float("nan"))))
        keep.append(fi)
    F = len(keep)
    Yk = (Y[:, keep] == 1).astype(np.float32)
    Mk = (Y[:, keep] != 0).astype(np.float32)

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    logger.info("Fitting batched probes on %s (%d features × %d layers "
                "× 2 variants, + cross + shuffled)", dev, F, len(layers))

    def fit_logits(Xtr, Ytr, Mtr, Xte, C=1.0):
        Xt = torch.tensor(Xtr, device=dev)
        mu = Xt.mean(0, keepdim=True)
        sd = Xt.std(0, keepdim=True) + 1e-6
        Xt = (Xt - mu) / sd
        Yt = torch.tensor(Ytr, device=dev)
        Mt = torch.tensor(Mtr, device=dev)
        npos = (Yt * Mt).sum(0).clamp(min=1)
        nneg = ((1 - Yt) * Mt).sum(0).clamp(min=1)
        ntot = Mt.sum(0)
        Wgt = Mt * (Yt * (ntot / (2 * npos))
                    + (1 - Yt) * (ntot / (2 * nneg)))
        W = torch.zeros(Xt.shape[1], F, device=dev, requires_grad=True)
        b = torch.zeros(F, device=dev, requires_grad=True)
        opt = torch.optim.LBFGS([W, b], max_iter=200, history_size=10,
                                line_search_fn="strong_wolfe")
        bce = torch.nn.functional.binary_cross_entropy_with_logits

        def closure():
            opt.zero_grad()
            loss = (bce(Xt @ W + b, Yt, reduction="none") * Wgt).sum() \
                + (0.5 / C) * (W ** 2).sum()
            loss.backward()
            return loss

        opt.step(closure)
        with torch.no_grad():
            Xe = (torch.tensor(Xte, device=dev) - mu) / sd
            return (Xe @ W + b).cpu().numpy()

    def aucs_from(logits, te_ids):
        out = np.full(F, np.nan)
        for j in range(F):
            m = Mk[te_ids, j] > 0
            yy = Yk[te_ids, j][m]
            if len(np.unique(yy)) < 2:
                continue
            out[j] = roc_auc_score(yy, logits[m, j])
        return out

    perm = rng.permutation(len(tr_r))
    results = {}
    pooled_in_logits = {}
    shuf_aucs = {}
    for tag, Xd in (("m", Xm), ("p", Xp)):
        in_aucs, x_aucs = [], []
        for li, ly in enumerate(layers):
            lo = fit_logits(Xd[ly][tr_r], Yk[tr_r], Mk[tr_r],
                            Xd[ly][te_r])
            in_aucs.append(aucs_from(lo, te_r))
            if tag == "p":
                pooled_in_logits[li] = lo
                lo_sh = fit_logits(Xd[ly][tr_r], Yk[tr_r][perm],
                                   Mk[tr_r][perm], Xd[ly][te_r])
                shuf_aucs[li] = aucs_from(lo_sh, te_r)
            lo_x = fit_logits(Xd[ly][tr_x], Yk[tr_x], Mk[tr_x],
                              Xd[ly][te_x])
            x_aucs.append(aucs_from(lo_x, te_x))
            logger.info("  %s layer %d done", tag, ly + 1)
        results[tag] = (np.stack(in_aucs), np.stack(x_aucs))  # [L, F]

    rows = []
    for j, m in enumerate(meta):
        row = dict(m)
        for tag in ("m", "p"):
            in_a, x_a = results[tag]
            bl = int(np.nanargmax(in_a[:, j]))
            row[f"ly_{tag}"] = layers[bl]
            row[f"auc_{tag}"] = float(in_a[bl, j])
            row[f"auc_{tag}_x"] = float(x_a[bl, j])
            if tag == "p":
                msk = Mk[te_r, j] > 0
                yy = Yk[te_r, j][msk]
                pred = (pooled_in_logits[bl][msk, j] > 0).astype(int)
                row["bacc_p"] = float(
                    balanced_accuracy_score(yy, pred))
                row["auc_shuf"] = float(shuf_aucs[bl][j])
        rows.append(row)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-utts", type=int, default=0,
                    help="cap total utterances (0 = all 2,341)")
    ap.add_argument("--layers", default="auto",
                    help="'auto' or comma list of 1-indexed layers")
    ap.add_argument("--encoder", default="mms300m",
                    choices=["mms300m", "mmsfa"],
                    help="probe target: pristine SSL (mms300m, the "
                         "decisive run) or the CTC-fine-tuned alignment "
                         "encoder (mmsfa, comparison)")
    ap.add_argument("--g2p-patch", default="final",
                    choices=["none", "final", "final-medial"],
                    help="inherent-vowel deletion patch on epitran "
                         "output: 'final' = word-final rule (safe, "
                         "default); 'none' = raw epitran baseline; "
                         "'final-medial' = +condition-blind medial rule "
                         "(NEEDS LINGUIST REVIEW)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()
    if args.output is None:
        args.output = Path(
            f"outputs/stage0_probe_report.{args.encoder}."
            f"{args.g2p_patch}.md")

    Xm, Xp, Y, G, feat_names, phone_counter, layers = build_dataset(
        args.max_utts, args.device, args.layers, args.encoder,
        patch_mode=args.g2p_patch)
    rows = run_probes(Xm, Xp, Y, G, feat_names, layers,
                      device=args.device)
    rows.sort(key=lambda r: -r["auc_p"])

    def med(key):
        vals = np.array([r[key] for r in rows
                         if not np.isnan(r[key])])
        return float(np.median(vals)) if len(vals) else float("nan")

    med_m, med_p = med("auc_m"), med("auc_p")
    med_mx, med_px = med("auc_m_x"), med("auc_p_x")

    enc_desc = (
        "facebook/mms-300m — PRISTINE pretrained-only SSL (claim C1 as "
        "written)" if args.encoder == "mms300m" else
        "MMS_FA wav2vec2 — CTC-fine-tuned (mildly optimistic vs pristine "
        "SSL; comparison run)")
    lines = ["# Stage 0 smoke test — feature decodability on FIELD audio\n"]
    lines.append(f"Encoder: {enc_desc}. "
                 "Data: 07_Shatkhira + 15_Sunamganj GT-aligned field "
                 "utterances (8 kHz-origin). "
                 f"G2P patch: **{args.g2p_patch}**. "
                 "Probes: batched torch-LBFGS logistic (GPU), "
                 "standardized, class-balanced, L2 C=1.0 — ESTIMATOR "
                 "CHANGE vs the earlier sklearn runs; diff only against "
                 "same-estimator baselines.\n")
    lines.append("NOTE on gates: the plan's pre-registered FIELD gate is "
                 "median AUC >= 0.85 (GREEN). Sub-gate results proceed "
                 "via the pre-specified AMBER mechanism (soft feature "
                 "lattices + per-condition recalibration); that is a "
                 "fallback pathway, not a pass.\n")
    lines.append(f"- Phones probed: **{len(Y)}** "
                 f"({len(phone_counter)} types) | features kept: "
                 f"**{len(rows)}**")
    lines.append(f"- Median AUC in-domain — mid-frame: **{med_m:.3f}**, "
                 f"pooled: **{med_p:.3f}** (field gate 0.85)")
    lines.append(f"- Median AUC cross-interview (train Shatkhira → test "
                 f"Sunamganj) — mid-frame: **{med_mx:.3f}**, pooled: "
                 f"**{med_px:.3f}**\n")
    lines.append("| feature | ly_m | AUC_m | AUC_m_x | ly_p | AUC_p | "
                 "AUC_p_x | bacc_p | shuf | n_in | pos_in | n_x | pos_x |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        lines.append(
            f"| {r['feature']} | {r['ly_m'] + 1} | {r['auc_m']:.3f} | "
            f"{r['auc_m_x']:.3f} | {r['ly_p'] + 1} | {r['auc_p']:.3f} | "
            f"{r['auc_p_x']:.3f} | {r['bacc_p']:.3f} | "
            f"{r['auc_shuf']:.3f} | {r['n_test']} | {r['pos_frac']:.2f} | "
            f"{r['n_test_x']} | {r['pos_frac_x']:.2f} |")
    lines.append("\nColumns: _m = mid-frame embedding, _p = mean-over-span "
                 "(pooled), _x = cross-interview transfer at the variant's "
                 "in-domain best layer. n/pos given for BOTH eval cells so "
                 "underpowered rows (e.g. ant, sg) are visible directly.")
    lines.append("\nTop-25 phone types: " + ", ".join(
        f"{p}({c})" for p, c in phone_counter.most_common(25)))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:8]))
    logger.info("Full report: %s", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
