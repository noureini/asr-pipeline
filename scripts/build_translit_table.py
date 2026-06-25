"""
Day-0a (path A): build a Bengali-script -> Latin transliteration
table for the common English code-switched terms in Bangladeshi
field-survey speech.

Candidate dictionary is curated from domain knowledge (common
survey vocabulary: NGO/loan/vaccine/ward/...). We then VALIDATE
each entry against gt_unigrams_counts.tsv to drop entries that
don't actually appear in our GT — keeping the table honest and
data-grounded.

Output (persistent — survives WSL /tmp wipes on reboot):
  <repo>/data/translit.json        final {bn_form: latin_form} map
  <repo>/data/translit.review.tsv  human-readable review file
"""
from __future__ import annotations

import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Curated candidates — common English-in-Bengali-script terms in
# Bangladeshi survey speech (climate/health/livelihood/identity).
CANDIDATES: dict[str, str] = {
    # NGOs / formal institutions
    "এনজিও": "NGO",
    "এনআইডি": "NID",
    "এমপি": "MP",
    "ইউনিয়ন": None,             # naturalized administrative term
                                  # (ইউনিয়ন পরিষদ) — keep Bengali
    "ওয়ার্ড": None,              # naturalized administrative term —
                                  # keep Bengali
    "নাম্বার": "number",
    "অফিস": "office",
    "ব্যাংক": "bank",
    "পাসপোর্ট": "passport",
    "ফর্ম": "form",
    "প্রজেক্ট": "project",
    # প্রজেক্টে dropped — locative -ে suffix; keep the inflected
    # form Bengali (only bare lemma transliterates)
    # Health
    "ভ্যাকসিন": "vaccine",
    "টিকা": None,                 # Bengali native word; NOT translit
    "হসপিটাল": "hospital",
    "হাসপাতাল": None,             # Bengali native
    "ডাক্তার": None,               # Bengali native (loan but settled)
    "ক্লিনিক": "clinic",
    "মেডিক্যাল": "medical",
    "মেডিসিন": "medicine",
    "কোভিড": "COVID",
    "করোনা": "corona",
    "ভাইরাস": "virus",
    "মাস্ক": "mask",
    # Education
    "স্কুল": "school",
    "কলেজ": "college",
    "ইউনিভার্সিটি": "university",
    "ক্লাস": "class",
    "টিচার": "teacher",
    "স্টুডেন্ট": "student",
    # Tech / media
    "টিভি": "TV",
    "মোবাইল": "mobile",
    "ফোন": "phone",
    "ইন্টারনেট": "internet",
    "কম্পিউটার": "computer",
    "ফেসবুক": "Facebook",
    "ইমেইল": "email",
    "ভিডিও": "video",
    # Finance / livelihood (the survey domain)
    "লোন": "loan",
    "ক্রেডিট": "credit",
    "ইনকাম": "income",
    "ক্যাশ": "cash",
    "টাকা": None,                  # Bengali native
    "বিজনেস": "business",
    "মার্কেট": "market",
    "ফ্যামিলি": "family",
    "ইন্স্যুরেন্স": "insurance",
    # Disaster / climate
    "সাইক্লোন": "cyclone",
    "ফ্লাড": "flood",
    "ক্লাইমেট": "climate",
    "পলিসি": "policy",
    "ক্যাম্প": "camp",
    "শেল্টার": "shelter",
    "রিলিফ": "relief",
    "ইমার্জেন্সি": "emergency",
    # Common code-switch verbs/markers
    "ওকে": "ok",
    "অকে": "ok",
    "প্রবলেম": "problem",
    "সলভ": "solve",
    "সিস্টেম": "system",
    "প্ল্যান": "plan",
    "চান্স": "chance",
    "ব্যাকগ্রাউন্ড": "background",
    # The specific consent-script terms our GT has
    "ডাটা": "data",
    "এনালাইসিস": "analysis",
    "অ্যানালাইসিস": "analysis",
    "টেকনিক্যাল": "technical",
    "অ্যাসিস্ট্যান্ট": "assistant",
    "লিমিটেড": "Limited",
    "রিসার্চ": "research",
    "ইন্টারভিউ": "interview",
    "মেন্টাল": "mental",
    "মডেল": "model",
    "রেকর্ড": "record",
    "রেকর্ডিং": "recording",
}

GT_COUNTS = DATA_DIR / "gt_lm" / "gt_unigrams_counts.tsv"

counts: dict[str, int] = {}
if GT_COUNTS.exists():
    for ln in GT_COUNTS.read_text(encoding="utf-8").splitlines():
        try:
            w, c = ln.rstrip().split("\t")
            counts[w] = int(c)
        except ValueError:
            continue

# Three categories: kept, zero-count (drop), explicitly None (Bengali
# native, kept untranslated)
kept: dict[str, str] = {}
zero: list[tuple[str, str]] = []
native: list[str] = []

for bn, lat in CANDIDATES.items():
    if lat is None:
        native.append(bn)
        continue
    c = counts.get(bn, 0)
    if c > 0:
        kept[bn] = lat
    else:
        zero.append((bn, lat))

# Also surface any high-frequency GT tokens that look like
# transliterations but aren't in our candidate list — for human review
TRANSLIT_HINT = set("্য্যাশঈষঋৃৌঐৈ")  # uncommon-in-pure-Bengali clusters
def likely_translit(w: str) -> bool:
    # crude heuristic: contains atypical char OR ends in unusual cluster
    return any(ch in TRANSLIT_HINT for ch in w) and len(w) >= 4

possible_new = [(w, c) for w, c in counts.items()
                if c >= 3 and w not in CANDIDATES
                and likely_translit(w)]
possible_new.sort(key=lambda x: -x[1])

OUT_JSON = DATA_DIR / "translit.json"
OUT_TSV = DATA_DIR / "translit.review.tsv"
OUT_JSON.write_text(json.dumps(kept, ensure_ascii=False, indent=2),
                    encoding="utf-8")

lines = ["# bn_form\tlatin\tgt_count\tstatus"]
for bn, lat in sorted(kept.items(), key=lambda x: -counts.get(x[0], 0)):
    lines.append(f"{bn}\t{lat}\t{counts.get(bn,0)}\tKEPT")
for bn, lat in sorted(zero, key=lambda x: x[0]):
    lines.append(f"{bn}\t{lat}\t0\tdrop_zero_count")
for bn in sorted(native):
    lines.append(f"{bn}\t-\t{counts.get(bn,0)}\tnative_keep_bengali")
lines.append("")
lines.append("# === HIGH-FREQ GT TOKENS NOT IN TABLE — review for translit additions ===")
for w, c in possible_new[:30]:
    lines.append(f"{w}\t?\t{c}\tcandidate_for_review")
OUT_TSV.write_text("\n".join(lines) + "\n", encoding="utf-8")

print(f"kept entries:        {len(kept)}")
print(f"zero-count (dropped):{len(zero)}")
print(f"native (kept Bengali):{len(native)}")
print(f"candidates for review:{len(possible_new)}")
print()
print(f"-> {OUT_JSON}")
print(f"-> {OUT_TSV}")
print()
print("Top 20 KEPT entries by GT frequency:")
for bn, lat in sorted(kept.items(),
                      key=lambda x: -counts.get(x[0], 0))[:20]:
    print(f"  {bn:18s} -> {lat:14s}  (gt_count={counts.get(bn,0)})")
