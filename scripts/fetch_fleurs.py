"""
Fetch a small FLEURS test set for any language, to compare candidate models.

FLEURS (google/fleurs) has the same ~350 sentences read in 102 languages,
with human English translations, so one script gives a comparable test set
for any language the pipeline might add. The audio archive is streamed and
only the first N clips with DISTINCT sentences are kept (several speakers
read each sentence), so only a few MB are downloaded.

Writes to OUT_DIR:  *.wav (16 kHz)  ref.tsv (file, transcript, english)

Usage:
    uv run --frozen python scripts/fetch_fleurs.py ne_np ~/fleurs_ne -n 100
FLEURS codes: ne_np (Nepali), bn_in (Bengali), hi_in, ur_pk, sw_ke, am_et …
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import tarfile
import urllib.request
from pathlib import Path

BASE = "https://huggingface.co/datasets/google/fleurs/resolve/main/data"


def read_tsv(url: str) -> list[list[str]]:
    with urllib.request.urlopen(url) as r:
        lines = r.read().decode("utf-8").splitlines()
    return list(csv.reader(lines, delimiter="\t", quoting=csv.QUOTE_NONE))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("lang", help="FLEURS config, e.g. ne_np")
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("-n", type=int, default=100, help="number of clips")
    args = ap.parse_args()
    out = args.out_dir.expanduser()
    out.mkdir(parents=True, exist_ok=True)

    rows = {r[1]: r for r in read_tsv(f"{BASE}/{args.lang}/test.tsv")}
    english = {r[0]: r[2] for r in read_tsv(f"{BASE}/en_us/test.tsv")}

    kept: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    with urllib.request.urlopen(f"{BASE}/{args.lang}/audio/test.tar.gz") as r, \
            tarfile.open(fileobj=r, mode="r|gz") as tar:
        for member in tar:
            name = os.path.basename(member.name)
            row = rows.get(name)
            if not member.isfile() or row is None or row[0] in seen:
                continue
            (out / name).write_bytes(tar.extractfile(member).read())
            seen.add(row[0])
            kept.append((name, row[3], english.get(row[0], "")))
            if len(kept) == args.n:
                break

    with open(out / "ref.tsv", "w", encoding="utf-8", newline="") as f:
        csv.writer(f, delimiter="\t").writerows(kept)
    print(f"{len(kept)} clips (distinct sentences) + ref.tsv in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
