"""List .gguf files in a Hugging Face repo, OR search for GGUF repos.

Uses huggingface_hub (handles auth from HF_TOKEN / .env / cached login),
so it works on gated repos (Sarvam models are often gated).

Usage:
    # list files in a known repo:
    uv run python scripts/list_gguf.py unsloth/gemma-3-12b-it-GGUF

    # search for repos when you don't know the exact name:
    uv run python scripts/list_gguf.py --search sarvam-30b
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _load_env_token() -> str | None:
    # 1) already in environment
    for k in ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        if os.environ.get(k):
            return os.environ[k]
    # 2) parse a .env at repo root
    env = Path(__file__).resolve().parents[1] / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key in ("HF_TOKEN", "HUGGINGFACE_TOKEN",
                       "HUGGING_FACE_HUB_TOKEN") and val:
                return val
    return None


def main() -> int:
    args = sys.argv[1:]
    if not args:
        print("Usage: list_gguf.py <repo-id> | --search <term>",
              file=sys.stderr)
        return 1

    from huggingface_hub import HfApi
    token = _load_env_token()
    api = HfApi(token=token)
    if token:
        print("(using HF token from env/.env)")

    if args[0] == "--search":
        term = " ".join(args[1:]) or "sarvam"
        print(f"Searching GGUF repos matching '{term}' ...")
        found = list(api.list_models(search=term, library="gguf",
                                     limit=40))
        if not found:
            # broaden: search without the gguf filter
            found = list(api.list_models(search=term, limit=40))
        for m in found:
            name = m.id
            if "gguf" in name.lower() or "gguf" in str(
                    getattr(m, "tags", [])).lower():
                print(f"  {name}")
        print(f"({len(found)} repos matched; GGUF-looking ones above)")
        return 0

    repo = args[0]
    try:
        files = api.list_repo_files(repo_id=repo)
    except Exception as e:
        print(f"Failed to list {repo}: {type(e).__name__}: {e}",
              file=sys.stderr)
        print("If 401/403: the repo is gated — accept its license on "
              "the HF website with your account, or it doesn't exist "
              "(try --search).", file=sys.stderr)
        return 1
    ggufs = sorted(f for f in files if f.endswith(".gguf"))
    if not ggufs:
        print(f"No .gguf files in {repo}. All files:")
        for f in sorted(files)[:40]:
            print(f"  {f}")
        return 1
    print(f"GGUF files in {repo}:")
    for f in ggufs:
        print(f"  {f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
