"""
Download the fusion models for one language at the exact pinned commits.

Fetches into the Hugging Face cache:
  - witness 1: the profile's CTC model (.nemo checkpoint or HF CTC model)
  - witness 2: the profile's Whisper fine-tune
  - the judge: Gemma GGUF from the shared config

then builds the CTranslate2 copy of the Whisper witness (witness_whisper.
ct2_path), which the fusion stage uses for ~1.9x faster batched N-best. Run
once with a network connection; later pipeline runs are fully offline.

Usage:
    uv run python scripts/download_models.py               # Bengali (ben)
    uv run python scripts/download_models.py -l nep        # languages/nep.yaml
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

from asr_pipeline.config import load_config
from asr_pipeline.engines.ctc_engine import CTCEngine
from asr_pipeline.profiles import REPO_ROOT, load_profile

# Only the files inference needs (skips training artefacts).
_HF_MODEL_FILES = ["*.json", "*.txt", "*.model", "model.safetensors"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("-l", "--language", default="ben",
                    help="ISO 639-3 code of languages/<code>.yaml")
    ap.add_argument("-c", "--config", type=Path,
                    default=Path("fusion_production.yaml"),
                    help="shared config (judge model)")
    ap.add_argument("--witnesses-only", action="store_true",
                    help="skip the ~7 GB judge (e.g. to test a new profile)")
    args = ap.parse_args()

    profile = load_profile(args.language)
    ctc, wh = profile.witness_ctc, profile.witness_whisper
    fc = load_config(args.config).postprocessing.fusion
    print(f"Language profile: {profile.code} ({profile.name})")

    print(f"[1/4] CTC      {ctc.model} @ {ctc.revision or 'latest'} "
          f"({ctc.backend})")
    if ctc.backend == "nemo":
        engine = CTCEngine(model_id=ctc.model, language=profile.code,
                           nemo_file=ctc.nemo_file)
        hf_hub_download(ctc.model, engine.nemo_filename, revision=ctc.revision)
    else:
        snapshot_download(ctc.model, revision=ctc.revision,
                          allow_patterns=_HF_MODEL_FILES)

    print(f"[2/4] whisper  {wh.model} @ {wh.revision or 'latest'}")
    whisper_dir = snapshot_download(wh.model, revision=wh.revision,
                                    allow_patterns=_HF_MODEL_FILES)

    if args.witnesses_only:
        print("[3/4] judge    skipped (--witnesses-only)")
    else:
        print(f"[3/4] judge    {fc.gguf_repo}/{fc.gguf_file} @ "
              f"{fc.gguf_revision or 'latest'}  (~7 GB)")
        hf_hub_download(fc.gguf_repo, fc.gguf_file,
                        revision=fc.gguf_revision)

    if not wh.ct2_path:
        print("[4/4] CT2 copy skipped (witness_whisper.ct2_path is empty)")
        return 0
    ct2 = Path(wh.ct2_path)
    if not ct2.is_absolute():
        ct2 = REPO_ROOT / ct2
    if (ct2 / "model.bin").exists():
        print(f"[4/4] CT2 copy already at {ct2}")
    else:
        print(f"[4/4] converting Whisper to CTranslate2 -> {ct2}")
        subprocess.run(
            ["ct2-transformers-converter", "--model", whisper_dir,
             "--output_dir", str(ct2), "--quantization", "float16",
             "--copy_files", "tokenizer.json", "preprocessor_config.json"],
            check=True)

    print("Done. Models are cached; runs can now use HF_HUB_OFFLINE=1.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
