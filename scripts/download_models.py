"""
Download the Bengali fusion models at the exact commits pinned in a config.

Fetches into the Hugging Face cache:
  - titu FastConformer-CTC (.nemo)            -> witness 1
  - regional Whisper-medium (safetensors)      -> witness 2
  - Gemma-3-12B-it Q4_K_XL (GGUF)              -> judge

then builds the CTranslate2 copy of the Whisper witness (whisper_ct2_path),
which the fusion stage uses for ~1.9x faster batched N-best. Run once with a
network connection; later pipeline runs are fully offline.

Usage:
    uv run python scripts/download_models.py                 # fusion_production.yaml
    uv run python scripts/download_models.py -c my_config.yaml
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

from asr_pipeline.config import load_config

# Only the files inference needs (skips training artefacts).
_WHISPER_FILES = ["*.json", "*.txt", "model.safetensors"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("-c", "--config", type=Path,
                    default=Path("fusion_production.yaml"))
    args = ap.parse_args()

    cfg = load_config(args.config)
    titu, fc = cfg.titu, cfg.postprocessing.fusion

    print(f"[1/4] titu     {titu.model_id} @ {titu.revision or 'latest'}")
    hf_hub_download(titu.model_id, f"{titu.model_id.split('/')[-1]}.nemo",
                    revision=titu.revision)

    print(f"[2/4] whisper  {fc.whisper_model} @ {fc.whisper_revision or 'latest'}")
    whisper_dir = snapshot_download(fc.whisper_model,
                                    revision=fc.whisper_revision,
                                    allow_patterns=_WHISPER_FILES)

    print(f"[3/4] judge    {fc.gguf_repo}/{fc.gguf_file} @ "
          f"{fc.gguf_revision or 'latest'}  (~7 GB)")
    hf_hub_download(fc.gguf_repo, fc.gguf_file, revision=fc.gguf_revision)

    ct2 = Path(fc.whisper_ct2_path) if fc.whisper_ct2_path else None
    if ct2 is None:
        print("[4/4] CT2 copy skipped (whisper_ct2_path is empty)")
    elif (ct2 / "model.bin").exists():
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
