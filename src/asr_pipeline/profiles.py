"""Language profiles for the two-witness fusion pipeline.

Everything language-specific lives in one YAML file per language,
`languages/<code>.yaml` (code = the ISO 639-3 code passed with `-l`):

  - witness 1: an acoustic CTC model (NeMo `.nemo` or a Hugging Face CTC model)
  - witness 2: a fluent Whisper fine-tune, with its Whisper language code
  - the judge prompt (dialect glossary, domain priors, number rules, roles)
  - the scripts a transcript may contain, and the "unclear" marker

The judge (Gemma), VAD, decoding and guard thresholds are shared and stay in
fusion_production.yaml. To add a language, copy `languages/_template.yaml`.
"""
from __future__ import annotations

import unicodedata
from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field, field_validator

# Repo root (two levels above this file): profiles and prompt paths in a
# profile are resolved against it unless absolute.
REPO_ROOT = Path(__file__).resolve().parents[2]
PROFILES_DIR = REPO_ROOT / "languages"


class CTCWitness(BaseModel):
    """Witness 1: acoustically literal CTC model with word timestamps."""

    backend: Literal["nemo", "hf"] = "nemo"
    model: str
    revision: Optional[str] = None          # pinned Hugging Face commit
    # nemo: checkpoint file in the repo (default "<repo-name>.nemo")
    nemo_file: Optional[str] = None
    # nemo hybrid models (CTC + RNN-T/TDT heads): which decoder to use.
    # "ctc" keeps the witness acoustically literal; RNN-T/TDT add an
    # internal language model. None = the checkpoint's default decoder.
    decoder: Optional[Literal["ctc", "rnnt"]] = None
    # hf: MMS-style adapter language (e.g. "npi"), None for plain models
    target_lang: Optional[str] = None
    batch_size: int = 8


class WhisperWitness(BaseModel):
    """Witness 2: fluent seq2seq Whisper fine-tune (N-best readings)."""

    model: str
    revision: Optional[str] = None          # pinned Hugging Face commit
    language: str                           # Whisper language code ("bn", "ne")
    # Local CTranslate2 conversion (faster batched N-best). Built by
    # scripts/download_models.py; "" → HF transformers path only.
    ct2_path: str = ""


class LanguageProfile(BaseModel):
    code: str                               # ISO 639-3, matches `-l`
    name: str
    witness_ctc: CTCWitness
    witness_whisper: WhisperWitness
    prompt_file: str
    # Line tag in the judge's reply ([bn], [ne] …) and in transcripts.
    tag: str
    # Unicode scripts a transcript may contain (first word of the Unicode
    # character name: BENGALI, DEVANAGARI, LATIN …). Anything else is
    # treated as a cross-lingual hallucination and blanked.
    scripts: list[str] = Field(default_factory=lambda: ["LATIN"])
    unclear_marker: str = "[unclear]"       # written for unintelligible text

    @field_validator("scripts")
    @classmethod
    def _upper(cls, v: list[str]) -> list[str]:
        return [s.upper() for s in v]

    @property
    def prompt_path(self) -> Path:
        p = Path(self.prompt_file)
        return p if p.is_absolute() else REPO_ROOT / p

    def allows(self, ch: str) -> bool:
        """True if `ch` may appear in a transcript of this language."""
        try:
            script = unicodedata.name(ch).split()[0]
        except ValueError:                  # unassigned codepoint
            return False
        return script in self.scripts


def available_profiles() -> list[str]:
    return sorted(p.stem for p in PROFILES_DIR.glob("*.yaml")
                  if not p.stem.startswith("_"))


def load_profile(code: str) -> LanguageProfile:
    """Load `languages/<code>.yaml`; a clear error lists what exists."""
    path = PROFILES_DIR / f"{code}.yaml"
    if not path.exists():
        have = ", ".join(available_profiles()) or "none"
        raise FileNotFoundError(
            f"No language profile for '{code}' ({path}). Available: {have}. "
            f"To add one, copy languages/_template.yaml to "
            f"languages/{code}.yaml.")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    profile = LanguageProfile(**data)
    if profile.code != code:
        raise ValueError(
            f"{path.name} declares code '{profile.code}', expected '{code}'")
    return profile
