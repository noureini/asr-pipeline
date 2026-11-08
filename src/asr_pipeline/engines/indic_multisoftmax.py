# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modified for asr_pipeline: only the multi-softmax compatibility patches are
# kept (taken from nemo_curator/models/audio/indic_conformer_hybrid.py,
# NVIDIA-NeMo/Curator, main branch, fetched 2026-11-08); the Curator stage
# code and the loguru dependency were removed.

"""Load AI4Bharat IndicConformer *hybrid* checkpoints in stock NeMo.

These checkpoints were trained with AI4Bharat's NeMo fork, which adds a
*multi-softmax* head: one shared Conformer encoder, and per-language output
heads selected by ``language_id``. Stock NeMo cannot restore them:

  * ``RNNTDecoder(multisoftmax=...)``                  -> unexpected kwarg
  * ``RNNTJoint(multilingual=..., language_keys=...)`` -> unexpected kwargs +
    a per-language ``ModuleDict`` final layer
  * ``ConvASRDecoder(multisoftmax=...)``               -> unexpected kwarg

:func:`apply_multisoftmax_patches` patches just those classes (and the
"multilingual" tokenizer type). The patches are idempotent and additive: for a
normal NeMo model every patched path falls back to the original behaviour.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("asr_pipeline")

_PATCHED = False
# Passes ``multilingual`` / ``language_keys`` from the patched RNNTJoint.__init__
# into _joint_net_modules, which the original __init__ calls first.
_JOINT_CTX: dict[str, Any] = {}


def apply_multisoftmax_patches() -> None:
    """Idempotently patch ConvASRDecoder / RNNTJoint / RNNTDecoder for multi-softmax."""
    global _PATCHED
    if _PATCHED:
        return

    import torch
    from nemo.collections.asr.modules import conv_asr, rnnt
    from nemo.collections.asr.parts.mixins.mixins import ASRBPEMixin

    # ------------------------------------------------------------------
    # Tokenizer routing: the fork tags the aggregate tokenizer ``type:
    # multilingual``; stock NeMo only routes ``agg`` to the aggregate path and
    # sends everything else to the monolingual path (which needs a top-level
    # ``dir`` key and fails). Treat ``multilingual`` as an aggregate tokenizer.
    # ------------------------------------------------------------------
    _orig_setup_tokenizer = ASRBPEMixin._setup_tokenizer

    def _setup_tokenizer(self: Any, tokenizer_cfg: Any) -> None:
        ttype = tokenizer_cfg.get("type")
        if ttype is not None and str(ttype).lower() == "multilingual":
            self._setup_aggregate_tokenizer(tokenizer_cfg)
            # Stock NeMo keys its aggregate-tokenizer handling (vocabulary as a
            # list, CTC vocab wiring) off ``tokenizer_type == "agg"``; the fork
            # used "multilingual" for the same thing. Normalise so the model's
            # own __init__ takes the aggregate branch.
            self.tokenizer_type = "agg"
            self._derive_tokenizer_properties()
            return
        _orig_setup_tokenizer(self, tokenizer_cfg)

    ASRBPEMixin._setup_tokenizer = _setup_tokenizer

    # ------------------------------------------------------------------
    # ConvASRDecoder (auxiliary CTC head)
    # ------------------------------------------------------------------
    _ConvASRDecoder = conv_asr.ConvASRDecoder
    _conv_orig_init = _ConvASRDecoder.__init__

    def _conv_init(
        self: Any, *args: Any, multisoftmax: bool = False, language_masks: Any = None, **kwargs: Any
    ) -> None:
        # Structure is identical to stock NeMo; the extra kwargs only gate the
        # per-language masking applied in forward(). Drop them before delegating.
        _conv_orig_init(self, *args, **kwargs)
        self.multisoftmax = multisoftmax
        self.language_masks = language_masks

    def _conv_forward(self: Any, encoder_output: Any, language_ids: Any = None) -> Any:
        # Mirrors AI4Bharat fork conv_asr.ConvASRDecoder.forward (no @typecheck so
        # language_ids is accepted). decoder_layers -> [B, T, C]; optional mask to
        # the language's contiguous token block + blank, then log_softmax.
        if self.is_adapter_available():
            encoder_output = encoder_output.transpose(1, 2)
            encoder_output = self.forward_enabled_adapters(encoder_output)
            encoder_output = encoder_output.transpose(1, 2)

        if self.temperature != 1.0:
            decoder_output = self.decoder_layers(encoder_output).transpose(1, 2) / self.temperature
        else:
            decoder_output = self.decoder_layers(encoder_output).transpose(1, 2)

        if language_ids is not None:
            sample_mask = torch.tensor([self.language_masks[lang] for lang in language_ids], dtype=torch.bool)
            mask = sample_mask.unsqueeze(1).repeat(1, decoder_output.shape[1], 1).to(decoder_output.device)
            decoder_output = torch.masked_select(decoder_output, mask).view(
                decoder_output.shape[0], decoder_output.shape[1], -1
            )
        return torch.nn.functional.log_softmax(decoder_output, dim=-1)

    _ConvASRDecoder.__init__ = _conv_init
    _ConvASRDecoder.forward = _conv_forward

    # ------------------------------------------------------------------
    # RNNTDecoder (shared prediction network) — only absorbs the extra kwargs.
    # ------------------------------------------------------------------
    _RNNTDecoder = rnnt.RNNTDecoder
    _dec_orig_init = _RNNTDecoder.__init__

    def _dec_init(
        self: Any, *args: Any, multisoftmax: bool = False, language_masks: Any = None, **kwargs: Any
    ) -> None:
        _dec_orig_init(self, *args, **kwargs)
        self.multisoftmax = multisoftmax
        self.language_masks = language_masks

    _RNNTDecoder.__init__ = _dec_init

    # ------------------------------------------------------------------
    # RNNTJoint — per-language ModuleDict final layer + language routing.
    # ------------------------------------------------------------------
    _RNNTJoint = rnnt.RNNTJoint
    _joint_orig_init = _RNNTJoint.__init__
    _joint_orig_jnm = _RNNTJoint._joint_net_modules

    def _joint_init(
        self: Any,
        *args: Any,
        multilingual: bool = False,
        language_keys: Any = None,
        language_masks: Any = None,
        token_id_offsets: Any = None,
        offset_token_ids_by_token_id: Any = None,
        **kwargs: Any,
    ) -> None:
        # _joint_net_modules runs *inside* the original __init__ before we can set
        # instance attrs, so stash what it needs in module-level scratch.
        _JOINT_CTX["multilingual"] = multilingual
        _JOINT_CTX["language_keys"] = list(language_keys) if language_keys is not None else None
        try:
            _joint_orig_init(self, *args, **kwargs)
        finally:
            _JOINT_CTX.clear()
        self.multilingual = multilingual
        self.language_keys = list(language_keys) if language_keys is not None else None
        self.language_masks = language_masks
        self.token_id_offsets = token_id_offsets
        self.offset_token_ids_by_token_id = offset_token_ids_by_token_id

    def _joint_net_modules(
        self: Any,
        num_classes: int,
        pred_n_hidden: int,
        enc_n_hidden: int,
        joint_n_hidden: int,
        activation: str,
        dropout: float,
    ) -> Any:
        if not _JOINT_CTX.get("multilingual"):
            return _joint_orig_jnm(self, num_classes, pred_n_hidden, enc_n_hidden, joint_n_hidden, activation, dropout)
        language_keys = _JOINT_CTX["language_keys"]
        pred = torch.nn.Linear(pred_n_hidden, joint_n_hidden)
        enc = torch.nn.Linear(enc_n_hidden, joint_n_hidden)
        act = activation.lower()
        if act == "relu":
            act_mod: Any = torch.nn.ReLU(inplace=True)
        elif act == "sigmoid":
            act_mod = torch.nn.Sigmoid()
        elif act == "tanh":
            act_mod = torch.nn.Tanh()
        else:
            msg = f"Unsupported activation for joint step: {activation}"
            raise ValueError(msg)
        # Per-language head: V/num_langs (+1 for blank). self._vocab_size is the
        # full aggregate vocab; it is set before this method is called.
        per_lang = self._vocab_size // len(language_keys) + 1
        final_layer = torch.nn.ModuleDict({lang: torch.nn.Linear(joint_n_hidden, per_lang) for lang in language_keys})
        logger.debug("Multilingual RNNT joint: %d heads x %d classes", len(language_keys), per_lang)
        layers = [act_mod] + ([torch.nn.Dropout(p=dropout)] if dropout else []) + [final_layer]
        return pred, enc, torch.nn.Sequential(*layers)

    def _joint_after_projection(self: Any, f: Any, g: Any, language_ids: Any = None) -> Any:
        # Mirrors fork RNNTJoint.joint_after_projection with language routing.
        f = f.unsqueeze(dim=2)  # (B, T, 1, H)
        g = g.unsqueeze(dim=1)  # (B, 1, U, H)
        inp = f + g  # (B, T, U, H)
        del f, g
        if self.is_adapter_available():
            inp = self.forward_enabled_adapters(inp)

        if language_ids is not None:
            for module in self.joint_net[:-1]:
                inp = module(inp)
            if len(set(language_ids)) == 1:
                res = self.joint_net[-1][language_ids[0]](inp)
            else:
                res = torch.stack(
                    [self.joint_net[-1][lang](single) for single, lang in zip(inp, language_ids, strict=True)]
                )
        else:
            res = self.joint_net(inp)
        del inp

        if self.preserve_memory:
            torch.cuda.empty_cache()
        if self.log_softmax is None:
            if not res.is_cuda:
                res = (
                    (res / self.temperature).log_softmax(dim=-1)
                    if self.temperature != 1.0
                    else res.log_softmax(dim=-1)
                )
        elif self.log_softmax:
            res = (res / self.temperature).log_softmax(dim=-1) if self.temperature != 1.0 else res.log_softmax(dim=-1)
        return res

    _RNNTJoint.__init__ = _joint_init
    _RNNTJoint._joint_net_modules = _joint_net_modules
    _RNNTJoint.joint_after_projection = _joint_after_projection

    _PATCHED = True
    logger.debug("Applied AI4Bharat multi-softmax patches to NeMo")
