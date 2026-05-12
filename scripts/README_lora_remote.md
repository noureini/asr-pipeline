# Bengali ASR LoRA — Remote Pipeline

End-to-end training pipeline for fine-tuning a Bengali transcription corrector
on top of frozen ZIPA + FST output. Runs standalone on any GPU box with
clone → run → download workflow.

## What it does

```
Audio → ZIPA (frozen) → IPA tokens → FST → "noisy Bengali"
                                              ↓
                            LoRA-tuned Aya 8B / Qwen / etc.
                                              ↓
                                      "clean Bengali"
```

The LoRA is fine-tuned on (noisy_FST_output, ground_truth_Bengali) pairs
extracted from public Bengali ASR datasets.

## Hardware

| GPU VRAM | Recommended model |
|---|---|
| 8 GB | `unsloth/Qwen2.5-3B-Instruct-bnb-4bit` |
| 16 GB | `unsloth/aya-expanse-8b-bnb-4bit` (default) |
| 24 GB | `unsloth/aya-expanse-8b-bnb-4bit` w/ r=32 |
| 40 GB+ | `unsloth/aya-expanse-32b-bnb-4bit` for max quality |
| 98 GB | Aya 32B + r=32 + bigger batch — ideal |

## Quick start (98 GB box)

```bash
# 1. Clone
git clone https://github.com/<your_user>/asr_pipeline.git
cd asr_pipeline

# 2. (optional) HF token if you use gated repos
export HF_TOKEN="hf_xxxxxxxxx"

# 3. Run end-to-end
bash scripts/run_lora_pipeline.sh --full
```

## Modes

| Mode | Sources | Time | Total pairs |
|---|---|---|---|
| `--quick` | FLEURS + Bengali_AI_Speech + banspeech | ~2h extract + 2h train | ~42K |
| `--full` | + SKNahin (24h budget) | ~25h extract + 8h train | ~900K+ |

## Resumability

Every stage is resume-safe:

- **Dataset extraction**: skips IDs already in the JSONL files
- **Training**: Unsloth saves per-epoch checkpoints to `lora_models/<name>/checkpoints/`

If the run crashes or you Ctrl+C, just re-run the same command. It picks up.

## Custom configs

### Use a bigger model (98 GB box can handle Aya 32B)

```bash
python scripts/train_lora_remote.py \
  --dataset ./lora_data/lora_dataset_full.jsonl \
  --output-dir ./lora_models/aya32b_v1 \
  --model unsloth/aya-expanse-32b-bnb-4bit \
  --lora-r 32 \
  --batch-size 2 \
  --grad-accum 4 \
  --epochs 3
```

### Use Qwen instead of Aya (Apache license, no NC)

```bash
python scripts/train_lora_remote.py \
  --dataset ./lora_data/lora_dataset_full.jsonl \
  --output-dir ./lora_models/qwen32b_v1 \
  --model unsloth/Qwen2.5-32B-Instruct-bnb-4bit \
  --lora-r 32 --batch-size 2 --grad-accum 4 --epochs 3
```

### Run only a subset of sources

```bash
python scripts/build_lora_dataset_remote.py \
  --output-dir ./lora_data \
  --sources fleurs bengali_ai_speech \
  --max-hours 2
```

### Tighter SKNahin quality filter

```bash
python scripts/build_lora_dataset_remote.py \
  --output-dir ./lora_data \
  --sources sknahin \
  --sknahin-wer-max 0.5 \
  --max-hours 6
```

## Outputs

After successful run:

```
lora_data/
├── fleurs.jsonl              ~3K formal news with EN translation
├── bengali_ai_speech.jsonl   ~32K conversational
├── banspeech.jsonl           ~8K multi-domain (audiobook/news/drama/etc.)
├── sknahin.jsonl             N samples (depends on time budget)
└── lora_dataset_full.jsonl   combined training file

lora_models/aya8b_v1/
├── checkpoints/              per-epoch checkpoints
├── final/                    LoRA adapter (~80 MB) — for HF/Unsloth inference
└── gguf/
    └── unsloth.Q4_K_M.gguf   ~5 GB standalone model — for llama.cpp/Ollama
```

## Inference

```python
from llama_cpp import Llama

m = Llama(
    model_path="lora_models/aya8b_v1/gguf/unsloth.Q4_K_M.gguf",
    n_ctx=2048, n_gpu_layers=-1,  # all on GPU
)

# After your audio → ZIPA → FST chain produces noisy Bengali:
noisy = "একজনশুদুমাত্রআশ্চর্যহতেপারে"

resp = m.create_chat_completion(
    messages=[
        {"role": "system", "content":
            "You are a Bengali transcription corrector..."},
        {"role": "user", "content": noisy},
    ],
    temperature=0.0, max_tokens=512,
)
print(resp["choices"][0]["message"]["content"])
```

## Troubleshooting

### `datasets` version errors

If you see `trust_remote_code is not supported`, the pinned versions weren't
installed. Run:

```bash
uv pip install --force-reinstall \
  "datasets==2.21.0" "huggingface_hub==0.24.7" "fsspec<=2024.6.1"
```

### CUDA out-of-memory during training

Reduce batch size or use a smaller model:

```bash
python scripts/train_lora_remote.py --batch-size 1 --grad-accum 8 ...
```

### Resume after crash

Just re-run the same command. Both extraction and training are resume-safe.

### GGUF export fails

Adapter is still saved in `lora_models/<name>/final/`. Convert manually:

```bash
git clone https://github.com/ggerganov/llama.cpp /tmp/llama.cpp
python /tmp/llama.cpp/convert_hf_to_gguf.py \
  ./lora_models/aya8b_v1/merged_16bit \
  --outfile ./lora_models/aya8b_v1/gguf/model.Q4_K_M.gguf
```

## Strategy notes

### Why these data sources

- **FLEURS** — formal news Bengali (your existing 1.5K from local extraction)
- **Bengali_AI_Speech** (32K) — conversational, matches survey speech
- **banspeech** (8K) — multi-domain (audiobook, drama, news, lectures, parliament)
- **SKNahin** (3.73M total, filtered) — aggregator including CV, OpenSLR, UCLA,
  MADASR, Shrutilipi, Kathbath, IndicTTS, Gali, FLEURS

### Why filter SKNahin by `wer < 1.0` instead of `is_better=True`

`is_better=True` keeps only easy audio (clean + transcript matches model
prediction well). For real survey transcription (noisy field recordings,
phone calls, regional accents), training only on easy data → model overfits
to clean inputs and fails on hard ones. `wer < 1.0` keeps challenging-but-
correctly-labeled samples for robust generalization.

### Why Aya Expanse for Bengali

Cohere's Aya Expanse was purpose-built for multilingual coverage including
Bengali. Stronger Bengali pre-training than Qwen or Llama at the same size.
**License: CC-BY-NC** (research only). Use Qwen-32B if commercial.
