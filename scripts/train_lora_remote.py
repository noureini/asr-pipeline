"""
LoRA fine-tune on built dataset, designed for remote compute.

With 98GB VRAM you have options:
  - Aya Expanse 8B (~6GB QLoRA)        — fast, proven
  - Aya Expanse 32B (~22GB QLoRA)      — best quality, slower
  - Qwen2.5-32B-Instruct (~22GB QLoRA) — Apache license alternative

Defaults to Aya 8B for fastest iteration. Bump to 32B once you've validated
the smaller model works.

Outputs:
  <output_dir>/checkpoints/  per-epoch checkpoints
  <output_dir>/final/         final LoRA adapter
  <output_dir>/gguf/          GGUF Q4_K_M (standalone, for llama.cpp)

Usage:
  uv run python scripts/train_lora_remote.py \
    --dataset ./lora_data/lora_dataset_full.jsonl \
    --output-dir ./lora_models/aya8b_v1 \
    --model unsloth/aya-expanse-8b-bnb-4bit \
    --epochs 3
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logger = logging.getLogger("train_lora_remote")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--model", default="unsloth/aya-expanse-8b-bnb-4bit",
                   help="Use unsloth/aya-expanse-32b-bnb-4bit if VRAM allows + license OK.")
    p.add_argument("--max-seq-len", type=int, default=1024)
    p.add_argument("--lora-r", type=int, default=16,
                   help="r=16 for narrow corrections, r=32 for wider coverage")
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--lora-dropout", type=float, default=0.0)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=4,
                   help="Bigger on 98GB VRAM. Drop to 2 if OOM with 32B.")
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--export-gguf", default="q4_k_m",
                   help="GGUF quantization or 'none' to skip")
    p.add_argument("--export-merged-16bit", action="store_true",
                   help="Also save full merged fp16 model (large)")
    p.add_argument("--system-prompt", default=(
        "You are a Bengali transcription corrector. The input is a noisy "
        "Bengali transcription where word boundaries are missing and some "
        "characters are phonetically confused. Output ONLY the corrected, "
        "properly-segmented Bengali sentence."
    ))
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")

    # ─── Heavy imports ──────────────────────────────────────────────────
    from unsloth import FastLanguageModel, is_bfloat16_supported
    from unsloth.chat_templates import train_on_responses_only
    from datasets import Dataset
    from trl import SFTTrainer, SFTConfig

    # ─── Load model ─────────────────────────────────────────────────────
    logger.info(f"Loading {args.model}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_len,
        dtype=None,
        load_in_4bit=True,
    )

    # ─── Add LoRA ───────────────────────────────────────────────────────
    logger.info(f"LoRA r={args.lora_r} alpha={args.lora_alpha}")
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
        use_rslora=False,
        loftq_config=None,
    )

    # ─── Load dataset ───────────────────────────────────────────────────
    logger.info(f"Loading dataset from {args.dataset}...")
    rows = []
    with open(args.dataset, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            rows.append({"noisy": d["noisy"], "bengali": d["bengali"]})
    logger.info(f"  {len(rows)} pairs loaded")

    # ─── Apply chat template ────────────────────────────────────────────
    def to_chat(ex):
        messages = [
            {"role": "system", "content": args.system_prompt},
            {"role": "user", "content": ex["noisy"]},
            {"role": "assistant", "content": ex["bengali"]},
        ]
        return {"text": tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False)}

    ds = Dataset.from_list(rows).map(
        to_chat, remove_columns=["noisy", "bengali"],
    )
    logger.info(f"  formatted with chat template")

    # ─── Trainer ────────────────────────────────────────────────────────
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trainer = SFTTrainer(
        model=model, tokenizer=tokenizer,
        train_dataset=ds, dataset_text_field="text",
        max_seq_length=args.max_seq_len,
        dataset_num_proc=4, packing=False,
        args=SFTConfig(
            output_dir=str(args.output_dir / "checkpoints"),
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            warmup_ratio=args.warmup_ratio,
            lr_scheduler_type="cosine",
            optim="adamw_8bit",
            weight_decay=args.weight_decay,
            bf16=is_bfloat16_supported(),
            fp16=not is_bfloat16_supported(),
            logging_steps=10,
            save_strategy="epoch",
            save_total_limit=2,
            seed=args.seed,
            report_to="none",
        ),
    )

    # Detect chat-template markers based on model family
    model_lower = args.model.lower()
    if "aya" in model_lower or "cohere" in model_lower:
        instruction_part = "<|START_OF_TURN_TOKEN|><|USER_TOKEN|>"
        response_part = "<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>"
    elif "qwen" in model_lower:
        instruction_part = "<|im_start|>user\n"
        response_part = "<|im_start|>assistant\n"
    else:
        # Generic fallback
        instruction_part = "<|im_start|>user\n"
        response_part = "<|im_start|>assistant\n"

    trainer = train_on_responses_only(
        trainer,
        instruction_part=instruction_part,
        response_part=response_part,
    )

    logger.info("=" * 60)
    logger.info("Starting training...")
    logger.info("=" * 60)
    trainer.train()

    # ─── Save artifacts ─────────────────────────────────────────────────
    final_dir = args.output_dir / "final"
    logger.info(f"Saving LoRA adapter → {final_dir}")
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    if args.export_merged_16bit:
        merged_dir = args.output_dir / "merged_16bit"
        logger.info(f"Saving merged fp16 → {merged_dir}")
        model.save_pretrained_merged(str(merged_dir), tokenizer,
                                     save_method="merged_16bit")

    if args.export_gguf and args.export_gguf.lower() != "none":
        gguf_dir = args.output_dir / "gguf"
        logger.info(f"Exporting GGUF ({args.export_gguf}) → {gguf_dir}")
        try:
            model.save_pretrained_gguf(
                str(gguf_dir), tokenizer,
                quantization_method=args.export_gguf,
            )
            logger.info(f"  ✓ GGUF saved")
        except Exception as e:
            logger.error(f"  GGUF export failed: {e}")
            logger.error("  Adapter is still in final/. You can convert later.")

    logger.info("DONE.")


if __name__ == "__main__":
    main()
