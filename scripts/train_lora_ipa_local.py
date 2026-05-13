"""
LoRA fine-tune a small LLM on (IPA → Bengali + English) locally.

Designed for RTX 3060 Mobile 6GB. Uses Qwen2.5-1.5B-Instruct (default)
which fits comfortably with QLoRA + reasonable batch size.

Trains in ~30-60 min on the 25K-pair IPA dataset. Outputs:
  - LoRA adapter (~30 MB)
  - Optional GGUF (~1 GB)

Usage:
  uv run python scripts/train_lora_ipa_local.py \
    --train ./lora_data_ipa/lora_dataset_full_train.jsonl \
    --val   ./lora_data_ipa/lora_dataset_full_val.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


SYSTEM_PROMPT = (
    "You are a Bengali linguist and translator. You receive IPA "
    "(International Phonetic Alphabet) transcriptions of Bengali speech.\n"
    "Your job:\n"
    "1. Read the IPA phonetically\n"
    "2. Determine the correct Bengali text\n"
    "3. Translate to English\n\n"
    "Output format:\n"
    "BN: [correct Bengali text]\n"
    "EN: [English translation]"
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--val", type=Path, required=True)
    p.add_argument("--model",
                   default="unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit",
                   help="Small model that fits 6GB VRAM. Alternatives: "
                        "unsloth/Qwen2.5-3B-Instruct-bnb-4bit (tighter), "
                        "unsloth/aya-expanse-8b-bnb-4bit (very tight)")
    p.add_argument("--output-dir", type=Path,
                   default=Path("./models/qwen_ipa_lora"))
    p.add_argument("--max-seq-len", type=int, default=1024)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--export-gguf", default="q4_k_m",
                   help="GGUF quantization or 'none' to skip")
    args = p.parse_args()

    # Apply patch in case unsloth has the _IS_MLX issue
    try:
        import unsloth_zoo
    except ImportError:
        pass
    import unsloth as _u
    init_path = os.path.join(os.path.dirname(_u.__file__), '__init__.py')
    with open(init_path) as f:
        content = f.read()
    if '_IS_MLX' not in content:
        with open(init_path, 'a') as f:
            f.write('\n_IS_MLX = False\n')
        # Clear pyc cache
        pycache = os.path.dirname(init_path) + "/__pycache__"
        if os.path.exists(pycache):
            for f in os.listdir(pycache):
                if f.startswith('__init__'):
                    os.remove(os.path.join(pycache, f))
        print("✓ unsloth patched, please re-run if reload fails")

    from unsloth import FastLanguageModel, is_bfloat16_supported
    from unsloth.chat_templates import train_on_responses_only
    from datasets import Dataset
    from trl import SFTTrainer, SFTConfig

    # ─── Load model ─────────────────────────────────────────────
    print(f"Loading {args.model}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_len,
        dtype=None,
        load_in_4bit=True,
    )

    # ─── Add LoRA ───────────────────────────────────────────────
    print(f"LoRA r={args.lora_r} alpha={args.lora_alpha}")
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.lora_alpha,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )

    # ─── Load + format data ─────────────────────────────────────
    print(f"Loading train: {args.train}")
    train_rows = []
    with open(args.train, encoding="utf-8") as f:
        for line in f:
            train_rows.append(json.loads(line))
    print(f"  {len(train_rows)} samples")

    print(f"Loading val: {args.val}")
    val_rows = []
    with open(args.val, encoding="utf-8") as f:
        for line in f:
            val_rows.append(json.loads(line))
    print(f"  {len(val_rows)} samples")

    def format_chat(ex):
        user_msg = ex["ipa"]
        assistant_msg = f"BN: {ex['bengali']}"
        if ex.get("english"):
            assistant_msg += f"\nEN: {ex['english']}"
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_msg},
        ]
        return {"text": tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False)}

    train_ds = Dataset.from_list(train_rows).map(
        format_chat, remove_columns=list(train_rows[0].keys())
    )
    val_ds = Dataset.from_list(val_rows).map(
        format_chat, remove_columns=list(val_rows[0].keys())
    )

    print("\n── Sample training text ──")
    print(train_ds[0]["text"][:500])
    print("──────────────────────────\n")

    # ─── Trainer ────────────────────────────────────────────────
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        dataset_text_field="text",
        max_seq_length=args.max_seq_len,
        dataset_num_proc=2,
        packing=False,
        args=SFTConfig(
            output_dir=str(args.output_dir / "checkpoints"),
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            warmup_ratio=0.05,
            lr_scheduler_type="cosine",
            optim="adamw_8bit",
            weight_decay=0.01,
            bf16=is_bfloat16_supported(),
            fp16=not is_bfloat16_supported(),
            logging_steps=10,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
            seed=args.seed,
            report_to="none",
        ),
    )

    # Detect chat template markers
    model_lower = args.model.lower()
    if "qwen" in model_lower:
        instruction_part = "<|im_start|>user\n"
        response_part = "<|im_start|>assistant\n"
    elif "aya" in model_lower or "cohere" in model_lower:
        instruction_part = "<|START_OF_TURN_TOKEN|><|USER_TOKEN|>"
        response_part = "<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>"
    else:
        instruction_part = "<|im_start|>user\n"
        response_part = "<|im_start|>assistant\n"

    trainer = train_on_responses_only(
        trainer,
        instruction_part=instruction_part,
        response_part=response_part,
    )

    print("\n=== Training ===\n")
    trainer.train()

    # ─── Save ───────────────────────────────────────────────────
    final_dir = args.output_dir / "final"
    print(f"\nSaving LoRA adapter → {final_dir}")
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    # ─── GGUF export ────────────────────────────────────────────
    if args.export_gguf and args.export_gguf.lower() != "none":
        gguf_dir = args.output_dir / "gguf"
        print(f"\nExporting GGUF ({args.export_gguf}) → {gguf_dir}")
        try:
            model.save_pretrained_gguf(
                str(gguf_dir), tokenizer,
                quantization_method=args.export_gguf,
            )
            print(f"  ✓ GGUF saved")
        except Exception as e:
            print(f"  ⚠ GGUF export failed: {e}")
            print(f"  Adapter still saved at {final_dir}")

    print("\nDONE.")


if __name__ == "__main__":
    main()
