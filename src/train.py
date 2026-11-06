"""
QLoRA fine-tuning script for Text-to-SQL (PT-BR).

Fine-tunes a small causal LM (default: Qwen/Qwen2.5-0.5B-Instruct) on the
synthetic dataset produced by `data_prep.py`, using 4-bit quantization
(bitsandbytes) + LoRA adapters (PEFT).

Design decisions (ablated separately, see configs/):
  - LoRA rank, alpha, dropout
  - Learning rate
  - Number of epochs
  - Target modules (attention only vs attention + MLP)

Usage:
    python src/train.py --config configs/base.yaml

Loss is computed on the SQL answer only: prompt tokens are masked out of the
labels. Precision follows the GPU: bf16 where it is native, fp16 on a T4.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

from common import compute_dtype

IGNORE_INDEX = -100
MAX_LENGTH = 1024


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_tokenizer(model_name: str):
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok


def build_model(model_name: str, use_4bit: bool = True):
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype(),
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=compute_dtype(),
            device_map="auto",
            trust_remote_code=True,
        )
    return model


def format_example(example: dict, tokenizer) -> dict:
    """Apply chat template, tokenize, and mask the prompt out of the labels."""
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    prompt = tokenizer.apply_chat_template(
        example["messages"][:-1],
        tokenize=False,
        add_generation_prompt=True,
    )
    tokens = tokenizer(text, truncation=True, max_length=MAX_LENGTH, padding=False)
    prompt_len = len(tokenizer(prompt, truncation=True, max_length=MAX_LENGTH)["input_ids"])
    tokens["labels"] = [IGNORE_INDEX] * prompt_len + tokens["input_ids"][prompt_len:]
    return tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_name = cfg.get("run_name", Path(args.config).stem)
    output_dir = args.output_dir or f"runs/{run_name}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print(f"[train] Run: {run_name}")
    print(f"[train] Output dir: {output_dir}")
    print(f"[train] Config: {json.dumps(cfg, indent=2)}")

    # --- Tokenizer and model -------------------------------------------------
    tokenizer = build_tokenizer(cfg["model_name"])
    model = build_model(cfg["model_name"], use_4bit=cfg.get("use_4bit", True))

    # --- LoRA ---------------------------------------------------------------
    lora_config = LoraConfig(
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=cfg["target_modules"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # --- Data ---------------------------------------------------------------
    data_files = {
        "train": str(Path(args.data_dir) / "train.jsonl"),
        "val": str(Path(args.data_dir) / "val.jsonl"),
    }
    raw = load_dataset("json", data_files=data_files)
    tokenized = raw.map(
        lambda e: format_example(e, tokenizer),
        remove_columns=raw["train"].column_names,
        desc="Tokenizing",
    )

    # --- Training -----------------------------------------------------------
    grad_accum_steps = cfg.get("grad_accum_steps", 4)
    steps_per_epoch = math.ceil(len(tokenized["train"]) / (cfg["batch_size"] * grad_accum_steps))
    total_steps = steps_per_epoch * cfg["num_epochs"]

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=cfg["num_epochs"],
        per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=cfg["batch_size"],
        gradient_accumulation_steps=grad_accum_steps,
        learning_rate=cfg["learning_rate"],
        lr_scheduler_type=cfg.get("lr_scheduler", "cosine"),
        warmup_steps=int(total_steps * cfg.get("warmup_ratio", 0.03)),
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        save_total_limit=2,
        bf16=compute_dtype() == torch.bfloat16,
        fp16=compute_dtype() == torch.float16,
        optim=cfg.get("optim", "paged_adamw_8bit"),
        report_to="none",
        remove_unused_columns=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    collator = DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=IGNORE_INDEX)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["val"],
        data_collator=collator,
    )

    train_result = trainer.train()

    # Save adapter + tokenizer
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save metrics
    metrics = {
        "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "train_loss": float(train_result.training_loss),
        "train_runtime_sec": float(train_result.metrics.get("train_runtime", 0)),
        "config": cfg,
    }
    with open(Path(output_dir) / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Save loss history for plotting
    log_history = trainer.state.log_history
    with open(Path(output_dir) / "log_history.json", "w") as f:
        json.dump(log_history, f, indent=2)

    print(f"\n[train] Done. Adapter saved to {output_dir}")


if __name__ == "__main__":
    main()
