"""
Interactive / scripted inference with the fine-tuned LoRA adapter.

Example:
    python src/inference.py \
        --adapter runs/base \
        --base_model Qwen/Qwen2.5-0.5B-Instruct \
        --schema "CREATE TABLE clientes (id INT, nome VARCHAR, cidade VARCHAR);" \
        --question "Quantos clientes temos em São Paulo?"
"""
from __future__ import annotations

import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import compute_dtype
from evaluate import clean_sql


def run(base_model: str, adapter: str, schema: str, question: str) -> str:
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model, dtype=compute_dtype(), device_map="auto", trust_remote_code=True
    )
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()

    messages = [
        {"role": "system", "content": "Você é um assistente que converte perguntas em português para SQL."},
        {"role": "user", "content": f"### Schema:\n{schema}\n\n### Pergunta:\n{question}"},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=128, do_sample=False,
                             pad_token_id=tokenizer.pad_token_id)
    text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return clean_sql(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", type=str, default=None)
    ap.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--schema", type=str, required=True)
    ap.add_argument("--question", type=str, required=True)
    args = ap.parse_args()

    sql = run(args.base_model, args.adapter, args.schema, args.question)
    print(f"\nQuestion: {args.question}")
    print(f"SQL:      {sql}")


if __name__ == "__main__":
    main()
