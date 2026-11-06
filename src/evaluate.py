"""
Evaluation script: runs the fine-tuned model over the test split and reports
multiple metrics to capture both surface-level and semantic correctness.

Metrics
-------
  exact_match     : normalized SQL string match (stripped, lowercased)
  token_f1        : token-level F1 vs. gold SQL (tolerates minor differences)
  valid_sql       : the SQL compiles in SQLite against the example's own schema

Usage:
    python src/evaluate.py --adapter runs/base --data data/test.jsonl \
                           --base_model Qwen/Qwen2.5-0.5B-Instruct

Leave out --adapter to score the base model; results go to runs/baseline.
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
from collections import Counter
from pathlib import Path

SCHEMA_PATTERN = re.compile(r"### Schema:\n(.*?)\n\n### Pergunta:", re.DOTALL)
FENCE_PATTERN = re.compile(r"```(?:sql)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def normalize(sql: str) -> str:
    sql = sql.strip().rstrip(";").strip()
    sql = re.sub(r"\s+", " ", sql)
    return sql.lower()


def token_f1(pred: str, gold: str) -> float:
    p = Counter(normalize(pred).split())
    g = Counter(normalize(gold).split())
    if not p or not g:
        return 0.0
    common = sum((p & g).values())
    if common == 0:
        return 0.0
    precision = common / sum(p.values())
    recall = common / sum(g.values())
    return 2 * precision * recall / (precision + recall)


def clean_sql(text: str) -> str:
    """Keep the SQL only: the fenced block if there is one, else the first paragraph."""
    fenced = FENCE_PATTERN.search(text)
    if fenced:
        return fenced.group(1).strip()
    return text.strip().split("\n\n")[0].strip()


def extract_schema(user_message: str) -> str:
    match = SCHEMA_PATTERN.search(user_message)
    if match is None:
        raise ValueError("user message has no ### Schema block")
    return match.group(1)


def sql_parses(sql: str, schema: str) -> bool:
    """Does the SQL compile against the schema? EXPLAIN resolves tables and columns."""
    conn = sqlite3.connect(":memory:")
    try:
        conn.executescript(schema)
        conn.execute(f"EXPLAIN {sql.strip().rstrip(';')}")
        return True
    except sqlite3.Error:
        return False
    finally:
        conn.close()


def generate(model, tokenizer, messages: list[dict], max_new_tokens: int = 128) -> str:
    import torch

    prompt = tokenizer.apply_chat_template(
        messages[:-1],  # drop the gold assistant turn
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return clean_sql(text)


def main():
    # Heavy imports stay here so the metrics above can be tested without a GPU stack.
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from common import compute_dtype

    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", type=str, default=None, help="LoRA adapter dir; omit for baseline")
    ap.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--data", type=str, default="data/test.jsonl")
    ap.add_argument("--output", type=str, default=None)
    ap.add_argument("--limit", type=int, default=0, help="Limit N examples (0 = all)")
    args = ap.parse_args()

    out_path = Path(args.output or Path(args.adapter or "runs/baseline") / "eval_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[eval] Loading base model: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=compute_dtype(),
        device_map="auto",
        trust_remote_code=True,
    )
    if args.adapter:
        print(f"[eval] Loading adapter: {args.adapter}")
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    examples = [json.loads(l) for l in open(args.data, encoding="utf-8")]
    if args.limit:
        examples = examples[: args.limit]

    predictions = []
    em_hits = 0
    f1_sum = 0.0
    parse_hits = 0
    per_template = {}

    for i, ex in enumerate(examples):
        gold = ex["messages"][-1]["content"]
        pred = generate(model, tokenizer, ex["messages"])
        em = int(normalize(pred) == normalize(gold))
        f1 = token_f1(pred, gold)
        parses = int(sql_parses(pred, extract_schema(ex["messages"][-2]["content"])))
        em_hits += em
        f1_sum += f1
        parse_hits += parses

        tpl = ex.get("template", "unknown")
        d = per_template.setdefault(tpl, {"n": 0, "em": 0, "f1": 0.0})
        d["n"] += 1
        d["em"] += em
        d["f1"] += f1

        predictions.append({
            "question": ex["messages"][-2]["content"][-300:],
            "gold": gold,
            "pred": pred,
            "em": em,
            "f1": round(f1, 3),
            "parses": parses,
            "template": tpl,
        })

        if (i + 1) % 25 == 0:
            print(f"  [{i+1}/{len(examples)}] EM={em_hits/(i+1):.3f}  F1={f1_sum/(i+1):.3f}")

    n = len(examples)
    summary = {
        "n": n,
        "exact_match": em_hits / n,
        "token_f1": f1_sum / n,
        "valid_sql": parse_hits / n,
        "per_template": {
            t: {"n": d["n"], "em": d["em"] / d["n"], "f1": d["f1"] / d["n"]}
            for t, d in per_template.items()
        },
    }
    print("\n[eval] Summary:")
    print(json.dumps(summary, indent=2))

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "predictions": predictions}, f, indent=2, ensure_ascii=False)
    print(f"[eval] Wrote {out_path}")


if __name__ == "__main__":
    main()
