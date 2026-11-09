# Text-to-SQL PT-BR

QLoRA fine-tune of Qwen2.5-0.5B-Instruct that turns Brazilian Portuguese questions into SQL. The repo holds the synthetic data generator, four training configs, the evaluation and a Colab notebook that runs all of it.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/baconshow/text2sql-ptbr-lora/blob/main/notebooks/colab_train.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

![architecture](assets/architecture.svg)

## Results

One notebook run on Colab, one seed, 88 test examples. Each training run took 4 to 5 minutes. The raw table is in [`results/ablation_table.md`](results/ablation_table.md).

| Run | LoRA r | Target modules | LR | Trainable params | Exact match | Token F1 | Valid SQL |
|---|---:|---|---:|---:|---:|---:|---:|
| no fine-tune | - | - | - | - | 5/88 (0.057) | 0.542 | 59/88 (0.670) |
| `base` | 16 | attention | 2e-4 | 2.16 M | 87/88 (0.989) | 0.997 | 87/88 (0.989) |
| `rank8` | 8 | attention | 2e-4 | 1.08 M | 86/88 (0.977) | 0.994 | 88/88 (1.000) |
| `rank32_allmodules` | 32 | attention + MLP | 2e-4 | 17.60 M | 88/88 (1.000) | 1.000 | 88/88 (1.000) |
| `lr_high` | 16 | attention | 5e-4 | 2.16 M | 88/88 (1.000) | 1.000 | 88/88 (1.000) |

How to read this:

- **The fine-tune works on this task.** Without it, the model adds joins, invents columns or answers in prose. With it, the model writes the SQL shape the dataset uses.
- **The perfect scores are not general Text-to-SQL skill.** Test queries are new, but they share templates and schemas with train, so this benchmark is easy once the model learns its format. See [Dataset](#dataset).
- **The runs do not rank.** They differ by at most 2 of 88 examples, on one seed. That gap is too small to pick a best rank or learning rate.

All three misses involve `estoque`. Train has no `MIN(estoque)`, and has `SUM(estoque)` only with `GROUP BY`:

| Run | Gold | Prediction |
|---|---|---|
| `base` | `SELECT MIN(estoque) FROM produtos;` | `SELECT MIN(presto) FROM produtos;` |
| `rank8` | `SELECT MIN(estoque) FROM produtos;` | `SELECT MIN(preco) FROM produtos;` |
| `rank8` | `SELECT SUM(estoque) FROM produtos;` | `SELECT estoque FROM produtos;` |

![training loss](assets/loss_curves.png)

Loss falls below 0.05 within 20 steps in all four runs, which also says the task is easy to fit.

## Design

| Decision | Choice | Why |
|---|---|---|
| Model | Qwen2.5-0.5B-Instruct | Fits a free T4 and runs on modest hardware. |
| Quantization | 4-bit NF4, double quant | The base model stays frozen and small in memory. |
| Loss | SQL answer tokens only | Prompt tokens are masked, so the model is not graded on copying the schema. |
| Precision | bf16 on Ampere or newer, fp16 on T4 | The T4 has no native bf16. |
| Data | Synthetic, about 900 unique examples | Spider and WikiSQL are in English. |
| Split | By gold SQL | A query in the test split never appears in train, even with other wording. |
| Metrics | Exact match, token F1, valid SQL | Valid SQL compiles the answer in SQLite against the example's own schema. |

## Dataset

`src/data_prep.py` is deterministic (`seed=42`). It samples one of 4 domains (e-commerce, HR, finance, healthcare) and one of 15 single-table templates: `SELECT *`, `COUNT`, `COUNT(DISTINCT)`, `AVG`, `SUM`, `MIN`, `MAX`, `ORDER BY ... LIMIT`, `GROUP BY`, `LIKE`, `BETWEEN`, and numeric and text `WHERE` filters.

The templates and value pools run out at about 900 unique examples, so `--n_samples 3000` stops there and says so. There are no joins, and test queries use the same templates and schemas as train. Scores measure new queries in known shapes, not new schemas.

```json
{
  "messages": [
    {"role": "system", "content": "Você é um assistente que converte perguntas em português para SQL."},
    {"role": "user", "content": "### Schema:\nCREATE TABLE ...\n\n### Pergunta:\nTraga todos os registros de funcionários."},
    {"role": "assistant", "content": "SELECT * FROM funcionarios;"}
  ],
  "domain": "rh",
  "template": "select_all"
}
```

## Run it

On Colab: open the notebook with the badge above, pick a T4 runtime and use `Run all`.

Locally, with a CUDA GPU:

```bash
git clone https://github.com/baconshow/text2sql-ptbr-lora.git
cd text2sql-ptbr-lora
pip install -r requirements.txt

python src/data_prep.py --output data --n_samples 3000 --seed 42

for cfg in base ablation_rank8 ablation_rank32_allmodules ablation_lr_high; do
    python src/train.py --config configs/$cfg.yaml
done

python src/evaluate.py --data data/test.jsonl
for run in base rank8 rank32_allmodules lr_high; do
    python src/evaluate.py --adapter runs/$run --data data/test.jsonl
done

python src/make_report.py --runs base rank8 rank32_allmodules lr_high
```

Try one question:

```bash
python src/inference.py --adapter runs/base \
    --schema "CREATE TABLE clientes (id INT, nome VARCHAR(100), cidade VARCHAR(50));" \
    --question "Quantos clientes temos em São Paulo?"
```

Tests need no GPU:

```bash
pip install pytest
pytest tests
```

## Layout

```
configs/      one YAML per run
src/          data_prep, train, evaluate, inference, make_report
notebooks/    the Colab runner
tests/        dataset, split, metrics and report checks
results/      ablation table, written by make_report
```

## Next

- Joins and sampled schemas, to test schemas the model has not seen. The current test split is too easy to separate the runs.
- More seeds, so differences between runs can mean something.
- Execution accuracy: run the SQL on synthetic rows and compare result sets.

## License

MIT
