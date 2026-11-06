import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from data_prep import SCHEMAS, generate, split_by_sql, to_chat_message
from evaluate import clean_sql, extract_schema, normalize, sql_parses, token_f1
from make_report import build_table

SCHEMA = "CREATE TABLE clientes (id INT, nome VARCHAR(100), cidade VARCHAR(50));"


def test_generate_returns_no_duplicates():
    examples = generate(500, seed=42)
    keys = [(e.domain, e.question, e.sql) for e in examples]
    assert len(keys) == len(set(keys))


def test_generate_is_deterministic():
    assert generate(100, seed=7) == generate(100, seed=7)


def test_split_keeps_each_sql_in_one_split():
    splits = split_by_sql(generate(500, seed=42), 0.1, 0.1, seed=42)
    sql_sets = {name: {(e.domain, e.sql) for e in part} for name, part in splits.items()}
    assert sql_sets["test"] and sql_sets["val"]
    assert not sql_sets["train"] & sql_sets["test"]
    assert not sql_sets["train"] & sql_sets["val"]
    assert not sql_sets["val"] & sql_sets["test"]


def test_valid_sql_accepts_query_on_schema():
    assert sql_parses("SELECT nome FROM clientes WHERE cidade = 'Belo Horizonte';", SCHEMA)


def test_valid_sql_rejects_unknown_table_column_and_garbage():
    assert not sql_parses("SELECT * FROM pedidos;", SCHEMA)
    assert not sql_parses("SELECT salario FROM clientes;", SCHEMA)
    assert not sql_parses("isto nao e sql", SCHEMA)


def test_every_gold_query_is_valid_against_its_own_schema():
    for example in generate(500, seed=42):
        message = to_chat_message(example)["messages"]
        assert sql_parses(message[2]["content"], extract_schema(message[1]["content"])), example.sql


def test_extract_schema_returns_full_ddl():
    example = generate(1, seed=1)[0]
    user_message = to_chat_message(example)["messages"][1]["content"]
    assert extract_schema(user_message) == SCHEMAS[example.domain]["ddl"]


def test_clean_sql_handles_fences_and_trailing_prose():
    assert clean_sql("```sql\nSELECT 1;\n```\nThis query returns one.") == "SELECT 1;"
    assert clean_sql("SELECT 1;\n\nThis query returns one.") == "SELECT 1;"


def test_token_f1_and_normalize():
    assert normalize(" SELECT  1; ") == "select 1"
    assert token_f1("select 1", "SELECT 1;") == 1.0
    assert token_f1("a b", "c d") == 0.0


def test_report_table_uses_measured_values(tmp_path):
    summary = {"exact_match": 0.5, "token_f1": 0.75, "valid_sql": 1.0}
    (tmp_path / "baseline").mkdir()
    (tmp_path / "baseline" / "eval_results.json").write_text(json.dumps({"summary": summary}))
    (tmp_path / "base").mkdir()
    (tmp_path / "base" / "eval_results.json").write_text(json.dumps({"summary": summary}))
    metrics = {"trainable_params": 2162688, "config": {"lora_r": 16, "learning_rate": 2e-4}}
    (tmp_path / "base" / "metrics.json").write_text(json.dumps(metrics))

    table = build_table(tmp_path, ["base"])

    assert "| base | 2.16 M | 16 | 0.0002 | 0.500 | 0.750 | 1.000 |" in table
    assert "| no fine-tune |" in table
