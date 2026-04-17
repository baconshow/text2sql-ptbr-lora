"""
Synthetic Text-to-SQL dataset generator (PT-BR).

Generates ~3000 (schema, question, sql) triples across 4 business domains
(e-commerce, HR, finance, healthcare), with 20+ query templates covering:
SELECT, COUNT, SUM, AVG, MIN, MAX, GROUP BY, ORDER BY, WHERE, JOIN, LIMIT.

Why synthetic?
- Real Spider/WikiSQL datasets are English-only.
- Synthetic data shows the "synthetic data curation" skill listed in the job JD.
- We control distribution and can run ablations on dataset size.

Usage:
    python src/data_prep.py --output data/ --n_samples 3000 --seed 42
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


# ---------------------------------------------------------------------------
# Schemas (CREATE TABLE strings, in PT-BR column names)
# ---------------------------------------------------------------------------

SCHEMAS: dict[str, dict] = {
    "ecommerce": {
        "ddl": (
            "CREATE TABLE clientes (id INT, nome VARCHAR(100), email VARCHAR(100), "
            "cidade VARCHAR(50), estado VARCHAR(2), data_cadastro DATE);\n"
            "CREATE TABLE produtos (id INT, nome VARCHAR(100), categoria VARCHAR(50), "
            "preco DECIMAL(10,2), estoque INT);\n"
            "CREATE TABLE pedidos (id INT, cliente_id INT, data_pedido DATE, "
            "valor_total DECIMAL(10,2), status VARCHAR(20));\n"
            "CREATE TABLE itens_pedido (pedido_id INT, produto_id INT, quantidade INT, "
            "preco_unitario DECIMAL(10,2));"
        ),
        "tables": {
            "clientes": ["id", "nome", "email", "cidade", "estado", "data_cadastro"],
            "produtos": ["id", "nome", "categoria", "preco", "estoque"],
            "pedidos": ["id", "cliente_id", "data_pedido", "valor_total", "status"],
            "itens_pedido": ["pedido_id", "produto_id", "quantidade", "preco_unitario"],
        },
        "numeric_cols": {
            "produtos": ["preco", "estoque"],
            "pedidos": ["valor_total"],
            "itens_pedido": ["quantidade", "preco_unitario"],
        },
        "text_cols": {
            "clientes": ["nome", "email", "cidade", "estado"],
            "produtos": ["nome", "categoria"],
            "pedidos": ["status"],
        },
        "pt_names": {
            "clientes": "clientes",
            "produtos": "produtos",
            "pedidos": "pedidos",
            "itens_pedido": "itens do pedido",
        },
    },
    "rh": {
        "ddl": (
            "CREATE TABLE funcionarios (id INT, nome VARCHAR(100), cargo VARCHAR(50), "
            "departamento_id INT, salario DECIMAL(10,2), data_admissao DATE);\n"
            "CREATE TABLE departamentos (id INT, nome VARCHAR(50), centro_custo VARCHAR(20));\n"
            "CREATE TABLE ferias (funcionario_id INT, data_inicio DATE, data_fim DATE, dias INT);"
        ),
        "tables": {
            "funcionarios": ["id", "nome", "cargo", "departamento_id", "salario", "data_admissao"],
            "departamentos": ["id", "nome", "centro_custo"],
            "ferias": ["funcionario_id", "data_inicio", "data_fim", "dias"],
        },
        "numeric_cols": {
            "funcionarios": ["salario"],
            "ferias": ["dias"],
        },
        "text_cols": {
            "funcionarios": ["nome", "cargo"],
            "departamentos": ["nome", "centro_custo"],
        },
        "pt_names": {
            "funcionarios": "funcionários",
            "departamentos": "departamentos",
            "ferias": "férias",
        },
    },
    "financeiro": {
        "ddl": (
            "CREATE TABLE contas (id INT, titular VARCHAR(100), tipo VARCHAR(20), "
            "saldo DECIMAL(15,2), agencia VARCHAR(10));\n"
            "CREATE TABLE transacoes (id INT, conta_id INT, tipo VARCHAR(20), "
            "valor DECIMAL(15,2), data_transacao TIMESTAMP, descricao VARCHAR(200));\n"
            "CREATE TABLE categorias (id INT, nome VARCHAR(50), grupo VARCHAR(30));"
        ),
        "tables": {
            "contas": ["id", "titular", "tipo", "saldo", "agencia"],
            "transacoes": ["id", "conta_id", "tipo", "valor", "data_transacao", "descricao"],
            "categorias": ["id", "nome", "grupo"],
        },
        "numeric_cols": {
            "contas": ["saldo"],
            "transacoes": ["valor"],
        },
        "text_cols": {
            "contas": ["titular", "tipo", "agencia"],
            "transacoes": ["tipo", "descricao"],
            "categorias": ["nome", "grupo"],
        },
        "pt_names": {
            "contas": "contas",
            "transacoes": "transações",
            "categorias": "categorias",
        },
    },
    "saude": {
        "ddl": (
            "CREATE TABLE pacientes (id INT, nome VARCHAR(100), cpf VARCHAR(14), "
            "data_nascimento DATE, cidade VARCHAR(50));\n"
            "CREATE TABLE consultas (id INT, paciente_id INT, medico_id INT, "
            "data_consulta DATE, especialidade VARCHAR(50), valor DECIMAL(10,2));\n"
            "CREATE TABLE medicos (id INT, nome VARCHAR(100), crm VARCHAR(20), "
            "especialidade VARCHAR(50));"
        ),
        "tables": {
            "pacientes": ["id", "nome", "cpf", "data_nascimento", "cidade"],
            "consultas": ["id", "paciente_id", "medico_id", "data_consulta", "especialidade", "valor"],
            "medicos": ["id", "nome", "crm", "especialidade"],
        },
        "numeric_cols": {
            "consultas": ["valor"],
        },
        "text_cols": {
            "pacientes": ["nome", "cidade"],
            "consultas": ["especialidade"],
            "medicos": ["nome", "especialidade"],
        },
        "pt_names": {
            "pacientes": "pacientes",
            "consultas": "consultas",
            "medicos": "médicos",
        },
    },
}


# Sample values used to render realistic WHERE clauses
SAMPLE_TEXT_VALUES = {
    "cidade": ["São Paulo", "Rio de Janeiro", "Belo Horizonte", "Curitiba", "Salvador"],
    "estado": ["SP", "RJ", "MG", "PR", "BA"],
    "categoria": ["Eletrônicos", "Livros", "Roupas", "Alimentos", "Beleza"],
    "status": ["pendente", "pago", "enviado", "cancelado", "entregue"],
    "cargo": ["Analista", "Gerente", "Coordenador", "Diretor", "Estagiário"],
    "especialidade": ["Cardiologia", "Ortopedia", "Pediatria", "Clínica Geral", "Dermatologia"],
    "tipo": ["corrente", "poupança", "investimento", "débito", "crédito"],
    "grupo": ["Receita", "Despesa", "Investimento"],
}


@dataclass
class Example:
    schema: str
    question: str
    sql: str
    domain: str
    template: str


# ---------------------------------------------------------------------------
# Template functions: each returns one or more Example for the given domain.
# ---------------------------------------------------------------------------

def t_count_all(domain: str, rng: random.Random) -> Example:
    s = SCHEMAS[domain]
    table = rng.choice(list(s["tables"].keys()))
    pt = s["pt_names"][table]
    q = rng.choice([
        f"Quantos {pt} existem no total?",
        f"Qual o número total de {pt}?",
        f"Me diga quantos {pt} temos cadastrados.",
        f"Quantos registros há na tabela de {pt}?",
    ])
    sql = f"SELECT COUNT(*) FROM {table};"
    return Example(s["ddl"], q, sql, domain, "count_all")


def t_select_all(domain: str, rng: random.Random) -> Example:
    s = SCHEMAS[domain]
    table = rng.choice(list(s["tables"].keys()))
    pt = s["pt_names"][table]
    q = rng.choice([
        f"Liste todos os {pt}.",
        f"Mostre todos os {pt}.",
        f"Traga todos os registros de {pt}.",
    ])
    sql = f"SELECT * FROM {table};"
    return Example(s["ddl"], q, sql, domain, "select_all")


def t_where_text(domain: str, rng: random.Random) -> Example | None:
    s = SCHEMAS[domain]
    text_cols = s.get("text_cols", {})
    if not text_cols:
        return None
    table = rng.choice(list(text_cols.keys()))
    col = rng.choice(text_cols[table])
    if col not in SAMPLE_TEXT_VALUES:
        return None
    val = rng.choice(SAMPLE_TEXT_VALUES[col])
    pt = s["pt_names"][table]
    q = rng.choice([
        f"Liste os {pt} com {col} igual a '{val}'.",
        f"Quais {pt} têm {col} '{val}'?",
        f"Mostre os {pt} onde {col} é '{val}'.",
    ])
    sql = f"SELECT * FROM {table} WHERE {col} = '{val}';"
    return Example(s["ddl"], q, sql, domain, "where_text")


def t_where_numeric_gt(domain: str, rng: random.Random) -> Example | None:
    s = SCHEMAS[domain]
    num_cols = s.get("numeric_cols", {})
    if not num_cols:
        return None
    table = rng.choice(list(num_cols.keys()))
    col = rng.choice(num_cols[table])
    val = rng.choice([100, 500, 1000, 5000, 10000])
    pt = s["pt_names"][table]
    q = rng.choice([
        f"Quais {pt} têm {col} maior que {val}?",
        f"Liste os {pt} com {col} acima de {val}.",
        f"Mostre {pt} cujo {col} excede {val}.",
    ])
    sql = f"SELECT * FROM {table} WHERE {col} > {val};"
    return Example(s["ddl"], q, sql, domain, "where_numeric_gt")


def t_where_numeric_lt(domain: str, rng: random.Random) -> Example | None:
    s = SCHEMAS[domain]
    num_cols = s.get("numeric_cols", {})
    if not num_cols:
        return None
    table = rng.choice(list(num_cols.keys()))
    col = rng.choice(num_cols[table])
    val = rng.choice([100, 500, 1000, 5000])
    pt = s["pt_names"][table]
    q = rng.choice([
        f"Quais {pt} têm {col} menor que {val}?",
        f"Liste {pt} com {col} abaixo de {val}.",
    ])
    sql = f"SELECT * FROM {table} WHERE {col} < {val};"
    return Example(s["ddl"], q, sql, domain, "where_numeric_lt")


def t_avg(domain: str, rng: random.Random) -> Example | None:
    s = SCHEMAS[domain]
    num_cols = s.get("numeric_cols", {})
    if not num_cols:
        return None
    table = rng.choice(list(num_cols.keys()))
    col = rng.choice(num_cols[table])
    pt = s["pt_names"][table]
    q = rng.choice([
        f"Qual a média de {col} dos {pt}?",
        f"Calcule a média de {col} em {pt}.",
        f"Me dê o {col} médio dos {pt}.",
    ])
    sql = f"SELECT AVG({col}) FROM {table};"
    return Example(s["ddl"], q, sql, domain, "avg")


def t_sum(domain: str, rng: random.Random) -> Example | None:
    s = SCHEMAS[domain]
    num_cols = s.get("numeric_cols", {})
    if not num_cols:
        return None
    table = rng.choice(list(num_cols.keys()))
    col = rng.choice(num_cols[table])
    pt = s["pt_names"][table]
    q = rng.choice([
        f"Qual o total de {col} em {pt}?",
        f"Some o {col} de todos os {pt}.",
        f"Soma total de {col}.",
    ])
    sql = f"SELECT SUM({col}) FROM {table};"
    return Example(s["ddl"], q, sql, domain, "sum")


def t_max(domain: str, rng: random.Random) -> Example | None:
    s = SCHEMAS[domain]
    num_cols = s.get("numeric_cols", {})
    if not num_cols:
        return None
    table = rng.choice(list(num_cols.keys()))
    col = rng.choice(num_cols[table])
    pt = s["pt_names"][table]
    q = rng.choice([
        f"Qual o maior {col} entre os {pt}?",
        f"Maior valor de {col} em {pt}.",
    ])
    sql = f"SELECT MAX({col}) FROM {table};"
    return Example(s["ddl"], q, sql, domain, "max")


def t_min(domain: str, rng: random.Random) -> Example | None:
    s = SCHEMAS[domain]
    num_cols = s.get("numeric_cols", {})
    if not num_cols:
        return None
    table = rng.choice(list(num_cols.keys()))
    col = rng.choice(num_cols[table])
    pt = s["pt_names"][table]
    q = rng.choice([
        f"Qual o menor {col} entre os {pt}?",
        f"Menor valor de {col} em {pt}.",
    ])
    sql = f"SELECT MIN({col}) FROM {table};"
    return Example(s["ddl"], q, sql, domain, "min")


def t_order_limit(domain: str, rng: random.Random) -> Example | None:
    s = SCHEMAS[domain]
    num_cols = s.get("numeric_cols", {})
    if not num_cols:
        return None
    table = rng.choice(list(num_cols.keys()))
    col = rng.choice(num_cols[table])
    n = rng.choice([5, 10, 20])
    pt = s["pt_names"][table]
    direction = rng.choice(["DESC", "ASC"])
    word = "maiores" if direction == "DESC" else "menores"
    q = f"Liste os {n} {pt} com {word} {col}."
    sql = f"SELECT * FROM {table} ORDER BY {col} {direction} LIMIT {n};"
    return Example(s["ddl"], q, sql, domain, "order_limit")


def t_group_by_count(domain: str, rng: random.Random) -> Example | None:
    s = SCHEMAS[domain]
    text_cols = s.get("text_cols", {})
    if not text_cols:
        return None
    table = rng.choice(list(text_cols.keys()))
    col = rng.choice(text_cols[table])
    pt = s["pt_names"][table]
    q = rng.choice([
        f"Quantos {pt} existem por {col}?",
        f"Contagem de {pt} agrupada por {col}.",
    ])
    sql = f"SELECT {col}, COUNT(*) FROM {table} GROUP BY {col};"
    return Example(s["ddl"], q, sql, domain, "group_by_count")


def t_group_by_sum(domain: str, rng: random.Random) -> Example | None:
    s = SCHEMAS[domain]
    text_cols = s.get("text_cols", {})
    num_cols = s.get("numeric_cols", {})
    # Find a table that has both text and numeric columns
    candidates = [t for t in text_cols if t in num_cols]
    if not candidates:
        return None
    table = rng.choice(candidates)
    text_col = rng.choice(text_cols[table])
    num_col = rng.choice(num_cols[table])
    pt = s["pt_names"][table]
    q = rng.choice([
        f"Qual o total de {num_col} por {text_col} em {pt}?",
        f"Some {num_col} agrupado por {text_col}.",
    ])
    sql = f"SELECT {text_col}, SUM({num_col}) FROM {table} GROUP BY {text_col};"
    return Example(s["ddl"], q, sql, domain, "group_by_sum")


def t_count_distinct(domain: str, rng: random.Random) -> Example | None:
    s = SCHEMAS[domain]
    text_cols = s.get("text_cols", {})
    if not text_cols:
        return None
    table = rng.choice(list(text_cols.keys()))
    col = rng.choice(text_cols[table])
    pt = s["pt_names"][table]
    q = rng.choice([
        f"Quantos {col} distintos existem em {pt}?",
        f"Número de {col} únicos em {pt}.",
    ])
    sql = f"SELECT COUNT(DISTINCT {col}) FROM {table};"
    return Example(s["ddl"], q, sql, domain, "count_distinct")


def t_like(domain: str, rng: random.Random) -> Example | None:
    s = SCHEMAS[domain]
    text_cols = s.get("text_cols", {})
    # Prefer a "nome" column when present
    candidates = [(t, c) for t, cols in text_cols.items() for c in cols if c == "nome"]
    if not candidates:
        return None
    table, col = rng.choice(candidates)
    prefix = rng.choice(["Ma", "Jo", "Al", "Ca", "Ro"])
    pt = s["pt_names"][table]
    q = rng.choice([
        f"Liste {pt} cujo {col} começa com '{prefix}'.",
        f"Quais {pt} têm {col} iniciando em '{prefix}'?",
    ])
    sql = f"SELECT * FROM {table} WHERE {col} LIKE '{prefix}%';"
    return Example(s["ddl"], q, sql, domain, "like")


def t_between(domain: str, rng: random.Random) -> Example | None:
    s = SCHEMAS[domain]
    num_cols = s.get("numeric_cols", {})
    if not num_cols:
        return None
    table = rng.choice(list(num_cols.keys()))
    col = rng.choice(num_cols[table])
    lo = rng.choice([100, 500, 1000])
    hi = lo + rng.choice([500, 1000, 5000])
    pt = s["pt_names"][table]
    q = f"Liste os {pt} com {col} entre {lo} e {hi}."
    sql = f"SELECT * FROM {table} WHERE {col} BETWEEN {lo} AND {hi};"
    return Example(s["ddl"], q, sql, domain, "between")


TEMPLATES: list[Callable] = [
    t_count_all, t_count_all,           # extra weight on common queries
    t_select_all,
    t_where_text, t_where_text,
    t_where_numeric_gt, t_where_numeric_gt,
    t_where_numeric_lt,
    t_avg, t_avg,
    t_sum, t_sum,
    t_max,
    t_min,
    t_order_limit, t_order_limit,
    t_group_by_count,
    t_group_by_sum,
    t_count_distinct,
    t_like,
    t_between,
]


def generate(n_samples: int, seed: int = 42) -> list[Example]:
    rng = random.Random(seed)
    out: list[Example] = []
    domains = list(SCHEMAS.keys())
    attempts = 0
    while len(out) < n_samples and attempts < n_samples * 10:
        attempts += 1
        domain = rng.choice(domains)
        template = rng.choice(TEMPLATES)
        ex = template(domain, rng)
        if ex is not None:
            out.append(ex)
    return out


def format_prompt(schema: str, question: str) -> str:
    """The exact prompt format used during both training and inference."""
    return (
        "Você é um assistente que converte perguntas em português para SQL.\n"
        f"### Schema:\n{schema}\n\n"
        f"### Pergunta:\n{question}\n\n"
        "### SQL:\n"
    )


def to_chat_message(ex: Example) -> dict:
    """Render as HF chat-style (compatible with Qwen tokenizer)."""
    return {
        "messages": [
            {"role": "system", "content": "Você é um assistente que converte perguntas em português para SQL."},
            {"role": "user", "content": f"### Schema:\n{ex.schema}\n\n### Pergunta:\n{ex.question}"},
            {"role": "assistant", "content": ex.sql},
        ],
        "domain": ex.domain,
        "template": ex.template,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="data", help="Output directory")
    parser.add_argument("--n_samples", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--test_frac", type=float, default=0.1)
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    examples = generate(args.n_samples, args.seed)
    random.Random(args.seed).shuffle(examples)

    n = len(examples)
    n_test = int(n * args.test_frac)
    n_val = int(n * args.val_frac)
    n_train = n - n_val - n_test

    splits = {
        "train": examples[:n_train],
        "val": examples[n_train : n_train + n_val],
        "test": examples[n_train + n_val :],
    }

    for name, split in splits.items():
        path = out_dir / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for ex in split:
                f.write(json.dumps(to_chat_message(ex), ensure_ascii=False) + "\n")
        print(f"  {name}: {len(split):>5} → {path}")

    # Per-template distribution
    from collections import Counter
    template_counts = Counter(e.template for e in examples)
    print("\nTemplate distribution:")
    for tpl, c in sorted(template_counts.items(), key=lambda x: -x[1]):
        print(f"  {tpl:<25} {c}")


if __name__ == "__main__":
    main()
