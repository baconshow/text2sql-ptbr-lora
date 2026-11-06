"""Build results/ablation_table.md and assets/loss_curves.png from finished runs.

Usage:
    python src/make_report.py --runs base rank8 rank32_allmodules lr_high
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def table_row(label: str, params: str, rank: str, lr: str, summary: dict) -> str:
    cells = [
        label, params, rank, lr,
        f"{summary['exact_match']:.3f}",
        f"{summary['token_f1']:.3f}",
        f"{summary['valid_sql']:.3f}",
    ]
    return "| " + " | ".join(cells) + " |"


def build_table(runs_dir: Path, runs: list[str]) -> str:
    lines = [
        "| Run | Trainable params | LoRA r | LR | Exact match | Token F1 | Valid SQL |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    baseline = load_json(runs_dir / "baseline" / "eval_results.json")["summary"]
    lines.append(table_row("no fine-tune", "-", "-", "-", baseline))
    for run in runs:
        metrics = load_json(runs_dir / run / "metrics.json")
        summary = load_json(runs_dir / run / "eval_results.json")["summary"]
        config = metrics["config"]
        params = f"{metrics['trainable_params'] / 1e6:.2f} M"
        lines.append(table_row(run, params, str(config["lora_r"]),
                               str(config["learning_rate"]), summary))
    return "\n".join(lines) + "\n"


def plot_losses(runs_dir: Path, runs: list[str], output: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    for run in runs:
        logs = load_json(runs_dir / run / "log_history.json")
        points = [(entry["step"], entry["loss"]) for entry in logs if "loss" in entry]
        ax.plot(*zip(*points), label=run)
    ax.set_xlabel("Step")
    ax.set_ylabel("Train loss")
    ax.set_title("QLoRA training loss")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.savefig(output, dpi=150, bbox_inches="tight")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--runs_dir", type=Path, default=Path("runs"))
    args = parser.parse_args()

    Path("results").mkdir(exist_ok=True)
    Path("assets").mkdir(exist_ok=True)
    table = build_table(args.runs_dir, args.runs)
    Path("results/ablation_table.md").write_text(table, encoding="utf-8")
    plot_losses(args.runs_dir, args.runs, Path("assets/loss_curves.png"))
    print(table)


if __name__ == "__main__":
    main()
