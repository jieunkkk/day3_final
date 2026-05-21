import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_leaderboard(registry, output_dir: Path | str, top_n: int = 20) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = registry.load_registry().sort_values("pr_auc_mean", ascending=False).head(top_n)
    if df.empty:
        return output_dir / "leaderboard.png"

    fig, ax = plt.subplots(figsize=(12, max(6, top_n * 0.35)))
    sns.barplot(data=df, y="experiment_id", x="pr_auc_mean", hue="stage", dodge=False, ax=ax)
    ax.set_xlabel("OOF PR-AUC")
    ax.set_title("Experiment Leaderboard (Top by PR-AUC)")
    plt.tight_layout()
    path = output_dir / "leaderboard.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_ablation_waterfall(registry, output_dir: Path | str, stages: list[int] | None = None) -> Path:
    output_dir = Path(output_dir)
    df = registry.load_registry()
    if df.empty:
        return output_dir / "ablation_waterfall.png"

    stages = stages or [1, 2, 3, 4, 5]
    best_per_stage = []
    for s in stages:
        sub = df[df["stage"] == s]
        if not sub.empty:
            best_per_stage.append(sub.sort_values("pr_auc_mean", ascending=False).iloc[0])

    if not best_per_stage:
        return output_dir / "ablation_waterfall.png"

    labels = [f"Stage {int(r['stage'])}\n{r['experiment_id']}" for r in best_per_stage]
    values = [float(r["pr_auc_mean"]) for r in best_per_stage]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(len(values)), values, marker="o", linewidth=2)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel("Best OOF PR-AUC")
    ax.set_title("Progressive Improvement by Stage")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = output_dir / "ablation_waterfall.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def export_summary_report(registry, output_dir: Path | str) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = registry.load_registry().sort_values("pr_auc_mean", ascending=False)
    summary = {
        "n_experiments": len(df),
        "best_experiment": df.iloc[0]["experiment_id"] if len(df) else None,
        "best_pr_auc": float(df.iloc[0]["pr_auc_mean"]) if len(df) else None,
        "top_10": df.head(10)[["experiment_id", "description", "pr_auc_mean", "f1_mean", "recall_mean", "mcc_mean"]].to_dict("records"),
    }
    path = output_dir / "summary_report.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return path
