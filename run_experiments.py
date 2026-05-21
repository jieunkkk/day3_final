#!/usr/bin/env python
"""Run SECOM comparative experiments with 5-fold stratified OOF validation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.data_loader import load_secom
from src.experiment.grid_builder import build_all_experiments, build_stage_experiments
from src.experiment.registry import ExperimentRegistry
from src.experiment.runner import ExperimentRunner
from src.experiment.stats_tests import run_full_statistical_report
from src.reporting.summary import export_summary_report, plot_ablation_waterfall, plot_leaderboard


def parse_args():
    p = argparse.ArgumentParser(description="SECOM ML experiment runner")
    p.add_argument("--stage", type=int, nargs="*", default=None, help="Stages to run (1-5). Default: all")
    p.add_argument("--all", action="store_true", help="Run full grid (excludes slow M4/M5/MICE/RFE by default)")
    p.add_argument("--include-slow", action="store_true", help="Include slow imputers and RFE experiments")
    p.add_argument("--experiments-dir", type=str, default=str(ROOT / "experiments"))
    p.add_argument("--reports-dir", type=str, default=str(ROOT / "reports"))
    p.add_argument("--skip-existing", action="store_true", help="Skip experiments already in registry")
    p.add_argument("--limit", type=int, default=None, help="Max number of pending experiments to run")
    return p.parse_args()


def get_configs(args) -> list:
    if args.all or args.stage is None:
        return build_all_experiments(skip_slow=not args.include_slow)
    configs = []
    for s in args.stage:
        configs.extend(build_stage_experiments(s))
    return configs


def main():
    args = parse_args()
    X, y = load_secom()
    registry = ExperimentRegistry(args.experiments_dir)
    runner = ExperimentRunner(n_splits=5, random_state=42)

    configs = get_configs(args)
    existing = set()
    if args.skip_existing:
        df = registry.load_registry()
        if not df.empty:
            existing = set(df["experiment_id"].tolist())

    pending = [c for c in configs if c.experiment_id not in existing]
    if args.limit is not None:
        pending = pending[: args.limit]
    print(f"Total configs: {len(configs)}, pending: {len(pending)}")

    if pending:
        runner.run_many(pending, X.values, y.values, registry=registry)

    reports_dir = Path(args.reports_dir)
    run_full_statistical_report(registry, reports_dir / "stats", baseline_id="E001")
    plot_leaderboard(registry, reports_dir)
    plot_ablation_waterfall(registry, reports_dir)
    export_summary_report(registry, reports_dir)

    df = registry.load_registry().sort_values("pr_auc_mean", ascending=False)
    print("\n=== Top 10 Experiments (OOF PR-AUC) ===")
    cols = ["experiment_id", "stage", "description", "pr_auc_mean", "f1_mean", "recall_mean", "mcc_mean"]
    print(df[cols].head(10).to_string(index=False))
    print(f"\nResults saved to: {args.experiments_dir}")
    print(f"Reports saved to: {args.reports_dir}")


if __name__ == "__main__":
    main()
