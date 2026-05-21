#!/usr/bin/env python
"""Precompute permutation importance CSVs for Streamlit dashboard (optional)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis.dashboard_data import _importance_path, compute_feature_importance
from src.experiment.registry import ExperimentRegistry


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exp", nargs="*", help="Experiment IDs (default: all in registry)")
    p.add_argument("--force", action="store_true", help="Overwrite existing CSVs")
    args = p.parse_args()

    reg = ExperimentRegistry(ROOT / "experiments").load_registry()
    exp_ids = args.exp or reg["experiment_id"].tolist()

    for exp_id in exp_ids:
        out = _importance_path(exp_id)
        if out.exists() and not args.force:
            print(f"skip {exp_id}")
            continue
        print(f"compute {exp_id}...")
        df = compute_feature_importance(exp_id, n_repeats=5)
        df.to_csv(out, index=False)
        print(f"  -> {out}")


if __name__ == "__main__":
    main()
