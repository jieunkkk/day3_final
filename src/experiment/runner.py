import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from src.experiment.config_schema import ExperimentConfig
from src.experiment.metrics import bootstrap_ci, compute_metrics, find_optimal_threshold
from src.pipelines.factory import MISSING_MAP, build_pipeline
from src.pipelines.feature_select import remove_isolation_forest_outliers

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def _apply_row_filters(X, y, cfg: ExperimentConfig):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    missing = MISSING_MAP[cfg.missing_policy]
    if missing["row_threshold"] is not None:
        mask = pd.DataFrame(X).isna().mean(axis=1).values <= missing["row_threshold"]
        X, y = X[mask], y[mask]
    if cfg.outlier_policy == "O3":
        X, y = remove_isolation_forest_outliers(X, y, random_state=cfg.random_state)
    return X, y


class ExperimentRunner:
    def __init__(self, n_splits: int = 5, random_state: int = 42):
        self.n_splits = n_splits
        self.random_state = random_state
        self.skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    def run(self, cfg: ExperimentConfig, X, y) -> dict[str, Any]:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        oof_proba = np.zeros(len(y), dtype=float)
        fold_records = []

        for fold_idx, (train_idx, val_idx) in enumerate(self.skf.split(X, y)):
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]

            X_train, y_train = _apply_row_filters(X_train, y_train, cfg)
            if len(np.unique(y_train)) < 2:
                raise ValueError(f"{cfg.experiment_id} fold {fold_idx}: train set has single class after filtering")

            pipeline = build_pipeline(cfg, y_train)
            pipeline.fit(X_train, y_train)
            proba = pipeline.predict_proba(X_val)[:, 1]
            oof_proba[val_idx] = proba

            fold_metrics = compute_metrics(y_val, proba, threshold=0.5)
            fold_metrics["fold"] = fold_idx + 1
            fold_records.append(fold_metrics)

        optimal_threshold = find_optimal_threshold(y, oof_proba, metric="f1")
        oof_metrics = compute_metrics(y, oof_proba, threshold=optimal_threshold)
        oof_metrics_default = compute_metrics(y, oof_proba, threshold=0.5)

        pr_auc_scores = [r["pr_auc"] for r in fold_records]
        f1_scores = [r["f1"] for r in fold_records]
        pr_mean, pr_low, pr_high = bootstrap_ci(
            y, oof_proba,
            lambda yt, yp: compute_metrics(yt, yp, optimal_threshold)["pr_auc"],
        )

        return {
            "experiment_id": cfg.experiment_id,
            "config": cfg.to_dict(),
            "oof_proba": oof_proba.tolist(),
            "oof_y_true": y.tolist(),
            "oof_pred": (np.array(oof_proba) >= optimal_threshold).astype(int).tolist(),
            "optimal_threshold": optimal_threshold,
            "oof_metrics": oof_metrics,
            "oof_metrics_default_threshold": oof_metrics_default,
            "fold_scores": fold_records,
            "fold_summary": {
                "pr_auc_mean": float(np.mean(pr_auc_scores)),
                "pr_auc_std": float(np.std(pr_auc_scores)),
                "f1_mean": float(np.mean(f1_scores)),
                "f1_std": float(np.std(f1_scores)),
            },
            "bootstrap": {
                "pr_auc_mean": pr_mean,
                "pr_auc_ci_low": pr_low,
                "pr_auc_ci_high": pr_high,
            },
        }

    def run_many(self, configs: list[ExperimentConfig], X, y, registry=None) -> list[dict]:
        results = []
        for cfg in tqdm(configs, desc="Experiments"):
            result = self.run(cfg, X, y)
            results.append(result)
            if registry is not None:
                registry.save_experiment(result)
        return results
