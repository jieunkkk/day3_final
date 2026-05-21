import json
from pathlib import Path

import pandas as pd


class ExperimentRegistry:
    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "configs").mkdir(exist_ok=True)
        (self.root / "oof").mkdir(exist_ok=True)
        (self.root / "fold_scores").mkdir(exist_ok=True)
        self.registry_path = self.root / "registry.csv"

    @staticmethod
    def _read_registry(path: Path) -> pd.DataFrame:
        if not path.exists() or path.stat().st_size == 0:
            return pd.DataFrame()
        try:
            df = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            return pd.DataFrame()
        return df if not df.empty else pd.DataFrame()

    def save_experiment(self, result: dict) -> None:
        exp_id = result["experiment_id"]
        config_path = self.root / "configs" / f"{exp_id}.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(result["config"], f, ensure_ascii=False, indent=2)

        oof_path = self.root / "oof" / f"{exp_id}.csv"
        pd.DataFrame({
            "y_true": result["oof_y_true"],
            "y_proba": result["oof_proba"],
            "y_pred": result["oof_pred"],
        }).to_csv(oof_path, index=False)

        fold_path = self.root / "fold_scores" / f"{exp_id}.csv"
        pd.DataFrame(result["fold_scores"]).to_csv(fold_path, index=False)

        row = {
            "experiment_id": exp_id,
            "stage": result["config"]["stage"],
            "tier": result["config"]["tier"],
            "description": result["config"]["description"],
            "model_name": result["config"]["model_name"],
            "missing_policy": result["config"]["missing_policy"],
            "outlier_policy": result["config"]["outlier_policy"],
            "feature_policy": result["config"]["feature_policy"],
            "scale_policy": result["config"]["scale_policy"],
            "sampling_policy": result["config"]["sampling_policy"],
            "pr_auc_mean": result["oof_metrics"]["pr_auc"],
            "f1_mean": result["oof_metrics"]["f1"],
            "recall_mean": result["oof_metrics"]["recall"],
            "mcc_mean": result["oof_metrics"]["mcc"],
            "roc_auc_mean": result["oof_metrics"]["roc_auc"],
            "optimal_threshold": result["optimal_threshold"],
            "pr_auc_fold_mean": result["fold_summary"]["pr_auc_mean"],
            "pr_auc_fold_std": result["fold_summary"]["pr_auc_std"],
            "f1_fold_mean": result["fold_summary"]["f1_mean"],
            "f1_fold_std": result["fold_summary"]["f1_std"],
        }
        df = self._read_registry(self.registry_path)
        if not df.empty and "experiment_id" in df.columns:
            df = df[df["experiment_id"] != exp_id]
        else:
            df = pd.DataFrame()
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        df.to_csv(self.registry_path, index=False)

    def load_registry(self) -> pd.DataFrame:
        return self._read_registry(self.registry_path)

    def load_fold_scores(self, experiment_id: str) -> pd.DataFrame:
        return pd.read_csv(self.root / "fold_scores" / f"{experiment_id}.csv")

    def load_oof(self, experiment_id: str) -> pd.DataFrame:
        return pd.read_csv(self.root / "oof" / f"{experiment_id}.csv")
