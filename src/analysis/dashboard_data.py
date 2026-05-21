import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.inspection import permutation_importance

from src.config import EXPERIMENTS_DIR, REPORTS_DIR

IMPORTANCE_DIR = REPORTS_DIR / "importance"
from src.data_loader import load_secom
from src.experiment.config_schema import ExperimentConfig
from src.experiment.registry import ExperimentRegistry


@lru_cache(maxsize=4)
def load_registry_df() -> pd.DataFrame:
    reg = ExperimentRegistry(EXPERIMENTS_DIR)
    return reg.load_registry().sort_values("pr_auc_mean", ascending=False)


def load_experiment_config(exp_id: str) -> ExperimentConfig:
    with open(EXPERIMENTS_DIR / "configs" / f"{exp_id}.json", encoding="utf-8") as f:
        d = json.load(f)
    return ExperimentConfig(**d)


def load_oof(exp_id: str) -> pd.DataFrame:
    oof = pd.read_csv(EXPERIMENTS_DIR / "oof" / f"{exp_id}.csv")
    oof["sample_idx"] = np.arange(len(oof))
    return oof


def load_pairwise_stats() -> tuple[pd.DataFrame, pd.DataFrame]:
    stats_dir = EXPERIMENTS_DIR.parent / "reports" / "stats"
    vs_base = pd.read_csv(stats_dir / "pairwise_vs_baseline.csv") if (stats_dir / "pairwise_vs_baseline.csv").exists() else pd.DataFrame()
    top_pair = pd.read_csv(stats_dir / "pairwise_top_models.csv") if (stats_dir / "pairwise_top_models.csv").exists() else pd.DataFrame()
    return vs_base, top_pair


def get_feature_names(X: pd.DataFrame) -> list[str]:
    return [str(c) for c in X.columns]


def _importance_path(exp_id: str) -> Path:
    IMPORTANCE_DIR.mkdir(parents=True, exist_ok=True)
    return IMPORTANCE_DIR / f"{exp_id}.csv"


def load_feature_importance(exp_id: str, *, allow_compute: bool = True) -> pd.DataFrame:
    """Load precomputed importance, else fast MI proxy (Streamlit Cloud friendly)."""
    path = _importance_path(exp_id)
    if path.exists():
        return pd.read_csv(path)

    if allow_compute:
        return compute_feature_importance(exp_id)

    return fast_feature_importance()


@lru_cache(maxsize=1)
def fast_feature_importance() -> pd.DataFrame:
    from sklearn.feature_selection import mutual_info_classif

    X, y = load_secom()
    X_fill = X.fillna(X.median(numeric_only=True))
    mi = mutual_info_classif(X_fill, y, random_state=42)
    return pd.DataFrame({
        "feature": [str(c) for c in X.columns],
        "importance": mi,
        "importance_std": 0.0,
        "method": "mutual_info",
    }).sort_values("importance", ascending=False)


@lru_cache(maxsize=2)
def compute_feature_importance(exp_id: str, n_repeats: int = 5) -> pd.DataFrame:
    from src.pipelines.factory import build_pipeline

    cfg = load_experiment_config(exp_id)
    X, y = load_secom()
    feature_names = get_feature_names(X)
    pipe = build_pipeline(cfg, y.values)
    pipe.fit(X.values, y.values)

    result = permutation_importance(
        pipe, X.values, y.values,
        n_repeats=n_repeats,
        scoring="average_precision",
        random_state=42,
        n_jobs=-1,
    )
    return pd.DataFrame({
        "feature": feature_names,
        "importance": result.importances_mean,
        "importance_std": result.importances_std,
        "method": "permutation",
    }).sort_values("importance", ascending=False)


def compare_feature_groups(X: pd.DataFrame, y: pd.Series, importance_df: pd.DataFrame, top_k: int = 20, bottom_k: int = 20) -> dict:
    top_feats = importance_df.head(top_k)["feature"].tolist()
    bottom_feats = importance_df.tail(bottom_k)["feature"].tolist()
    top_feats = [f for f in top_feats if f in X.columns]
    bottom_feats = [f for f in bottom_feats if f in X.columns]

    rows = []
    for group_name, feats in [("Top (중요)", top_feats), ("Bottom (비중요)", bottom_feats)]:
        for feat in feats:
            pass_vals = X.loc[y == 0, feat].dropna()
            fail_vals = X.loc[y == 1, feat].dropna()
            if len(pass_vals) < 5 or len(fail_vals) < 3:
                continue
            stat, p = stats.mannwhitneyu(fail_vals, pass_vals, alternative="two-sided")
            pooled_std = np.sqrt((pass_vals.std() ** 2 + fail_vals.std() ** 2) / 2)
            d = (fail_vals.mean() - pass_vals.mean()) / pooled_std if pooled_std > 0 else 0
            rows.append({
                "group": group_name,
                "feature": feat,
                "pass_mean": pass_vals.mean(),
                "fail_mean": fail_vals.mean(),
                "delta_mean": fail_vals.mean() - pass_vals.mean(),
                "p_value": p,
                "cohens_d": d,
            })

    detail = pd.DataFrame(rows)
    if detail.empty:
        return {"detail": detail, "group_summary": pd.DataFrame()}

    group_summary = detail.groupby("group").agg(
        n_features=("feature", "count"),
        sig_features=("p_value", lambda s: int((s < 0.05).sum())),
        mean_abs_d=("cohens_d", lambda s: float(np.abs(s).mean())),
        median_p=("p_value", "median"),
    ).reset_index()

    top_sig = int((detail.loc[detail["group"] == "Top (중요)", "p_value"] < 0.05).sum())
    bot_sig = int((detail.loc[detail["group"] == "Bottom (비중요)", "p_value"] < 0.05).sum())
    stat, p = stats.mannwhitneyu(
        detail.loc[detail["group"] == "Top (중요)", "p_value"],
        detail.loc[detail["group"] == "Bottom (비중요)", "p_value"],
        alternative="less",
    )
    group_summary["interpretation"] = (
        f"중요 변수 {top_sig}개 vs 비중요 {bot_sig}개가 Pass/Fail 간 유의미(p<0.05). "
        f"두 그룹 p-value 분포 비교 p={p:.4f}"
    )
    return {"detail": detail, "group_summary": group_summary, "group_pvalue": p}


def deep_eda_hints(X: pd.DataFrame, y: pd.Series, importance_df: pd.DataFrame, top_k: int = 15) -> pd.DataFrame:
    top_feats = importance_df.head(top_k)["feature"].tolist()
    top_feats = [f for f in top_feats if f in X.columns]
    hints = []

    corr = X[top_feats].corr(method="spearman")
    high_pairs = []
    for i, a in enumerate(top_feats):
        for b in top_feats[i + 1:]:
            r = corr.loc[a, b]
            if abs(r) > 0.7:
                high_pairs.append((a, b, r))
    if high_pairs:
        hints.append({"category": "다중공선성", "finding": f"Top 특성 중 |Spearman|>0.7 쌍 {len(high_pairs)}개 — 앙상블/PCA 검토"})

    fail_rate_by_missing = X.isna().mean(axis=1)
    stat, p = stats.mannwhitneyu(fail_rate_by_missing[y == 1], fail_rate_by_missing[y == 0])
    hints.append({"category": "결측 패턴", "finding": f"Fail 웨이퍼의 행 결측률이 Pass와 {'유의미히 다름' if p < 0.05 else '비슷함'} (p={p:.4f})"})

    for feat in top_feats[:5]:
        fail_vals = X.loc[y == 1, feat].dropna()
        pass_vals = X.loc[y == 0, feat].dropna()
        if len(fail_vals) < 3:
            continue
        z_fail = (fail_vals - pass_vals.mean()) / (pass_vals.std() + 1e-9)
        outlier_fail = int((np.abs(z_fail) > 2.5).sum())
        if outlier_fail > 0:
            hints.append({"category": "분포 이상", "finding": f"센서 #{feat}: Fail 중 {outlier_fail}건이 Pass 분포 대비 이상치(z>2.5)"})

    mi_top = top_feats[:3]
    hints.append({"category": "공정 연계", "finding": f"핵심 센서 후보: {', '.join('#'+f for f in mi_top)} — SECOM은 익명 센서이므로 공정 엔지니어 매핑 필요"})

    return pd.DataFrame(hints)


def analyze_misclassifications(oof: pd.DataFrame, X: pd.DataFrame, y: pd.Series, threshold: float) -> dict:
    df = oof.copy()
    df["y_true"] = y.values
    df["error_type"] = "TN"
    df.loc[(df.y_true == 1) & (df.y_pred == 0), "error_type"] = "FN"
    df.loc[(df.y_true == 0) & (df.y_pred == 1), "error_type"] = "FP"
    df.loc[(df.y_true == 1) & (df.y_pred == 1), "error_type"] = "TP"

    df["margin"] = np.where(df.y_pred == 1, df.y_proba - threshold, threshold - df.y_proba)
    df["abs_dist_threshold"] = np.abs(df.y_proba - threshold)
    df["error_subtype"] = "Correct"
    df.loc[df.error_type == "FN", "error_subtype"] = np.where(
        df.loc[df.error_type == "FN", "abs_dist_threshold"] < 0.05, "FN_아쉬운(근접)", "FN_확신적오답"
    )
    df.loc[df.error_type == "FP", "error_subtype"] = np.where(
        df.loc[df.error_type == "FP", "abs_dist_threshold"] < 0.05, "FP_아쉬운(근접)", "FP_확신적오답"
    )

    errors = df[df.error_type.isin(["FN", "FP"])].copy()
    profile_cols = X.columns[:min(50, X.shape[1])]
    pass_centroid = X.loc[y == 0, profile_cols].median()
    fail_centroid = X.loc[y == 1, profile_cols].median()

    def centroid_dist(idx, label):
        row = X.iloc[idx][profile_cols]
        target = fail_centroid if label == 1 else pass_centroid
        return float(np.nanmean(np.abs(row - target)))

    df["centroid_dist"] = [centroid_dist(i, lbl) for i, lbl in zip(df.index, df.y_true)]

    label_suspect = []
    for idx, row in errors.iterrows():
        si = row["sample_idx"]
        if row["error_type"] == "FN":
            d_pass = centroid_dist(si, 0)
            d_fail = centroid_dist(si, 1)
            if d_pass < d_fail:
                label_suspect.append("FN이나 Pass 프로파일에 가까움 → 라벨 오류 가능")
            else:
                label_suspect.append("진짜 경계/미학습 케이스")
        else:
            d_pass = centroid_dist(si, 0)
            d_fail = centroid_dist(si, 1)
            if d_fail < d_pass:
                label_suspect.append("FP이나 Fail 프로파일에 가까움 → 라벨 오류 가능")
            else:
                label_suspect.append("Pass 중 이상치/오탐")
    errors["label_audit"] = label_suspect

    subtype_counts = df.loc[df.error_type.isin(["FN", "FP"]), "error_subtype"].value_counts()
    return {
        "full": df,
        "errors": errors,
        "subtype_counts": subtype_counts,
        "label_suspect_rate": float(errors["label_audit"].str.contains("라벨 오류").mean()) if len(errors) else 0,
    }
