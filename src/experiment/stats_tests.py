import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def _get_fold_scores(registry, exp_id: str, metric: str = "pr_auc") -> np.ndarray:
    fold_df = registry.load_fold_scores(exp_id)
    return fold_df[metric].values


def paired_comparison(scores_a: np.ndarray, scores_b: np.ndarray, alpha: float = 0.05) -> dict:
    delta = scores_a - scores_b
    shapiro_p = float(stats.shapiro(delta).pvalue) if len(delta) >= 3 else np.nan
    if shapiro_p >= 0.05:
        stat, p_value = stats.ttest_rel(scores_a, scores_b)
        test_name = "paired_t_test"
    else:
        stat, p_value = stats.wilcoxon(scores_a, scores_b)
        test_name = "wilcoxon"
    d = float(delta.mean() / delta.std()) if delta.std() > 0 else 0.0
    return {
        "test": test_name,
        "statistic": float(stat),
        "p_value": float(p_value),
        "significant": bool(p_value < alpha),
        "mean_delta": float(delta.mean()),
        "cohens_d": d,
        "shapiro_p": shapiro_p,
    }


def mcnemar_test(y_true, pred_a, pred_b) -> dict:
    y_true = np.asarray(y_true)
    pred_a = np.asarray(pred_a)
    pred_b = np.asarray(pred_b)
    correct_a = pred_a == y_true
    correct_b = pred_b == y_true
    b_count = int(np.sum(correct_a & ~correct_b))
    c_count = int(np.sum(~correct_a & correct_b))
    if b_count + c_count == 0:
        return {"b": b_count, "c": c_count, "p_value": 1.0, "significant": False}
    result = stats.binomtest(b_count, b_count + c_count, 0.5)
    p_value = float(result.pvalue)
    return {"b": b_count, "c": c_count, "p_value": p_value, "significant": p_value < 0.05}


def friedman_test(score_matrix: np.ndarray) -> dict:
    if score_matrix.shape[1] < 3:
        stat, p_value = stats.friedmanchisquare(*[score_matrix[:, i] for i in range(score_matrix.shape[1])])
    else:
        stat, p_value = stats.friedmanchisquare(*[score_matrix[:, i] for i in range(score_matrix.shape[1])])
    return {"statistic": float(stat), "p_value": float(p_value), "significant": bool(p_value < 0.05)}


def holm_correction(p_values: list[float], alpha: float = 0.05) -> list[bool]:
    m = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    rejected = [False] * m
    for rank, (idx, p) in enumerate(indexed):
        if p <= alpha / (m - rank):
            rejected[idx] = True
        else:
            break
    return rejected


def compare_experiments(registry, exp_a: str, exp_b: str, metric: str = "pr_auc") -> dict:
    scores_a = _get_fold_scores(registry, exp_a, metric)
    scores_b = _get_fold_scores(registry, exp_b, metric)
    paired = paired_comparison(scores_a, scores_b)
    oof_a = registry.load_oof(exp_a)
    oof_b = registry.load_oof(exp_b)
    mcnemar = mcnemar_test(oof_a["y_true"], oof_a["y_pred"], oof_b["y_pred"])
    return {
        "experiment_a": exp_a,
        "experiment_b": exp_b,
        "metric": metric,
        "mean_a": float(scores_a.mean()),
        "mean_b": float(scores_b.mean()),
        "paired_test": paired,
        "mcnemar": mcnemar,
    }


def run_full_statistical_report(registry, output_dir: Path | str, baseline_id: str = "E001") -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = registry.load_registry()
    if df.empty:
        return {}

    df = df.sort_values("pr_auc_mean", ascending=False).reset_index(drop=True)
    exp_ids = df["experiment_id"].tolist()

    pairwise_rows = []
    p_values = []
    pairs = []
    for i, exp_a in enumerate(exp_ids):
        if exp_a == baseline_id:
            continue
        cmp = compare_experiments(registry, baseline_id, exp_a)
        pairs.append((baseline_id, exp_a))
        p_values.append(cmp["paired_test"]["p_value"])
        pairwise_rows.append({
            "baseline": baseline_id,
            "experiment": exp_a,
            "pr_auc_baseline": cmp["mean_a"],
            "pr_auc_experiment": cmp["mean_b"],
            "delta": cmp["paired_test"]["mean_delta"],
            "p_value": cmp["paired_test"]["p_value"],
            "test": cmp["paired_test"]["test"],
            "cohens_d": cmp["paired_test"]["cohens_d"],
            "mcnemar_p": cmp["mcnemar"]["p_value"],
        })

    if p_values:
        holm = holm_correction(p_values)
        for row, sig in zip(pairwise_rows, holm):
            row["significant_holm"] = sig

    pairwise_df = pd.DataFrame(pairwise_rows)
    pairwise_df.to_csv(output_dir / "pairwise_vs_baseline.csv", index=False)

    top_n = min(8, len(exp_ids))
    top_ids = exp_ids[:top_n]
    score_matrix = np.column_stack([_get_fold_scores(registry, eid) for eid in top_ids])
    friedman = friedman_test(score_matrix)

    top_pairwise = []
    top_pvals = []
    for i in range(len(top_ids)):
        for j in range(i + 1, len(top_ids)):
            cmp = compare_experiments(registry, top_ids[i], top_ids[j])
            top_pairwise.append({
                "experiment_a": top_ids[i],
                "experiment_b": top_ids[j],
                "delta": cmp["paired_test"]["mean_delta"],
                "p_value": cmp["paired_test"]["p_value"],
            })
            top_pvals.append(cmp["paired_test"]["p_value"])
    if top_pvals:
        holm_top = holm_correction(top_pvals)
        for row, sig in zip(top_pairwise, holm_top):
            row["significant_holm"] = sig
    top_pairwise_df = pd.DataFrame(top_pairwise)
    top_pairwise_df.to_csv(output_dir / "pairwise_top_models.csv", index=False)

    summary = {
        "baseline_id": baseline_id,
        "best_experiment": exp_ids[0],
        "best_pr_auc": float(df.loc[0, "pr_auc_mean"]),
        "friedman_top_models": friedman,
        "n_experiments": len(df),
    }
    with open(output_dir / "statistical_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary
