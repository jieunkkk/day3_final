"""SECOM 반도체 공정 불량 예측 — Streamlit 대시보드"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis.dashboard_data import (
    analyze_misclassifications,
    compare_feature_groups,
    compute_feature_importance,
    deep_eda_hints,
    load_experiment_config,
    load_oof,
    load_pairwise_stats,
    load_registry_df,
)
from src.analysis.insights import future_recommendations, interpret_experiments, model_upgrade_plan
from src.config import POLICY_LABELS
from src.data_loader import load_secom

st.set_page_config(page_title="SECOM ML Dashboard", page_icon="🏭", layout="wide")

st.title("🏭 SECOM 반도체 공정 불량 예측 대시보드")
st.caption("5-Fold Stratified OOF · 51개 비교실험 · Tabular ML (Grad-CAM 해당 없음)")


@st.cache_data(show_spinner="데이터 로딩...")
def load_base_data():
    return load_registry_df(), *load_pairwise_stats(), *load_secom()


registry, pairwise_base, pairwise_top, X, y = load_base_data()

with st.sidebar:
    st.header("설정")
    metric_options = {
        "PR-AUC (불균형 권장)": "pr_auc_mean",
        "F1 (Fail)": "f1_mean",
        "Recall (Fail)": "recall_mean",
        "MCC": "mcc_mean",
    }
    sort_label = st.selectbox("모델 선택 기준", list(metric_options.keys()), index=1)
    sort_col = metric_options[sort_label]
    registry_sorted = registry.sort_values(sort_col, ascending=False)
    exp_ids = registry_sorted["experiment_id"].tolist()
    default_idx = exp_ids.index("E206") if "E206" in exp_ids else 0
    selected_exp = st.selectbox("분석 대상 모델", exp_ids, index=default_idx)
    cfg = load_experiment_config(selected_exp)
    oof = load_oof(selected_exp)
    row = registry.loc[registry["experiment_id"] == selected_exp].iloc[0]
    threshold = float(row["optimal_threshold"])
    st.metric("OOF PR-AUC", f"{row['pr_auc_mean']:.3f}")
    st.metric("OOF F1", f"{row['f1_mean']:.3f}")
    st.metric("최적 Threshold", f"{threshold:.2f}")

@st.cache_data(show_spinner="Feature Importance 계산 (1~3분)...")
def cached_importance(exp_id: str):
    return compute_feature_importance(exp_id)

imp_df = cached_importance(selected_exp)
cmp = compare_feature_groups(X, y, imp_df, top_k=20, bottom_k=20)
mis = analyze_misclassifications(oof, X, y, threshold)

tab1, tab2, tab3 = st.tabs(["📊 실험 결과", "🔬 입력 변수 분석", "🔍 오분류 심화 분석"])

# ── Tab 1 ──
with tab1:
    st.header("실험 결과 & 통계 검증")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("총 실험 수", len(registry))
    c2.metric("최고 PR-AUC", f"{registry['pr_auc_mean'].max():.3f}")
    c3.metric("최고 F1", f"{registry['f1_mean'].max():.3f}")
    sig_count = int(pairwise_base["significant_holm"].sum()) if not pairwise_base.empty and "significant_holm" in pairwise_base.columns else 0
    c4.metric("vs Dummy 유의미", f"{sig_count}개")

    display_cols = [
        "experiment_id", "stage", "description", "model_name",
        "missing_policy", "outlier_policy", "feature_policy", "sampling_policy",
        "pr_auc_mean", "f1_mean", "recall_mean", "mcc_mean", "optimal_threshold",
    ]
    show = registry.sort_values("pr_auc_mean", ascending=False)[display_cols].copy()
    if not pairwise_base.empty:
        sig_map = pairwise_base.set_index("experiment")["significant_holm"].to_dict()
        show["vs_E001_유의"] = show["experiment_id"].map(lambda x: sig_map.get(x, False))
    st.dataframe(show, use_container_width=True, height=400)

    st.subheader("방법론 해석")
    for item in interpret_experiments(registry, pairwise_base):
        st.markdown(f"**{item['title']}** — {item['body']}")

    st.subheader("Stage별 최고 성능")
    rows = []
    for stage_val in sorted(registry["stage"].unique()):
        sub = registry[registry["stage"] == stage_val]
        best = sub.sort_values("pr_auc_mean", ascending=False).iloc[0]
        rows.append({"stage": stage_val, "best_exp": best["experiment_id"], "best_pr_auc": best["pr_auc_mean"], "best_f1": best["f1_mean"]})
    stage_agg = pd.DataFrame(rows)
    st.plotly_chart(px.bar(stage_agg, x="stage", y="best_pr_auc", text="best_exp", title="Stage별 Best PR-AUC"), use_container_width=True)

    if not pairwise_base.empty:
        st.subheader("통계 검증 (vs Dummy E001)")
        pv = pairwise_base.head(15).copy()
        pv["delta"] = pv["pr_auc_experiment"] - pv["pr_auc_baseline"]
        st.plotly_chart(
            px.scatter(pv, x="pr_auc_experiment", y="delta", color="significant_holm",
                       hover_data=["experiment", "p_value"], title="Fold PR-AUC 개선 (Δ = experiment − baseline)"),
            use_container_width=True,
        )
        st.dataframe(pv[["experiment", "pr_auc_experiment", "delta", "p_value", "cohens_d", "significant_holm"]], use_container_width=True)

    st.subheader("향후 실험 제안")
    for r in future_recommendations(registry):
        st.markdown(f"- {r}")

# ── Tab 2 ──
with tab2:
    st.header("입력 변수 (센서) 분석")
    st.info("Tabular 데이터 → Feature Importance + 통계 검定 (Grad-CAM 해당 없음)")

    top_k = st.slider("Top/Bottom K", 5, 30, 20)
    cmp = compare_feature_groups(X, y, imp_df, top_k=top_k, bottom_k=top_k)
    fig_imp = px.bar(imp_df.head(25), x="importance", y="feature", orientation="h", title=f"Top 25 Permutation Importance ({selected_exp})")
    fig_imp.update_layout(yaxis={"categoryorder": "total ascending"}, height=600)
    st.plotly_chart(fig_imp, use_container_width=True)

    st.markdown(
        f"**전처리**: {POLICY_LABELS['missing'].get(cfg.missing_policy)} · "
        f"{POLICY_LABELS['outlier'].get(cfg.outlier_policy)} · "
        f"{POLICY_LABELS['feature'].get(cfg.feature_policy)} · "
        f"{POLICY_LABELS['sampling'].get(cfg.sampling_policy)}"
    )

    cmp = compare_feature_groups(X, y, imp_df, top_k=top_k, bottom_k=top_k)
    st.subheader("중요 vs 비중요 — Pass/Fail Mann-Whitney 비교")
    if not cmp["detail"].empty:
        st.dataframe(cmp["detail"].round(4), use_container_width=True)
        top_sig = (cmp["detail"].loc[cmp["detail"]["group"] == "Top (중요)", "p_value"] < 0.05).mean()
        bot_sig = (cmp["detail"].loc[cmp["detail"]["group"] == "Bottom (비중요)", "p_value"] < 0.05).mean()
        st.markdown(f"**해석**: 중요 변수 유의 비율 {top_sig*100:.0f}% vs 비중요 {bot_sig*100:.0f}% — "
                    f"{'중요 변수가 통계적으로 더 강한 분리력' if top_sig > bot_sig else '분리력 약함, feature engineering 필요'}.")

    st.subheader("심화 EDA 힌트")
    st.dataframe(deep_eda_hints(X, y, imp_df), use_container_width=True)

    top_feats = [f for f in imp_df.head(10)["feature"] if f in X.columns]
    if len(top_feats) >= 2:
        st.plotly_chart(px.imshow(X[top_feats].corr(method="spearman"), text_auto=".2f", title="Top 10 Spearman 상관"), use_container_width=True)

# ── Tab 3 ──
with tab3:
    st.header("오분류 심화 분석")
    err = mis["full"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("FN", int((err["error_type"] == "FN").sum()))
    c2.metric("FP", int((err["error_type"] == "FP").sum()))
    c3.metric("라벨 오류 의심", f"{mis['label_suspect_rate']*100:.1f}%")
    c4.metric("Threshold", f"{threshold:.2f}")

    subtype_df = mis["subtype_counts"].reset_index()
    subtype_df.columns = ["유형", "건수"]
    st.plotly_chart(px.bar(subtype_df, x="유형", y="건수", color="유형"), use_container_width=True)

    col_a, col_b = st.columns(2)
    errors = mis["errors"]
    with col_a:
        st.markdown("**아쉬운 FN** (threshold 조정 가능)")
        st.dataframe(errors[errors["error_subtype"] == "FN_아쉬운(근접)"][["sample_idx", "y_proba", "label_audit"]].head(10))
    with col_b:
        st.markdown("**확신적 FN** (모델 한계)")
        st.dataframe(errors[errors["error_subtype"] == "FN_확신적오답"][["sample_idx", "y_proba", "label_audit"]].head(10))

    st.subheader("케이스 직접 검토")
    filters = st.multiselect("유형", errors["error_subtype"].unique().tolist(), default=list(errors["error_subtype"].unique()[:2]))
    pool = errors[errors["error_subtype"].isin(filters)]
    if len(pool):
        si = st.selectbox("샘플", pool["sample_idx"].tolist(),
                          format_func=lambda i: f"#{i} proba={pool.loc[pool['sample_idx']==i,'y_proba'].iloc[0]:.3f}")
        si = int(si)
        st.markdown(f"실제 **{'Fail' if y.iloc[si] else 'Pass'}** | proba **{oof.loc[oof['sample_idx']==si,'y_proba'].iloc[0]:.3f}**")
        st.markdown(f"**라벨 감사**: {errors.loc[errors['sample_idx']==si,'label_audit'].iloc[0]}")
        imp_top = [f for f in imp_df.head(12)["feature"] if f in X.columns]
        prof = pd.DataFrame({
            "sensor": imp_top,
            "z_vs_pass": ((X.iloc[si][imp_top] - X.loc[y == 0, imp_top].median()) / (X.loc[y == 0, imp_top].std() + 1e-9)).values,
        })
        st.plotly_chart(px.bar(prof, x="sensor", y="z_vs_pass", color="z_vs_pass", color_continuous_scale="RdBu_r", title=f"Sample #{si} Top 센서 z-score"), use_container_width=True)

    st.plotly_chart(px.scatter(errors, x="y_proba", y="abs_dist_threshold", color="error_subtype", hover_data=["sample_idx", "label_audit"]), use_container_width=True)

    st.subheader("모델 고도화 계획")
    for p in model_upgrade_plan(registry, cmp, mis):
        st.markdown(f"**{p['phase']}**")
        for a in p["actions"]:
            st.markdown(f"- {a}")
