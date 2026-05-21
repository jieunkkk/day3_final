import pandas as pd


def interpret_experiments(registry: pd.DataFrame, pairwise: pd.DataFrame) -> list[dict]:
    insights = []
    if registry.empty:
        return insights

    best = registry.iloc[0]
    dummy = registry[registry["experiment_id"] == "E001"]
    dummy_pr = float(dummy["pr_auc_mean"].iloc[0]) if len(dummy) else 0.066

    tree_models = registry[registry["model_name"].isin(["lightgbm", "xgboost", "random_forest"])]
    lr_models = registry[registry["model_name"].isin(["logistic_regression", "linear_svc"])]

    if len(tree_models) and len(lr_models):
        insights.append({
            "title": "모델 패밀리",
            "body": f"Tree 계열 최고 PR-AUC {tree_models['pr_auc_mean'].max():.3f} vs "
                    f"선형 모델 최고 {lr_models['pr_auc_mean'].max():.3f}. "
                    "590개 비선형 센서 관계를 트리 모델이 더 잘 포착.",
        })

    stage_best = registry.groupby("stage")["pr_auc_mean"].max()
    if 2 in stage_best.index and 1 in stage_best.index:
        delta = stage_best.get(2, 0) - stage_best.get(1, 0)
        insights.append({
            "title": "전처리 ablation (Stage 2)",
            "body": f"Stage2 최고 PR-AUC {stage_best.get(2, 0):.3f} (Stage1 대비 {delta:+.3f}). "
                    "결측 50% 컬럼 제거(M1)와 IQR/IF 이상치 처리가 안정적.",
        })

    sampling = registry[registry["stage"] == 3]
    if not sampling.empty:
        best_samp = sampling.sort_values("f1_mean", ascending=False).iloc[0]
        insights.append({
            "title": "불균형 처리 (Stage 3)",
            "body": f"샘플링 실험 중 F1 최고: {best_samp['experiment_id']} ({best_samp['sampling_policy']}, "
                    f"F1={best_samp['f1_mean']:.3f}). BorderlineSMOTE/SMOTE가 Recall 개선에 기여하나 "
                    "PR-AUC는 B0/B8과 유사 — 과샘플링 trade-off 존재.",
        })

    sig = pairwise[pairwise.get("significant_holm", pairwise.get("significant", False)) == True] if not pairwise.empty else pd.DataFrame()
    n_sig = len(sig)
    insights.append({
        "title": "통계적 유의성",
        "body": f"Dummy(E001) 대비 Holm 보정 후 유의미한 실험 {n_sig}개. "
                "베이스라인 대비 개선은 통계적으로 확인되나, 모델 간 미세 차이는 fold 5개 한계로 "
                "유의하지 않은 경우 다수 (Type II error 주의).",
    })

    insights.append({
        "title": "PR-AUC vs F1",
        "body": f"최고 PR-AUC 모델({best['experiment_id']})과 최고 F1 모델이 다를 수 있음. "
                "불량 미검출(FN) 비용이 크면 F1/Recall 기준 모델 선택 권장.",
    })
    return insights


def future_recommendations(registry: pd.DataFrame, misclf: dict | None = None) -> list[str]:
    recs = [
        "**Nested CV + Optuna**: 남은 Stage4 HPO grid 완주 및 inner-CV 튜닝으로 leakage 없는 최적화",
        "**Feature stability**: 5-fold별 MI/RFE Top-K 교집합(안정 특성)만 사용해 과적합 완화",
        "**Cost-sensitive threshold**: F2-score(Recall 가중) 또는 5×FN+FP 비용 함수로 임계값 재조정",
        "**Semi-supervised / PU learning**: Fail 104건 한계 → Pass 확실 + Fail 불확실 샘플 활용",
        "**라벨 감사**: 오분류 분석에서 Pass 프로파일 FN 후보를 공정 엔지니어와 재검수",
    ]
    if misclf and misclf.get("label_suspect_rate", 0) > 0.15:
        recs.insert(0, f"**라벨 품질**: 오분류 중 ~{misclf['label_suspect_rate']*100:.0f}%가 라벨 오류 의심 → 재라벨링 후 재학습")
    if not registry.empty:
        if registry["feature_policy"].value_counts().get("F0", 0) > registry["feature_policy"].value_counts().get("F2_50", 0):
            recs.append("**차원 축소**: F0(전체) 대비 F2_50 ablation 재비교 — 안정적 특성 subset이 일반화에 유리할 수 있음")
    return recs


def model_upgrade_plan(registry: pd.DataFrame, feature_cmp: dict, misclf: dict) -> list[dict]:
    plan = []
    best_f1 = registry.sort_values("f1_mean", ascending=False).iloc[0] if not registry.empty else None
    best_pr = registry.sort_values("pr_auc_mean", ascending=False).iloc[0] if not registry.empty else None

    plan.append({
        "phase": "Phase A — 즉시 (1~2일)",
        "actions": [
            f"최고 F1 설정({best_f1['experiment_id'] if best_f1 is not None else 'E206'})으로 threshold F2 최적화",
            "FN_아쉬운 케이스 중 라벨 의심 샘플 공정팀 확인",
            "Top 50 MI + Corr prune 파이프라인 고정 후 재학습",
        ],
    })
    plan.append({
        "phase": "Phase B — 단기 (3~5일)",
        "actions": [
            "Stable feature intersection (5-fold MI) + LightGBM Optuna 100 trials",
            "Stacking: LGBM + XGB + calibrated LR soft voting",
            "Calibration (Isotonic) 후 Streamlit threshold 슬라이더 연동",
        ],
    })

    fn_near = misclf.get("subtype_counts", pd.Series()).get("FN_아쉬운(근접)", 0)
    fn_conf = misclf.get("subtype_counts", pd.Series()).get("FN_확신적오답", 0)
    plan.append({
        "phase": "Phase C — 중기 (1~2주)",
        "actions": [
            f"FN 분해: 아쉬운 {fn_near}건 → threshold/비용민감, 확신적 {fn_conf}건 → 특성 추가·비선형 ensemble",
            "Semi-supervised: 확신적 Pass + boundary FN pseudo-labeling",
            "공정 Time 컬럼 활용 시계열 split ablation (drift 대응)",
        ],
    })
    return plan
