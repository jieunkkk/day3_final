from itertools import product

from src.experiment.config_schema import ExperimentConfig

DEFAULT_LGBM = {
    "n_estimators": 500,
    "num_leaves": 31,
    "learning_rate": 0.05,
    "colsample_bytree": 0.5,
    "min_child_samples": 20,
}

DEFAULT_XGB = {
    "n_estimators": 500,
    "max_depth": 5,
    "learning_rate": 0.05,
    "colsample_bytree": 0.5,
    "min_child_weight": 3,
}


def build_stage1_baselines() -> list[ExperimentConfig]:
    return [
        ExperimentConfig("E001", stage=1, tier=0, description="Dummy majority class",
                           model_name="dummy"),
        ExperimentConfig("E002", stage=1, tier=1, description="LR + M0 minimal preprocess",
                           missing_policy="M0", model_name="logistic_regression",
                           scale_policy="S1", sampling_policy="B1"),
        ExperimentConfig("E003", stage=1, tier=1, description="LR + M1 recommended missing",
                           missing_policy="M1", model_name="logistic_regression",
                           scale_policy="S1", sampling_policy="B1"),
        ExperimentConfig("E004", stage=1, tier=2, description="RandomForest + M1",
                           missing_policy="M1", model_name="random_forest", sampling_policy="B1"),
        ExperimentConfig("E005", stage=1, tier=2, description="XGBoost + M1 + scale_pos_weight",
                           missing_policy="M1", model_name="xgboost",
                           sampling_policy="B2", model_params=DEFAULT_XGB),
        ExperimentConfig("E006", stage=1, tier=2, description="LightGBM + M1 + scale_pos_weight",
                           missing_policy="M1", model_name="lightgbm",
                           sampling_policy="B2", model_params=DEFAULT_LGBM),
    ]


def build_stage2_preprocessing_ablation(best_model: str = "lightgbm") -> list[ExperimentConfig]:
    configs = []
    counter = 100

    missing_policies = ["M0", "M1", "M2", "M3", "M4", "M5"]
    for mp in missing_policies:
        counter += 1
        configs.append(ExperimentConfig(
            f"E{counter}", stage=2, tier=2,
            description=f"Missing ablation {mp}",
            missing_policy=mp, model_name=best_model,
            sampling_policy="B2", model_params=DEFAULT_LGBM if best_model == "lightgbm" else DEFAULT_XGB,
        ))

    outlier_policies = ["O0", "O1", "O2", "O3", "O4"]
    for op in outlier_policies:
        counter += 1
        configs.append(ExperimentConfig(
            f"E{counter}", stage=2, tier=2,
            description=f"Outlier ablation {op}",
            missing_policy="M1", outlier_policy=op, model_name=best_model,
            sampling_policy="B2", model_params=DEFAULT_LGBM,
        ))

    feature_policies = ["F0", "F1", "F2_20", "F2_50", "F2_100", "F3_20", "F3_50", "F4"]
    for fp in feature_policies:
        counter += 1
        configs.append(ExperimentConfig(
            f"E{counter}", stage=2, tier=2,
            description=f"Feature ablation {fp}",
            missing_policy="M1", feature_policy=fp, model_name=best_model,
            sampling_policy="B2", model_params=DEFAULT_LGBM,
        ))

    for sp in ["S0", "S1", "S2"]:
        counter += 1
        configs.append(ExperimentConfig(
            f"E{counter}", stage=2, tier=2,
            description=f"Scale ablation {sp}",
            missing_policy="M1", scale_policy=sp, model_name=best_model,
            sampling_policy="B2", model_params=DEFAULT_LGBM,
        ))

    return configs


def build_stage3_sampling_ablation(best_preprocess: dict | None = None) -> list[ExperimentConfig]:
    bp = best_preprocess or {"missing_policy": "M1", "outlier_policy": "O0", "feature_policy": "F0", "scale_policy": "S0"}
    configs = []
    counter = 200
    sampling_policies = ["B0", "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8"]

    for model in ["lightgbm", "xgboost"]:
        params = DEFAULT_LGBM if model == "lightgbm" else DEFAULT_XGB
        for sp in sampling_policies:
            counter += 1
            mp = dict(bp)
            if sp in {"B4", "B5", "B6", "B7", "B8"} and model == "lightgbm":
                model_params = {k: v for k, v in params.items()}
            elif sp == "B2":
                model_params = params
            else:
                model_params = params
            configs.append(ExperimentConfig(
                f"E{counter}", stage=3, tier=3,
                description=f"Sampling {sp} with {model}",
                missing_policy=mp.get("missing_policy", "M1"),
                outlier_policy=mp.get("outlier_policy", "O0"),
                feature_policy=mp.get("feature_policy", "F0"),
                scale_policy=mp.get("scale_policy", "S0"),
                sampling_policy=sp, model_name=model, model_params=model_params,
            ))

    for k in [3, 5, 7]:
        counter += 1
        configs.append(ExperimentConfig(
            f"E{counter}", stage=3, tier=3,
            description=f"SMOTE k_neighbors={k}",
            missing_policy=bp.get("missing_policy", "M1"),
            feature_policy=bp.get("feature_policy", "F0"),
            sampling_policy=f"B4_{k}", model_name="lightgbm", model_params=DEFAULT_LGBM,
        ))

    return configs


def build_stage4_hpo_grid(best_preprocess: dict | None = None, best_sampling: str = "B2") -> list[ExperimentConfig]:
    bp = best_preprocess or {"missing_policy": "M1", "outlier_policy": "O0", "feature_policy": "F2_50", "scale_policy": "S0"}
    configs = []
    counter = 300

    lgbm_grid = [
        {"num_leaves": 15, "learning_rate": 0.03, "colsample_bytree": 0.3, "min_child_samples": 50, "reg_lambda": 10},
        {"num_leaves": 31, "learning_rate": 0.05, "colsample_bytree": 0.5, "min_child_samples": 20, "reg_lambda": 1},
        {"num_leaves": 63, "learning_rate": 0.05, "colsample_bytree": 0.7, "min_child_samples": 10, "reg_lambda": 1},
        {"num_leaves": 31, "learning_rate": 0.01, "colsample_bytree": 0.5, "min_child_samples": 30, "reg_alpha": 1},
        {"num_leaves": 15, "learning_rate": 0.05, "colsample_bytree": 0.3, "min_child_samples": 20, "reg_alpha": 0.1},
    ]
    xgb_grid = [
        {"max_depth": 4, "learning_rate": 0.03, "colsample_bytree": 0.3, "min_child_weight": 5, "reg_lambda": 10},
        {"max_depth": 5, "learning_rate": 0.05, "colsample_bytree": 0.5, "min_child_weight": 3, "reg_lambda": 5},
        {"max_depth": 6, "learning_rate": 0.05, "colsample_bytree": 0.7, "min_child_weight": 1, "reg_lambda": 1},
        {"max_depth": 4, "learning_rate": 0.01, "colsample_bytree": 0.5, "min_child_weight": 10, "reg_alpha": 1},
        {"max_depth": 5, "learning_rate": 0.03, "colsample_bytree": 0.3, "min_child_weight": 3, "gamma": 0.1},
    ]

    for i, params in enumerate(lgbm_grid):
        counter += 1
        merged = {**DEFAULT_LGBM, **params}
        configs.append(ExperimentConfig(
            f"E{counter}", stage=4, tier=4,
            description=f"LGBM HPO grid {i+1}",
            missing_policy=bp["missing_policy"], outlier_policy=bp.get("outlier_policy", "O0"),
            feature_policy=bp.get("feature_policy", "F2_50"), scale_policy=bp.get("scale_policy", "S0"),
            sampling_policy=best_sampling, model_name="lightgbm", model_params=merged,
        ))

    for i, params in enumerate(xgb_grid):
        counter += 1
        merged = {**DEFAULT_XGB, **params}
        configs.append(ExperimentConfig(
            f"E{counter}", stage=4, tier=4,
            description=f"XGB HPO grid {i+1}",
            missing_policy=bp["missing_policy"], outlier_policy=bp.get("outlier_policy", "O0"),
            feature_policy=bp.get("feature_policy", "F2_50"), scale_policy=bp.get("scale_policy", "S0"),
            sampling_policy=best_sampling, model_name="xgboost", model_params=merged,
        ))

    return configs


def build_stage5_combined_best() -> list[ExperimentConfig]:
    """Hand-picked strong combinations from ablation patterns."""
    combos = [
        ("E500", "M1 O0 F0 B2 LGBM default", "M1", "O0", "F0", "S0", "B2", DEFAULT_LGBM),
        ("E501", "M1 O1 F2_50 B2 LGBM", "M1", "O1", "F2_50", "S0", "B2", DEFAULT_LGBM),
        ("E502", "M1 O0 F2_50 B4 SMOTE LGBM", "M1", "O0", "F2_50", "S0", "B4", DEFAULT_LGBM),
        ("E503", "M1 O0 F2_50 B2 XGB", "M1", "O0", "F2_50", "S0", "B2", DEFAULT_XGB),
        ("E504", "M1 O2 F1 B2 LGBM", "M1", "O2", "F1", "S0", "B2", DEFAULT_LGBM),
        ("E505", "M1 O0 F2_100 B6 SMOTETomek LGBM", "M1", "O0", "F2_100", "S0", "B6", DEFAULT_LGBM),
    ]
    configs = []
    for exp_id, desc, mp, op, fp, sp, samp, params in combos:
        model = "xgboost" if "XGB" in desc else "lightgbm"
        configs.append(ExperimentConfig(
            exp_id, stage=5, tier=4, description=desc,
            missing_policy=mp, outlier_policy=op, feature_policy=fp,
            scale_policy=sp, sampling_policy=samp, model_name=model, model_params=params,
        ))
    return configs


def build_stage_experiments(stage: int, **kwargs) -> list[ExperimentConfig]:
    builders = {
        1: build_stage1_baselines,
        2: lambda: build_stage2_preprocessing_ablation(kwargs.get("best_model", "lightgbm")),
        3: lambda: build_stage3_sampling_ablation(kwargs.get("best_preprocess")),
        4: lambda: build_stage4_hpo_grid(kwargs.get("best_preprocess"), kwargs.get("best_sampling", "B2")),
        5: build_stage5_combined_best,
    }
    return builders[stage]()


def build_all_experiments(skip_slow: bool = True) -> list[ExperimentConfig]:
    configs = []
    configs.extend(build_stage1_baselines())
    configs.extend(build_stage2_preprocessing_ablation())
    configs.extend(build_stage3_sampling_ablation())

    if skip_slow:
        slow_missing = {"M4", "M5"}
        slow_feature = {"F3_20", "F3_50"}
        configs = [
            c for c in configs
            if c.missing_policy not in slow_missing and c.feature_policy not in slow_feature
        ]

    configs.extend(build_stage4_hpo_grid())
    configs.extend(build_stage5_combined_best())

    seen = set()
    unique = []
    for c in configs:
        key = (c.experiment_id,)
        if key not in seen:
            seen.add(key)
            unique.append(c)
    return unique
