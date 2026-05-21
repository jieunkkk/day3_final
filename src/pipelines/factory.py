import lightgbm as lgb
import xgboost as xgb
from imblearn.pipeline import Pipeline as ImbPipeline
from lightgbm import LGBMClassifier
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier

from src.experiment.config_schema import ExperimentConfig
from src.pipelines.feature_select import CorrelationPruner, MutualInfoSelector, RFESelector
from src.pipelines.missing import (
    DropConstantFeatures,
    DropHighMissingColumns,
    IQRClipOutliers,
    ImputerTransformer,
    WinsorizeOutliers,
)
from src.pipelines.sampler import get_sampler


MISSING_MAP = {
    "M0": {"col_threshold": None, "row_threshold": None, "imputer": "median"},
    "M1": {"col_threshold": 0.5, "row_threshold": None, "imputer": "median"},
    "M2": {"col_threshold": 0.9, "row_threshold": None, "imputer": "median"},
    "M3": {"col_threshold": 0.5, "row_threshold": 0.3, "imputer": "median"},
    "M4": {"col_threshold": 0.5, "row_threshold": None, "imputer": "knn"},
    "M5": {"col_threshold": 0.5, "row_threshold": None, "imputer": "iterative"},
}


def _build_preprocessing_steps(cfg: ExperimentConfig) -> list:
    missing = MISSING_MAP[cfg.missing_policy]
    steps = [
        ("drop_constant", DropConstantFeatures()),
        ("drop_missing_cols", DropHighMissingColumns(threshold=missing["col_threshold"])),
    ]
    steps.append(("imputer", ImputerTransformer(strategy=missing["imputer"])))

    if cfg.outlier_policy == "O1":
        steps.append(("winsorize", WinsorizeOutliers()))
    elif cfg.outlier_policy == "O2":
        steps.append(("iqr_clip", IQRClipOutliers()))
    elif cfg.outlier_policy == "O4":
        steps.append(("robust_scale_outlier", RobustScaler()))

    if cfg.feature_policy == "F1":
        steps.append(("corr_prune", CorrelationPruner(threshold=0.95)))
    elif cfg.feature_policy.startswith("F2"):
        k = int(cfg.feature_policy.split("_")[-1]) if "_" in cfg.feature_policy else 50
        steps.append(("corr_prune", CorrelationPruner(threshold=0.95)))
        steps.append(("mi_select", MutualInfoSelector(k=k, random_state=cfg.random_state)))
    elif cfg.feature_policy.startswith("F3"):
        k = int(cfg.feature_policy.split("_")[-1]) if "_" in cfg.feature_policy else 50
        steps.append(("rfe_select", RFESelector(
            estimator=LGBMClassifier(n_estimators=100, random_state=cfg.random_state, verbose=-1),
            n_features=k,
        )))
    elif cfg.feature_policy == "F4":
        steps.append(("corr_prune", CorrelationPruner(threshold=0.95)))
        steps.append(("mi_select", MutualInfoSelector(k=50, random_state=cfg.random_state)))

    if cfg.scale_policy == "S1":
        steps.append(("scaler", StandardScaler()))
    elif cfg.scale_policy == "S2":
        steps.append(("scaler", RobustScaler()))
    elif cfg.scale_policy == "S0" and cfg.outlier_policy != "O4":
        pass

    return steps


def _pos_weight_from_params(cfg: ExperimentConfig, y) -> float | None:
    if cfg.sampling_policy in {"B4", "B5", "B6", "B7", "B8"} or cfg.sampling_policy.startswith("B4_"):
        return None
    if "scale_pos_weight" in cfg.model_params:
        return cfg.model_params["scale_pos_weight"]
    if cfg.sampling_policy == "B2":
        n_pos = max(int(y.sum()), 1)
        n_neg = len(y) - n_pos
        return n_neg / n_pos
    return None


def build_classifier(cfg: ExperimentConfig, y=None):
    params = dict(cfg.model_params)
    pos_weight = _pos_weight_from_params(cfg, y) if y is not None else params.get("scale_pos_weight")

    if cfg.model_name == "dummy":
        return DummyClassifier(strategy="most_frequent")

    if cfg.model_name == "logistic_regression":
        return LogisticRegression(
            penalty=params.get("penalty", "l2"),
            C=params.get("C", 1.0),
            class_weight="balanced" if cfg.sampling_policy == "B1" else None,
            max_iter=2000,
            random_state=cfg.random_state,
            solver="liblinear" if params.get("penalty", "l2") == "l1" else "lbfgs",
        )

    if cfg.model_name == "linear_svc":
        base = LinearSVC(
            C=params.get("C", 1.0),
            class_weight="balanced" if cfg.sampling_policy == "B1" else None,
            random_state=cfg.random_state,
            max_iter=5000,
        )
        return CalibratedClassifierCV(base, cv=3)

    if cfg.model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=params.get("n_estimators", 300),
            max_depth=params.get("max_depth", 10),
            min_samples_leaf=params.get("min_samples_leaf", 5),
            class_weight="balanced" if cfg.sampling_policy == "B1" else None,
            random_state=cfg.random_state,
            n_jobs=-1,
        )

    if cfg.model_name == "xgboost":
        xgb_params = {
            "n_estimators": params.get("n_estimators", 500),
            "max_depth": params.get("max_depth", 5),
            "learning_rate": params.get("learning_rate", 0.05),
            "subsample": params.get("subsample", 0.8),
            "colsample_bytree": params.get("colsample_bytree", 0.5),
            "min_child_weight": params.get("min_child_weight", 3),
            "gamma": params.get("gamma", 0.0),
            "reg_alpha": params.get("reg_alpha", 0.1),
            "reg_lambda": params.get("reg_lambda", 5.0),
            "random_state": cfg.random_state,
            "eval_metric": "logloss",
            "verbosity": 0,
        }
        if pos_weight is not None:
            xgb_params["scale_pos_weight"] = pos_weight
        return XGBClassifier(**xgb_params)

    if cfg.model_name == "lightgbm":
        lgb_params = {
            "n_estimators": params.get("n_estimators", 500),
            "num_leaves": params.get("num_leaves", 31),
            "max_depth": params.get("max_depth", -1),
            "learning_rate": params.get("learning_rate", 0.05),
            "subsample": params.get("subsample", 0.8),
            "colsample_bytree": params.get("colsample_bytree", 0.5),
            "min_child_samples": params.get("min_child_samples", 20),
            "reg_alpha": params.get("reg_alpha", 0.1),
            "reg_lambda": params.get("reg_lambda", 1.0),
            "random_state": cfg.random_state,
            "verbose": -1,
        }
        if pos_weight is not None:
            lgb_params["scale_pos_weight"] = pos_weight
        elif cfg.sampling_policy == "B1":
            lgb_params["class_weight"] = "balanced"
        return LGBMClassifier(**lgb_params)

    raise ValueError(f"Unknown model: {cfg.model_name}")


def build_pipeline(cfg: ExperimentConfig, y=None):
    steps = _build_preprocessing_steps(cfg)
    sampler = get_sampler(cfg.sampling_policy, cfg.random_state)
    if sampler is not None:
        steps.append(("sampler", sampler))
    steps.append(("classifier", build_classifier(cfg, y)))
    return ImbPipeline(steps)
