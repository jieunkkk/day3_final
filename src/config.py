from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "secom"
EXPERIMENTS_DIR = ROOT / "experiments"
REPORTS_DIR = ROOT / "reports"

BASELINE_ID = "E001"

POLICY_LABELS = {
    "missing": {
        "M0": "결측 그대로 + median impute",
        "M1": "결측 50%↑ 컬럼 제거 + median",
        "M2": "결측 90%↑ 컬럼 제거 + median",
        "M3": "M1 + 행 결측 30%↑ 제거",
        "M4": "M1 + KNN impute",
        "M5": "M1 + MICE impute",
    },
    "outlier": {
        "O0": "이상치 처리 없음",
        "O1": "Winsorize (1~99%)",
        "O2": "IQR clip",
        "O3": "Isolation Forest 행 제거",
        "O4": "RobustScaler",
    },
    "feature": {
        "F0": "전체 특성",
        "F1": "상관 0.95↑ 중복 제거",
        "F2_20": "MI Top 20",
        "F2_50": "MI Top 50",
        "F2_100": "MI Top 100",
        "F3_20": "RFE 20",
        "F3_50": "RFE 50",
        "F4": "Corr prune + MI 50",
    },
    "sampling": {
        "B0": "샘플링 없음",
        "B1": "class_weight balanced",
        "B2": "scale_pos_weight",
        "B3": "RandomUnderSampler",
        "B4": "SMOTE",
        "B5": "BorderlineSMOTE",
        "B6": "SMOTETomek",
        "B7": "ADASYN",
        "B8": "weight + SMOTE",
    },
}
