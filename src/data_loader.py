from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "secom" / "uci-secom.csv"


def load_secom(data_path: Path | str | None = None) -> tuple[pd.DataFrame, pd.Series]:
    path = Path(data_path) if data_path else DATA_PATH
    df = pd.read_csv(path)
    y = (df["Pass/Fail"] == 1).astype(int)
    X = df.drop(columns=["Pass/Fail", "Time"])
    X = X.apply(pd.to_numeric, errors="coerce")
    return X, y
