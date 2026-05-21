import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer


class DropConstantFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        X = self._to_frame(X)
        self.keep_indices_ = [i for i, c in enumerate(X.columns) if X[c].nunique(dropna=True) > 1]
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return X[:, self.keep_indices_]

    def _to_frame(self, X):
        if isinstance(X, pd.DataFrame):
            return X
        return pd.DataFrame(X)


class DropHighMissingColumns(BaseEstimator, TransformerMixin):
    def __init__(self, threshold: float | None = None):
        self.threshold = threshold

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        if self.threshold is None:
            self.keep_indices_ = list(range(X.shape[1]))
        else:
            miss = X.isna().mean()
            self.keep_indices_ = [i for i, c in enumerate(X.columns) if miss[c] <= self.threshold]
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return X[:, self.keep_indices_]


class ImputerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, strategy: str = "median"):
        self.strategy = strategy
        self.imputer_ = None

    def fit(self, X, y=None):
        if self.strategy == "median":
            self.imputer_ = SimpleImputer(strategy="median")
        elif self.strategy == "knn":
            self.imputer_ = KNNImputer(n_neighbors=5)
        elif self.strategy == "iterative":
            self.imputer_ = IterativeImputer(max_iter=10, random_state=42)
        else:
            raise ValueError(f"Unknown imputer strategy: {self.strategy}")
        self.imputer_.fit(X)
        return self

    def transform(self, X):
        return self.imputer_.transform(X)


class WinsorizeOutliers(BaseEstimator, TransformerMixin):
    def __init__(self, lower: float = 0.01, upper: float = 0.99):
        self.lower = lower
        self.upper = upper
        self.bounds_ = None

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.bounds_ = np.quantile(X, [self.lower, self.upper], axis=0)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        low, high = self.bounds_
        return np.clip(X, low, high)


class IQRClipOutliers(BaseEstimator, TransformerMixin):
    def __init__(self, factor: float = 1.5):
        self.factor = factor
        self.bounds_ = None

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        q1 = np.nanquantile(X, 0.25, axis=0)
        q3 = np.nanquantile(X, 0.75, axis=0)
        iqr = q3 - q1
        self.bounds_ = (q1 - self.factor * iqr, q3 + self.factor * iqr)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        low, high = self.bounds_
        return np.clip(X, low, high)
