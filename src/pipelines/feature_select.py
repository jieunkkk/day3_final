import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import RFE, mutual_info_classif
from sklearn.inspection import permutation_importance


class CorrelationPruner(BaseEstimator, TransformerMixin):
    def __init__(self, threshold: float = 0.95):
        self.threshold = threshold
        self.keep_indices_ = None

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        n_features = X.shape[1]
        if n_features <= 1:
            self.keep_indices_ = np.arange(n_features)
            return self
        corr = np.corrcoef(X, rowvar=False)
        corr = np.nan_to_num(corr, nan=0.0)
        upper = np.triu(np.ones_like(corr, dtype=bool), k=1)
        to_drop = set()
        for i in range(n_features):
            if i in to_drop:
                continue
            high_corr = np.where((np.abs(corr[i]) > self.threshold) & upper[i])[0]
            to_drop.update(high_corr.tolist())
        self.keep_indices_ = np.array([i for i in range(n_features) if i not in to_drop])
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return X[:, self.keep_indices_]


class MutualInfoSelector(BaseEstimator, TransformerMixin):
    def __init__(self, k: int = 50, random_state: int = 42):
        self.k = k
        self.random_state = random_state
        self.keep_indices_ = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        k = min(self.k, X.shape[1])
        scores = mutual_info_classif(X, y, random_state=self.random_state)
        self.keep_indices_ = np.argsort(scores)[-k:]
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return X[:, self.keep_indices_]


class RFESelector(BaseEstimator, TransformerMixin):
    def __init__(self, estimator, n_features: int = 50):
        self.estimator = estimator
        self.n_features = n_features
        self.selector_ = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        n = min(self.n_features, X.shape[1])
        self.selector_ = RFE(estimator=self.estimator, n_features_to_select=n, step=0.1)
        self.selector_.fit(X, y)
        return self

    def transform(self, X):
        return self.selector_.transform(X)


def remove_isolation_forest_outliers(X, y, contamination: float = 0.05, random_state: int = 42):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    iso = IsolationForest(contamination=contamination, random_state=random_state)
    mask = iso.fit_predict(X) == 1
    return X[mask], y[mask]
