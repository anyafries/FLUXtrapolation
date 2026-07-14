"""
XGBoost with a fixed conservative feature set.

Only uses: TA, VPD, SW_IN, EVI, NDWI_SWIR2, and all PFT_* columns.
Feature selection happens in fit() given a feature_names list.
"""

from xgboost import XGBRegressor


_CONSERVATIVE_FEATURES = frozenset(['TA', 'VPD', 'SW_IN', 'EVI', 'NDWI_SWIR2'])
_CONSERVATIVE_PREFIX = 'PFT_'


class XGBConservative:
    """XGBoost restricted to the conservative meteorological + PFT feature set."""

    def __init__(self, **kwargs):
        self._xgb = XGBRegressor(**kwargs)
        self._feature_indices = None

    def fit(self, X, y, feature_names=None, eval_set=None, **kwargs):
        if feature_names is not None:
            self._feature_indices = [
                i for i, name in enumerate(feature_names)
                if name in _CONSERVATIVE_FEATURES or name.startswith(_CONSERVATIVE_PREFIX)
            ]
            X = X[:, self._feature_indices]
            if eval_set is not None:
                eval_set = [(ev[0][:, self._feature_indices], ev[1]) for ev in eval_set]
        return self._xgb.fit(X, y, eval_set=eval_set, **kwargs)

    def predict(self, X):
        if self._feature_indices is not None:
            X = X[:, self._feature_indices]
        return self._xgb.predict(X)
