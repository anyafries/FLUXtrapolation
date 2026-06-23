"""
AnchorBoosting wrapper for the FLUXNET benchmark.

Wraps `anchorboosting.AnchorBooster` (LightGBM-based boosting of the anchor
loss) with the (X, y, envs) fit signature used elsewhere in this repo.

Reference: https://github.com/mlondschien/anchor_boosting
"""

import numpy as np

from anchorboosting import AnchorBooster
from sklearn.preprocessing import LabelEncoder


class AnchorBoosting:
    """
    Anchor boosting for worst-environment generalization.

    Defaults follow the recommendations from the AnchorBooster docstring:
    shallow trees (``max_depth=3``), 1000 boosting rounds, and a small
    ``min_gain_to_split`` to avoid splitting zero-variance leaves.

    Args:
        gamma: Anchor regularization strength (>= 1; 1 = standard regression).
        num_boost_round: Number of boosting iterations.
        learning_rate: Boosting learning rate.
        max_depth: Maximum tree depth.
        min_gain_to_split: Minimum gain to split a leaf.
        **lgbm_params: Additional LightGBM parameters (e.g. ``num_leaves``,
            ``lambda_l2``).
    """

    def __init__(self, gamma=1.0, num_boost_round=1000, learning_rate=0.1,
                 max_depth=3, min_gain_to_split=0.1, **lgbm_params):
        self.gamma = gamma
        self.num_boost_round = num_boost_round
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_gain_to_split = min_gain_to_split
        self.lgbm_params = lgbm_params

    def fit(self, X, y, envs):
        if not isinstance(envs[0], str) and hasattr(envs[0], "__len__"):
            envs = ["".join(map(str, e)) for e in envs]
        Z = LabelEncoder().fit_transform(envs).astype(np.int64)

        # Boolean features were cast to float64 in the dataloader; recover them
        # as the columns whose values lie in {0, 1}.
        categorical_feature = [
            j for j in range(X.shape[1])
            if np.isin(np.unique(X[:, j]), [0.0, 1.0]).all()
        ]

        self.model_ = AnchorBooster(
            gamma=self.gamma,
            num_boost_round=self.num_boost_round,
            learning_rate=self.learning_rate,
            objective="regression",
            max_depth=self.max_depth,
            min_gain_to_split=self.min_gain_to_split,
            verbosity=-1,
            **self.lgbm_params,
        )
        self.model_.fit(X, y, Z=Z, categorical_feature=categorical_feature or None)
        return self

    def predict(self, X):
        return self.model_.predict(X)
