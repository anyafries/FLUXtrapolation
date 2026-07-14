"""
AnchorBoosting wrapper for the FLUXNET benchmark.

Wraps `anchorboosting.AnchorBooster` (LightGBM-based boosting of the anchor
loss) with the (X, y, envs) fit signature used elsewhere in this repo.

Two anchor configurations are exposed through ``anchor_type``:

  - 'onehot'     : the environments, one-hot / categorical encoded, as the
                   anchor. A 1d integer array is passed as Z, which
                   ``AnchorBooster`` treats as a categorical anchor whose
                   projection equals the one-hot encoding. (Model ``anchorboosting``.)
  - 'continuous' : a continuous linear anchor built from the tower coordinates
                   ([tower_lat, tower_lon]). For the time-split setting (where
                   the environment is a (site_id, year) pair) a time axis is
                   appended: the observation date as days since 2015-01-01 when
                   supplied via the ``dates`` fit argument, otherwise the year
                   offset (year - 2015) as a fallback. An intercept column is
                   prepended so the projection residualizes the mean.
                   (Model ``anchorboosting-c``.)

The continuous anchor looks up per-site tower coordinates on demand from the
per-site CSVs under ``{data_path}/sites/{site_id}.csv`` (cached across calls).
If you train with a non-default ``--path``, pass a matching ``data_path``.

Reference: https://github.com/mlondschien/anchor_boosting
"""

import os

import numpy as np
import pandas as pd

from anchorboosting import AnchorBooster
from sklearn.preprocessing import LabelEncoder


# Cache of {(data_path, site_id): (lat, lon)} so we don't re-read CSVs every fit.
_COORD_CACHE = {}


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
        anchor_type: 'onehot' (categorical environment anchor) or 'continuous'
            (linear lat/lon[/time] anchor).
        data_path: Directory holding per-site CSVs (``{data_path}/sites/*.csv``),
            used only by the continuous anchor to look up tower coordinates.
        **lgbm_params: Additional LightGBM parameters (e.g. ``num_leaves``,
            ``lambda_l2``).
    """

    def __init__(self, gamma=1.0, num_boost_round=1000, learning_rate=0.1,
                 max_depth=3, min_gain_to_split=0.1, anchor_type='onehot',
                 data_path='data', **lgbm_params):
        self.gamma = gamma
        self.num_boost_round = num_boost_round
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_gain_to_split = min_gain_to_split
        self.anchor_type = anchor_type
        self.data_path = data_path
        self.lgbm_params = lgbm_params

    # ------------------------------------------------------------------
    # Anchor construction
    # ------------------------------------------------------------------
    def _site_coords(self, site_id):
        """Look up (tower_lat, tower_lon) for a site from its CSV (cached)."""
        key = (self.data_path, site_id)
        if key not in _COORD_CACHE:
            csv_path = os.path.join(self.data_path, "sites", f"{site_id}.csv")
            if not os.path.exists(csv_path):
                raise FileNotFoundError(
                    f"Could not find site file {csv_path!r} needed to build the "
                    f"continuous anchor. Pass data_path=<your --path> to the model."
                )
            row = pd.read_csv(csv_path, usecols=["tower_lat", "tower_lon"], nrows=1)
            _COORD_CACHE[key] = (float(row["tower_lat"][0]), float(row["tower_lon"][0]))
        return _COORD_CACHE[key]

    def _build_continuous_Z(self, envs, dates=None):
        """Build the continuous linear anchor [intercept, lat, lon(, time)]."""
        envs = list(envs)
        # In the time-split setting envs are (site_id, year) tuples and we add a
        # time axis; in the spatial settings they are plain site_id strings.
        has_time = not isinstance(envs[0], str) and hasattr(envs[0], "__len__")
        if has_time:
            site_ids = [str(e[0]) for e in envs]
            years = np.array([float(e[1]) for e in envs], dtype=np.float64)
        else:
            site_ids = [str(e) for e in envs]
            years = None

        coords = {s: self._site_coords(s) for s in set(site_ids)}
        lat = np.array([coords[s][0] for s in site_ids], dtype=np.float64)
        lon = np.array([coords[s][1] for s in site_ids], dtype=np.float64)

        cols = [np.ones(len(site_ids), dtype=np.float64), lat, lon]  # intercept + lat/lon
        if has_time:
            if dates is not None:
                # Days since 2015-01-01, supplied per-row by the benchmark.
                cols.append(np.asarray(dates, dtype=np.float64))
            else:
                # Fallback when no row-level date is available: years since 2015.
                cols.append(years - 2015.0)
        return np.column_stack(cols)

    # ------------------------------------------------------------------
    # sklearn-style API
    # ------------------------------------------------------------------
    def fit(self, X, y, envs, dates=None):
        if self.anchor_type == 'continuous':
            Z = self._build_continuous_Z(envs, dates)
            categorical_feature = None
        else:
            if not isinstance(envs[0], str) and hasattr(envs[0], "__len__"):
                envs = ["".join(map(str, e)) for e in envs]
            Z = LabelEncoder().fit_transform(envs).astype(np.int64)

            # Boolean features were cast to float64 in the dataloader; recover them
            # as the columns whose values lie in {0, 1}.
            categorical_feature = [
                j for j in range(X.shape[1])
                if np.isin(np.unique(X[:, j]), [0.0, 1.0]).all()
            ] or None

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
        self.model_.fit(X, y, Z=Z, categorical_feature=categorical_feature)
        return self

    def predict(self, X):
        return self.model_.predict(X)
