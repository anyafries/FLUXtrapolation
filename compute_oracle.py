"""
Compute a per-site in-domain oracle baseline for regret.

For each site we fit a RandomForest on that site's own test rows and read its
out-of-bag (OOB) predictions -- an honest, per-row estimate of the best RMSE
achievable if the model had been trained directly on that site. The resulting
per-(setting, target, scale, env) RMSE is cached to a JSON, which eval.py joins
to compute regret = RMSE - bestRMSE for every model.

Usage:
    python compute_oracle.py --setting spatial-easy40 --target GPP
    python compute_oracle.py            # all settings x targets
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from dataloader import load_data, get_data_split
from utils.eval_utils import compute_metrics
from utils.utils import setup_logging, get_best_rmse_path

logger = setup_logging(__name__)

ALL_SETTINGS = ['time-split', 'spatial-easy40', 'TA40']
ALL_TARGETS = ['ET', 'GPP', 'NEE']


def clean_env(e):
    """Normalize an env value so its string form matches the metrics CSVs.

    For the time-split setting env is a (site_id, year) tuple; under numpy 2.x a
    numpy-int year stringifies as 'np.int64(2019)', whereas the on-disk metrics
    CSVs store the plain '('AU-ASM', 2019)'. Cast tuple ints to Python int so the
    later astype(str) reproduces the CSV form. String envs pass through unchanged.
    """
    if isinstance(e, tuple):
        return tuple(int(x) if isinstance(x, np.integer) else x for x in e)
    return e


def compute_oob_predictions(xtest, ytest, sites, min_obs, n_estimators,
                            min_samples_leaf):
    """
    Fit a RandomForest per site and return its OOB predictions.

    Args:
        xtest: Feature matrix for the test rows.
        ytest: Target vector for the test rows.
        sites: Per-row site_id labels (array aligned with xtest/ytest).
        min_obs: Minimum rows required to fit a site's oracle.
        n_estimators, min_samples_leaf: RandomForest hyperparameters.

    Returns:
        np.ndarray of OOB predictions (np.nan for skipped sites / never-OOB rows).
    """
    ytest = np.asarray(ytest).ravel()
    oob_pred = np.full(len(ytest), np.nan)

    for site in np.unique(sites):
        # Only fit on rows with an observed target (test rows keep NaN targets
        # where qc_mask == 0); NaN-target rows stay NaN and are ignored downstream.
        mask = (sites == site) & np.isfinite(ytest)
        n = int(mask.sum())
        if n < min_obs:
            logger.info(f"  Skipping {site}: only {n} valid rows (< {min_obs})")
            continue
        rf = RandomForestRegressor(
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            oob_score=True,
            n_jobs=-1,
            random_state=42,
        )
        rf.fit(xtest[mask], ytest[mask])
        oob_pred[mask] = rf.oob_prediction_

    return oob_pred


def compute_oracle_for_experiment(df, setting, target, min_obs, n_estimators,
                                  min_samples_leaf):
    """Fit the per-site OOB oracle for one (setting, target) and return its
    per-(scale, env) metrics DataFrame."""
    split = get_data_split(
        df, setting, path=args.path, target=target,
        remove_missing_target=True, return_colnames=True,
    )
    _, _, test, _, _ = split
    xtest, ytest, envs_test, sites_test, times_test = test
    sites = np.asarray(sites_test.values)

    logger.info(f"Fitting per-site oracle: setting={setting}, target={target} "
                f"({len(np.unique(sites))} sites, {len(sites)} rows)")

    oob_pred = compute_oob_predictions(
        xtest, ytest, sites, min_obs, n_estimators, min_samples_leaf
    )

    preds_df = pd.DataFrame({
        'y_true': np.asarray(ytest).ravel(),
        'y_pred': oob_pred,
        'env': [clean_env(e) for e in envs_test.values],
        'site_id': sites_test.values,
        'time': times_test.values,
    })

    return compute_metrics(preds_df, 'oracle', setting, target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute per-site OOB-oracle baseline RMSE for regret."
    )
    parser.add_argument("--path", type=str, default='data',
                        help="Path to the data directory")
    parser.add_argument("--setting", type=str,
                        choices=['time-split', 'spatial-easy40', 'TA40', 'all'],
                        default='all', help="Experiment setting")
    parser.add_argument("--target", type=str,
                        choices=['GPP', 'NEE', 'ET', 'all'],
                        default='all', help="Target variable")
    parser.add_argument("--n_estimators", type=int, default=200,
                        help="Number of trees per site oracle")
    parser.add_argument("--min_samples_leaf", type=int, default=5,
                        help="Minimum samples per leaf")
    parser.add_argument("--min_obs", type=int, default=50,
                        help="Minimum rows required to fit a site's oracle")
    parser.add_argument("--rerun", action='store_true',
                        help="Recompute even if the baseline JSON exists")

    args = parser.parse_args()

    out_path = get_best_rmse_path()
    if os.path.exists(out_path) and not args.rerun:
        logger.info(f"Oracle baseline already exists at {out_path}. "
                    f"Use --rerun to recompute.")
        raise SystemExit(0)

    settings = ALL_SETTINGS if args.setting == 'all' else [args.setting]
    targets = ALL_TARGETS if args.target == 'all' else [args.target]

    logger.info(f"Loading data from {args.path}...")
    df = load_data(args.path)

    best_cols = ['setting', 'target', 'scale', 'env']
    all_best = []
    for target in targets:
        for setting in settings:
            metrics_df = compute_oracle_for_experiment(
                df, setting, target, args.min_obs,
                args.n_estimators, args.min_samples_leaf,
            )
            if metrics_df is None:
                logger.warning(f"No oracle metrics for {setting}/{target}")
                continue
            best = metrics_df[best_cols + ['rmse']].rename(
                columns={'rmse': 'best_rmse'}
            )
            all_best.append(best)

    if not all_best:
        logger.error("No oracle baselines computed; nothing to save.")
        raise SystemExit(1)

    best_df = pd.concat(all_best, ignore_index=True)
    # Match the metrics-CSV env representation (tuples -> their string repr) so
    # eval.py can merge on env; also makes the records JSON-serializable.
    best_df['env'] = best_df['env'].astype(str)
    records = best_df.to_dict(orient='records')

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(records, f, indent=4)
    logger.info(f"Saved oracle baseline ({len(records)} rows) to {out_path}")
