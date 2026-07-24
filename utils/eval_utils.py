"""
Evaluation metrics for FLUXNET benchmark.

This module contains functions to evaluate model predictions using
various metrics. For temporal aggregation, see aggregation.py.
"""

import json
import numpy as np
import pandas as pd
import os

from utils.aggregation import AGGREGATIONS
from utils.utils import (
    setup_logging,
    get_metrics_path,
    get_params_path,
    get_model_path,
    save_csv,
    load_csv
)

logger = setup_logging(__name__)

# -----------------------------------------------------------------------
# ----------------------- Individual Metric Functions -------------------
# -----------------------------------------------------------------------
# Each function takes ytrue, ypred arrays and returns a scalar metric.

def rmse(ytrue, ypred):
    """Root Mean Squared Error."""
    return np.sqrt(np.nanmean((ypred - ytrue) ** 2))


def mse(ytrue, ypred):
    """Mean Squared Error."""
    return np.nanmean((ypred - ytrue) ** 2)


def mae(ytrue, ypred):
    """Mean Absolute Error."""
    return np.nanmean(np.abs(ypred - ytrue))


def relative_mae(ytrue, ypred):
    """Relative MAE: mae / mean(obs)."""
    mask = np.isfinite(ytrue) & np.isfinite(ypred)
    if mask.sum() == 0:
        return np.nan
    mean_obs = np.nanmean(ytrue[mask])
    if mean_obs == 0:
        return np.nan
    return np.nanmean(np.abs(ypred[mask] - ytrue[mask])) / mean_obs


def bias(ytrue, ypred):
    """Mean bias (obs - pred)."""
    return np.nanmean(ytrue - ypred)


def relative_bias(ytrue, ypred):
    """Relative bias: bias / mean(obs)."""
    mask = np.isfinite(ytrue) & np.isfinite(ypred)
    if mask.sum() == 0:
        return np.nan
    mean_obs = np.nanmean(ytrue[mask])
    if mean_obs == 0:
        return np.nan
    return np.nanmean(ytrue[mask] - ypred[mask]) / mean_obs


def nse(ytrue, ypred):
    """
    Nash-Sutcliffe Efficiency (NSE / MEF in QuickEval).

    NSE = 1 - sum((obs - pred)^2) / sum((obs - mean(obs))^2)

    Returns:
        float: NSE value (ranges from -inf to 1, where 1 is perfect)
    """
    mask = np.isfinite(ytrue) & np.isfinite(ypred)
    ytrue_m = ytrue[mask]
    ypred_m = ypred[mask]

    ss_res = np.sum((ytrue_m - ypred_m) ** 2)
    ss_tot = np.sum((ytrue_m - np.mean(ytrue_m)) ** 2)

    if ss_tot == 0:
        return np.nan
    return 1 - (ss_res / ss_tot)


def r2_score(ytrue, ypred):
    """Coefficient of determination (R²)."""
    mask = np.isfinite(ytrue) & np.isfinite(ypred)
    if mask.sum() < 2:
        return np.nan
    if (np.std(ytrue[mask]) == 0) or (np.std(ypred[mask]) == 0):
        return np.nan
    corr = np.corrcoef(ytrue[mask], ypred[mask])[0, 1]
    return corr ** 2


# Default metrics to compute
DEFAULT_METRICS = {
    'mse': mse,
    'rmse': rmse,
    'mae': mae,
    'nse': nse,
    'r2_score': r2_score,
    'bias': bias,
    'relative_mae': relative_mae,
    'relative_bias': relative_bias
}

# -----------------------------------------------------------------------
# -------------------------- Metrics I/O --------------------------------
# -----------------------------------------------------------------------

def compute_and_save_metrics(predictions_df, setting, target, model_name, val_strategy):
    """Save metrics DataFrame to CSV."""
    metrics_path = get_metrics_path(setting, target, model_name, val_strategy)
    metrics_df = compute_metrics(predictions_df, model_name, setting, target)
    save_csv(metrics_df, metrics_path)
    return metrics_df


def load_metrics(setting, target, model_name, val_strategy):
    """Load metrics file for a given experiment."""
    metrics_path = get_metrics_path(setting, target, model_name, val_strategy)
    return load_csv(metrics_path)


def save_best_params(best_params, setting, target, model_name, val_strategy,
                     val_score=None):
    """Save the best hyperparameter dictionary (and its validation score) to JSON."""
    path = get_params_path(setting, target, model_name, val_strategy)
    # Ensure the directory exists (in case it's the first file being saved)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, 'w') as f:
        json.dump({'params': best_params, 'val_score': val_score}, f, indent=4)
    logger.info(f"Saved best parameters to {path}")


def save_model(model, params, input_dim, setting, target, model_name, val_strategy):
    """
    Save a fitted deep-learning model's weights for later reuse.

    Stores each nn.Module submodule's state_dict plus the params and input_dim
    needed to rebuild the architecture (see get_model). Only meaningful for the
    torch-based models (mlp, gdro, coral, mmd).
    """
    import torch

    path = get_model_path(setting, target, model_name, val_strategy)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {name: module.state_dict()
             for name, module in vars(model).items()
             if isinstance(module, torch.nn.Module)}
    torch.save({'params': params, 'input_dim': input_dim, 'state': state}, path)
    logger.info(f"Saved model weights to {path}")


# -----------------------------------------------------------------------
# ----------------------- Metric Computation ----------------------------
# -----------------------------------------------------------------------

def compute_metrics(predictions_df, model_name, setting, target, scales=None, metrics=None):
    """
    Compute metrics at all temporal scales and environments for an experiment.
    """
    if predictions_df is None:
        logger.warning("No predictions DataFrame provided, cannot compute metrics.")
        return None
    
    scales = scales or list(AGGREGATIONS.keys())
    metrics = metrics or DEFAULT_METRICS
    multi_year_scales = {'seasonal', 'iav', 'anom'}
    
    all_results = []

    for scale in scales:
        try:
            # 1. Setup and Aggregate Data
            df_scale = predictions_df.copy()
            if scale in multi_year_scales and 'site_id' in df_scale.columns:
                df_scale['env'] = df_scale['site_id']

            if scale not in AGGREGATIONS:
                raise ValueError(f"Unknown scale: {scale}. Available: {list(AGGREGATIONS.keys())}")
                
            agg_df = AGGREGATIONS[scale](df_scale)

            # 2. Group by environment and compute metrics
            scale_results = []
            for env, group in agg_df.groupby('env'):
                y_true, y_pred = group['y_true'].values, group['y_pred'].values
                
                # Base row with metadata
                row = {
                    'target': target, 'setting': setting, 'model': model_name,
                    'scale': scale, 'env': env, 'n_samples': len(group)
                }

                # Compute scalar metrics
                valid_data = not (np.all(np.isnan(y_true)) or np.all(np.isnan(y_pred)))
                for name, func in metrics.items():
                    try:
                        row[name] = func(y_true, y_pred) if valid_data else np.nan
                    except Exception:
                        row[name] = np.nan
                        
                scale_results.append(row)
            
            all_results.extend(scale_results)

            # 3. Logging for the current scale
            if scale_results:
                n_samples = [r['n_samples'] for r in scale_results]
                summary_log = f"Computed {scale} metrics: {len(scale_results)} groups"
                logger.info(summary_log + ("\t" if len(summary_log) < 33 else "") +
                            f"\t(min samples: {min(n_samples)}, mean: {np.mean(n_samples):.1f}, max: {max(n_samples)})")

        except Exception as e:
            logger.warning(f"Could not compute {scale} metrics: {e}")

    if not all_results:
        return None

    # 4. Format and return DataFrame
    combined = pd.DataFrame(all_results)
    leading_cols = ['target', 'setting', 'model', 'scale', 'env', 'n_samples']
    other_cols = [c for c in combined.columns if c not in leading_cols]
    
    return combined[[c for c in leading_cols if c in combined.columns] + other_cols]