"""
Script to load and compare results from multiple experiments.

This script loads pre-computed metrics if available, or computes them from
predictions. Results are saved to results/metrics/ for subsequent runs.

Usage:
    python eval.py --setting spatial-easy --target GPP
"""

import json
import os
import pandas as pd

from dataloader import load_predictions
from utils.eval_utils import load_metrics, compute_and_save_metrics
from utils.plots import plot_metric_grid, plot_cdf_grid, create_html_leaderboard
from utils.utils import (
    setup_logging,
    find_available_experiments,
    get_best_rmse_path,
    get_metrics_path,
    save_csv,
)

logger = setup_logging(__name__)

display_names = {
    "time-split": "temporal",
    "spatial-easy40": "spatial",
    "TA40": "temperature"
}

def add_regret_column(metrics_df, setting, target, metrics_path):
    """Ensure metrics_df has a 'regret' column.

    If missing, join the per-site OOB-oracle baseline (best_rmse) from the cached
    JSON, compute regret = rmse - best_rmse, and rewrite the metrics CSV in place.
    If the column already exists the df is returned untouched (no rewrite); if the
    oracle baseline is absent a warning is logged and regret is skipped.
    """
    if metrics_df is None or 'regret' in metrics_df.columns:
        return metrics_df

    best_path = get_best_rmse_path()
    if not os.path.exists(best_path):
        logger.warning(f"No oracle baseline at {best_path}; skipping regret. "
                       f"Run compute_oracle.py first.")
        return metrics_df

    with open(best_path) as f:
        best_df = pd.DataFrame(json.load(f))
    best_df = best_df[(best_df['setting'] == setting) &
                      (best_df['target'] == target)][['scale', 'env', 'best_rmse']]

    merged = metrics_df.merge(best_df, on=['scale', 'env'], how='left')
    merged['regret'] = merged['rmse'] - merged['best_rmse']
    merged = merged.drop(columns=['best_rmse'])   # persist regret only
    save_csv(merged, metrics_path)                # rewrite in place
    return merged


def get_metrics(setting, target, model_name, val_strategy, rerun=False):
    """Get metrics for an experiment, computing if necessary.

    Also ensures the metrics carry a persisted 'regret' column (see
    add_regret_column), rewriting the CSV once if it predates that column.
    """
    metrics_path = get_metrics_path(setting, target, model_name, val_strategy)

    # Try loading existing metrics
    metrics_df = None if rerun else load_metrics(setting, target, model_name, val_strategy)

    # Compute and save metrics if not available
    if metrics_df is None:
        predictions_df = load_predictions(setting, target, model_name, val_strategy)
        metrics_df = compute_and_save_metrics(predictions_df, setting, target, model_name, val_strategy)

    return add_regret_column(metrics_df, setting, target, metrics_path)


def load_all_metrics(settings=None, targets=None, models=None, scales=None, 
                     val_strategy='mean', rerun=False):
    """
    Load or compute metrics for all specified experiments.

    Args:
        settings: List of settings to include (default: all found)
        targets: List of targets to include (default: all found)
        models: List of models to include (default: all found)
        scales: List of scales to include (default: all scales in data)
        val_strategy: Validation strategy to load ('mean', 'max', 'discrepancy')
        rerun: If True, recompute all metrics from predictions

    Returns:
        pd.DataFrame with all metrics combined
    """
    available = find_available_experiments()
    if not available:
        logger.error(f"No experiments found.")
        return pd.DataFrame()

    # Filter by val_strategy and other criteria
    available = [(s, t, m, vs) for s, t, m, vs in available if vs == val_strategy]
    if settings:
        available = [(s, t, m, vs) for s, t, m, vs in available if s in settings]
    if targets:
        available = [(s, t, m, vs) for s, t, m, vs in available if t in targets]
    if models:
        available = [(s, t, m, vs) for s, t, m, vs in available if m in models]

    logger.info(f"Processing {len(available)} experiments (val_strategy={val_strategy})")

    all_results = []
    for setting, target, model_name, vs in available:
        metrics_df = get_metrics(setting, target, model_name, vs, rerun=rerun)
        if metrics_df is not None:
            if scales and 'scale' in metrics_df.columns:
                metrics_df = metrics_df[metrics_df['scale'].isin(scales)]
            drop_scales = ['daily', 'monthly']
            metrics_df = metrics_df[~metrics_df['scale'].isin(drop_scales)]
            all_results.append(metrics_df)

    if all_results:
        return pd.concat(all_results, ignore_index=True)

    return pd.DataFrame()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Load and compare results from FLUXNET experiments"
    )
    parser.add_argument("--plots_dir", type=str, default='results/plots')
    parser.add_argument("--setting", type=str, default=None,
                        help="Filter by setting (e.g., 'spatial-easy40')")
    parser.add_argument("--target", type=str, default=None,
                        help="Filter by target (e.g., 'GPP')")
    parser.add_argument("--model", type=str, default=None,
                        help="Filter by model (e.g., 'lr')")
    parser.add_argument("--scale", type=str, default=None,
                        help="Filter by scale (e.g., 'daily', 'weekly', 'monthly')")
    parser.add_argument("--rerun", action='store_true',
                        help="Recompute metrics from predictions")
    parser.add_argument("--val_strategy", type=str,
                        choices=['mean', 'max', 'discrepancy'], default='mean',
                        help="Validation strategy to load results for (default: mean)")
    parser.add_argument("--metric", type=str, default='rmse',
                        help="Metric to plot (default: rmse)")

    args = parser.parse_args()

    # Parse filters
    plots_dir = os.path.join(args.plots_dir, args.val_strategy)
    settings = [args.setting] if args.setting else None
    targets = [args.target] if args.target else None
    models = [args.model] if args.model else None
    scales = [args.scale] if args.scale else None
    metric = args.metric
    
    # Load results
    results = load_all_metrics(
        settings=settings,
        targets=targets,
        models=models,
        scales=scales,
        val_strategy=args.val_strategy,
        rerun=args.rerun,
    )
    results = results[results['setting'].isin(display_names.keys())]  
    results['scale'] = results['scale'].replace({'spatial': 'site-mean'})

    # Generate plots for all targets
    for target in results['target'].unique():
        plot_metric_grid(results, target, metric=metric, outdir=plots_dir)
        plot_metric_grid(results, target, 
                        agg=lambda x: x.quantile(0.9),
                        outdir=plots_dir, metric=metric)
        plot_cdf_grid(results, target, scale='hourly', metric=metric, 
                      settings_names=display_names, outdir=plots_dir)
        plot_cdf_grid(results, target, scale='daily', metric=metric, 
                      settings_names=display_names, outdir=plots_dir)
        plot_cdf_grid(results, target, scale='weekly', metric=metric, 
                      settings_names=display_names, outdir=plots_dir)
        create_html_leaderboard(results, target, metric='rmse',
                                aggfunc='median',
                                settings_names=display_names,
                                filename=f'{plots_dir}/leaderboard_{target}.html')
        create_html_leaderboard(results, target, metric='rmse', 
                                aggfunc=lambda x: x.quantile(0.9),
                                settings_names=display_names,
                                filename=f'{plots_dir}/leaderboard_q90_{target}.html')