"""
Per-scale RMSE across scenarios, with site-bootstrap error bars.

For a given target, produces two figures -- one aggregating the held-out-env
RMSE by the median, one by the 0.9 quantile (q90). Each figure has:

  * x axis: temporal scale (hourly, weekly, seasonal, anom, iav, site-mean)
  * y axis: RMSE (ET is shown x100, matching paper_plots.py)
  * three panels: the three OOD scenarios (temporal / spatial / temperature)
  * colour: model

Error bars come from a nonparametric bootstrap over the *held-out sites*: for
each (scenario, model, scale) cell we resample the set of test sites with
replacement, pool the per-env RMSE rows of the drawn sites, and recompute the
aggregate (median or q90). The whiskers are the percentile CI of that bootstrap
distribution; the marker is the point estimate on the real (unresampled) rows.

Usage:
    python paper_experiments/scale_rmse_bootstrap_plot.py --target ET
    python paper_experiments/scale_rmse_bootstrap_plot.py --target all --n_boot 2000
"""

import argparse
import ast
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval import load_all_metrics  # noqa: E402
from utils.utils import setup_logging  # noqa: E402

logger = setup_logging(__name__)

# --- scenarios (panels) -----------------------------------------------------
SETTINGS = ['time-split', 'spatial-easy40', 'TA40']
SETTING_NAMES = {
    'time-split': 'temporal',
    'spatial-easy40': 'spatial',
    'TA40': 'temperature',
}

# --- scales (x axis) --------------------------------------------------------
# RAW_SCALES are the names on disk; 'spatial' is renamed to 'site-mean' below.
RAW_SCALES = ['hourly', 'weekly', 'seasonal', 'anom', 'iav', 'spatial']
SCALES = ['hourly', 'weekly', 'seasonal', 'anom', 'iav', 'site-mean']

# --- models (colour) --------------------------------------------------------
# The "paper set". Models with no metrics on disk (e.g. lstm until it is run)
# are simply skipped -- load_all_metrics only returns what exists.
MODELS = ['xgb', 'mlp', 'lstm', 'coral', 'gdro', 'mmd', 'lr', 'constant']
MODEL_ORDER = ['xgb', 'lightgbm', 'mlp', 'lstm', 'gdro', 'coral', 'mmd',
               'lr', 'robust-lr', 'ridge', 'constant']
# husl gives evenly-spaced, distinct hues for all models (tab10 only has 10,
# which would collide xgb and constant).
_palette = sns.color_palette("husl", n_colors=len(MODEL_ORDER))
MODEL_COLORS = {m: _palette[i] for i, m in enumerate(MODEL_ORDER)}

PLOTS_DIR = 'paper_experiments/plots'
STYLE_FILE_PATH = 'utils/neurips.mplstyle'

AGGS = {
    'median': np.nanmedian,
    'q90': lambda x: np.nanquantile(x, 0.9),
}


def _site(env):
    """Held-out site name from an env cell (('SITE', year) or bare 'SITE')."""
    s = str(env)
    if s.startswith('('):
        return ast.literal_eval(s)[0]
    return s


def bootstrap_ci(rmse_by_site, aggfunc, n_boot, ci, rng):
    """Percentile CI of `aggfunc` under resampling of sites with replacement.

    rmse_by_site: dict site -> 1d array of per-env RMSEs for that site.
    Returns (lo, hi); (nan, nan) if there is nothing to resample.
    """
    sites = list(rmse_by_site.keys())
    if not sites:
        return np.nan, np.nan
    n = len(sites)
    idx = np.arange(n)
    stats = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        draw = rng.choice(idx, size=n, replace=True)
        vals = np.concatenate([rmse_by_site[sites[k]] for k in draw])
        stats[b] = aggfunc(vals)
    half = (100.0 - ci) / 2.0
    return np.nanpercentile(stats, half), np.nanpercentile(stats, 100.0 - half)


def plot_target(results, target, aggname, n_boot, ci, seed, outdir):
    aggfunc = AGGS[aggname]
    rt = results[results['target'] == target].copy()
    if target == 'ET':
        rt['rmse'] = rt['rmse'] * 100.0  # match paper_plots.py units
    rt['site'] = rt['env'].map(_site)

    models = [m for m in MODEL_ORDER if m in set(rt['model'])]
    if not models:
        logger.warning(f"No models with data for target {target}; skipping.")
        return

    x = np.arange(len(SCALES))
    # Small horizontal dodge so overlapping markers/whiskers stay legible.
    if len(models) > 1:
        offsets = np.linspace(-0.28, 0.28, len(models))
    else:
        offsets = np.array([0.0])

    fig, axes = plt.subplots(1, len(SETTINGS), figsize=(7.0, 2.2),
                             sharex=True)
    if len(SETTINGS) == 1:
        axes = [axes]

    hourly_rows = []  # (setting, model, point, lo, hi) for the printed summary
    all_rows = []     # every scale, written to CSV

    for ax, setting in zip(axes, SETTINGS):
        rs = rt[rt['setting'] == setting]
        for mi, model in enumerate(models):
            rm = rs[rs['model'] == model]
            xs, ys, lo_err, hi_err = [], [], [], []
            for xi, scale in enumerate(SCALES):
                cell = rm[rm['scale'] == scale]
                vals = cell['rmse'].to_numpy(dtype=float)
                vals = vals[~np.isnan(vals)]
                if vals.size == 0:
                    continue
                point = aggfunc(vals)
                rmse_by_site = {
                    s: g['rmse'].to_numpy(dtype=float)
                    for s, g in cell.groupby('site')
                }
                rng = np.random.default_rng(seed + 1000 * mi + xi)
                lo, hi = bootstrap_ci(rmse_by_site, aggfunc, n_boot, ci, rng)
                width = hi - lo
                pct = 100.0 * width / abs(point) if point else float('nan')
                all_rows.append({
                    'target': target, 'agg': aggname, 'scenario':
                    SETTING_NAMES.get(setting, setting), 'setting': setting,
                    'model': model, 'scale': scale, 'n_sites': len(rmse_by_site),
                    'point': point, 'ci_lo': lo, 'ci_hi': hi,
                    'ci_width': width, 'ci_width_pct_of_point': pct,
                })
                if scale == 'hourly':
                    hourly_rows.append((setting, model, point, lo, hi))
                xs.append(xi + offsets[mi])
                ys.append(point)
                lo_err.append(max(point - lo, 0.0) if np.isfinite(lo) else 0.0)
                hi_err.append(max(hi - point, 0.0) if np.isfinite(hi) else 0.0)
            if not xs:
                continue
            ax.errorbar(
                xs, ys, yerr=[lo_err, hi_err],
                marker='o', markersize=3, linestyle='-', linewidth=0.7,
                color=MODEL_COLORS.get(model, 'gray'), label=model,
                capsize=1.5, elinewidth=0.7, alpha=0.9,
            )
        ax.set_title(SETTING_NAMES.get(setting, setting))
        ax.set_xticks(x)
        ax.set_xticklabels(SCALES, rotation=45, ha='right')
        ax.set_xlabel('temporal scale')

    ylabel = 'RMSE (x100)' if target == 'ET' else 'RMSE'
    axes[0].set_ylabel(f'{aggname} {ylabel}')

    # Single shared legend from the first panel's handles.
    handles, labels = axes[0].get_legend_handles_labels()
    # De-duplicate while keeping order.
    seen, h2, l2 = set(), [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen.add(l)
            h2.append(h)
            l2.append(l)
    fig.legend(h2, l2, loc='upper center', bbox_to_anchor=(0.5, 1.12),
               ncol=len(l2), frameon=False, handlelength=1.0,
               handletextpad=0.4, columnspacing=1.0)

    # NB: neurips.mplstyle uses text.usetex, so a literal '%' must be escaped
    # (otherwise TeX treats the rest of the line as a comment and drops it).
    fig.suptitle(f'{target} -- {aggname} RMSE '
                 f'({ci:.0f}\\% site-bootstrap CI, n={n_boot})',
                 y=1.18, fontsize='small')

    os.makedirs(outdir, exist_ok=True)
    out = f'{outdir}/scale_rmse_{aggname}_{target}.png'
    fig.savefig(out, bbox_inches='tight', dpi=300)
    plt.close(fig)
    logger.info(f"Saved {out}")
    print(f"Saved {out}")

    # CI width at the hourly scale, absolute and as % of the point estimate.
    unit = ' (x100)' if target == 'ET' else ''
    print(f"\n[{target} | {aggname} RMSE{unit}] "
          f"hourly-scale {ci:.0f}% CI width per scenario:")
    print(f"  {'scenario':<12} {'model':<9} {'point':>8} "
          f"{'CI width':>9} {'% of point':>11}")
    for setting in SETTINGS:
        for (s, model, point, lo, hi) in hourly_rows:
            if s != setting:
                continue
            width = hi - lo
            pct = 100.0 * width / abs(point) if point else float('nan')
            print(f"  {SETTING_NAMES.get(setting, setting):<12} {model:<9} "
                  f"{point:>8.3f} {width:>9.3f} {pct:>10.1f}%")

    # Full table (all scales) to CSV, alongside the figure.
    csv_out = f'{outdir}/scale_rmse_{aggname}_{target}_ci.csv'
    pd.DataFrame(all_rows).to_csv(csv_out, index=False)
    logger.info(f"Saved {csv_out}")
    print(f"Saved {csv_out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target', choices=['ET', 'GPP', 'NEE', 'all'],
                        default='ET')
    parser.add_argument('--val_strategy',
                        choices=['mean', 'max', 'discrepancy'], default='mean')
    parser.add_argument('--n_boot', type=int, default=1000,
                        help='number of site-bootstrap resamples')
    parser.add_argument('--ci', type=float, default=95.0,
                        help='confidence interval width in percent')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--outdir', type=str, default=PLOTS_DIR)
    args = parser.parse_args()

    targets = ['ET', 'GPP', 'NEE'] if args.target == 'all' else [args.target]

    results = load_all_metrics(
        settings=SETTINGS,
        targets=targets,
        models=MODELS,
        scales=RAW_SCALES,
        val_strategy=args.val_strategy,
        rerun=False,
    )
    if results.empty:
        logger.error("No metrics loaded; nothing to plot.")
        return
    results['scale'] = results['scale'].replace({'spatial': 'site-mean'})

    if os.path.exists(STYLE_FILE_PATH):
        plt.style.use(STYLE_FILE_PATH)

    for target in targets:
        for aggname in ('median', 'q90'):
            plot_target(results, target, aggname,
                        args.n_boot, args.ci, args.seed, args.outdir)


if __name__ == "__main__":
    main()
