"""
Plots for the TabPFN ablations (see tabpfn_ablation_run.py).

One figure per ablation: skill score relative to a baseline model, as a
function of context size or ensemble size. The leftmost panel averages over
every (setting, scale) cell, the remaining panels average over the six scales
within one scenario. Solid lines are the median of the domain-level RMSE,
dashed lines the 90th percentile. Blue horizontal lines mark XGBoost, fitted on
the full training split.

Usage:
    python paper_experiments/tabpfn_ablation_plot.py
    python paper_experiments/tabpfn_ablation_plot.py --target GPP --baseline constant
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval import load_all_metrics
from utils.utils import setup_logging

from tabpfn_ablation_run import (
    BASE_CONTEXT,
    N_ESTIMATORS,
    SETTINGS,
    VAL_STRATEGY,
    VERSIONS,
    context_sizes,
    format_size,
    run_name,
)

logger = setup_logging(__name__)

SETTING_NAMES = {
    'time-split': 'temporal',
    'spatial-easy40': 'spatial',
    'TA40': 'temperature',
}
RAW_SCALES = ['hourly', 'weekly', 'seasonal', 'anom', 'iav', 'spatial']
SCALES = ['hourly', 'weekly', 'seasonal', 'anom', 'iav', 'site-mean']

# (label, aggfunc, linestyle, marker) -- the two summaries of the domain-level
# RMSE distribution shown in every panel. Linestyle and marker both encode the
# aggregation, so that the two are still told apart where the lines are short or
# the dashes hard to see.
AGGS = [
    ('median', 'median', '-', 'o'),
    ('90th percentile', lambda x: x.quantile(0.9), '--', 's'),
]
REFERENCE_MODEL = 'xgb'
REFERENCE_COLOR = '#1f77b4'

# Amber -> red -> plum for the TabPFN versions, darkening with the version so
# that the ordering is readable, against a blue reference model. Sampled from
# `inferno` rather than from one hue: colour is the only channel left for the
# version, and three shades of orange are not far enough apart to separate the
# curves where they overlap.
VERSION_COLORS = dict(zip(VERSIONS,
                          plt.get_cmap('inferno')(np.linspace(0.8, 0.3, len(VERSIONS)))))
MARKER_SIZE = 3
VERSION_LABELS = {'v2.5': 'TabPFN2.5', 'v2.6': 'TabPFN2.6', 'v3': 'TabPFN3',
                  'v3-ood': 'TabPFN3-ood'}

# The NeurIPS text block is 5.5 in wide. Keeping the cropped figure inside it
# is what lets the figure be included at its natural size, which in turn is what
# keeps the font sizes of the style file. LEGEND_Y and LEGEND_GAP are the top of
# the legend block and the space between the two legends, in figure coordinates.
FIG_WIDTH = 5.5
FIG_HEIGHT = 1.9
PAD_INCHES = 0.01
LEGEND_Y = -0.04
LEGEND_GAP = 0.04
LEGEND_KW = dict(frameon=False, handlelength=1.3, handletextpad=0.5,
                 columnspacing=1.0, borderaxespad=0.0)

PLOTS_DIR = os.path.join(os.path.dirname(__file__), 'plots', 'tabpfn_ablation')
STYLE_FILE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                               'utils', 'neurips.mplstyle')


def get_skill_scores(results, target, aggfunc, baseline):
    """
    Cell-by-cell skill score 1 - RMSE / RMSE_baseline for one target.

    Cells are (setting, scale) pairs; the RMSE of a cell is `aggfunc` applied
    over the domains of that setting. Returns a DataFrame indexed by model.
    """
    subset = results[results['target'] == target]
    pivot = subset.pivot_table(index='model', columns=['setting', 'scale'],
                               values='rmse', aggfunc=aggfunc)
    cols = [(s, sc) for s in SETTINGS for sc in SCALES if (s, sc) in pivot.columns]
    pivot = pivot[cols]
    if baseline not in pivot.index:
        raise ValueError(f"Baseline model `{baseline}` not found in the results. "
                         f"Available: {sorted(pivot.index)}")
    return 1 - pivot / pivot.loc[baseline]


def panel_score(skill_df, model, setting=None):
    """Mean skill score over all cells, or over the cells of one setting."""
    if model not in skill_df.index:
        return np.nan
    cols = [c for c in skill_df.columns if setting is None or c[0] == setting]
    row = skill_df.loc[model, cols]
    return row.mean() if row.notna().any() else np.nan


def get_curves(ablation, versions):
    """
    (version, x values, model names) for each line of the given ablation.

    For the estimators ablation the context differs per version, so it is
    carried into the legend label.
    """
    curves = []
    for version in versions:
        if ablation == 'context':
            xs = context_sizes(version)
            names = [run_name(version, x, 1) for x in xs]
            label = VERSION_LABELS[version]
        else:
            context = BASE_CONTEXT[version]
            xs = list(N_ESTIMATORS)
            names = [run_name(version, context, x) for x in xs]
            label = f"{VERSION_LABELS[version]} ({format_size(context)})"
        curves.append((version, label, xs, names))
    return curves


def format_context_tick(value, _pos=None):
    """'10^4', '3 x 10^4', ... for a log x-axis of context sizes."""
    exponent = int(np.floor(np.log10(value)))
    mantissa = value / 10 ** exponent
    if np.isclose(mantissa, 1):
        return rf"$10^{{{exponent}}}$"
    return rf"${mantissa:.0f}\cdot 10^{{{exponent}}}$"


def place_legends(fig, groups, y=LEGEND_Y, gap=LEGEND_GAP):
    """
    Draw one titled legend per group, side by side and centred below the figure.

    `groups` are (title, handles) pairs, each laid out as a single row of
    entries under its title. Groups fall back to two rows when one row would
    push the pair wider than the figure: the cropped figure has to stay within
    the 5.5 in text block for the font sizes of the style file to survive.
    """
    budget = (FIG_WIDTH - 2 * PAD_INCHES) / FIG_WIDTH
    for nrows in (1, 2):
        legends = [fig.legend(handles=handles, title=title, loc='upper left',
                              ncol=int(np.ceil(len(handles) / nrows)),
                              bbox_to_anchor=(0.0, y), **LEGEND_KW)
                   for title, handles in groups]
        fig.draw_without_rendering()
        widths = [lg.get_window_extent().width / fig.bbox.width for lg in legends]
        total = sum(widths) + gap * (len(widths) - 1)
        if total <= budget or nrows == 2:
            break
        for lg in legends:  # too wide as one row -- lay the entries out as two
            lg.remove()

    x = (1.0 - total) / 2
    for legend, width in zip(legends, widths):
        legend.set_bbox_to_anchor((x, y), transform=fig.transFigure)
        x += width + gap


def plot_ablation(results, ablation, target, baseline, versions):
    """Draw the four-panel skill-score figure for one ablation and save it."""
    os.makedirs(PLOTS_DIR, exist_ok=True)
    curves = get_curves(ablation, versions)

    skill_by_agg = {name: get_skill_scores(results, target, aggfunc, baseline)
                    for name, aggfunc, _, _ in AGGS}
    panels = [(None, 'overall')] + \
             [(s, SETTING_NAMES[s]) for s in SETTINGS]

    fig, axes = plt.subplots(1, len(panels), figsize=(FIG_WIDTH, FIG_HEIGHT),
                             sharex=True, sharey=True,
                             gridspec_kw={'wspace': 0.12})
    fig.subplots_adjust(left=0.085, right=0.99, top=0.86, bottom=0.26)
    plotted = []
    for ax, (setting, title) in zip(axes, panels):
        for agg_name, _, linestyle, marker in AGGS:
            skill_df = skill_by_agg[agg_name]

            reference = panel_score(skill_df, REFERENCE_MODEL, setting)
            if np.isfinite(reference):
                ax.axhline(reference, color=REFERENCE_COLOR, linestyle=linestyle,
                           linewidth=1.0, zorder=1)

            for version, label, xs, names in curves:
                ys = [panel_score(skill_df, n, setting) for n in names]
                ax.plot(xs, ys, linestyle=linestyle, linewidth=1,
                        marker=marker, markersize=MARKER_SIZE,
                        color=VERSION_COLORS[version], zorder=2)
                plotted += [{'ablation': ablation, 'target': target,
                             'panel': title, 'agg': agg_name, 'version': version,
                             'x': x, 'model': n, 'skill_score': y}
                            for x, n, y in zip(xs, names, ys)]

        ax.set_title(title)
        ax.axhline(0.0, color='black', linewidth=0.5, linestyle=':', zorder=0)
        if ablation == 'context':
            ax.set_xscale('log')
            ticks = sorted({x for _, _, xs, _ in curves for x in xs})
            ax.set_xticks(ticks)
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_context_tick))
            ax.xaxis.set_minor_locator(ticker.NullLocator())
        else:
            ax.set_xticks(N_ESTIMATORS)
            ax.xaxis.set_major_formatter(ticker.ScalarFormatter())

    xlabel = ('context size (observations)' if ablation == 'context'
              else 'ensemble members')
    # One shared x label: the panels are too narrow to carry one each.
    fig.supxlabel(xlabel, y=0.035, fontsize=plt.rcParams['axes.labelsize'])
    axes[0].set_ylabel('skill score')

    # Two legends below the figure, one per visual channel: colour encodes the
    # model, linestyle the summary of the domain-level RMSE distribution.
    model_handles = [Line2D([], [], color=VERSION_COLORS[v], linewidth=1,
                            label=label)
                     for v, label, _, _ in curves]
    model_handles += [Line2D([], [], color=REFERENCE_COLOR, linewidth=1,
                              label='XGBoost')]
    agg_handles = [Line2D([], [], color='black', linestyle=ls, linewidth=1,
                          marker=mk, markersize=MARKER_SIZE, label=agg_name)
                   for agg_name, _, ls, mk in AGGS]
    place_legends(fig, [('Model', model_handles),
                        ('RMSE aggregation', agg_handles)])

    stem = os.path.join(PLOTS_DIR, f'{ablation}_{target}_{baseline}-baseline')
    for extension in ('pdf', 'png'):
        fig.savefig(f'{stem}.{extension}', bbox_inches='tight',
                    pad_inches=PAD_INCHES, dpi=300)
    plt.close(fig)
    pd.DataFrame(plotted).to_csv(f'{stem}.csv', index=False)
    logger.info(f"Saved {stem}.pdf / .png / .csv")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation", type=str, default='both',
                        choices=['context', 'estimators', 'both'])
    parser.add_argument("--target", type=str, default='ET',
                        choices=['GPP', 'NEE', 'ET'])
    parser.add_argument("--baseline", type=str, default='lr',
                        help="Model the skill score is computed against")
    parser.add_argument("--versions", type=str, nargs='+', default=VERSIONS,
                        choices=VERSIONS)
    args = parser.parse_args()

    ablations = ['context', 'estimators'] if args.ablation == 'both' else [args.ablation]
    ablation_models = sorted({name for ablation in ablations
                              for _, _, _, names in get_curves(ablation, args.versions)
                              for name in names})

    results = load_all_metrics(
        settings=SETTINGS,
        targets=[args.target],
        models=ablation_models + [REFERENCE_MODEL, args.baseline],
        scales=RAW_SCALES,
        val_strategy=VAL_STRATEGY,
        rerun=False,
    )
    if results.empty:
        raise SystemExit("No metrics found -- run tabpfn_ablation_run.py first.")
    results['scale'] = results['scale'].replace({'spatial': 'site-mean'})

    missing = [m for m in ablation_models if m not in set(results['model'])]
    if missing:
        logger.warning(f"No results yet for {len(missing)} configuration(s): "
                       f"{', '.join(missing)}")

    plt.style.use(STYLE_FILE_PATH)
    for ablation in ablations:
        plot_ablation(results, ablation, args.target, args.baseline, args.versions)
