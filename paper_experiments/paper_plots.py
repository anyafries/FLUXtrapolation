"""
Script to plot results from multiple experiments for the paper.
"""

import matplotlib.pyplot as plt
import os
import pandas as pd
import sys

from paper_plot_utils import (
    create_latex_leaderboard,
    get_relative_errors_by_flux,
    plot_saturation_by_flux,
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval import load_all_metrics
from utils.plots import plot_cdf
from utils.utils import setup_logging

logger = setup_logging(__name__)


SETTINGS = ['time-split', 'spatial-easy40', 'TA40']
SETTING_NAMES = {
    'time-split': 'temporal',
    'spatial-easy40': 'spatial',
    'TA40': 'temperature',
}
TARGETS = ['ET', 'GPP', 'NEE'] 
MODELS = ['lr', 'xgb', 'mlp', 'mmd', 'coral', 'gdro', 'constant', 'lstm', 'tabpfn-v2.6', 'tabpfn-v3-ood']
# Unused: model ordering and colors come from utils.plots.
# MODEL_ORDER = ['xgb', 'mlp', 'gdro', 'coral', 'mmd', 'lstm', 'tabpfn-v2.6', 'tabpfn-v3-ood', 'lr', 'constant']
# color_palette = sns.color_palette("tab10", n_colors=len(MODEL_ORDER))
# MODEL_COLORS = {model: color_palette[i] for i, model in enumerate(MODEL_ORDER)}

RAW_SCALES = ['hourly', 'weekly', 'seasonal', 'anom', 'iav', 'spatial']
SCALES = ['hourly', 'weekly', 'seasonal', 'anom', 'iav', 'site-mean'] 
SCALE_COLORS = {
    "hourly":   "#4a2377",
    "weekly":   "#8cc5e3",
    "seasonal": "#f55f99",
    "anom":     "#0d7d87",
    "iav":      "#f47a00",
    "site-mean":  "#1b9e77",
}

PLOTS_DIR = f'paper_experiments/plots'
STYLE_FILE_PATH = 'utils/neurips.mplstyle'

if __name__ == "__main__":
    results = load_all_metrics(
        settings=SETTINGS,
        targets=TARGETS,
        models=MODELS,
        scales=RAW_SCALES,
        val_strategy='mean',
        rerun=False,
    )
    print(results.head())
    results['scale'] = results['scale'].replace({'spatial': 'site-mean'})

    plt.style.use(STYLE_FILE_PATH)

    # flux_saturation_results = []
    for target in ['GPP', 'ET', 'NEE']:
        if target == 'ET':
            # multiply RMSE by 100 
            results_tex = results.copy()
            results_tex['rmse'] = results_tex['rmse'] * 100
        else: 
            results_tex = results.copy()

        # Create LaTeX leaderboard for each target 
        for aggname, aggfunc in [
            ('median', 'median'), ('90q', lambda x: x.quantile(0.9))
        ]:
            tables = ['supp']
            if target == 'ET' and aggname == '90q':  
                tables.append('main')
            for table in tables:
                if table == 'main':
                    filename = f'{PLOTS_DIR}/table_{aggname}_{target}.tex'
                else: 
                    filename = f'{PLOTS_DIR}/table_{aggname}_{target}_supp.tex'
                create_latex_leaderboard(
                    results_tex, target, metric='rmse', 
                    lower_is_better=True,
                    scale_order=SCALES,
                    settings_order=SETTINGS,
                    settings_names=SETTING_NAMES,
                    aggfunc=aggfunc,
                    filename=filename,
                    display_mode='value',
                    main_table=table=='main',
                )

    # CDF for hourly, weekly, seasonal for TA40 ET
    models_for_cdf = ['lr', 'xgb', 'mmd', 'gdro', 'lstm', 'tabpfn-v3-ood', 'constant']
    for scale in ['hourly', 'weekly', 'seasonal', 'site-mean']:
        fig, axes = plt.subplots(1, 3, figsize=(4.9, 1.7), sharey=True)
        for i, target in enumerate(TARGETS):
            results_t = results[(results['target'] == target) & results['model'].isin(models_for_cdf)].copy()
            if target == 'ET':
                results_t['rmse'] = results_t['rmse'] * 100
            ax = axes[i] 
            ax.axhline(0.5, color='gray', linestyle='--', linewidth=0.5)
            ax.axhline(0.9, color='gray', linestyle='--', linewidth=0.5)
            plot_cdf(results_t, scale=scale, target=target, 
                    setting='TA40',  metric='rmse', ax=ax,
                    linestyle='-', linewidth=0.8)
            if scale == "weekly" and target == "ET":
                ax.set_xlim(1, 10)
            ax.set_title(target, pad=1)
            ax.set_xlabel('RMSE', labelpad=0.5)
            ax.set_ylabel('Cumulative Probability' if i == 0 else '', labelpad=0.5)
            # make yticks smaller fontsize
            ax.tick_params(axis='y', labelsize=7)

        if axes[0].get_legend() is not None:
            axes[0].get_legend().remove()
        if axes[1].get_legend() is not None:
            axes[1].get_legend().remove()
        leg = axes[2].legend(
            title="", 
            frameon=True, 
            handlelength=0.6,
            handleheight=0.4,
            handletextpad=0.3,
            labelspacing=0.2,
            borderpad=0.2,
            loc='center left',
            bbox_to_anchor=(1.15, 0.5)
        )
        leg.get_frame().set_linewidth(0.5)
        leg.get_frame().set_edgecolor('lightgray')
        plt.tight_layout()
        plt.subplots_adjust(right=0.8)
        plt.savefig(f'{PLOTS_DIR}/cdf_{scale}.png', bbox_inches='tight', dpi=300, pad_inches=0.02)
        print(f"CDF plot {scale} scale saved to {PLOTS_DIR}/cdf_{scale}.png")