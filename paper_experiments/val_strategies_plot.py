import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval import load_all_metrics
from utils.plots import get_model_colors, get_ordered_models
from utils.utils import setup_logging

logger = setup_logging(__name__)

# --- aggregation: swap to lambda x: x.quantile(0.9) for 90th percentile ---
AGG_FUNC = 'median'
AGG_FUNC_NAME = 'median'

RAW_SCALES = ['hourly', 'weekly', 'seasonal', 'anom', 'iav', 'spatial']
SCALES = ['hourly', 'weekly', 'seasonal', 'anom', 'iav', 'site-mean']

PLOTS_DIR = os.path.join(os.path.dirname(__file__), 'plots', 'val_strategies')
STYLE_FILE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'utils', 'neurips.mplstyle')

VAL_SPLITS = {
    'default':  'spatial-easy40',
    'iid':      'spatial-easy40-iid',
    'temporal': 'spatial-easy40-temporal',
    'oracle':   'spatial-easy40-oracle',
}


def plot_val_comparison(data, x_col, x_order, title_suffix, filename_suffix):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    for target in data['target'].unique():
        df_t = data[data['target'] == target]
        agg = (df_t.groupby(['scale', 'model', x_col])['rmse']
                   .agg(AGG_FUNC)
                   .reset_index())

        fig, axes = plt.subplots(2, 3, figsize=(7, 4), sharey=False,
                                 gridspec_kw={'hspace': 0.4, 'wspace': 0.3})
        legend_ax = None
        agg[x_col] = pd.Categorical(agg[x_col], categories=x_order, ordered=True)
        for ax, scale in zip(axes.flatten(), SCALES):
            df_s = agg[agg['scale'] == scale].sort_values(x_col)
            present_models = get_ordered_models(df_s['model'].unique())
            sns.lineplot(
                data=df_s, x=x_col, y='rmse', hue='model',
                hue_order=present_models,
                palette=get_model_colors(present_models),
                markers=True, marker='o', ax=ax, linewidth=1, markersize=3,
            )
            ax.set_title(scale)
            ax.set_xlabel('')
            ax.set_ylabel(f'{AGG_FUNC_NAME} RMSE')
            ax.tick_params(axis='x', labelsize=7)
            if ax.get_legend():
                ax.get_legend().remove()
            if legend_ax is None and len(present_models) > 0:
                legend_ax = ax

        if legend_ax is not None:
            handles, labels = legend_ax.lines, get_ordered_models(agg['model'].unique())
            handles = [legend_ax.lines[i] for i in range(len(labels))]
            fig.legend(handles, labels, loc='lower center', ncol=5,
                       fontsize=7, bbox_to_anchor=(0.5, -0.05))

        fig.suptitle(f'{target} — {title_suffix}')
        out_path = os.path.join(PLOTS_DIR, f'{AGG_FUNC_NAME}_{filename_suffix}_{target}.pdf')
        fig.savefig(out_path, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Saved {out_path}")


if __name__ == '__main__':
    plt.style.use(STYLE_FILE_PATH)

    # Plot 1: fix val_split=default, vary val_strategy
    logger.info("Loading data for Plot 1 (val_strategy comparison)...")
    dfs = []
    for vs in ['mean', 'max', 'discrepancy']:
        df = load_all_metrics(settings=['spatial-easy40'], val_strategy=vs,
                              scales=RAW_SCALES)
        df['val_strategy'] = vs
        dfs.append(df)
    data1 = pd.concat(dfs, ignore_index=True)
    data1['scale'] = data1['scale'].replace({'spatial': 'site-mean'})

    plot_val_comparison(
        data1,
        x_col='val_strategy',
        x_order=['mean', 'max', 'discrepancy'],
        title_suffix='val strategy (default val split)',
        filename_suffix='val_strategy',
    )

    # Plot 2: fix val_strategy=mean, vary val_split
    logger.info("Loading data for Plot 2 (val_split comparison)...")
    dfs = []
    for split_name, setting in VAL_SPLITS.items():
        df = load_all_metrics(settings=[setting], val_strategy='mean',
                              scales=RAW_SCALES)
        if df.empty:
            logger.warning(f"No data for setting={setting}, skipping.")
            continue
        df['val_split'] = split_name
        dfs.append(df)
    data2 = pd.concat(dfs, ignore_index=True)
    data2['scale'] = data2['scale'].replace({'spatial': 'site-mean'})

    plot_val_comparison(
        data2,
        x_col='val_split',
        x_order=list(VAL_SPLITS.keys()),
        title_suffix='val split (mean strategy)',
        filename_suffix='val_split',
    )
