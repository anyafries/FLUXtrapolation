"""Plotting functions for FLUXNET benchmark results."""


import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os
import pandas as pd
import seaborn as sns

from utils.utils import setup_logging

logger = setup_logging(__name__)

PLOTS_DIR = 'results/plots'
SCALES = ['hourly', 'daily', 'weekly', 'monthly', 'seasonal', 'anom', 'iav']

# Model ordering: lr first, xgb second, then alphabetically
MODEL_ORDER = ['xgb', 'lightgbm', 'mlp',
               'gdro', 'coral', 'mmd',
            #    'maxrm_mse', 'maxrm_regret',
               'lr', 'robust-lr', 'ridge',  'constant']
color_palette = sns.color_palette("tab10", n_colors=len(MODEL_ORDER))
MODEL_COLORS = {model: color_palette[i] for i, model in enumerate(MODEL_ORDER)}

# Extra colors for methods not in MODEL_ORDER, assigned in order of appearance
_EXTRA_PALETTE = sns.color_palette("Set2") + sns.color_palette("tab20")


def get_model_colors(models):
    """Return a color mapping covering all models.

    Known models (in MODEL_ORDER) keep their fixed color; any new methods are
    assigned distinct colors from a fallback palette, appended in order.
    """
    colors = dict(MODEL_COLORS)
    extras = [m for m in get_ordered_models(models) if m not in MODEL_COLORS]
    for i, model in enumerate(extras):
        colors[model] = _EXTRA_PALETTE[i % len(_EXTRA_PALETTE)]
    return colors

# Setting ordering: time-split, spatial-easy, spatial-hard
SETTINGS_ORDER = ['time-split', 'spatial-easy40', 'TA40',
                  'spatial-easy', 'spatial-hard', 
                  'LST', 'TA', 'VPD', 
                  'PFT_CRO', 'PFT_ENF', 'PFT_GRA', 'PFT_WET', 
                  'forest', 'grass-savanna', 'schrub-savanna',
                  'europe', 'rest-of-world',
                  ] + [f'hard-{i}' for i in range(1, 6)] + ['time-space']

# Metrics where higher is better (affects sorting direction and labels)
HIGHER_IS_BETTER = {'nse', 'r2_score', 'pearson_corr'}


def get_ordered_models(models):
    """Order models: lr, xgb, then alphabetically."""
    models = list(models)
    ordered = [m for m in MODEL_ORDER if m in models]
    remaining = sorted([m for m in models if m not in MODEL_ORDER])
    return ordered + remaining


def get_ordered_settings(settings):
    """Order settings: time-split, spatial-easy, spatial-hard, then alphabetically."""
    settings = list(settings)
    ordered = [s for s in SETTINGS_ORDER if s in settings]
    remaining = sorted([s for s in settings if s not in SETTINGS_ORDER])
    return ordered + remaining


def is_higher_better(metric):
    """Check if higher values are better for this metric."""
    return metric.lower() in HIGHER_IS_BETTER


def plot_metric_by_setting(results, target, metric, scale, ax, agg='median', 
                           legend=False, ymax=None):
    """
    Plot metric across settings for one scale (single subplot).

    Args:
        results: DataFrame with columns target, setting, model, scale, env, metric
        target: Target variable to filter (e.g., 'GPP')
        metric: Metric column name (e.g., 'rmse')
        scale: Temporal scale to filter (e.g., 'daily')
        agg: Aggregation function to apply (default: 'median')
        ax: Matplotlib axes to plot on
    """
    subset = results[(results['target'] == target) & (results['scale'] == scale)]
    if subset.empty:
        ax.set_title(f"{scale} (no data)")
        return

    data = (
        subset
        .groupby(['setting', 'model'])[metric]
        .agg(agg)
        .reset_index()
    )

    # hue_order = get_ordered_models(data['model'].unique())
    categories = SETTINGS_ORDER + [s for s in np.sort(data['setting'].unique()) if s not in SETTINGS_ORDER]
    data['setting'] = pd.Categorical(data['setting'], categories=categories, ordered=True)
    data = data.sort_values('setting')
    palette = get_model_colors(data['model'].unique())
    for i, plot_func in enumerate([sns.lineplot, sns.scatterplot]):
        plot_func(data=data, x='setting', y=metric, ax=ax, hue='model',
                  palette=palette, legend=legend&(i==1))

    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories, rotation=90, ha='center')
    ax.set_title(scale)
    ax.set_xlabel('')
    ax.set_ylim(bottom=0)
    if ymax is not None:
        ax.set_ylim(top=ymax)


def plot_metric_grid(results, target, metric='rmse', agg='median', outdir=PLOTS_DIR):
    """
    Create 3x2 grid showing metric across settings for all scales.

    Args:
        results: DataFrame with results
        target: Target variable (e.g., 'GPP')
        metric: Metric to plot (default: 'rmse')
        agg: Aggregation function to apply (default: 'median')
        outdir: Output directory for saved plot
    """
    os.makedirs(outdir, exist_ok=True)

    fig, axes = plt.subplots(4, 2, figsize=(8, 8), sharex=True)
    axes = axes.flatten()

    for i, scale in enumerate(SCALES):
        plot_metric_by_setting(results, target, metric, scale, axes[i],
                               agg=agg, legend=(i == len(SCALES) - 1), 
                               ymax=1 if metric.lower() == 'nse' else None)
    axes[len(SCALES) - 1].legend(title='')

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    fig.suptitle(f"{target}")
    plt.tight_layout()

    if callable(agg):
        agg = 'quantile'
    outfile = os.path.join(outdir, f"{metric}_by_scale_{agg}_{target}.png")
    plt.savefig(outfile, dpi=150)
    plt.close(fig)
    logger.info(f"Saved: {outfile}")


def plot_cdf(results, target, metric, scale, setting, ax, xmax=None,
             setting_name=None, linestyle='-', linewidth=2):
    """
    Plot CDF of metric for one target/scale/setting (single subplot).

    Args:
        results: DataFrame with results
        target: Target variable to filter
        metric: Metric column name
        scale: Temporal scale to filter
        setting: Setting to filter
        ax: Matplotlib axes to plot on
    """
    subset = results[
        (results['target'] == target) &
        (results['setting'] == setting) &
        (results['scale'] == scale)
    ]

    if subset.empty:
        ax.set_title(f"{setting} (no data)")
        return

    higher_better = is_higher_better(metric)
    models = get_ordered_models(subset['model'].unique())
    model_colors = get_model_colors(models)

    for model_name in models:
        model_data = subset[subset['model'] == model_name]
        values = model_data[metric].dropna().values
        if len(values) == 0:
            continue

        if higher_better:
            # Sort descending for higher-is-better metrics
            sorted_values = np.sort(values)[::-1]
        else:
            # Sort ascending for lower-is-better metrics
            sorted_values = np.sort(values)

        ax.plot(sorted_values, np.linspace(0, 1, len(sorted_values)),
                label=model_name, color=model_colors[model_name],
                linestyle=linestyle, linewidth=linewidth)

    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.3)
    ax.set_xlabel(f'{metric.upper()} (x)')

    env = "sites-years" if setting == 'time-split' else "sites"
    if higher_better:
        ax.set_ylabel(f'% of {env} with {metric.upper()} >= x')
    else:
        ax.set_ylabel(f'% of {env} with {metric.upper()} <= x')

    if metric.lower() == 'nse':
        ax.set_xlim(-0.5, 1.0)
    # else:
    #     ax.set_xlim(0, xmax)
    setting_name = setting_name if setting_name else setting
    ax.set_title(setting_name)
    ax.legend()


def plot_quantile(results, target, metric, scale, setting, ax, y_limit=None):
    """
    Plot Quantile function (Sorted Performance Curve) of metric.
    
    Args:
        results: DataFrame with results
        target: Target variable to filter
        metric: Metric column name
        scale: Temporal scale to filter
        setting: Setting to filter
        ax: Matplotlib axes to plot on
        y_limit: Optional float to cap the Y-axis (useful for exploding errors)
    """
    subset = results[
        (results['target'] == target) &
        (results['setting'] == setting) &
        (results['scale'] == scale)
    ]

    if subset.empty:
        ax.set_title(f"{setting} (no data)")
        return

    higher_better = is_higher_better(metric)
    models = get_ordered_models(subset['model'].unique())
    model_colors = get_model_colors(models)

    for model_name in models:
        model_data = subset[subset['model'] == model_name]
        values = model_data[metric].dropna().values

        if len(values) == 0:
            continue

        if higher_better:
            # Sort descending: best (high) to worst (low)
            sorted_values = np.sort(values)[::-1]
        else:
            # Sort ascending: best (low) to worst (high)
            sorted_values = np.sort(values)

        percentiles = np.linspace(0, 1, len(sorted_values))
        ax.plot(percentiles, sorted_values, label=model_name,
                color=model_colors[model_name])

    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    env = "sites" if setting.startswith('spatial') else "site-years"
    if higher_better:
        ax.set_xlabel(f'% of {env} with {metric.upper()} >= y')
    else:
        ax.set_xlabel(f'% of {env} with {metric.upper()} <= y')

    if metric.lower() == 'nse':
        ax.set_ylim(-0.5, 1.0)
        ax.axhline(0, color='k', linestyle=':', linewidth=1)

    ax.set_ylabel(f'{metric.upper()} (y)')
    ax.set_title(setting)

    if y_limit:
        ax.set_ylim(top=y_limit)

    ax.legend()


def plot_cdf_grid(results, target, metric='rmse', scale='daily', 
                  settings_names=None, outdir=PLOTS_DIR):
    """
    Create subplots showing CDF for each available setting.

    Args:
        results: DataFrame with results
        target: Target variable (e.g., 'GPP')
        metric: Metric to plot (default: 'rmse')
        scale: Temporal scale (default: 'daily')
        outdir: Output directory for saved plot
    """
    os.makedirs(outdir, exist_ok=True)

    # Get available settings for this target/scale
    subset = results[(results['target'] == target) & (results['scale'] == scale)]
    settings = get_ordered_settings(subset['setting'].unique())

    if len(settings) == 0:
        logger.warning(f"No data for {target} at {scale} scale")
        return

    n_settings = len(settings)
    fig, axes = plt.subplots(1, n_settings, figsize=(4 * n_settings, 4), 
                             sharey=True, sharex=True)

    if n_settings == 1:
        axes = [axes]

    xmax = results[
        (results['target'] == target) &
        (results['scale'] == scale)
    ][metric].max()

    for i, setting in enumerate(settings):
        setting_name = settings_names.get(setting, setting) if settings_names else setting
        axes[i].axhline(0.5, color='gray', linestyle=':', linewidth=1)
        axes[i].axhline(0.9, color='gray', linestyle=':', linewidth=1)
        plot_cdf(results, target, metric, scale, setting, axes[i], 
                 setting_name=setting_name, xmax=xmax)

    fig.suptitle(f"{target} ({scale})")
    plt.tight_layout()

    outfile = os.path.join(outdir, f"cdf_{target}_{metric}_{scale}.png")
    plt.savefig(outfile, dpi=150)
    plt.close(fig)
    logger.info(f"Saved: {outfile}")


# ---------------- Preparing data for leaderboard -----------------

def get_pivot_df_with_scores(df, target, metric, aggfunc='median', 
                             lower_is_better=True,
                             scale_order=['hourly', 'daily', 'weekly', 'monthly', 'seasonal', 'anom', 'iav'],
                             model_order=None,
                             settings_order=None,
                             baseline_model='lr'): # <-- Added baseline_model parameter
    assert lower_is_better, "This function currently assumes that lower metric values are better"
    subset = df[(df['target'] == target) & (df['scale'] != 'spatial')]
    
    pivot_df = subset.pivot_table(
        index='model', 
        columns=['setting', 'scale'], 
        values=metric, 
        aggfunc=aggfunc
    )

    if settings_order is None:
        settings_order = get_ordered_settings(pivot_df.columns.get_level_values(0).unique())
    if scale_order is None:
        scale_order = pivot_df.columns.get_level_values(1).unique()
    
    ordered_cols = []
    for s in settings_order:
        for sc in scale_order:
            if (s, sc) in pivot_df.columns:
                ordered_cols.append((s, sc))
        for col in pivot_df.columns:
            if col[0] == s and col not in ordered_cols:
                ordered_cols.append(col)
    pivot_df = pivot_df[ordered_cols]

    skill_scores_df, overall_scores = get_weighted_skill_scores(pivot_df, baseline_model=baseline_model)

    if model_order is not None:
        pivot_df = pivot_df.reindex(model_order)
        if skill_scores_df is not None:
            skill_scores_df = skill_scores_df.reindex(model_order)
        if overall_scores is not None:
            overall_scores = overall_scores.reindex(model_order)
    elif overall_scores is not None:
        sort_index = overall_scores.sort_values(ascending=False).index
        pivot_df = pivot_df.reindex(sort_index)
        skill_scores_df = skill_scores_df.reindex(sort_index)
        overall_scores = overall_scores.reindex(sort_index)

    return pivot_df, overall_scores, skill_scores_df


# ---------------- Leaderboard helpers -----------------

def format_sig_figs(val, n=2):
    """Formats a number to n significant figures."""
    if pd.isna(val) or val == 0:
        return str(val)
    # This handles scientific notation and standard float formatting automatically
    return f"{val:.{n}g}"


def get_hex_relative_color(val, best_val, rel_threshold=0.2, lower_is_better=True):
    """
    Colors values based on their distance from the best value in the column.
    Green = Best
    White = Best + (Best * rel_threshold) [for lower_is_better]
    """
    if pd.isna(val) or pd.isna(best_val):
        return "FFFFFF"

    # Define the 'Limit' where the color fades to white
    if lower_is_better:
        # e.g., Best is 0.1, threshold is 1.0 (double). Limit is 0.2.
        limit = best_val * (1 + rel_threshold)
        if val <= best_val: ratio = 1.0
        elif val >= limit: ratio = 0.0
        else:
            # Linear interpolation between best and limit
            ratio = (limit - val) / (limit - best_val)
    else:
        # e.g., Best is 0.8, threshold is 0.5 (half). Limit is 0.4.
        limit = best_val * (1 - rel_threshold)
        if val >= best_val: ratio = 1.0
        elif val <= limit: ratio = 0.0
        else:
            ratio = (val - limit) / (best_val - limit)

    # Gradient: Green (99, 190, 123) to White (255, 255, 255)
    r = int(255 - (ratio * (255 - 99)))
    g = int(255 - (ratio * (255 - 190)))
    b = int(255 - (ratio * (255 - 123)))
    
    return f"{r:02X}{g:02X}{b:02X}"


def get_weighted_skill_scores(df, baseline_model='constant'):
    """
    Computes Continuous Skill Scores relative to a baseline.
    Returns:
      - skill_scores_df: The cell-by-cell skill scores.
      - overall_scores: A Pandas Series of the final weighted average per model.
    """
    SCALE_WEIGHTS = {
        'hourly': 1.0, 'daily': 1.0, 'weekly': 1.0, 'monthly': 1.0,
        'seasonal': 1.0, 'anom': 1.0, 'iav': 1.0, 'site-mean': 1.0,
    }

    if baseline_model not in df.index:
        print(f"Warning: Baseline '{baseline_model}' not found.")
        return None, None
        
    baseline_errors = df.loc[baseline_model]
    
    # 1 - (Model_Error / Baseline_Error)
    skill_scores_df = 1 - (df / baseline_errors)
    
    aligned_weights = pd.Series(index=df.columns, dtype=float)
    for col in df.columns:
        scale_name = col[-1] if isinstance(col, tuple) else col
        aligned_weights[col] = SCALE_WEIGHTS.get(scale_name, 1.0) 
        
    def compute_weighted_mean(row):
        mask = row.notna() & aligned_weights.notna()
        if not mask.any():
            return np.nan
        return np.average(row[mask], weights=aligned_weights[mask])

    overall_scores = skill_scores_df.apply(compute_weighted_mean, axis=1)
    
    return skill_scores_df, overall_scores


# ---------------- HTML leaderboard with colored cells -----------------

# TODO: add option for higher-is-better metrics
# TODO: make more modular/general -> only for one scale for example
# https://pandas.pydata.org/docs/user_guide/style.html
import pandas as pd
def create_html_leaderboard(
    df,
    target,
    metric,
    filename,
    aggfunc='median',
    lower_is_better=True,
    scale_order=['hourly', 'daily', 'weekly', 'monthly', 'seasonal', 'anom', 'iav'],
    model_order=None,
    settings_order=None,
    settings_names=None,
    rel_threshold=0.2,
    display_mode='value',
    baseline_model='lr'
):
    # --- 1. Data Preparation (Same as before) ---
    pivot_df, overall_scores, skill_scores_df = get_pivot_df_with_scores(
        df, target, metric, aggfunc, lower_is_better,
        scale_order, model_order, settings_order, baseline_model
    )

    if settings_names is not None:
        renamed_cols = [(settings_names.get(s, s), sc) for s, sc in pivot_df.columns]
        pivot_df.columns = pd.MultiIndex.from_tuples(renamed_cols)
        if skill_scores_df is not None:
            skill_scores_df.columns = pivot_df.columns

    if overall_scores is not None:
        pivot_df.insert(0, ('Summary', 'Skill score \u2191'), overall_scores)

    display_df = pd.DataFrame(index=pivot_df.index, columns=pivot_df.columns)
    
    for col in pivot_df.columns:
        for row in pivot_df.index:
            val = pivot_df.loc[row, col]
            if pd.isna(val):
                display_df.loc[row, col] = "-"
            elif col[0] == 'Summary':
                display_df.loc[row, col] = f"<b>{val:.2f}</b>"
            elif display_mode == 'skill_score' and skill_scores_df is not None and col in skill_scores_df.columns:
                ss = skill_scores_df.loc[row, col]
                display_df.loc[row, col] = f"{ss:.2f}" if pd.notna(ss) else "-"
            else:
                display_df.loc[row, col] = format_sig_figs(val, n=2)

    # --- 2. Styling with Rotated Headers & Sans-Serif ---
    table_styles = [
        # Set Global Font to Sans-Serif
        {'selector': 'table', 'props': [
            ('border-collapse', 'collapse'), 
            ('font-family', 'Arial, Helvetica, sans-serif'), 
            ('font-size', '12px')
        ]},
        {'selector': 'th, td', 'props': [('border', '1px solid #d3d3d3'), ('padding', '8px')]},
        
        # Style for the Scale Headers (Level 1) to rotate them
        {'selector': 'th.col_heading.level1', 'props': [
            ('height', '80px'),
            ('vertical-align', 'bottom'),
            ('padding', '5px'),
            ('min-width', '25px')
        ]},
        # The actual rotation logic for the text inside the cell
        {'selector': 'th.col_heading.level1 span', 'props': [
            ('writing-mode', 'vertical-rl'),
            ('transform', 'rotate(180deg)'),
            ('text-align', 'left'),
            ('display', 'inline-block')
        ]},
        
        {'selector': 'th.col_heading.level0', 'props': [('background-color', '#ececec'), ('font-weight', 'bold')]},
        {'selector': 'th.row_heading', 'props': [('background-color', '#ffffff'), ('text-align', 'left'), ('font-weight', 'bold')]}
    ]

    def style_from_original(df_dummy):
        styles = pd.DataFrame('', index=pivot_df.index, columns=pivot_df.columns)
        for col in pivot_df.columns:
            if col[0] == 'Summary':
                styles[col] = 'background-color: #f8f9fa;'
                continue
            col_data = pivot_df[col]
            best_val = col_data.min() if lower_is_better else col_data.max()
            for row in pivot_df.index:
                val = pivot_df.loc[row, col]
                if pd.notna(val):
                    hex_color = get_hex_relative_color(val, best_val, rel_threshold=rel_threshold, lower_is_better=lower_is_better)
                    styles.loc[row, col] = f'background-color: #{hex_color}; color: #1a1a1a;'
        return styles

    styler = (
        display_df.style
        .set_table_styles(table_styles)
        .apply(style_from_original, axis=None)
    )

    # --- 3. Save ---
    html_output = f'<meta charset="UTF-8">\n{styler.to_html()}'
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(html_output)
    
    return html_output

