import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.plots import (
    get_pivot_df_with_scores,
    format_sig_figs,
    get_hex_relative_color,
)


# ---------------- Latex leaderboard -----------------
def create_latex_leaderboard(
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
    display_mode='rank',
    baseline_model='lr',
    main_table=True,
):
    # --- 1. Data Preparation (Transposed) ---
    pivot_df, overall_scores, skill_scores_df = get_pivot_df_with_scores(
        df, target, metric, aggfunc, lower_is_better,
        scale_order, model_order, settings_order, baseline_model
    )

    # --- 2. Build the Cell Strings (Colours + Text) ---
    latex_df = pd.DataFrame(index=pivot_df.index, columns=pivot_df.columns)

    for col in pivot_df.columns:
        col_data = pivot_df[col]
        best_val = col_data.min() if lower_is_better else col_data.max()

        for row_idx in pivot_df.index:
            val = col_data[row_idx]

            if pd.isna(val):
                latex_df.loc[row_idx, col] = "-"
                continue

            if display_mode == 'value':
                cell_text = format_sig_figs(val, n=2)
            elif display_mode == 'skill_score':
                if skill_scores_df is not None and pd.notna(skill_scores_df.loc[row_idx, col]):
                    cell_text = f"{skill_scores_df.loc[row_idx, col]:.2f}"
                else:
                    cell_text = "-"
            else:
                cell_text = ""

            hex_color = get_hex_relative_color(
                val,
                best_val,
                rel_threshold=rel_threshold,
                lower_is_better=lower_is_better
            )

            latex_df.loc[row_idx, col] = f"\\cellcolor[HTML]{{{hex_color}}} {cell_text}"

    # --- 3. Append Summary Column ---
    if overall_scores is not None:
        latex_df[('Summary', 'Skill score $\\uparrow$')] = overall_scores.apply(
            lambda x: f"\\textbf{{{x:.2f}}}" if pd.notna(x) else "-"
        )

    latex_df.index.name = None

    # --- 4. Rename Settings ---
    if settings_names is not None:
        renamed_columns = []

        for setting, scale in latex_df.columns:
            if setting == 'Summary':
                renamed_columns.append((setting, scale))
            else:
                display_setting = settings_names.get(setting, setting)
                renamed_columns.append((display_setting, scale))

        latex_df.columns = pd.MultiIndex.from_tuples(renamed_columns)

    # --- 5. Format Column Headers & Build Column Format ---
    new_cols = []
    col_format = "l"

    prev_setting = latex_df.columns[0][0]

    for setting, scale in latex_df.columns:
        if setting != prev_setting:
            col_format += "@{\\hspace{1.5em}}"
            prev_setting = setting

        col_format += "c"

        if setting == 'Summary':
            if main_table:
                new_cols.append(('', ''))
            else:
                new_cols.append(("\\multirow{2}{*}{\\rotatebox{90}{Skill score $\\uparrow$}}", ''))
        else:
            new_cols.append((setting, f"\\rotatebox{{90}}{{{scale}}}"))

    latex_df.columns = pd.MultiIndex.from_tuples(new_cols)

    # --- 6. Export to LaTeX ---
    latex_str = latex_df.to_latex(
        escape=False,
        na_rep="-",
        column_format=col_format,
        multicolumn_format="c"
    )

    # --- 7. Add Title Row ---
    has_summary = latex_df.columns[-1][0] == ''
    n_total_cols = len(latex_df.columns)
    n_data_cols = n_total_cols - int(has_summary)

    direction = "$\\downarrow$" if lower_is_better else "$\\uparrow$"
    if aggfunc == 'median':
        agg_name = "Median"
    else:
        agg_name = "90th percentile of"
    title = f"{agg_name} domain-level {target} {metric.upper()} {direction}"

    lines = latex_str.splitlines()
    toprule_idx = next(i for i, line in enumerate(lines) if "\\toprule" in line)

    if main_table:
        title_row = f"& \\multicolumn{{{n_data_cols}}}{{c}}{{{title}}}"
        if has_summary:
            title_row += " & \\multirow{3}{*}[-1em]{\\rotatebox{90}{Skill score $\\uparrow$}}"
        title_row += r" \\"
        lines.insert(toprule_idx + 1, title_row)

    latex_str = "\n".join(lines)

    # --- 8. Wrap in Small Font ---
    latex_str = "{\\small\n\\setlength{\\tabcolsep}{1.4pt}\n" + latex_str + "\n}\n"
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(latex_str)
    print(f"LaTeX leaderboard saved to {filename}")

    return latex_str




# ---------------- Flux saturation -----------------

def get_relative_errors_by_flux(df, target, metric, aggfunc='median',
                               lower_is_better=True,
                               scale_order=['hourly', 'daily', 'weekly', 'monthly', 'seasonal', 'anom', 'iav'],
                               model_order=None,
                               settings_order=None,
                               baseline_model='lr'):
    # 1. Get pivot table with scores to ensure consistent ordering and baseline selection                           
    pivot_df, overall_scores, _ = get_pivot_df_with_scores(
        df, target, metric, aggfunc, lower_is_better,
        scale_order, model_order, settings_order, baseline_model
    )

    # only keep the top 5 models by overall skill score for the error-by-flux plot
    if overall_scores is not None:
        top_models = overall_scores.head(4).index.tolist()
        pivot_df = pivot_df.loc[top_models]

    # divide each column by the value in the lowest-error model for that column to get relative errors
    relative_errors = pd.DataFrame(index=pivot_df.index, columns=pivot_df.columns)
    for col in pivot_df.columns:
        col_data = pivot_df[col]
        best_in_col = col_data.min() if lower_is_better else col_data.max()
        relative_errors[col] = col_data / best_in_col

    return relative_errors.median(axis=0).reset_index().rename(columns={0: 'relative_error'})




def plot_saturation_by_flux(fig, errs_df, filename, settings, scales, scale_colors, setting_names):
    
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.15], wspace=0.3, hspace=0.35)
    
    ax_time  = fig.add_subplot(gs[0, 0])
    ax_space = fig.add_subplot(gs[1, 0], sharex=ax_time, sharey=ax_time)
    ax_temp  = fig.add_subplot(gs[:, 1])

    axes = {
        "time-split": ax_time,
        "spatial-easy40": ax_space,
        "TA40": ax_temp,
    }    

    for setting in settings:
        subset = errs_df[errs_df["setting"] == setting]

        ax = axes[setting]
        sns.barplot(
            data=subset,
            x="target",
            y="relative_error",
            hue="scale",
            hue_order=scales,
            palette=scale_colors,
            ax=ax,
        )

        ax.set_title(setting_names[setting], pad=1, fontsize=9, weight='bold')
        ax.set_xlabel("")
        ax.set_ylabel("")

        if setting in ["time-split", "spatial-easy40"]:
            ax.set_ylim(1.0, 1.11)
        else:
            ax.set_ylim(1.0, 1.25)

        if setting == "time-split":
            ax.tick_params(labelbottom=False)

        if setting == "TA40":
            # locate it top right with coordinates
            ax.legend(
                title="", frameon=False, loc=(0.59, 0.58),
                handlelength=0.6,
                handleheight=0.4,
                handletextpad=0.3,
                labelspacing=0.2,
                borderpad=0.1,
                )
        else:
            ax.get_legend().remove()

    fig.supylabel("Q90 RMSE: Top-4 median / best", x=0.005, fontsize=10)        
    plt.savefig(filename, dpi=300)
    plt.close(fig)
    print(f"Relative error by scale plot saved to {filename}")
