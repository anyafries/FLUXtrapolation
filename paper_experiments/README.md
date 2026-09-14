# Paper Experiments

Scripts to reproduce all figures and tables in the paper.

## Running instructions

All scripts must be run from the **project root** (`fluxnet_bench/`).

---

## Pilot study and benchmarking the baselines

The benchmark must already have been run to load baseline results (see the main README).

For the benchmark summaries (i.e., **Tables 1, 5–10 and the CDF plots**), run

```bash
python paper_experiments/paper_plots.py
```
**Outputs** saved to `paper_experiments/plots/`:
- `table_{aggname}_{target}.tex` — main paper LaTeX tables (median and 90th percentile RMSE)
- `table_{aggname}_{target}_supp.tex` — supplementary tables
- `cdf_{target}_{scale}.png` — CDF plots per flux target and temporal scale

---

## Quantifying the differences between the extrapolation scenarios

To compare the P(X) and P(Y|X) distribution shift between the extrapolation scenarios (i.e., **Figures 1,2,3 and Table 3**) using a domain classifier and density ratio reweighting, run

```bash
python paper_experiments/distribution_distances/analyze.py
```

**Key booleans** near the top of the file (lines ~27–56):

| Flag | Default | Effect |
|---|---|---|
| `GET_MARGINAL_METRICS` | `False` | Compute P(X) shift metrics (Accuracy-based) |
| `GET_CONDITIONAL_METRICS` | `True` | Compute P(Y\|X) shift metrics (reweighted RMSE) |
| `COMPUTE_STAT_SIG` | `True` | Run permutation/bootstrap significance tests (slow) |
| `FULL_BOOTSTRAP` | `True` | Run full bootstrap for confidence intervals (slow) |
| `MAKE_MARGINAL_PLOTS` | `False` | Create marginal histogram PDFs |
| `MAKE_CONDITIONAL_PLOTS` | `False` | Create conditional slice PDFs |

**Outputs:**
- Metric tables printed to stdout
- Optional plots (when enabled) saved to `paper_experiments/distribution_distances/plots/`

---

## RMSE across temporal scales, with bootstrap error bars

Plots RMSE for every model across the temporal scales (hourly → site-mean), with
one panel per extrapolation scenario (temporal, spatial, temperature). Two
figures are produced: one where each model's per-site errors are summarized by
the median, one by the 90th percentile.

```bash
python paper_experiments/scale_rmse_bootstrap_plot.py --target ET
```

`--target` can be `ET`, `GPP`, `NEE`, or `all`. Other options: `--n_boot`
(resamples, default 1000), `--ci` (interval width, default 95), `--val_strategy`.

**How the error bars are made.** The dot for each model is its RMSE summarized
across the held-out sites (median or 90th percentile). To show how much that
number depends on *which* sites we happened to test on, we resample the held-out
sites with replacement and recompute it — 1000 times. We resample whole sites
(not individual records) because the readings from one site are related to each
other. The error bar spans the middle 95% of those 1000 values.

The script also prints, for the hourly scale, how wide each interval is — both in
raw RMSE and as a percentage of the dot it belongs to.

**Outputs** saved to `paper_experiments/plots/`:
- `scale_rmse_median_{target}.png` — RMSE summarized by the median
- `scale_rmse_q90_{target}.png` — RMSE summarized by the 90th percentile

---

## ACFs

Generates autocorrelation function (ACF) plots for ET, GPP, and NEE across hourly, daily, and weekly temporal scales for four example sites (**Figure 5**).

```bash
python paper_experiments/acfs.py
```

**Outputs** saved to `paper_experiments/plots/`:
- `acf_{site}.png` — one file per site (4 total)

---

## Map and split visualization

Generate world maps of FluxNet site locations and time-series plots illustrating the train/validation/test splits:

```bash
Rscript paper_experiments/plot_regions_and_splits.R
```

**Outputs** saved to `paper_experiments/plots/`:
- `fluxnet_map.png` — world map of all FluxNet sites
- `site_split_space.png` — map showing spatial train/test split
- `site_split_ta.png` — map showing temperature-based train/test split
- `time_split.png` — combined time-series panel showing temporal split
- `time_split_{site}.png` — individual time-series for example sites
