"""
TabPFN ablations: context size and ensemble size.

Two ablations, both run across the three benchmark settings:

  context     -- fix n_estimators=1 and vary the number of training rows the
                 model is conditioned on (`subsample_samples`). Each version is
                 run at its own default context size and at every smaller one,
                 so the versions overlap at the short-context end and can be
                 compared directly.
  estimators  -- fix each version at its base context size and vary
                 `n_estimators` (TabPFN's ensemble of input permutations).

Unlike train_model.py there is no hyperparameter search: every configuration is
its own "model", fitted once and evaluated on the test split. Results are
written to the usual results/{models,metrics} tree under a model name like
`tabpfn-v2.6-ctx30k` or `tabpfn-v3-ctx100k-n4`, so they load through
eval.load_all_metrics next to xgb/lr. The `n_estimators=1` point of the
estimators ablation is the same run as the base point of the context ablation
and is therefore computed only once.

Usage:
    python paper_experiments/tabpfn_ablation_run.py --ablation context
    python paper_experiments/tabpfn_ablation_run.py --ablation estimators --device cuda:0

One setting per GPU is the intended way to run this: each shard loads the data
once and writes its own runtime log, so the three can run concurrently. Pass
--skip-defaults to a shard whose setting is still being produced by
train_model.py, so the two do not write the same files.
"""

import argparse
import gc
import os
import sys
import time

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataloader import load_data, get_data_split, save_predictions
from models import get_random_params
from models.tabpfn import TabPFN, _CKPTS, _MODEL_ARTIFACTS
from utils.eval_utils import compute_and_save_metrics, save_best_params
from utils.utils import setup_logging, get_metrics_path, RESULTS_DIR

logger = setup_logging(__name__)

SETTINGS = ['time-split', 'spatial-easy40', 'TA40']
VERSIONS = ['v2.5', 'v2.6', 'v3', 'v3-ood']

# Results are saved with this val_strategy tag so they sit alongside the
# baselines in eval.load_all_metrics(val_strategy='mean'). Nothing is actually
# selected on validation here -- each configuration is evaluated as given.
VAL_STRATEGY = 'mean'
RANDOM_STATE = 42

# --- context-size ablation -----------------------------------------------
# Extra context sizes per version, on top of the version's default context.
EXTRA_CONTEXT_SIZES = {
    'v2.5': [10_000, 30_000],
    'v2.6': [10_000, 30_000],
    'v3':   [10_000, 30_000, 100_000],
    'v3-ood': [10_000, 30_000, 100_000],
}

# --- ensemble-size ablation ----------------------------------------------
# Context held fixed while n_estimators varies. The v3 checkpoints are capped at
# 100k rather than their 1M default: memory scales with context x n_estimators.
BASE_CONTEXT = {'v2.5': 30_000, 'v2.6': 100_000, 'v3': 100_000,
                'v3-ood': 100_000}
N_ESTIMATORS = [1, 2, 4]

def runtime_csv_path(settings):
    """
    Path of the runtime log.

    One setting per file when a single setting is run, so that per-setting
    shards running concurrently do not interleave rows or each write their own
    header into a shared file.
    """
    stem = 'tabpfn_ablation_runtimes'
    if len(settings) == 1:
        stem += f'_{settings[0]}'
    return os.path.join(RESULTS_DIR, 'experiments', f'{stem}.csv')


def default_model_name(version):
    """Benchmark model name for a version, as used by train_model.py."""
    return f"tabpfn-{version}"


def format_size(n):
    """Compact, filename-safe rendering of a context size (10000 -> '10k')."""
    if n % 1_000_000 == 0:
        return f"{n // 1_000_000}M"
    if n % 1_000 == 0:
        return f"{n // 1_000}k"
    return str(n)


def read_default_configs():
    """
    The (context, n_estimators) that `train_model.py --model_name tabpfn-<v>`
    runs for each version, read from the benchmark's own parameter grid so that
    those runs are reused rather than duplicated under a second name.

    A version whose grid is not a single configuration is left out and simply
    gets a suffixed name like any other point of the ablation.
    """
    defaults = {}
    for version in VERSIONS:
        params = get_random_params(default_model_name(version))
        config = params[0] if len(params) == 1 else {}
        if 'subsample_samples' in config and 'n_estimators' in config:
            defaults[version] = (config['subsample_samples'], config['n_estimators'])
        else:
            logger.warning(
                f"{default_model_name(version)} is not a single configuration in "
                f"get_random_params; its default context will be run under a "
                f"ctx-suffixed name instead."
            )
    return defaults


def context_sizes(version):
    """Context sizes for the context ablation: the extras plus the default."""
    sizes = set(EXTRA_CONTEXT_SIZES[version])
    default = DEFAULT_CONFIGS.get(version)
    if default is not None:
        sizes.add(default[0])
    return sorted(sizes)


def run_name(version, context, n_estimators):
    """
    Model name used for the results files.

    The version's default configuration keeps the plain benchmark name
    (`tabpfn-v2.5`, `tabpfn-v2.6`, `tabpfn-v3`, `tabpfn-v3-ood`), so a run
    already made with train_model.py is reused rather than refitted. Every other configuration
    gets a suffix.
    """
    if DEFAULT_CONFIGS.get(version) == (context, n_estimators):
        return default_model_name(version)
    name = f"tabpfn-{version}-ctx{format_size(context)}"
    if n_estimators != 1:
        name += f"-n{n_estimators}"
    return name


DEFAULT_CONFIGS = read_default_configs()


def get_configs(ablation, versions):
    """List of (version, context, n_estimators), de-duplicated in order."""
    configs = []
    if ablation in ('context', 'both'):
        configs += [(v, c, 1) for v in versions for c in context_sizes(v)]
    if ablation in ('estimators', 'both'):
        configs += [(v, BASE_CONTEXT[v], n) for v in versions for n in N_ESTIMATORS]
    seen, unique = set(), []
    for cfg in configs:
        if cfg not in seen:
            seen.add(cfg)
            unique.append(cfg)
    return unique


def check_checkpoints(versions):
    """Fail early (before loading 2.6 GB of data) if a checkpoint is missing."""
    missing = [(v, _MODEL_ARTIFACTS / _CKPTS[v]) for v in versions
               if not (_MODEL_ARTIFACTS / _CKPTS[v]).exists()]
    if missing:
        for version, path in missing:
            logger.error(f"Missing TabPFN {version} checkpoint: {path}")
        raise FileNotFoundError(
            "Download the missing checkpoints into model_artifacts/ (see README) "
            "or restrict the run with --versions."
        )


def log_runtime(row, path):
    """Append one timing row to the ablation runtime log."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame([row]).to_csv(
        path, mode='a', header=not os.path.exists(path), index=False
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default='data',
                        help="Path to the data directory")
    parser.add_argument("--ablation", type=str, default='both',
                        choices=['context', 'estimators', 'both'],
                        help="Which ablation to run")
    parser.add_argument("--setting", type=str, default='all',
                        choices=SETTINGS + ['all'], help="Experiment setting")
    parser.add_argument("--target", type=str, default='ET',
                        choices=['GPP', 'NEE', 'ET', 'all'],
                        help="Target variable to predict")
    parser.add_argument("--versions", type=str, nargs='+', default=VERSIONS,
                        choices=VERSIONS, help="TabPFN versions to include")
    parser.add_argument("--device", type=str, default=None,
                        help="Device for TabPFN, e.g. cuda:0 or cpu. Default "
                             "lets TabPFN pick, which claims every visible GPU.")
    parser.add_argument("--rerun", action='store_true',
                        help="Rerun configurations whose metrics already exist")
    parser.add_argument("--skip-defaults", action='store_true',
                        help="Drop the configurations that are a version's "
                             "benchmark default, i.e. the ones that would be "
                             "saved as plain tabpfn-v2.5/-v2.6/-v3/-v3-ood. "
                             "Use this when train_model.py is producing those "
                             "runs for the same setting concurrently.")

    args = parser.parse_args()
    settings = SETTINGS if args.setting == 'all' else [args.setting]
    targets = ['GPP', 'NEE', 'ET'] if args.target == 'all' else [args.target]
    configs = get_configs(args.ablation, args.versions)
    runtime_csv = runtime_csv_path(settings)

    if args.skip_defaults:
        configs = [c for c in configs
                   if run_name(*c) != default_model_name(c[0])]

    check_checkpoints(args.versions)

    # Work out everything that still needs running before touching the data,
    # so a fully-cached invocation costs nothing.
    todo = {}
    for setting in settings:
        for target in targets:
            pending = [
                cfg for cfg in configs
                if args.rerun or not os.path.exists(
                    get_metrics_path(setting, target, run_name(*cfg), VAL_STRATEGY)
                )
            ]
            if pending:
                todo[(setting, target)] = pending

    n_runs = sum(len(v) for v in todo.values())
    logger.info(f"{len(configs)} configuration(s) x {len(settings)} setting(s) "
                f"x {len(targets)} target(s): {n_runs} run(s) pending.")
    if not todo:
        logger.info("Nothing to do. Use --rerun to recompute.")
        sys.exit(0)

    logger.info(f"Loading data from {args.path}...")
    df = load_data(args.path)

    run_idx = 0
    for (setting, target), pending in todo.items():
        logger.info(f"=== {setting} / {target}: {len(pending)} run(s) ===")
        train, val, test, feature_cols, _ = get_data_split(
            df,
            setting,
            target=target,
            remove_missing_target=True,
            path=args.path,
            return_colnames=True,
        )
        xtrain, ytrain, _ = train
        xtest = test[0]
        logger.info(f"Train rows: {len(xtrain)}, test rows: {len(xtest)}, "
                    f"features: {len(feature_cols)}")

        for version, context, n_estimators in pending:
            run_idx += 1
            name = run_name(version, context, n_estimators)
            logger.info(f"[{run_idx}/{n_runs}] {name} on {setting}/{target} "
                        f"(context={min(context, len(xtrain))}, "
                        f"n_estimators={n_estimators})")

            model = TabPFN(
                version=version,
                n_estimators=n_estimators,
                subsample_samples=context,
                random_state=RANDOM_STATE,
                device=args.device,
            )
            t0 = time.perf_counter()
            model.fit(X=xtrain, y=ytrain)
            fit_seconds = time.perf_counter() - t0

            t0 = time.perf_counter()
            ypred = model.predict(xtest)
            predict_seconds = time.perf_counter() - t0
            logger.info(f"    fit {fit_seconds:.1f}s, predict {predict_seconds:.1f}s")

            params = {
                'version': version,
                'n_estimators': n_estimators,
                'subsample_samples': context,
                'random_state': RANDOM_STATE,
            }
            save_best_params(params, setting, target, name,
                             val_strategy=VAL_STRATEGY, val_score=None)
            preds_df = save_predictions(test, ypred, setting, target, name,
                                        val_strategy=VAL_STRATEGY)
            compute_and_save_metrics(preds_df, setting, target, name,
                                     val_strategy=VAL_STRATEGY)
            log_runtime({
                'model': name, 'setting': setting, 'target': target,
                'version': version, 'context': context,
                'n_estimators': n_estimators,
                'n_train': len(xtrain), 'n_test': len(xtest),
                'fit_seconds': round(fit_seconds, 2),
                'predict_seconds': round(predict_seconds, 2),
            }, runtime_csv)

            # TabPFN keeps the whole context on the GPU; release it before the
            # next (possibly larger) configuration is built.
            del model, ypred, preds_df
            gc.collect()
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass

    logger.info(f"Done. Runtimes appended to {runtime_csv}")
