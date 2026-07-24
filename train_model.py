import argparse
import os
import pandas as pd
import torch

from sklearn.metrics import root_mean_squared_error

from dataloader import load_data, get_data_split, save_predictions
from models import get_model, get_random_params
from tests.test_models import test_model
from utils.eval_utils import compute_and_save_metrics, save_best_params, save_model
from utils.utils import setup_logging, get_metrics_path

logger = setup_logging(__name__)

# Safe, no-crash determinism knobs (near no-ops for the Linear-only nets, but
# harmless). Reproducibility of the deep models comes primarily from saved weights.
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

ALL_SETTINGS = ['time-split', 'spatial-easy40', 'TA40']
ALL_TARGETS = ['ET', 'GPP', 'NEE']

DEEP_MODELS = ['mlp', 'gdro', 'coral', 'mmd', 'mmd-median', 'lstm']
STANDARDIZE_MODELS = ['robust-lr', 'ridge', 'mlp', 'gdro', 'coral', 'mmd', 'mmd-median']
ASTORCH_MODELS = ['mlp', 'gdro', 'coral', 'mmd', 'mmd-median']

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default='data',
                        help="Path to the data directory")
    parser.add_argument("--rerun", action='store_true',
                        help="Rerun existing results")
    parser.add_argument("--setting", type=str,
                        choices=['time-split', 'spatial-easy40', 'TA40', 'all'],
                        default='all', help="Experiment setting")
    parser.add_argument("--target", type=str, 
                        choices=['GPP', 'NEE', 'ET', 'all'],
                        default='all', help="Target variable to predict")
    parser.add_argument("--model_name", type=str,
                        default='lr', help="Model to use for the experiment")

    args = parser.parse_args()
    model_name = args.model_name

    # Determine which settings and targets to run
    settings = ALL_SETTINGS if args.setting == 'all' else [args.setting]
    targets = ALL_TARGETS if args.target == 'all' else [args.target]

    # Load data
    data_path = os.path.join(args.path, "daily_with_mask.csv")
    logger.info(f"Loading data from {args.path}...")
    df = load_data(args.path)

    # Run experiments
    for target in targets:
        for setting in settings:
            logger.info(f"Running experiment: setting={setting}, target={target}, model={model_name}")
            if not args.rerun and all(
                os.path.exists(get_metrics_path(setting, target, model_name, s))
                for s in ['mean', 'max', 'discrepancy']
            ):
                logger.info(f"Results for {setting}/{target}/{model_name} already exist. Use --rerun to rerun.")
                continue

            split = get_data_split(
                df,
                setting,
                target=target,
                remove_missing_target=True,
                path=args.path,
                standardize=model_name in STANDARDIZE_MODELS,
                astorch=model_name in ASTORCH_MODELS,
                return_colnames=True,
                sequence=model_name == 'lstm',
            )
            train, val, test, feature_cols, _ = split
            xtrain, ytrain, envs_train = train
            xval, yval, envs_val = val
            xtest = test[0]

            # Hyperparameter tuning loop
            logger.info(f"Starting random search for {model_name} on {setting}-{target}...")
            param_grid = get_random_params(model_name)
            best = {s: {'score': float('inf'), 'params': None, 'model': None}
                    for s in ['mean', 'max', 'discrepancy']}
            
            for i, params in enumerate(param_grid):
                # Get model and train
                model = get_model(model_name, params)
                use_eval_set, use_envs, use_feature_names = test_model(model)

                # Build fit arguments dynamically
                fit_args = {'X': xtrain, 'y': ytrain}
                if use_eval_set:
                    fit_args['eval_set'] = [(xval, yval)]
                if use_envs:
                    fit_args['envs'] = envs_train.values
                if model_name in ['xgb', 'xgb_conservative']:
                    fit_args['verbose'] = False
                if use_feature_names:
                    fit_args['feature_names'] = feature_cols
                if model_name == 'lightgbm':
                    import lightgbm as lgb
                    fit_args['eval_metric'] = 'rmse'
                    fit_args['callbacks'] = [
                        lgb.early_stopping(stopping_rounds=10),
                        lgb.log_evaluation(period=0) # Keeps logs clean
                    ]

                # Train model
                model.fit(**fit_args)

                # Evaluate on validation set
                val_preds = model.predict(xval)
                val_df = pd.DataFrame({
                    'y_true': yval.ravel(), 
                    'y_pred': val_preds, 
                    'env': envs_val.values
                })
                site_rmse = val_df.groupby('env')[['y_true', 'y_pred']].apply(
                    lambda g: root_mean_squared_error(g['y_true'], g['y_pred'])
                )
                scores = {
                    'mean': site_rmse.mean(),
                    'max': site_rmse.max(),
                    'discrepancy': site_rmse.max() - site_rmse.min(),
                }
                for s, score in scores.items():
                    if score < best[s]['score']:
                        best[s].update({'score': score, 'params': params, 'model': model})
                logger.info(
                    f"Params {i+1}/{len(param_grid)} complete. "
                    f"mean={scores['mean']:.4f}, max={scores['max']:.4f}, "
                    f"discrepancy={scores['discrepancy']:.4f}"
                )

            # Final evaluation on test set with best model per strategy
            logger.info("Model selection complete.")
            logger.info(f"Saving predictions and metrics for {setting}/{target}/{model_name}...")
            for strategy, b in best.items():
                logger.info(f"  [{strategy}] Best val score: {b['score']:.4f}, params: {b['params']}")
                save_best_params(b['params'], setting, target, model_name,
                                 val_strategy=strategy, val_score=b['score'])
                if model_name in DEEP_MODELS:
                    save_model(b['model'], b['params'], len(feature_cols),
                               setting, target, model_name, val_strategy=strategy)
                ypred = b['model'].predict(xtest)
                preds_df = save_predictions(test, ypred, setting, target, model_name, val_strategy=strategy)
                compute_and_save_metrics(preds_df, setting, target, model_name, val_strategy=strategy)