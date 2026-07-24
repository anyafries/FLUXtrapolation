import argparse
import os
import pandas as pd
import sys
from sklearn.metrics import root_mean_squared_error

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataloader import load_data, get_data_split, save_predictions
from models import get_model, get_random_params
from tests.test_models import test_model
from utils.eval_utils import compute_and_save_metrics, save_best_params
from utils.utils import setup_logging, get_metrics_path


logger = setup_logging(__name__)

ALL_TARGETS = ['GPP', 'NEE', 'ET']
ALL_VAL_SPLITS = ['iid', 'temporal', 'oracle']


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default='data',
                        help="Path to the data directory")
    parser.add_argument("--rerun", action='store_true',
                        help="Rerun existing results")
    parser.add_argument("--target", type=str, 
                        choices=['GPP', 'NEE', 'ET', 'all'],
                        default='ET', help="Target variable to predict")
    parser.add_argument("--model_name", type=str, 
                        default='lr', help="Model to use for the experiment")
    parser.add_argument("--val_split", type=str,
                        choices=['iid', 'temporal', 'oracle', 'all'], default='all',
                        help="Validation split strategy to use for hyperparameter tuning (default: all)")

    args = parser.parse_args()
    model_name = args.model_name

    # Determine which settings and targets to run
    settings = ['spatial-easy40']
    targets = ALL_TARGETS if args.target == 'all' else [args.target]
    val_splits = ALL_VAL_SPLITS if args.val_split == 'all' else [args.val_split]

    # Load data
    logger.info(f"Loading data from {args.path}...")
    df = load_data(args.path)

    # Run experiments
    for setting in settings:
        for target in targets:
            for val_split in val_splits:
                save_setting_name = f"{setting}-{val_split}"
                logger.info(f"Running experiment: {save_setting_name}, target={target}, model={model_name}")

                if not args.rerun and all(
                    os.path.exists(get_metrics_path(save_setting_name, target, model_name, s))
                    for s in ['mean', 'max', 'discrepancy']
                ):
                    logger.info(f"Results for {save_setting_name} already exist. Skipping.")
                    continue

                split = get_data_split(
                    df,
                    setting=setting,
                    target=target,
                    validation_split=val_split,
                    remove_missing_target=True,
                    path=args.path,
                    standardize=model_name in ['robust-lr', 'ridge'],
                    astorch=model_name in ['mlp', 'gdro', 'coral', 'mmd'],
                    return_colnames=True,
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
                logger.info(f"Saving predictions and metrics for {save_setting_name}/{target}/{model_name}...")
                for strategy, b in best.items():
                    logger.info(f"  [{strategy}] Best val score: {b['score']:.4f}")
                    save_best_params(b['params'], save_setting_name, target, model_name,
                                     val_strategy=strategy, val_score=b['score'])
                    ypred = b['model'].predict(xtest)
                    preds_df = save_predictions(test, ypred, save_setting_name, target, model_name, val_strategy=strategy)
                    compute_and_save_metrics(preds_df, save_setting_name, target, model_name, val_strategy=strategy)