"""
Model definitions for FLUXNET benchmark.

Available models:
  - 'lr'            : Linear Regression (sklearn)
  - 'ridge'         : Ridge Regression (sklearn)
  - 'xgb'              : XGBoost Regressor
  - 'xgb_conservative' : XGBoost with conservative feature set (TA, VPD, SW_IN, EVI, NDWI_SWIR2, PFT_*)
  - 'mlp'           : Multi-layer Perceptron
  - 'gdro'          : Group DRO (worst-group loss minimization)
  - 'coral'         : CORAL domain adaptation
  - 'mmd'           : MMD domain adaptation
  - 'maxrm_mse'     : MaxRM Random Forest with MSE risk
  - 'maxrm_regret'  : MaxRM Random Forest with regret risk
"""

import json
import numpy as np
import os
import random

from utils.utils import setup_logging, get_params_path

logger = setup_logging(__name__)


# ------------------------------------------------------------------------
# Loading models by name
# ------------------------------------------------------------------------

def get_model(model_name, params=None):
    """
    Factory function to get a model instance by name.

    Args:
        model_name (str): Name of the model. See module docstring for options.
        params (dict, optional): Parameters to initialize the model.
            If None, defaults will be used.

    Returns:
        A model instance with fit() and predict() methods

    Raises:
        NotImplementedError: If the model name is not recognized
    """
    if model_name == 'xgb':
        from xgboost import XGBRegressor
        return XGBRegressor(**params)

    elif model_name == 'xgb_conservative':
        from .xgb_conservative import XGBConservative
        return XGBConservative(**params)
    
    elif model_name == 'lr':
        from sklearn.linear_model import LinearRegression
        return LinearRegression()
    
    elif model_name == 'robust-lr':
        from sklearn.linear_model import HuberRegressor
        return HuberRegressor(**params)
    
    elif model_name == 'constant':
        from sklearn.dummy import DummyRegressor
        return DummyRegressor(strategy='mean')
    
    elif model_name == 'ridge':
        from sklearn.linear_model import Ridge
        return Ridge(**params)

    elif model_name == 'lightgbm':
        from lightgbm import LGBMRegressor
        return LGBMRegressor(**params)
    
    elif model_name == 'mlp':
        from .mlp import MLP
        return MLP(**params)
    
    elif model_name == 'gdro':
        from .gdro import GroupDRO
        return GroupDRO(**params)
    
    elif model_name == 'coral':
        from .coral import CORAL
        return CORAL(**params)
    
    elif model_name == 'mmd':
        from .coral import MMD
        return MMD(**params)
    
    elif model_name == 'anchorboosting':
        from .anchorboosting import AnchorBoosting
        return AnchorBoosting(**params)

    elif model_name == 'anchorboosting-c':
        from .anchorboosting import AnchorBoosting
        return AnchorBoosting(anchor_type='continuous', **params)

    elif model_name == 'maxrm-mse':
        from .maxrm_rf import MaxRM_RF
        return MaxRM_RF(risk='mse', **params)
    
    elif model_name == 'maxrm-regret':
        from .maxrm_rf import MaxRM_RF
        return MaxRM_RF(risk='regret', **params)
    
    else:
        raise NotImplementedError(
            f"Model `{model_name}` not implemented. "
            f"Available models: 'xgb', 'xgb_conservative', 'lr', 'ridge', 'mlp', 'gdro', 'coral', 'mmd', 'maxrm_mse', 'maxrm_regret'"
        )


# ------------------------------------------------------------------------
# Hyperparameter grids for each model
# ------------------------------------------------------------------------

def sample_log_uniform(low, high, rng):
    """Best practice for LR and Alpha: samples across orders of magnitude."""
    return float(10 ** rng.uniform(np.log10(low), np.log10(high)))


def load_best_mlp_params(setting, target, val_strategy):
    """Helper to fetch the best MLP parameters already saved."""
    path = get_params_path(setting, target, 'mlp', val_strategy)
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return None


def get_random_params(model_name, n_iter=10, setting=None, target=None):
    """
    Generates N random parameter sets for a given model.
    
    Args:
        model_name (str): The model identifier.
        n_iter (int): Number of random configurations to generate.
        setting (str): Current setting (used for inheriting MLP params).
        target (str): Current target (used for inheriting MLP params).
    """
    np_rng = np.random.default_rng(42)
    rng = random.Random(42)

    if model_name in ['lr', 'constant']:
        return [{}]

    best_mlp = None
    if model_name in ['gdro', 'coral', 'mmd']:
        if setting is not None and target is not None:
            best_mlp = load_best_mlp_params(setting, target, 
                                            val_strategy='mean')
            if best_mlp:
                logger.info(f"Inheriting MLP params for {model_name}: {best_mlp.get('hidden_dims')}")
                n_iter = max(1, n_iter // 2)  # Reduce iterations since we're inheriting MLP params

    random_configs = []
    for _ in range(n_iter):
        params = {}

        if model_name == 'ridge':
            params = {
                'alpha': sample_log_uniform(1e-3, 1e2, np_rng)
            }

        elif model_name == 'robust-lr':
            params = {
                'alpha': sample_log_uniform(1e-3, 1e2, np_rng),
                'epsilon': float(np_rng.uniform(1.1, 2.0))
            }

        elif model_name in ['maxrm_mse', 'maxrm_regret']:
            params = {
                'n_estimators': rng.choice([100, 200, 500, 1000]),
                'min_samples_leaf': rng.randint(5, 50),
                'random_state': 42,
            }

        elif model_name in ['xgb', 'xgb_conservative']:
            params = {
                'n_estimators': rng.randint(100, 1000),
                'max_depth': rng.randint(3, 10),
                'learning_rate': sample_log_uniform(0.01, 0.3, np_rng),
                'subsample': float(np_rng.uniform(0.6, 1.0)),
                'objective': 'reg:squarederror',
                'early_stopping_rounds': 10,
                'n_jobs': 4,
                'random_state': 42,
            }

        elif model_name == 'lightgbm':
            max_depth = rng.randint(3, 12)
            # num_leaves should be less than 2^max_depth
            max_leaves = max(min(65, 2**max_depth - 1), 15)

            params = {
                'n_estimators': rng.randint(100, 500),
                'learning_rate': sample_log_uniform(0.05, 0.2, np_rng),
                'num_leaves': rng.randint(15, max_leaves),
                'max_depth': max_depth,
                'min_child_samples': rng.randint(5, 30),
                # 'subsample': np.random.uniform(0.6, 1.0),
                'colsample_bytree': float(np_rng.uniform(0.6, 1.0)),
                'objective': 'regression',
                'random_state': 42,
                'verbosity': -1,
                'n_jobs': 4,
                'boosting_type': 'goss',
            }

        elif model_name in ['anchorboosting', 'anchorboosting-c']:
            params = {
                'gamma': sample_log_uniform(1.0, 100.0, np_rng),
                'num_boost_round': 1000,
                'learning_rate': 0.1,
                'max_depth': 3,
                'min_gain_to_split': 0.1,
                'lambda_l2': sample_log_uniform(1e-3, 1e1, np_rng),
            }

        elif model_name in ['mlp', 'gdro', 'coral', 'mmd']:
            # Pre-load MLP best params for derivative models
            if best_mlp and model_name != 'mlp':
                params = best_mlp.copy()
            else:
                # Base deep learning params
                params = {
                    'hidden_dims': rng.choice([[128, 64], [256, 128], [512, 256, 128]]),
                    'lr': sample_log_uniform(1e-5, 1e-2, np_rng),
                    'dropout': float(np_rng.uniform(0.0, 0.5)),
                    'n_epochs': 100,
                    'batch_size': rng.choice([512, 1024, 2048])
                }

            # Model-specific logic
            if model_name == 'gdro':
                params['group_weight_step'] = sample_log_uniform(1e-4, 1e-1, np_rng)
            elif model_name == 'coral':
                params['coral_lambda'] = sample_log_uniform(1e-2, 1e1, np_rng)
            elif model_name == 'mmd':
                params['mmd_lambda'] = sample_log_uniform(1e-2, 1e1, np_rng)

        random_configs.append(params)

    return random_configs