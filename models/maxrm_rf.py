"""
MaxRM Random Forest models for worst-case risk minimization.

Implements two risk criteria on top of adaXT's RandomForest:
  - 'mse':    MaxRM-MSE — minimizes worst-group mean squared error
  - 'regret': MaxRM-Regret — minimizes worst-group regret (loss relative to
              per-environment ERM oracle)

Reference implementation: [removed for anonymous review]
"""

import functools
import os
import sys
import time

import numpy as np

from adaXT.random_forest import RandomForest
from sklearn.preprocessing import LabelEncoder


def _install_tree_progress():
    """
    Report per-tree progress during modify_predictions_trees().

    adaXT runs the trees through a single blocking map_async().get(), so there
    is no progress hook to attach to. Instead we wrap the per-tree worker so
    each forked child reports when it starts and finishes a tree. Enable with
    MAXRM_PROGRESS=1.
    """
    orig = RandomForest._modify_single_tree_predictions
    if getattr(orig, '_progress_wrapped', False):
        return

    # functools.wraps is required, not cosmetic: adaXT ships the bound method
    # to workers via pickle, which serializes it by __name__ and resolves it
    # with getattr() in the child. Without the original name the lookup fails.
    @functools.wraps(orig)
    def wrapper(self, tree_data, *args, **kwargs):
        tree_idx = tree_data[2]
        n_leaves = len(tree_data[0].leaf_nodes)
        t0 = time.time()
        print(f"  [tree {tree_idx:4d}] start  ({n_leaves} leaves, "
              f"pid {os.getpid()})", file=sys.stderr, flush=True)
        result = orig(self, tree_data, *args, **kwargs)
        print(f"  [tree {tree_idx:4d}] done   ({time.time() - t0:.1f}s)",
              file=sys.stderr, flush=True)
        return result

    wrapper._progress_wrapped = True
    RandomForest._modify_single_tree_predictions = wrapper


if os.environ.get('MAXRM_PROGRESS'):
    _install_tree_progress()

def modify_predictions(
    model,
    train_ids_int,
    risk,
    sols_erm=None,
    sols_erm_trees=None,
    n_jobs=10,
    verbose=True,
):
    """
    Try modify_predictions_trees with several solvers, then fall back to
    opt_method='extragradient'. Returns True if successful, False otherwise.
    """
    # Forward verbose into modify_predictions_trees so its own
    # "Initial score / Optimized score / rolling back" diagnostics print,
    # not just this wrapper's solver-attempt lines.
    kwargs = {"method": risk, "n_jobs": n_jobs, "verbose": verbose,
              "bcd": True, "block_size": 15}

    if risk == "regret":
        kwargs["sols_erm"] = sols_erm
        kwargs["sols_erm_trees"] = sols_erm_trees

    solvers = [None, "ECOS", "SCS"]

    for solver in solvers:
        if verbose:
            solver_name = "default solver" if solver is None else solver
        try:
            if verbose:
                print(f"* Trying {solver_name}...")
            model.modify_predictions_trees(
                train_ids_int, **kwargs, solver=solver
            )
            return True
        except Exception as e:
            if verbose:
                print(f"* {solver_name} failed.")
                print(str(e))

    if verbose:
        print(
            f"* Fallback: all solvers failed. "
            "Retrying with opt_method='extragradient'."
        )
    try:
        model.modify_predictions_trees(
            train_ids_int, **kwargs, opt_method="extragradient"
        )
        return True
    except Exception:
        if verbose:
            print(f"* ERROR in modify_predictions_trees after all fallbacks")
        return False


class MaxRM_RF(RandomForest):
    """
    MaxRM Random Forest: adaXT RandomForest with worst-group risk minimization.

    After a standard RF fit, calls modify_predictions_trees() to re-weight
    leaf predictions so that the worst-group risk (MSE or regret) is minimized.

    Args:
        risk (str): 'mse' for MaxRM-MSE, 'regret' for MaxRM-Regret.

    Reference: [removed for anonymous review]
    """

    def __init__(self, n_estimators=100, seed=42,
                 min_samples_leaf=30, n_jobs=1, risk='mse',
                 max_depth=None, sampling_args=None):
        params = {
            'n_estimators': n_estimators,
            'min_samples_leaf': min_samples_leaf,
            'n_jobs': n_jobs,
            'forest_type': 'Regression',
            'seed': seed,
            'max_features': 1/3,
            # The value here is only a default; the hyperparameter grid passes
            # its own sampling_args and takes precedence.
            'sampling_args': sampling_args or {'size': 0.2},
        }
        # adaXT defaults max_depth to 2**31 - 1, so only set it when asked.
        if max_depth is not None:
            params['max_depth'] = max_depth
        super().__init__(**params)
        self.risk = risk
        self.n_estimators = n_estimators
        self._init_params = params

    def fit(self, X, y, envs):
        """
        Fit the forest, then adjust leaf predictions to minimize worst-group risk.

        Args:
            X: Feature matrix.
            y: Target vector.
            envs: Group/environment labels (integer-castable array).
        """
        print(f"Fitting maxRM-RF with the following environments:", flush=True)
        print(f"  Raw:  {envs[:10]} ...", flush=True)
        if isinstance(envs[0], tuple):
            # This happens for the time-series split, 
            # where envs are tuples of (site, year). 
            # We use the year as the environment label for MaxRM.
            envs = np.array([e[1] for e in envs])
        print(f"  Done: {np.unique(envs)}", flush=True)

        le = LabelEncoder()
        train_ids_int = le.fit_transform(envs)

        print(
            f"Fitting MaxRM_RF with risk={self.risk}: "
            f"{X.shape[0]} rows, {X.shape[1]} features, "
            f"{self.n_estimators} trees...",
            flush=True,
        )
        # adaXT seeds the bootstrap from `seed`, but per-split feature
        # subsampling (active whenever max_features is set) draws from the
        # global numpy RNG, which `seed` does not touch. Seed it here so the
        # forest is reproducible run-to-run. Note: this is only fully
        # deterministic at n_jobs=1; under fork with multiple trees the workers
        # share the inherited global state, so some jitter remains.
        np.random.seed(self._init_params['seed'])
        t_fit = time.time()
        super().fit(X, y)
        print(f"Forest fit in {time.time() - t_fit:.1f}s.", flush=True)

        if self.risk == 'regret':
            print("Computing ERM predictions for regret calculation...")
            sols_erm = np.zeros(len(train_ids_int))
            sols_erm_trees = np.zeros(
                (self.n_estimators, len(train_ids_int))
            )
            for env in np.unique(train_ids_int):
                mask = train_ids_int == env
                xtrain_env = X[mask]
                ytrain_env = y[mask]
                rf_env = RandomForest(**self._init_params)
                rf_env.fit(xtrain_env, ytrain_env)
                fitted_env = rf_env.predict(xtrain_env)
                sols_erm[mask] = fitted_env
                for i in range(self.n_estimators):
                    fitted_env_tree = rf_env.trees[i].predict(xtrain_env)
                    sols_erm_trees[i, mask] = fitted_env_tree

        n_leaves = [len(t.leaf_nodes) for t in self.trees]
        print(
            f"Modifying predictions to minimize worst-group risk: "
            f"{self.n_estimators} trees, {len(np.unique(train_ids_int))} envs, "
            f"~{int(np.mean(n_leaves))} leaves/tree "
            f"(cost scales as trees x envs x leaves)...",
            flush=True,
        )
        t0 = time.time()
        success = modify_predictions(
            model=self,
            train_ids_int=train_ids_int,
            risk=self.risk,
            sols_erm=sols_erm if self.risk == 'regret' else None,
            sols_erm_trees=sols_erm_trees if self.risk == 'regret' else None,
            n_jobs=self._init_params['n_jobs'],
            verbose=True,
        )
        print(f"Modified predictions in {time.time() - t0:.1f}s.", flush=True)
        if not success:
            print(f"WARNING: modify_predictions failed for MaxRM_RF with risk={self.risk}.")

    def predict(self, X):
        return super().predict(X)
