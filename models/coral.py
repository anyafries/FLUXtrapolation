"""
Deep CORAL and MMD models with sklearn-style fit/predict API.

CORAL implements Deep CORAL (Correlation Alignment) for domain generalization from:
  Sun & Saenko, "Deep CORAL: Correlation Alignment for Deep Domain Adaptation" (ECCV 2016)

MMD implements Maximum Mean Discrepancy domain generalization.
  Adapted from https://github.com/mlfoundations/tableshift

In the domain generalization setting, penalties are applied pairwise (or vs. global)
across all training environments, encouraging domain-invariant representations.
"""

import copy
from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .torch_utils import FluxnetDataset, ProportionateBatchSampler


class _AbstractDomainAlignment(ABC):
    """
    Base class for domain-alignment regularizers (CORAL, MMD).

    Subclasses implement `_compute_penalty(feats, env_batch, subset_envs)`
    which returns `(penalty_tensor, lambda_weight)`.
    """

    def __init__(self, hidden_dims=[128, 64], dropout=0.1, lr=1e-3,
                 n_epochs=500, batch_size=256, early_stopping_rounds=10):
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.early_stopping_rounds = early_stopping_rounds
        self.feature_extractor = None
        self.head = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _build_model(self, input_dim):
        layers = []
        prev_dim = input_dim
        for h in self.hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h),
                nn.ReLU(),
                nn.Dropout(self.dropout)
            ])
            prev_dim = h
        self.feature_extractor = nn.Sequential(*layers)
        self.head = nn.Linear(prev_dim, 1)

    def _forward(self, X):
        feats = self.feature_extractor(X)
        return self.head(feats), feats

    @abstractmethod
    def _compute_penalty(self, feats, env_batch, subset_envs):
        ...

    def fit(self, X, y, eval_set=None, envs=None):
        """
        Train with domain alignment penalty.

        Args:
            X: Features array (n_samples, n_features)
            y: Targets array (n_samples,)
            eval_set: Optional [(X_val, y_val)] for early stopping
            envs: Environment labels (required)
        """
        if envs is None:
            raise ValueError(f"{type(self).__name__} requires environment labels (envs)")

        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        self._build_model(X.shape[1])
        self.feature_extractor.to(self.device)
        self.head.to(self.device)

        unique_envs = np.unique(envs)
        env_to_idx = {e: i for i, e in enumerate(unique_envs)}
        env_indices = np.array([env_to_idx[e] for e in envs])
        dataset = FluxnetDataset(X, y, env_indices)
        sampler = ProportionateBatchSampler(env_indices, self.batch_size)
        loader = DataLoader(dataset, batch_sampler=sampler)

        params = list(self.feature_extractor.parameters()) + list(self.head.parameters())
        optimizer = torch.optim.Adam(params, lr=self.lr)
        _g = torch.Generator(device='cpu')
        _g.manual_seed(42)

        use_val = eval_set is not None
        if use_val:
            assert isinstance(eval_set[0][0], torch.Tensor), "Validation X must be a torch.Tensor"
            assert isinstance(eval_set[0][1], torch.Tensor), "Validation y must be a torch.Tensor"
            X_val_t = eval_set[0][0].to(self.device)
            y_val_t = eval_set[0][1].to(self.device)
            best_val_loss = float('inf')
            best_extractor_weights = None
            best_head_weights = None
            rounds_without_improvement = 0

        pbar = tqdm(range(self.n_epochs), desc=type(self).__name__, unit="epoch")
        for _ in pbar:
            self.feature_extractor.train()
            self.head.train()
            for X_batch, y_batch, env_batch in loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                env_batch = env_batch.to(self.device)
                
                optimizer.zero_grad()
                pred, feats = self._forward(X_batch)
                mse = F.mse_loss(pred, y_batch)

                unique_batch_envs = torch.unique(env_batch)
                perm = torch.randperm(len(unique_batch_envs), generator=_g)
                subset_envs = unique_batch_envs[perm[:self._n_pairs]]

                penalty, lam = self._compute_penalty(feats, env_batch, subset_envs)
                loss = mse + lam * penalty
                loss.backward()
                optimizer.step()

            if use_val:
                self.feature_extractor.eval()
                self.head.eval()
                with torch.no_grad():
                    val_pred = self.head(self.feature_extractor(X_val_t))
                    val_loss = F.mse_loss(val_pred, y_val_t).item()
                pbar.set_postfix(val_loss=f"{val_loss:.4f}")
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_extractor_weights = copy.deepcopy(self.feature_extractor.state_dict())
                    best_head_weights = copy.deepcopy(self.head.state_dict())
                    rounds_without_improvement = 0
                else:
                    rounds_without_improvement += 1
                    if rounds_without_improvement >= self.early_stopping_rounds:
                        self.feature_extractor.load_state_dict(best_extractor_weights)
                        self.head.load_state_dict(best_head_weights)
                        break

        if use_val and best_extractor_weights is not None:
            self.feature_extractor.load_state_dict(best_extractor_weights)
            self.head.load_state_dict(best_head_weights)

        return self

    def predict(self, X):
        self.feature_extractor.eval()
        self.head.eval()
        with torch.no_grad():
            # Move inputs to device
            X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
            
            # Predict, move back to CPU, convert to numpy, and flatten
            return self.head(self.feature_extractor(X_t)).cpu().numpy().ravel()


class CORAL(_AbstractDomainAlignment):
    """
    Deep CORAL regressor with sklearn-style fit/predict API.

    Minimizes MSE plus a CORAL penalty that aligns feature distribution
    means and covariances across training environments.
    """

    def __init__(self, hidden_dims=[128, 64], dropout=0.1, lr=1e-3,
                 n_epochs=500, batch_size=256, early_stopping_rounds=10,
                 coral_lambda=1.0, num_coral_pairs=10):
        super().__init__(hidden_dims=hidden_dims, dropout=dropout, lr=lr,
                         n_epochs=n_epochs, batch_size=batch_size,
                         early_stopping_rounds=early_stopping_rounds)
        self.coral_lambda = coral_lambda
        self.num_coral_pairs = num_coral_pairs
        self._n_pairs = num_coral_pairs

    def _compute_penalty(self, feats, env_batch, subset_envs):
        global_mean = feats.mean(0)
        cent_batch = feats - global_mean
        global_cov = (cent_batch.T @ cent_batch) / (len(feats) - 1)
        coral = torch.tensor(0.0, device=feats.device)
        for env_id in subset_envs:
            mask = env_batch == env_id
            if mask.sum() > 1:
                f = feats[mask]
                m = f.mean(0)
                c = (f - m).T @ (f - m) / (len(f) - 1)
                coral += (m - global_mean).pow(2).mean()
                coral += (c - global_cov).pow(2).mean()
        return coral / len(subset_envs), self.coral_lambda


class MMD(_AbstractDomainAlignment):
    """
    MMD regressor using multi-scale Gaussian kernel, sklearn-style fit/predict API.

    Minimizes MSE plus a pairwise MMD penalty between sampled environment pairs,
    encouraging domain-invariant feature representations.

    Adapted from https://github.com/mlfoundations/tableshift

    The Gaussian kernel bandwidths are chosen one of two ways (`bandwidth_mode`):
      - "fixed"  : the TableShift/DomainBed multi-scale grid
                   {1e-3, 1e-2, 1e-1, 1, 10, 100, 1000} (default).
      - "median" : data-driven via the median heuristic. Each forward pass the
                   median squared pairwise distance of the pooled batch
                   representations sets a base length-scale; it is smoothed
                   across batches with an EMA (stored in a registered buffer)
                   and a multi-kernel set is centered on it. Only the choice of
                   bandwidths differs from "fixed"; the penalty, architecture,
                   training loop and lambda search are identical.
    """

    # Multi-kernel multipliers centered on the median length-scale (symmetric
    # about 1, matching the multi-scale structure of the fixed grid).
    _MEDIAN_MULTS = (0.25, 0.5, 1.0, 2.0, 4.0)
    _N_MED = 512            # cap on rows used for the median estimate
    _EMA_MOMENTUM = 0.9     # EMA smoothing of the batch median
    _SIGMA_SQ_FLOOR = 1e-8  # floor to avoid divide-by-zero

    def __init__(self, hidden_dims=[128, 64], dropout=0.1, lr=1e-3,
                 n_epochs=500, batch_size=1024, early_stopping_rounds=10,
                 mmd_lambda=1.0, num_mmd_pairs=5, bandwidth_mode="fixed"):
        super().__init__(hidden_dims=hidden_dims, dropout=dropout, lr=lr,
                         n_epochs=n_epochs, batch_size=batch_size,
                         early_stopping_rounds=early_stopping_rounds)
        if bandwidth_mode not in ("fixed", "median"):
            raise ValueError(
                f"bandwidth_mode must be 'fixed' or 'median', got {bandwidth_mode!r}"
            )
        self.mmd_lambda = mmd_lambda
        self.num_mmd_pairs = num_mmd_pairs
        self._n_pairs = num_mmd_pairs
        self.bandwidth_mode = bandwidth_mode
        # Recorded for verification (median mode only); reset per fit.
        self.sigma_sq_ema_history = []
        self._sigma_initialized = False

    def _build_model(self, input_dim):
        super()._build_model(input_dim)
        # Register the EMA bandwidth as a buffer (not a parameter) so it is
        # saved/restored with the feature extractor and available at eval time.
        # Registered eagerly (before the first batch) so state_dict keys are
        # consistent for early-stopping restore and any later strict load.
        self.sigma_sq_ema_history = []
        self._sigma_initialized = False
        if self.bandwidth_mode == "median":
            self.feature_extractor.register_buffer(
                "mmd_sigma_sq_ema", torch.ones((), device=self.device))

    def _my_cdist(self, x1, x2):
        # Copied from https://github.com/mlfoundations/tableshift
        x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
        x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
        res = torch.addmm(x2_norm.T, x1, x2.T, alpha=-2).add_(x1_norm)
        return res.clamp_min_(1e-30)

    def _gaussian_kernel(self, x, y, gamma=[0.001, 0.01, 0.1, 1, 10, 100, 1000]):
        # Copied from https://github.com/mlfoundations/tableshift
        D = self._my_cdist(x, y)
        K = torch.zeros_like(D)
        for g in gamma:
            K.add_(torch.exp(D.mul(-g)))
        return K

    def _median_gammas(self, feats):
        """
        Data-driven kernel gammas via the median heuristic.

        Computes the median squared pairwise L2 distance on the pooled batch
        representations (all domains together, capped at ``_N_MED`` rows for
        cost), smooths it across batches with an EMA buffer, and returns the
        multi-kernel set of gammas centered on the median.

        The bandwidth is a CONSTANT w.r.t. autograd: the distances, median and
        EMA update all run under ``torch.no_grad()`` and the returned gammas are
        plain Python floats, so no gradient flows through the bandwidth.
        """
        with torch.no_grad():
            z = feats.detach()
            n = z.shape[0]
            if n > self._N_MED:
                idx = torch.randperm(n, device=z.device)[:self._N_MED]
                z = z[idx]
            m = z.shape[0]
            ema = self.feature_extractor.mmd_sigma_sq_ema
            if m > 1:
                D = self._my_cdist(z, z)
                off_diag = D[~torch.eye(m, dtype=torch.bool, device=z.device)]
                sigma_sq_batch = off_diag.median().clamp_min(self._SIGMA_SQ_FLOOR)
                if not self._sigma_initialized:
                    ema.fill_(float(sigma_sq_batch))
                    self._sigma_initialized = True
                else:
                    ema.mul_(self._EMA_MOMENTUM).add_(
                        sigma_sq_batch, alpha=1.0 - self._EMA_MOMENTUM)
                ema.clamp_min_(self._SIGMA_SQ_FLOOR)
            sigma_sq = float(ema)
        self.sigma_sq_ema_history.append(sigma_sq)
        return [mult / sigma_sq for mult in self._MEDIAN_MULTS]

    def _compute_penalty(self, feats, env_batch, subset_envs):
        if self.bandwidth_mode == "median":
            gamma = self._median_gammas(feats)
        else:
            gamma = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
        env_feats = [
            feats[env_batch == e]
            for e in subset_envs
            if (env_batch == e).sum() > 1
        ]
        penalty = torch.tensor(0.0, device=feats.device)
        n_pairs = 0
        for i in range(len(env_feats)):
            for j in range(i + 1, len(env_feats)):
                Kxx = self._gaussian_kernel(env_feats[i], env_feats[i], gamma=gamma).mean()
                Kyy = self._gaussian_kernel(env_feats[j], env_feats[j], gamma=gamma).mean()
                Kxy = self._gaussian_kernel(env_feats[i], env_feats[j], gamma=gamma).mean()
                penalty += Kxx + Kyy - 2 * Kxy
                n_pairs += 1
        if n_pairs > 0:
            penalty /= n_pairs
        return penalty, self.mmd_lambda
