"""
Sequence-to-sequence LSTM baseline with a sklearn-style fit/predict API.

A small, config-driven LSTM for the hourly FLUXNET upscaling benchmark, added
as a modern sequential counterpart to the (non-sequential) baselines. Motivated
by [reference removed temporarily for double-blind review].

The model predicts the target at every timestep of a fixed-length window
(seq2seq). Two subtleties, handled here and in dataloader.get_sequence_split:

  * Loss masking. The first `warmup` steps of every window only spin up the
    hidden state and are excluded from the loss; so are timesteps whose target
    is missing (qc_mask == 0). We never impute the target. The dataloader
    supplies windows with a boolean mask encoding both conditions.

  * Eval coverage. At prediction time the dataloader tiles each site so every
    test step gets exactly one prediction. Training loss and validation stay
    measured-only, but the TEST set is scored on every row -- gap-filled targets
    and the first `warmup` steps included -- to match the flat baselines (see
    dataloader._eval_window_specs and _build_eval). The earliest steps of each
    site are predicted with less than a full warmup of context.

The X passed to fit/predict is NOT a plain tensor: it is the dict of per-site
blocks + window indices produced by dataloader.get_sequence_split.
"""

import copy

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class _WindowDataset(Dataset):
    """Slices fixed-length training windows out of per-site blocks on the fly.

    Windows are materialised lazily (by index) rather than up front, so the
    heavily-overlapping training windows don't blow up memory. Each item is
    (X_window, y_window, mask_window); mask marks steps that count towards the
    loss (past warmup AND with a measured, non-padded target).
    """

    def __init__(self, blocks, window_index, window, warmup):
        self.blocks = blocks
        self.window_index = window_index
        self.window = window
        self.warmup = warmup

    def __len__(self):
        return len(self.window_index)

    def __getitem__(self, i):
        si, start = self.window_index[i]
        b = self.blocks[si]
        feats, target, valid = b['feats'], b['target'], b['valid']
        length = feats.shape[0]
        end = min(start + self.window, length)
        real = end - start  # real (non-padded) steps in this window

        F = feats.shape[1]
        Xw = np.zeros((self.window, F), dtype=np.float32)
        yw = np.zeros((self.window,), dtype=np.float32)
        mw = np.zeros((self.window,), dtype=np.float32)

        Xw[:real] = feats[start:end]
        seg_tgt = target[start:end]
        seg_valid = valid[start:end].astype(bool)
        # Loss mask: measured target, and strictly after the warmup region.
        mask = seg_valid.copy()
        mask[:self.warmup] = False
        # Targets are NaN where not measured; zero them so nan * 0 stays finite.
        yw[:real] = np.where(seg_valid, np.nan_to_num(seg_tgt), 0.0)
        mw[:real] = mask.astype(np.float32)
        return (
            torch.from_numpy(Xw),
            torch.from_numpy(yw),
            torch.from_numpy(mw),
        )


class _LSTMNet(nn.Module):
    """Multi-layer LSTM with a per-timestep linear head (seq2seq).

    The head is fed the current input x_t alongside the LSTM state (a skip
    connection), so it can recover the instantaneous driver->flux mapping the
    flat MLP learns directly and use the recurrent state only for a temporal
    correction. 
    """

    def __init__(self, input_dim, hidden_size, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size + input_dim, 1)

    def forward(self, x):
        # x: (B, L, F) -> out: (B, L)
        h, _ = self.lstm(x)
        h = torch.cat([self.dropout(h), x], dim=-1)  # skip current input to head
        return self.head(h).squeeze(-1)


class LSTMRegressor:
    """Seq2seq LSTM regressor with a sklearn-style fit/predict API.

    Args:
        hidden_size: LSTM hidden units (default 64).
        num_layers: number of stacked LSTM layers (default 1).
        dropout: dropout applied between LSTM layers and before the head.
        lr: Adam learning rate.
        n_epochs: max epochs (early stopping usually stops sooner).
        batch_size: number of windows per training batch.
        early_stopping_rounds: patience (epochs) on the validation loss. Higher
            than the flat MLP's (10) because the window-based loader yields far
            fewer, more autocorrelated gradient steps per epoch, so the LSTM
            needs more epochs to converge and stops prematurely with patience 10.
        loss: 'mse' or 'huber'.
    """

    def __init__(self, hidden_size=64, num_layers=1, dropout=0.1, lr=1e-3,
                 n_epochs=100, batch_size=64, early_stopping_rounds=20,
                 loss='mse', eval_batch_size=128, seed=42):
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.early_stopping_rounds = early_stopping_rounds
        self.loss = loss
        self.eval_batch_size = eval_batch_size
        self.seed = seed
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _loss_fn(self, pred, target, mask):
        """Mean per-step loss over masked (valid, post-warmup) steps only."""
        if self.loss == 'huber':
            per = nn.functional.huber_loss(pred, target, reduction='none')
        else:
            per = nn.functional.mse_loss(pred, target, reduction='none')
        denom = mask.sum().clamp(min=1.0)
        return (per * mask).sum() / denom

    def fit(self, X, y=None, eval_set=None, envs=None):
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

        input_dim = X['n_features']
        self.model = _LSTMNet(input_dim, self.hidden_size,
                              self.num_layers, self.dropout).to(self.device)

        dataset = _WindowDataset(X['blocks'], X['window_index'],
                                 X['window'], X['warmup'])
        g = torch.Generator()
        g.manual_seed(self.seed)
        loader = DataLoader(dataset, batch_size=self.batch_size,
                            shuffle=True, generator=g)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        use_val = eval_set is not None
        if use_val:
            X_val, y_val = eval_set[0]
            best_val = float('inf')
            best_weights = None
            rounds_without_improvement = 0

        pbar = tqdm(range(self.n_epochs), desc="LSTM", unit="epoch")
        for _ in pbar:
            self.model.train()
            for Xb, yb, mb in loader:
                Xb, yb, mb = Xb.to(self.device), yb.to(self.device), mb.to(self.device)
                optimizer.zero_grad()
                loss = self._loss_fn(self.model(Xb), yb, mb)
                loss.backward()
                optimizer.step()

            if use_val:
                # Early stop on masked val MSE over the same rows we later score.
                val_pred = self.predict(X_val)
                val_loss = float(np.mean((val_pred - y_val) ** 2)) \
                    if len(y_val) else float('inf')
                pbar.set_postfix(val_loss=f"{val_loss:.4f}")
                if val_loss < best_val:
                    best_val = val_loss
                    best_weights = copy.deepcopy(self.model.state_dict())
                    rounds_without_improvement = 0
                else:
                    rounds_without_improvement += 1
                    if rounds_without_improvement >= self.early_stopping_rounds:
                        break

        if use_val and best_weights is not None:
            self.model.load_state_dict(best_weights)
        return self

    @torch.no_grad()
    def predict(self, X):
        """Per-timestep predictions, gathered to the flat valid-only rows.

        Runs the tiled eval windows, scatters each window's owned slice back to
        its site, then gathers predictions at X['flat_index'] so the output is
        row-aligned with the y/envs/sites/times returned by get_sequence_split.
        """
        self.model.eval()
        window = X['window']
        blocks = X['blocks']
        # Per-site prediction buffers (NaN where never covered).
        preds = [np.full(b['feats'].shape[0], np.nan, dtype=np.float32)
                 for b in blocks]

        specs = X['eval_windows']
        for i in range(0, len(specs), self.eval_batch_size):
            batch = specs[i:i + self.eval_batch_size]
            Xb = np.zeros((len(batch), window, X['n_features']), dtype=np.float32)
            for j, (si, start, _lo, _hi) in enumerate(batch):
                feats = blocks[si]['feats']
                end = min(start + window, feats.shape[0])
                Xb[j, :end - start] = feats[start:end]
            out = self.model(torch.from_numpy(Xb).to(self.device)).cpu().numpy()
            for j, (si, start, own_lo, own_hi) in enumerate(batch):
                preds[si][start + own_lo:start + own_hi] = out[j, own_lo:own_hi]

        return np.array([preds[si][t] for (si, t) in X['flat_index']],
                        dtype=np.float32)
