"""
TabPFN prior-data fitted network regressor.
"""

from pathlib import Path

import numpy as np

from tqdm import tqdm

_MODEL_ARTIFACTS = Path(__file__).resolve().parents[1] / "model_artifacts"

_CKPTS = {
    'v2.5': 'tabpfn-v2.5-regressor-v2.5_default.ckpt',
    'v2.6': 'tabpfn-v2.6-regressor-v2.6_default.ckpt',
    'v3': 'tabpfn-v3-regressor-v3_default.ckpt',
    'v3-ood': 'tabpfn-v3-regressor-v3_20260506_ood.ckpt',
}

_DEFAULT_VERSION = 'v2.5'


class TabPFN:
    def __init__(
        self,
        version=None,
        model_path=None,
        n_estimators=1,
        softmax_temperature=1.0,
        subsample_samples=30_000,
        predict_batch_size=50_000,
        random_state=None,
        device=None,
    ):
        if model_path is None:
            version = version or _DEFAULT_VERSION
            if version not in _CKPTS:
                raise ValueError(
                    f"Unknown TabPFN version `{version}`. "
                    f"Available versions: {list(_CKPTS)}"
                )
            model_path = _MODEL_ARTIFACTS / _CKPTS[version]
        self.model_path = str(model_path)
        self.n_estimators = n_estimators
        self.softmax_temperature = softmax_temperature
        self.subsample_samples = int(subsample_samples)
        self.predict_batch_size = int(predict_batch_size)
        self.random_state = random_state
        # None keeps TabPFN's own default ("auto"), which claims *every* visible
        # CUDA device. Pass e.g. "cuda:0" or "cpu" to pin a single one.
        self.device = device
        self._model = None
        self._fitted = False

    def _maybe_subsample(self, X, y):
        if len(X) <= self.subsample_samples:
            return X, y
        rng = np.random.default_rng(self.random_state)
        idx = rng.choice(len(X), size=self.subsample_samples, replace=False)
        idx.sort()
        return X[idx], y[idx]

    def fit(self, X, y):
        ckpt = Path(self.model_path)
        if not ckpt.exists():
            raise FileNotFoundError(
                f"TabPFN checkpoint not found at {ckpt}. "
                "See README for how to add TabPFN checkpoints."
            )
        from tabpfn import TabPFNRegressor

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        X, y = self._maybe_subsample(X, y)
        kwargs = {} if self.device is None else {'device': self.device}
        self._model = TabPFNRegressor(
            model_path=self.model_path,
            n_estimators=self.n_estimators,
            softmax_temperature=self.softmax_temperature,
            memory_saving_mode=False,
            ignore_pretraining_limits=True,
            **kwargs,
        )
        self._model.fit(X, y)
        self._fitted = True
        return self

    def predict(self, X):
        if not self._fitted or self._model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        X = np.asarray(X, dtype=np.float64)
        bs = self.predict_batch_size
        if len(X) <= bs:
            return self._model.predict(X)
        out = np.empty(len(X), dtype=np.float64)
        for i in tqdm(range(0, len(X), bs), desc="TabPFN predict", unit="batch"):
            out[i:i + bs] = self._model.predict(X[i:i + bs])
        return out
