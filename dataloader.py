import os
import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from utils.utils import setup_logging, get_predictions_path, load_csv, save_csv

logger = setup_logging(__name__)

# Fixed site groups for the two spatial-extrapolation settings. Defined once and
# shared by BOTH the flat (row-wise) baselines and the sequence (LSTM) split so
# the two paths always see identical train/val/test membership.
_SPATIAL_TEST_GROUPS = {
    'spatial-easy40': ['US-Tw1', 'DE-Hai', 'US-Seg', 'US-Sne', 'US-Tw4', 'US-xDL', 'UK-AMo', 'AU-Dry', 'US-CGG', 'FR-Bil', 'US-Rpf', 'DK-Skj', 'RU-Fy2', 'DE-Rns', 'US-Tw3', 'RU-Fyo', 'US-Snf', 'CH-Cha', 'AR-CCg', 'CL-SDF', 'DE-Gri', 'FR-Tou', 'AU-Whr', 'AU-GWW', 'US-RGo', 'IT-BCi', 'ES-Abr', 'SE-Nor', 'DE-Hzd', 'US-CS2', 'US-StJ', 'CA-TP3', 'BE-Dor', 'US-xWD', 'US-Syv', 'DE-RuR', 'CZ-BK1', 'BE-Maa', 'BE-Vie', 'FI-Var'],
    'TA40': ['AU-Dry', 'AU-DaS', 'AU-Lit', 'BR-Npw', 'AU-Lon', 'AU-ASM', 'US-xDS', 'US-ONA', 'US-SP1', 'US-xJE', 'US-SRM', 'US-HB2', 'AU-GWW', 'US-SRS', 'US-SRG', 'IL-Yat', 'US-HB3', 'US-HB1', 'US-xDL', 'US-RGA', 'AU-Cum', 'US-xTA', 'AU-Cpr', 'US-Whs', 'US-Cst', 'US-Wkg', 'IT-BCi', 'US-Jo2', 'IT-Cp2', 'US-RGo', 'ES-Abr', 'US-NC4', 'ES-Agu', 'US-Akn', 'US-xJR', 'ES-Pdu', 'US-Ton', 'ES-LM2', 'IT-Noe', 'ES-LM1'],
}
_SPATIAL_VAL_GROUPS = {
    'spatial-easy40': ['DE-Tha', 'US-xTR', 'US-ICh', 'FR-Aur', 'US-NR1', 'CA-TPD', 'AU-Cum', 'US-RGA', 'CZ-Lnz', 'US-UC1', 'SE-Htm', 'AU-Rgf', 'ES-Agu', 'FR-Mej', 'CA-ARF', 'CA-TP1', 'CA-SCC', 'US-BZB', 'US-xCP', 'DK-Vng'],
    'TA40': ['US-Snf', 'US-GLE', 'US-CF2', 'FI-Let', 'CZ-Lnz', 'US-Rls', 'UK-AMo', 'FR-Gri', 'US-xTR', 'US-ALQ', 'CA-ER1', 'US-xBR', 'FI-Hyy', 'IE-Cra', 'DE-Obe', 'AU-War', 'US-RGB', 'CH-Cha', 'US-Syv', 'US-UMB'],
}

# ---------------------------------------------------
# ------------------ Loading data -------------------
# ---------------------------------------------------

def load_data(path):
    """Load the data from the specified path."""
    # each file in this folder is a site
    # load them all and concatenate into one dataframe
    path = os.path.join(path, "sites")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data path not found: {path}")
    dfs = []
    for filename in sorted(os.listdir(path)):
        if filename.endswith(".csv"):
            site_id = filename.split(".")[0]
            df_site = pd.read_csv(os.path.join(path, filename))
            df_site["site_id"] = site_id
            dfs.append(df_site)
    df = pd.concat(dfs, ignore_index=True)

    bool_cols = df.select_dtypes(include='bool').columns
    for col in bool_cols:
        nunique = df[col].nunique(dropna=False)
        assert nunique == 2, f"Expected boolean column {col} to have exactly 2 unique values, but found {nunique}"
    return df


# -----------------------------------------------------------------------
# ------------------ Functions for getting fold data --------------------
# -----------------------------------------------------------------------

def get_data_split(
    df,
    setting,
    path,
    target="GPP",
    remove_missing_target=False,
    keep_lonlat=False,
    keep_time=False,
    astorch=False,
    return_colnames=False,
    standardize=False,
    validation_split='default',
    sequence=False,
    window=720,
    warmup=168,
    train_stride=168,
):
    """
    Get the train/test data for a specific setting.
    Args:
        df (pd.DataFrame): The input dataframe containing the data.
        setting (str): The cross-validation setting.
        path (str): The path to the data directory (used for loading site lists).
        target (str, optional): The target variable name.
            Defaults to "GPP".
        remove_missing_target (bool, optional): Whether to remove rows with
            missing target values. Defaults to False.
        keep_lonlat (bool, optional): Whether to keep longitude and latitude
            features. Defaults to False.
        keep_time (bool, optional): Whether to keep time feature. Defaults to False.
        astorch (bool, optional): Whether to return data as PyTorch tensors.
            Defaults to False.
        return_colnames (bool, optional): Whether to return column names of
            features. Defaults to False.
        standardize (bool, optional): Whether to standardize features using
            training set statistics. Defaults to False.
        sequence (bool, optional): If True, return time-ordered sequence windows
            for the LSTM baseline instead of flattened rows (see
            get_sequence_split). Defaults to False.
        window, warmup, train_stride (int, optional): Sequence-mode windowing
            parameters (only used when sequence=True). Defaults 720/168/168.
    Returns:
        tuple: xtrain, ytrain, envs_train, xtest, ytest, envs_test
    """
    # The LSTM baseline needs time-ordered windows rather than flattened rows,
    # so it takes a completely separate path that nonetheless reuses the exact
    # same site/year membership, target/qc conventions, and RobustScaler.
    if sequence:
        return get_sequence_split(
            df, setting, path, target=target, validation_split=validation_split,
            window=window, warmup=warmup, train_stride=train_stride,
            return_colnames=return_colnames,
        )

    # Subset the correct data
    if setting == "time-split":
        sites_to_keep = pd.read_csv(os.path.join(path, "sites_with_2018.csv"))
        df_out = df.loc[df["site_id"].isin(sites_to_keep['site_id'].values)].copy()
    else: 
        df_out = df.copy()

    # Preserve time column if needed for metadata
    time_col = df_out["time"].copy()

    # drop columns
    cols_to_drop = []
    if not keep_lonlat:
        cols_to_drop += ["tower_lat", "tower_lon"]
    if not keep_time:
        cols_to_drop += ["time"]
    for col in cols_to_drop:
        if col in df_out.columns:
            df_out.drop(columns=[col], inplace=True)

    # split into train/test
    if setting == "time-split":
        if validation_split != 'default':
            raise NotImplementedError("Custom validation split not implemented for time-split setting")
        df_out['site_year'] = list(zip(df_out['site_id'], df_out['year']))
        # split years chronologically
        train = df_out.loc[df_out["year"] < 2018].copy()
        val = df_out.loc[df_out["year"] == 2018].copy()
        test = df_out.loc[df_out["year"] > 2018].copy()
        
    else:
        if setting not in _SPATIAL_TEST_GROUPS:
            raise ValueError(f"Setting `{setting}` not recognized in get_data_split")
        test_group = _SPATIAL_TEST_GROUPS[setting]
        test = df_out.loc[df_out["site_id"].isin(test_group)].copy()

        # get train, val depending on validation_split strategy
        if validation_split == 'default':
            val_group = _SPATIAL_VAL_GROUPS[setting]
            val = df_out.loc[df_out["site_id"].isin(val_group)].copy()
            train = df_out.loc[~df_out["site_id"].isin(test_group + val_group)].copy()
            
        elif validation_split == 'iid':
            # stratified random split of remaining sites into train/val
            train_val_pool = df_out.loc[~df_out["site_id"].isin(test_group)].copy()
            # Perform a stratified random split: every site is in both sets
            train, val = train_test_split(
                train_val_pool, 
                test_size=1/8,
                random_state=42,
                stratify=train_val_pool['site_id']
            )
            
        elif validation_split == 'temporal':
            train = df_out.loc[(~df_out["site_id"].isin(test_group)) & (df_out["year"] < 2022)].copy()
            val = df_out.loc[(~df_out["site_id"].isin(test_group)) & (df_out["year"] == 2022)].copy()

        elif validation_split == 'oracle':
            train = df_out.loc[~df_out["site_id"].isin(test_group)].copy()
            test_pool = df_out.loc[df_out["site_id"].isin(test_group)].copy()
            val, _ = train_test_split(
                test_pool, 
                train_size=0.10,     # Get 10% for validation
                random_state=42,
                stratify=test_pool['site_id']
            )
            test = df_out.loc[df_out["site_id"].isin(test_group)].drop(val.index).copy()
        
        if test.shape[0] == 0:
            logger.warning(f"* SKIPPING {test_group}: no test data")
            raise ValueError(f"No test data for group {test_group}")
    del df_out

    #  for columns GPP, NEE, ET, make the values np.nan where qc_mask==0
    for col in ["GPP", "NEE", "ET"]:
        train.loc[train["qc_mask"] == 0, col] = np.nan
        val.loc[val["qc_mask"] == 0, col] = np.nan
        if remove_missing_target:
            train = train.dropna(subset=[col])
            val = val.dropna(subset=[col])

    # ensure no row has missing values (excluding target if remove_missing_target is False)
    feature_cols = [col for col in train.columns if col not in ['GPP', 'NEE', 'ET']]
    incomplete_train = train[feature_cols].isna().any(axis=1).sum()
    incomplete_val = val[feature_cols].isna().any(axis=1).sum()
    incomplete_test = test[feature_cols].isna().any(axis=1).sum()
    assert incomplete_train == incomplete_val == incomplete_test == 0, \
        f"Expected no missing values in features, but found {incomplete_train} in train and {incomplete_val} in val, and {incomplete_test} in test"

    # clean up
    if setting == "time-split":
        env_col = "site_year"
    else:
        env_col = "site_id"
    envs_train = train[env_col]
    envs_val = val[env_col].copy()
    envs_test = test[env_col].copy()

    # Extract metadata before dropping columns
    sites_test = test["site_id"].copy()
    times_test = time_col.loc[test.index]

    for col in ["site_id", "year", "site_year", "qc_mask"]:
        if col in train.columns:
            train = train.drop(columns=[col])
            val = val.drop(columns=[col])
            test = test.drop(columns=[col])
    train = train.astype(np.float64)
    val = val.astype(np.float64)
    test = test.astype(np.float64) 

    xcols = ~train.columns.isin(['GPP', 'NEE', 'ET'])
    ycol = train.columns == target

    # split into x,y
    xtrain, ytrain = train.values[:, xcols], train.values[:, ycol].ravel()
    xval, yval = val.values[:, xcols], val.values[:, ycol].ravel()
    xtest, ytest = test.values[:, xcols], test.values[:, ycol].ravel()

    if standardize:
        scaler = RobustScaler()
        xtrain = scaler.fit_transform(xtrain)
        xval = scaler.transform(xval)
        xtest = scaler.transform(xtest)

    if astorch:
        xtrain = torch.tensor(xtrain, dtype=torch.float32)
        ytrain = torch.tensor(ytrain, dtype=torch.float32).view(-1, 1)
        xval = torch.tensor(xval, dtype=torch.float32)
        yval = torch.tensor(yval, dtype=torch.float32).view(-1, 1)
        xtest = torch.tensor(xtest, dtype=torch.float32)
        ytest = torch.tensor(ytest, dtype=torch.float32).view(-1, 1)

    out = (
        (xtrain, ytrain, envs_train), 
        (xval, yval, envs_val),
        (xtest, ytest, envs_test, sites_test, times_test)
    )
    if return_colnames:
        out = out + (train.columns[xcols].tolist(), train.columns[ycol].tolist()[0])
    return out


# -----------------------------------------------------------------------
# ------------------ Sequence (LSTM) split + windowing ------------------
# -----------------------------------------------------------------------
# The flat baselines treat every hourly row independently. The LSTM instead
# consumes fixed-length windows of the hourly series. The helpers below build
# those windows *within* each (site, split) block so a window never crosses a
# site boundary (nor, for the temporal setting, a train/val/test year boundary).
#
# Any seq2seq net can reuse these helpers, but the eval tiling is causal: each
# step gets `warmup` steps of past context only, so bidirectional/attention
# models would need different windowing.

# Non-target / non-covariate columns that must never be fed to the model.
_NON_FEATURE_COLS = ['time', 'site_id', 'year', 'qc_mask',
                     'tower_lat', 'tower_lon', 'GPP', 'NEE', 'ET']


def _train_window_starts(length, window, stride):
    """Start offsets of overlapping training windows within one site block.

    Overlapping (stride < window) windows augment training: each timestep is
    seen in several windows with a different amount of preceding context. The
    final window is right-aligned so the tail of the series is not dropped.
    Sites shorter than one window yield a single (zero-padded) window.
    """
    if length <= 0:
        return []
    if length < window:
        return [0]
    starts = list(range(0, length - window + 1, stride))
    if starts[-1] != length - window:
        starts.append(length - window)
    return starts


def _eval_window_specs(length, window, warmup, cover_warmup=False):
    """Non-overlapping *coverage* of one site block for evaluation.

    Each returned (start, own_lo, own_hi) window emits predictions only for the
    slice [own_lo:own_hi] (window-local coords); the leading `warmup` steps of
    every window are context that spins up the hidden state. Consecutive windows
    are placed so their emitted slices tile the covered region exactly once.

    By default coverage is [warmup, length): every predicted step gets a full
    warmup of preceding context, and the first `warmup` steps are left
    unpredicted. With cover_warmup=True, coverage is [0, length) instead -- the
    first window emits from step 0, so the earliest steps are predicted with less
    than a full warmup of context. This is used to score the test set on EVERY
    row (matching the flat baselines), where a handful of low-context predictions
    at each site's start is preferable to leaving those rows unscored.
    """
    specs = []
    if length == 0:
        return specs
    if not cover_warmup and length <= warmup:
        return specs  # too short to emit anything after warmup
    owned = 0 if cover_warmup else warmup  # next absolute step needing a prediction
    while owned < length:
        start = max(0, owned - warmup)
        end = min(start + window, length)
        specs.append((start, owned - start, end - start))  # (start, own_lo, own_hi)
        owned = end
    return specs


def _build_site_blocks(split_df, feature_cols, target, scaler):
    """Turn one split's dataframe into per-site, time-ordered arrays.

    Returns a list of dicts (one per site) with standardized features, two target
    arrays, a boolean validity mask, timestamps and the site id. Sites are ordered
    deterministically.

    Two targets are kept:
      * 'target'      -- NaN wherever qc_mask is False (not a measured value). Used
                         for the training loss so the model never learns from
                         gap-filled (imputed) values.
      * 'target_eval' -- the raw target column, keeping the gap-filled value at
                         qc_mask == 0 steps. Used to score the test set on EVERY
                         row, exactly like the flat baselines (see _build_eval).
    """
    blocks = []
    for site_id, g in split_df.groupby('site_id', sort=True):
        g = g.sort_values('time')
        feats = scaler.transform(g[feature_cols].values).astype(np.float32)
        valid = g['qc_mask'].values.astype(bool)   # True == measured target
        tgt_eval = g[target].values.astype(np.float32).copy()  # keeps gap-filled
        tgt = tgt_eval.copy()
        tgt[~valid] = np.nan                        # never impute the training target
        blocks.append({
            'site_id': site_id,
            'feats': feats,
            'target': tgt,
            'target_eval': tgt_eval,
            'valid': valid,
            'time': g['time'].values,
            'year': g['year'].values,
        })
    return blocks


def get_sequence_split(df, setting, path, target="GPP", validation_split='default',
                       window=720, warmup=168, train_stride=168,
                       return_colnames=False):
    """Sequence-windowed analogue of get_data_split for the LSTM baseline.

    Reuses the identical train/val/test site (or year) membership, the same
    covariates, the same qc_mask target convention and a train-fit RobustScaler.
    The returned tuples mirror get_data_split's shape so train_model.py can drive
    the LSTM through the same loop:
        train = (Xtrain, ytrain, envs_train)
        val   = (Xval,   yval,   envs_val)
        test  = (Xtest,  ytest,  envs_test, sites_test, times_test)
    Here each X* is a dict of windows/blocks consumed by models.lstm.LSTMRegressor,
    while y*/envs*/sites*/times* are flat arrays aligned to the model's
    per-timestep predictions.

    Scored rows match the flat path. The test set is scored on EVERY row --
    gap-filled (qc_mask == 0) targets and the first `warmup` steps included --
    so per-site RMSE is directly comparable to the flat baselines. As with the
    flat baselines, the TRAINING loss and the validation set used for model
    selection stay measured-only (qc_mask == 1), so the model never learns from
    (or is tuned on) imputed values; only the reported test metric spans all rows.
    See _build_eval and _build_site_blocks.
    """
    if validation_split != 'default':
        raise NotImplementedError(
            "Sequence (LSTM) split only supports validation_split='default'")
    if window <= warmup:
        # eval-window tiling can't advance when window <= warmup (infinite loop).
        raise ValueError(
            f"window ({window}) must be strictly greater than warmup ({warmup})")

    df = df.copy()
    if setting == "time-split":
        sites_to_keep = pd.read_csv(os.path.join(path, "sites_with_2018.csv"))
        df = df.loc[df["site_id"].isin(sites_to_keep['site_id'].values)].copy()

    # --- identical membership to the flat path -------------------------------
    if setting == "time-split":
        train_df = df.loc[df["year"] < 2018].copy()
        val_df = df.loc[df["year"] == 2018].copy()
        test_df = df.loc[df["year"] > 2018].copy()
        env_is_site_year = True
    else:
        if setting not in _SPATIAL_TEST_GROUPS:
            raise ValueError(f"Setting `{setting}` not recognized in get_sequence_split")
        test_group = _SPATIAL_TEST_GROUPS[setting]
        val_group = _SPATIAL_VAL_GROUPS[setting]
        train_df = df.loc[~df["site_id"].isin(test_group + val_group)].copy()
        val_df = df.loc[df["site_id"].isin(val_group)].copy()
        test_df = df.loc[df["site_id"].isin(test_group)].copy()
        env_is_site_year = False

    feature_cols = [c for c in df.columns if c not in _NON_FEATURE_COLS]

    # RobustScaler fit on training covariates only. To match the flat baselines
    # exactly, fit on measured (qc_mask == 1) rows: the flat path drops
    # gap-filled-target rows via remove_missing_target BEFORE fitting the scaler,
    # so its median/IQR come from measured steps only. Features are complete
    # regardless of qc_mask, so this is purely about which rows define the stats.
    train_measured = train_df.loc[train_df['qc_mask'].astype(bool)]
    scaler = RobustScaler().fit(train_measured[feature_cols].values)

    train_blocks = _build_site_blocks(train_df, feature_cols, target, scaler)
    val_blocks = _build_site_blocks(val_df, feature_cols, target, scaler)
    test_blocks = _build_site_blocks(test_df, feature_cols, target, scaler)

    n_features = len(feature_cols)
    common = {'window': window, 'warmup': warmup, 'n_features': n_features}

    # --- training windows (overlapping) --------------------------------------
    train_index = []
    for si, b in enumerate(train_blocks):
        length = b['feats'].shape[0]
        for start in _train_window_starts(length, window, train_stride):
            # Skip windows with no valid target after warmup (nothing to learn from).
            end = min(start + window, length)
            if b['valid'][start + min(warmup, end - start):end].any():
                train_index.append((si, start))
    Xtrain = {**common, 'blocks': train_blocks, 'window_index': train_index,
              'train_stride': train_stride}

    def _build_eval(blocks, score_all=False):
        """Eval X-dict + flat metadata aligned to predictions.

        score_all=False (validation, model selection): scores only measured
        (qc_mask == 1) steps at t >= warmup -- the same rows the training loss
        sees, so hyperparameter selection is measured-only.

        score_all=True (test): scores EVERY step against 'target_eval', including
        gap-filled (qc_mask == 0) targets and the first `warmup` steps. This
        matches the flat baselines, whose test set keeps every row (only
        train/val drop gap-filled targets via remove_missing_target).
        """
        flat_index, flat_y, flat_env, flat_site, flat_time = [], [], [], [], []
        eval_windows = []
        for si, b in enumerate(blocks):
            length = b['feats'].shape[0]
            for (start, own_lo, own_hi) in _eval_window_specs(
                    length, window, warmup, cover_warmup=score_all):
                eval_windows.append((si, start, own_lo, own_hi))
            tgt = b['target_eval'] if score_all else b['target']
            first_t = 0 if score_all else warmup
            for t in range(first_t, length):
                # score_all: every step with a (measured or gap-filled) target;
                # otherwise: only measured (qc_mask == 1) steps.
                keep = not np.isnan(tgt[t]) if score_all else b['valid'][t]
                if keep:
                    flat_index.append((si, t))
                    flat_y.append(tgt[t])
                    flat_site.append(b['site_id'])
                    flat_time.append(b['time'][t])
                    flat_env.append((b['site_id'], int(b['year'][t]))
                                    if env_is_site_year else b['site_id'])
        X = {**common, 'blocks': blocks, 'eval_windows': eval_windows,
             'flat_index': flat_index}
        y = np.asarray(flat_y, dtype=np.float32)
        return X, y, pd.Series(flat_env), pd.Series(flat_site), pd.Series(flat_time)

    Xval, yval, envs_val, _, _ = _build_eval(val_blocks, score_all=False)
    Xtest, ytest, envs_test, sites_test, times_test = _build_eval(
        test_blocks, score_all=True)

    # ytrain is unused by the LSTM (targets live inside the windows) but returned
    # for signature parity with get_data_split.
    ytrain = np.concatenate([b['target'][b['valid']] for b in train_blocks]) \
        if train_blocks else np.empty(0, dtype=np.float32)
    # env matches the flat path: (site_id, year) for time-split, else site_id.
    # Aligned to ytrain (same per-block, valid-timestep order).
    if env_is_site_year:
        envs_train = pd.Series(
            [(b['site_id'], int(y))
             for b in train_blocks for y in b['year'][b['valid']]])
    else:
        envs_train = pd.Series(
            [b['site_id']
             for b in train_blocks for _ in range(int(b['valid'].sum()))])

    logger.info(
        f"[sequence] {setting}/{target}: train windows={len(train_index)}, "
        f"val steps={len(yval)}, test steps={len(ytest)} "
        f"(window={window}, warmup={warmup}, train_stride={train_stride})")

    out = (
        (Xtrain, ytrain, envs_train),
        (Xval, yval, envs_val),
        (Xtest, ytest, envs_test, sites_test, times_test),
    )
    if return_colnames:
        out = out + (feature_cols, target)
    return out


# -----------------------------------------------------------------------
# -------------------------- Predictions I/O ----------------------------
# -----------------------------------------------------------------------


def load_predictions(setting, target, model_name, val_strategy):
    """
    Load predictions file for a given experiment.

    Args:
        setting: Experiment setting (e.g., 'spatial-easy', 'time-split')
        target: Target variable (e.g., 'GPP', 'NEE')
        model_name: Model name (e.g., 'lr', 'xgb')
        val_strategy: Validation strategy used for model selection ('mean', 'max', 'discrepancy')

    Returns:
        pd.DataFrame with y_true, y_pred, and env columns
    """
    pred_path = get_predictions_path(setting, target, model_name, val_strategy)
    df = load_csv(pred_path)
    if df is None:
        raise FileNotFoundError(f"Predictions file not found: {pred_path}")
    return df


def save_predictions(test, ypred, setting, target, model_name, val_strategy):
    """Save predictions DataFrame to CSV."""
    # TODO: add mask?
    xtest, ytest, envs_test, sites_test, times_test = test
    predictions_df = pd.DataFrame({
        'y_true': ytest.ravel(),
        'y_pred': ypred,
        'env': envs_test,
        'site_id': sites_test,
        'time': times_test,
        # 'mask': mask,
    })

    pred_path = get_predictions_path(setting, target, model_name, val_strategy)
    save_csv(predictions_df, pred_path)
    return predictions_df