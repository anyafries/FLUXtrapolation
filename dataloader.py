import os
import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from utils.utils import setup_logging, get_predictions_path, load_csv, save_csv

logger = setup_logging(__name__)

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
    return_date=False,
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
        return_date (bool, optional): Whether to additionally return, for
            train/val/test, the observation time as days since 2015-01-01
            (float, aligned to the feature rows). Used by the anchorboosting
            model as a continuous time anchor. Appended as the last element of
            the returned tuple. Defaults to False.
    Returns:
        tuple: xtrain, ytrain, envs_train, xtest, ytest, envs_test
    """
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
        if setting == 'spatial-easy40':
            test_group = ['US-Tw1', 'DE-Hai', 'US-Seg', 'US-Sne', 'US-Tw4', 'US-xDL', 'UK-AMo', 'AU-Dry', 'US-CGG', 'FR-Bil', 'US-Rpf', 'DK-Skj', 'RU-Fy2', 'DE-Rns', 'US-Tw3', 'RU-Fyo', 'US-Snf', 'CH-Cha', 'AR-CCg', 'CL-SDF', 'DE-Gri', 'FR-Tou', 'AU-Whr', 'AU-GWW', 'US-RGo', 'IT-BCi', 'ES-Abr', 'SE-Nor', 'DE-Hzd', 'US-CS2', 'US-StJ', 'CA-TP3', 'BE-Dor', 'US-xWD', 'US-Syv', 'DE-RuR', 'CZ-BK1', 'BE-Maa', 'BE-Vie', 'FI-Var']
        elif setting == 'TA40':
            test_group = ['AU-Dry', 'AU-DaS', 'AU-Lit', 'BR-Npw', 'AU-Lon', 'AU-ASM', 'US-xDS', 'US-ONA', 'US-SP1', 'US-xJE', 'US-SRM', 'US-HB2', 'AU-GWW', 'US-SRS', 'US-SRG', 'IL-Yat', 'US-HB3', 'US-HB1', 'US-xDL', 'US-RGA', 'AU-Cum', 'US-xTA', 'AU-Cpr', 'US-Whs', 'US-Cst', 'US-Wkg', 'IT-BCi', 'US-Jo2', 'IT-Cp2', 'US-RGo', 'ES-Abr', 'US-NC4', 'ES-Agu', 'US-Akn', 'US-xJR', 'ES-Pdu', 'US-Ton', 'ES-LM2', 'IT-Noe', 'ES-LM1']
        else:
            raise ValueError(f"Setting `{setting}` not recognized in get_data_split")
        test = df_out.loc[df_out["site_id"].isin(test_group)].copy()

        # get train, val depending on validation_split strategy
        if validation_split == 'default':
            if setting == "spatial-easy40":
                val_group = ['DE-Tha', 'US-xTR', 'US-ICh', 'FR-Aur', 'US-NR1', 'CA-TPD', 'AU-Cum', 'US-RGA', 'CZ-Lnz', 'US-UC1', 'SE-Htm', 'AU-Rgf', 'ES-Agu', 'FR-Mej', 'CA-ARF', 'CA-TP1', 'CA-SCC', 'US-BZB', 'US-xCP', 'DK-Vng']
            elif setting == "TA40":
                val_group = ['US-Snf', 'US-GLE', 'US-CF2', 'FI-Let', 'CZ-Lnz', 'US-Rls', 'UK-AMo', 'FR-Gri', 'US-xTR', 'US-ALQ', 'CA-ER1', 'US-xBR', 'FI-Hyy', 'IE-Cra', 'DE-Obe', 'AU-War', 'US-RGB', 'CH-Cha', 'US-Syv', 'US-UMB']
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

    # Continuous time anchor: days since 2015-01-01, aligned to feature rows.
    dates = None
    if return_date:
        days = (
            pd.to_datetime(time_col) - pd.Timestamp("2015-01-01")
        ).dt.total_seconds() / 86400.0
        dates = (
            days.loc[train.index].to_numpy(dtype=np.float64),
            days.loc[val.index].to_numpy(dtype=np.float64),
            days.loc[test.index].to_numpy(dtype=np.float64),
        )

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
    if return_date:
        out = out + (dates,)
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