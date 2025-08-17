"""
Run a small OpenML benchmark for the custom MissingEstimator.
"""

import logging
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from missing_vals.openml import (
    determine_task_type,
    fetch_datasets_openml,
    DATASETS_REG,
    DATASETS_CLS,
)
from missing_vals.utils import augment_with_missing_values
from missing_vals.model import MissingEstimator
import json

# --------------------------------------------------------------------------- #
# Logging setup
# --------------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _sin(x, period):
    return np.sin(2 * np.pi * x / period)


def _cos(x, period):
    return np.cos(2 * np.pi * x / period)


def _encode_date_column(col: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert a date column to cyclical numerical features (sine/cosine).
    Adds:
      {col}_month_{sin,cos}, {col}_dow_{sin,cos}, {col}_hour_{sin,cos},
      {col}_minute_{sin,cos}, {col}_doy_{sin,cos}, {col}_is_weekend
    Drops the original date column.
    """
    df = df.copy()
    s = pd.to_datetime(df[col], errors="coerce")

    # Components (as floats to allow NaN)
    month = s.dt.month.astype("float")  # 1..12
    dow = s.dt.dayofweek.astype("float")  # 0..6 (Mon=0)
    hour = s.dt.hour.astype("float")  # 0..23
    minute = s.dt.minute.astype("float")  # 0..59
    doy = s.dt.dayofyear.astype("float")  # 1..365/366

    # Cyclical encodings
    df[f"{col}_month_sin"] = _sin(month - 1, 12)  # shift to 0..11
    df[f"{col}_month_cos"] = _cos(month - 1, 12)

    df[f"{col}_dow_sin"] = _sin(dow, 7)
    df[f"{col}_dow_cos"] = _cos(dow, 7)

    df[f"{col}_hour_sin"] = _sin(hour, 24)
    df[f"{col}_hour_cos"] = _cos(hour, 24)

    df[f"{col}_minute_sin"] = _sin(minute, 60)
    df[f"{col}_minute_cos"] = _cos(minute, 60)

    # Use 365.25 to smooth across leap years (doy is 1..366)
    df[f"{col}_doy_sin"] = _sin(doy - 1, 365.25)
    df[f"{col}_doy_cos"] = _cos(doy - 1, 365.25)

    # Useful binary feature
    df[f"{col}_is_weekend"] = s.dt.dayofweek.isin([5, 6]).astype("Int8")

    return df.drop(columns=[col])


def transform_data(
    train_X: pd.DataFrame, test_X: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    One-hot encode categoricals and z-score numerical features.
    The target **must already have been removed**.
    """
    categorical_cols = [
        c for c in train_X.columns if isinstance(train_X[c].dtype, pd.CategoricalDtype)
    ]
    numeric_cols = [c for c in train_X.columns if c not in categorical_cols]
    logger.debug(f"Categorical columns: {categorical_cols}")
    logger.debug(f"Numeric columns: {numeric_cols}")

    # Detect categorical columns that are actually dates
    date_categorical_cols = []
    for col in categorical_cols:
        # Try converting to datetime; if many values succeed, treat as date
        try:
            parsed = pd.to_datetime(train_X[col], errors="coerce")
            # If more than half the values are parsed as dates, consider it a date column
            if parsed.notna().mean() > 0.5:
                date_categorical_cols.append(col)
        except Exception:
            continue

    if date_categorical_cols:
        logger.info(f"Categorical columns detected as dates: {date_categorical_cols}")
        # Remove from categorical_cols and handle separately
        categorical_cols = [
            c for c in categorical_cols if c not in date_categorical_cols
        ]
        # Process each date column
        for col in date_categorical_cols:
            # Optionally encode date columns here
            logger.debug(f"Dropping date column: {col}")
            train_X = train_X.drop(columns=col)
            test_X = test_X.drop(columns=col)

    # one-hot (dense so we can wrap in a DataFrame)
    if categorical_cols:
        logger.debug(f"One-hot encoding columns: {categorical_cols}")
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        train_cat = ohe.fit_transform(train_X[categorical_cols])
        test_cat = ohe.transform(test_X[categorical_cols])

        cat_names = ohe.get_feature_names_out(categorical_cols)
        train_X = train_X.drop(columns=categorical_cols)
        test_X = test_X.drop(columns=categorical_cols)

        train_X = pd.concat(
            [
                train_X.reset_index(drop=True),
                pd.DataFrame(train_cat, columns=cat_names),
            ],
            axis=1,
        )
        test_X = pd.concat(
            [test_X.reset_index(drop=True), pd.DataFrame(test_cat, columns=cat_names)],
            axis=1,
        )

    # z-score
    if numeric_cols:
        logger.debug(f"Standardizing numeric columns: {numeric_cols}")
        scaler = StandardScaler()
        train_X[numeric_cols] = scaler.fit_transform(train_X[numeric_cols])
        test_X[numeric_cols] = scaler.transform(test_X[numeric_cols])

    return train_X, test_X


def append_score(store: dict[str, list[float]], task: str, scores: dict) -> None:
    """Collect the relevant metric for later averaging."""
    if task == "classification":  # accuracy and AUC
        store["acc"].append(scores["accuracy"])
        store["auc"].append(scores.get("roc_auc", np.nan))
    else:  # regression – R²
        store["r2"].append(scores["r2"])


def pretty_stats(values: list[float]) -> str:
    return f"{np.mean(values):.4f} ± {np.std(values):.4f}"


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    all_datasets = {**DATASETS_REG, **DATASETS_CLS}

    logger.info("▼ Downloading OpenML datasets (this is cached on first run)…")
    datasets = fetch_datasets_openml(
        list(all_datasets.values()), cache=True, cache_dir=Path("./openml_cache")
    )

    for ds_name, ds_id in all_datasets.items():
        logger.info(f"════════ Dataset: {ds_name}  (id={ds_id})")
        ds = datasets[ds_id]
        # Save the combined dataset as a CSV for debugging
        combined_df = pd.concat([ds.frame, ds.target.rename("target")], axis=1)
        combined_df.to_csv("debug.csv", index=False)
        logger.debug("Saved combined dataset to debug.csv")
        logger.info("✓ Datasets ready")
        task = determine_task_type(ds)  # 'classification' / 'regression'
        y = ds.target.copy(deep=True)
        X = ds.frame.drop(columns=y.name)
        error_datasets = []
        if task == "classification" and not pd.api.types.is_numeric_dtype(y):
            logger.debug(
                "Converting target to categorical codes for classification task."
            )
            y = y.astype("category").cat.codes  # make labels integer

        for imp in ["zero", "mean", "knn", "iterative", "promissing", "mpromissing"]:
            logger.info(f"Imputer: {imp}")

            stats = {"acc": [], "auc": [], "r2": []}

            # for split_seed in tqdm(range(5), leave=False):
            for split_seed in tqdm(range(1), leave=False):
                logger.debug(f"Split seed: {split_seed}")
                # 50 % hold-out split
                train_idx = X.sample(frac=0.5, random_state=split_seed).index
                X_train, y_train = X.loc[train_idx], y.loc[train_idx]
                X_test, y_test = X.drop(train_idx), y.drop(train_idx)

                logger.debug(f"Train size: {len(train_idx)}, Test size: {len(X_test)}")

                # optional artificial missingness
                X_train_aug = augment_with_missing_values(
                    pd.concat([X_train, y_train.rename("target")], axis=1),
                    augmentation_fraction=0.3,
                    exclude_columns=["target"],
                    random_state=split_seed,
                )
                y_train_aug = X_train_aug.pop("target")  # remove target again
                X_train_aug, X_test_proc = transform_data(X_train_aug, X_test)

                # model
                est = MissingEstimator(
                    imputer_name=imp,
                    epochs=10,
                    early_stopping=0.1,
                    patience=20,
                    random_state=split_seed,
                    output_activation="auto",  # let the estimator decide
                )
                est.fit(X_train_aug, y_train_aug)
                sc = est.score(X_test_proc, y_test)
                append_score(stats, task, sc)
                
                # ────────────────── summary for this imputer ──────────────────
                if task == "classification":
                    logger.info(f"      accuracy : {pretty_stats(stats['acc'])}")
                    logger.info(f"      ROC-AUC  : {pretty_stats(stats['auc'])}")
                else:
                    logger.info(f"      R² score : {pretty_stats(stats['r2'])}")
        