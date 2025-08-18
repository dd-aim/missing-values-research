"""
Run a small OpenML benchmark for the custom MissingEstimator using the centralized
run_missing_benchmark helper. Keeps existing preprocessing and dynamic architecture.
"""

from __future__ import annotations

from pathlib import Path
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

from missing_vals.openml import (
    determine_task_type,
    fetch_datasets_openml,
    DATASETS_REG,
    DATASETS_CLS,
)
from missing_vals.benchmark import run_missing_benchmark

# --------------------------------------------------------------------------- #
# Logging setup
# --------------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.WARNING,
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
    df = df.copy()
    s = pd.to_datetime(df[col], errors="coerce")

    month = s.dt.month.astype("float")
    dow = s.dt.dayofweek.astype("float")
    hour = s.dt.hour.astype("float")
    minute = s.dt.minute.astype("float")
    doy = s.dt.dayofyear.astype("float")

    df[f"{col}_month_sin"] = _sin(month - 1, 12)
    df[f"{col}_month_cos"] = _cos(month - 1, 12)

    df[f"{col}_dow_sin"] = _sin(dow, 7)
    df[f"{col}_dow_cos"] = _cos(dow, 7)

    df[f"{col}_hour_sin"] = _sin(hour, 24)
    df[f"{col}_hour_cos"] = _cos(hour, 24)

    df[f"{col}_minute_sin"] = _sin(minute, 60)
    df[f"{col}_minute_cos"] = _cos(minute, 60)

    df[f"{col}_doy_sin"] = _sin(doy - 1, 365.25)
    df[f"{col}_doy_cos"] = _cos(doy - 1, 365.25)

    df[f"{col}_is_weekend"] = s.dt.dayofweek.isin([5, 6]).astype("Int8")

    return df.drop(columns=[col])


def transform_data(
    train_X: pd.DataFrame, test_X: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    categorical_cols = [
        c for c in train_X.columns if isinstance(train_X[c].dtype, pd.CategoricalDtype)
    ]
    numeric_cols = [c for c in train_X.columns if c not in categorical_cols]

    date_categorical_cols = []
    for col in categorical_cols:
        try:
            parsed = pd.to_datetime(train_X[col], errors="coerce")
            if parsed.notna().mean() > 0.5:
                date_categorical_cols.append(col)
        except Exception:
            continue

    if date_categorical_cols:
        categorical_cols = [
            c for c in categorical_cols if c not in date_categorical_cols
        ]
        for col in date_categorical_cols:
            train_X = _encode_date_column(col, train_X)
            test_X = _encode_date_column(col, test_X)

    if categorical_cols:
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

    if numeric_cols:
        scaler = StandardScaler()
        train_X[numeric_cols] = scaler.fit_transform(train_X[numeric_cols])
        test_X[numeric_cols] = scaler.transform(test_X[numeric_cols])

    return train_X, test_X


def run_openml_experiment(
    classification_datasets: dict[str, int] = DATASETS_CLS,
    regression_datasets: dict[str, int] = DATASETS_REG,
    *,
    n_runs: int = 3,
    mechanisms: list[str] = ["MCAR", "MAR", "MNAR"],
    fractions: list[float] = [0.10, 0.25, 0.50, 0.90],
) -> None:
    """
    Run dataset-wise OpenML benchmarks using the centralized benchmark helper.

    - Tasks: classification and regression (handled per dataset)
    - Missingness: MCAR, MAR, MNAR
    - Fractions: 0.1, 0.25, 0.5, 0.9
    - Imputers: zero, mean, knn, iterative, promissing, mpromissing, compass
    - Protocol: single 50/50 split per dataset; centralized helper performs n_runs

    Model (per spec):
    - Two hidden layers: first has p/2 ReLU units (p = input dim), second has 2 units
    - Output: sigmoid (classification) or linear (regression)
    - Optimizer: SGD, epochs=100, batch_size=10, lr=0.1
    - Early stopping as requested

    Results are saved under results/openml/<task>_<dataset>/<MECHANISM>/*.csv
    """

    if not isinstance(classification_datasets, dict):
        raise ValueError("classification_datasets must be a dict[str, int]")
    if not isinstance(regression_datasets, dict):
        raise ValueError("regression_datasets must be a dict[str, int]")

    REQUESTED_IMPUTERS = [
        "zero",
        "mean",
        "knn",
        "iterative",
        "promissing",
        "mpromissing",
        "compass",
    ]

    # Model/training defaults (per spec)
    EPOCHS = 100
    BATCH_SIZE = 10
    LR = 0.01
    ACTIVATION = "relu"
    EARLY_STOPPING = 0.1
    PATIENCE = 50

    # Seeds for reproducibility (used to choose base split; run_missing_benchmark varies per run)
    base_seed = 0

    logger.info("Fetching OpenML datasets (cached if available)")
    cls_datasets = fetch_datasets_openml(
        list(classification_datasets.values()),
        cache=True,
        cache_dir=str(Path("./openml_cache")),
    )
    reg_datasets = fetch_datasets_openml(
        list(regression_datasets.values()),
        cache=True,
        cache_dir=str(Path("./openml_cache")),
    )

    def _process_dataset(task: str, name: str, ds_bunch) -> None:
        # Extract features/target
        y = ds_bunch.target.copy(deep=True)
        X = ds_bunch.frame.drop(columns=y.name)

        # Target handling
        if task == "classification" and not pd.api.types.is_numeric_dtype(y):
            y = y.astype("category").cat.codes
        elif task == "regression" and not pd.api.types.is_numeric_dtype(y):
            y = pd.to_numeric(y, errors="coerce")

        # 50/50 split (stratified for classification)
        strat = y if task == "classification" else None
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.5, random_state=base_seed, stratify=strat
        )

        # Transform features
        X_tr_proc, X_te_proc = transform_data(X_tr, X_te)
        Xtr = X_tr_proc.to_numpy()
        Xte = X_te_proc.to_numpy()
        ytr = pd.Series(y_tr).to_numpy()
        yte = pd.Series(y_te).to_numpy()

        # Dynamic hidden dims based on input dimension
        p_dim = Xtr.shape[1]
        hidden_dims_calc = (max(1, p_dim // 2), 2)

        # Estimator kwargs per dataset
        est_kwargs = dict(
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            lr=LR,
            activation=ACTIVATION,
            hidden_dims=hidden_dims_calc,
            output_activation=("sigmoid" if task == "classification" else "linear"),
            early_stopping=EARLY_STOPPING,
            patience=PATIENCE,
        )

        # Print hyperparameters to ensure alignment per dataset
        print(f"OpenML dataset '{name}' ({task}) configuration:")
        print(f"  p_dim: {p_dim}")
        print(f"  mechanisms: {mechanisms}")
        print(f"  fractions: {fractions}")
        print(f"  n_runs: {n_runs}")
        print("  estimator_kwargs:")
        for k, v in est_kwargs.items():
            print(f"    - {k}: {v}")
        print(f"  imputers: {REQUESTED_IMPUTERS}")

        # Call centralized benchmark per dataset
        run_missing_benchmark(
            X_train=Xtr,
            y_train=ytr,
            X_test=Xte,
            y_test=yte,
            fractions=fractions,
            mechanisms=mechanisms,
            n_runs=n_runs,
            imputers=REQUESTED_IMPUTERS,
            estimator_kwargs=est_kwargs,
            random_state=base_seed,
            dataset_name=f"{task}_{name}",
            results_root="results/openml",
        )

    # Process classification datasets
    for name, ds_id in classification_datasets.items():
        ds_bunch = cls_datasets.get(ds_id)
        if ds_bunch is None:
            logger.warning(
                f"Dataset id {ds_id} not fetched for classification; skipping"
            )
            continue
        _process_dataset("classification", name, ds_bunch)

    # Process regression datasets
    for name, ds_id in regression_datasets.items():
        ds_bunch = reg_datasets.get(ds_id)
        if ds_bunch is None:
            logger.warning(f"Dataset id {ds_id} not fetched for regression; skipping")
            continue
        _process_dataset("regression", name, ds_bunch)


if __name__ == "__main__":
    DATASETS_CLS = {
        "space-ga": 737,
        "pollen": 871,
        "wilt": 40983,
        "analcatdata-supreme": 728,
        "phoneme": 1489,
    }

    DATASETS_REG = {
        "stock-fardamento02": 42545,
        "delta-elevators": 198,
        "sulfur": 23515,
        "kin8nm": 189,
        "wine-quality": 287,
    }
    # Default: run with full dataset lists
    run_openml_experiment(
        regression_datasets=DATASETS_REG, classification_datasets=DATASETS_CLS
    )
