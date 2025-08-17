"""
Run a small OpenML benchmark for the custom MissingEstimator.
"""

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


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _encode_date_column(col: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert a date column to a numerical representation.
    """
    # First convert to datetime
    df[col] = pd.to_datetime(df[col], errors="coerce")

    df[f"{col}_year"] = df[col].dt.year
    df[f"{col}_month"] = df[col].dt.month
    df[f"{col}_day"] = df[col].dt.day
    df[f"{col}_dayofweek"] = df[col].dt.dayofweek  # 0=Monday
    df[f"{col}_hour"] = df[col].dt.hour
    df[f"{col}_minute"] = df[col].dt.minute
    df[f"{col}_second"] = df[col].dt.second
    df[f"{col}_is_weekend"] = df[col].dt.dayofweek.isin([5, 6]).astype(int)
    return df.drop(columns=[col])  # remove original date column


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
    print(
        f"  ── Categorical columns: {categorical_cols}"
        f"\n  ── Numeric columns: {numeric_cols}"
    )

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
        print(f"  ── Categorical columns detected as dates: {date_categorical_cols}")
        # Remove from categorical_cols and handle separately
        categorical_cols = [
            c for c in categorical_cols if c not in date_categorical_cols
        ]
        # Process each date column
        for col in date_categorical_cols:
            train_X = _encode_date_column(col, train_X)
            test_X = _encode_date_column(col, test_X)

    # one-hot (dense so we can wrap in a DataFrame)
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

    # z-score
    if numeric_cols:
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

    print("▼ Downloading OpenML datasets (this is cached on first run)…")
    datasets = fetch_datasets_openml(
        list(all_datasets.values()), cache=True, cache_dir=Path("./openml_cache")
    )

    for ds_name, ds_id in all_datasets.items():
        print(f"════════ Dataset: {ds_name}  (id={ds_id})")
        ds = datasets[ds_id]
        # Save the combined dataset as a CSV for debugging
        combined_df = pd.concat([ds.frame, ds.target.rename("target")], axis=1)
        combined_df.to_csv("debug.csv", index=False)
        print("✓ Datasets ready\n")
        task = determine_task_type(ds)  # 'classification' / 'regression'
        y = ds.target.copy(deep=True)
        X = ds.frame.drop(columns=y.name)

        if task == "classification" and not pd.api.types.is_numeric_dtype(y):
            y = y.astype("category").cat.codes  # make labels integer

        for imp in ["zero", "mean", "knn", "iterative", "promissing", "mpromissing"]:
            print(f"  ─ imputer = {imp}")

            stats = {"acc": [], "auc": [], "r2": []}

            for split_seed in tqdm(range(5), leave=False):
                # 50 % hold-out split
                train_idx = X.sample(frac=0.5, random_state=split_seed).index
                X_train, y_train = X.loc[train_idx], y.loc[train_idx]
                X_test, y_test = X.drop(train_idx), y.drop(train_idx)

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
                print(f"      accuracy : {pretty_stats(stats['acc'])}")
                print(f"      ROC-AUC  : {pretty_stats(stats['auc'])}\n")
            else:
                print(f"      R² score : {pretty_stats(stats['r2'])}\n")
