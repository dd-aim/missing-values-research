from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import logging

# NEW: imports for MI-based feature ranking
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.preprocessing import LabelEncoder

try:
    # jenga provides the MissingValues corruption
    from jenga.corruptions.generic import MissingValues
except Exception as e:
    MissingValues = None  # type: ignore


logger = logging.getLogger(__name__)
VALID_MECHANISMS = {"MCAR", "MAR", "MNAR"}

# NEW HELPERS: task inference, encoding, MI ranking, and mechanism-based column picker


def _infer_task(y: pd.Series) -> str:
    """Heuristic: return 'classification' if y looks categorical / small integer classes, else 'regression'."""
    s = pd.Series(y).dropna()
    if s.empty:
        return "regression"
    if (
        pd.api.types.is_object_dtype(s)
        or pd.api.types.is_categorical_dtype(s)
        or pd.api.types.is_bool_dtype(s)
    ):
        return "classification"
    if pd.api.types.is_integer_dtype(s) or pd.api.types.is_float_dtype(s):
        arr = s.to_numpy()
        integer_like = np.all(np.isclose(arr, np.round(arr), atol=1e-12))
        if integer_like:
            vals = np.unique(np.round(arr).astype(int))
            if 2 <= vals.size <= 20 and np.array_equal(
                vals, np.arange(vals.min(), vals.max() + 1)
            ):
                return "classification"
    return "regression"


def _ordinalize_categoricals(X: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Encode object/categorical/bool columns to integer codes for MI; return (X_encoded, discrete_mask)."""
    Xc = X.copy()
    discrete_mask = np.zeros(Xc.shape[1], dtype=bool)
    for j, col in enumerate(Xc.columns):
        if (
            pd.api.types.is_bool_dtype(Xc[col])
            or pd.api.types.is_object_dtype(Xc[col])
            or pd.api.types.is_categorical_dtype(Xc[col])
        ):
            Xc[col] = pd.Categorical(Xc[col]).codes.astype(int)
            discrete_mask[j] = True
    return Xc, discrete_mask


def _mi_rank(
    X: pd.DataFrame,
    y: pd.Series,
    task: Optional[str],
    rng: np.random.RandomState,
) -> list[str]:
    """Return feature names sorted by descending MI with y (ties jittered)."""
    if task is None:
        task = _infer_task(y)
    Xenc, discrete_mask = _ordinalize_categoricals(X)
    if task == "classification":
        y_enc = y.copy()
        if not (
            pd.api.types.is_integer_dtype(y_enc) or pd.api.types.is_bool_dtype(y_enc)
        ):
            y_enc = pd.Series(
                LabelEncoder().fit_transform(y_enc.astype(str)), index=y.index
            )
        mi = mutual_info_classif(
            Xenc.values,
            y_enc.values,
            discrete_features=discrete_mask if discrete_mask.any() else "auto",
            random_state=rng,
        )
    else:
        y_cont = pd.to_numeric(y, errors="coerce")
        mi = mutual_info_regression(
            Xenc.values,
            y_cont.values,
            discrete_features=discrete_mask if discrete_mask.any() else "auto",
            random_state=rng,
        )
    mi = mi + rng.uniform(-1e-9, 1e-9, size=mi.shape)  # break ties
    order = np.argsort(-mi)
    return [X.columns[i] for i in order]


def _auto_pick_column_for_mech(
    df: pd.DataFrame,
    mech: str,
    y: Optional[Union[pd.Series, np.ndarray]],
    target_col: Optional[str],
    rng: np.random.RandomState,
    task: Optional[str] = None,
) -> str:
    """Pick a single impacted feature per the paper: MCAR=random; MAR=2nd MI; MNAR=top MI."""
    # Identify features (exclude target if present)
    feature_cols = list(df.columns)
    if target_col and target_col in feature_cols:
        feature_cols.remove(target_col)

    if len(feature_cols) == 0:
        raise ValueError("No feature columns to corrupt.")

    mech = mech.upper()
    if mech == "MCAR" or y is None:
        # If no y given, fall back to random selection for any mechanism
        return rng.choice(feature_cols)

    y_series = pd.Series(y) if not isinstance(y, pd.Series) else y
    # If y is actually a column inside df and no explicit target_col was given, don’t corrupt it
    if target_col is None and "target" in df.columns and df["target"].equals(y_series):
        target_col = "target"

    X = (
        df.drop(columns=[target_col])
        if (target_col in df.columns)
        else df[feature_cols]
    )
    rank = _mi_rank(X, y_series.loc[X.index], task, rng)
    if mech == "MNAR":
        return rank[0]  # top MI
    if mech == "MAR":
        return rank[1] if len(rank) > 1 else rng.choice(rank)  # 2nd MI, fallback random
    return rng.choice(feature_cols)


def _to_dataframe(X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X.copy(deep=True)
    elif isinstance(X, np.ndarray):
        cols = [f"x{i}" for i in range(X.shape[1])]
        return pd.DataFrame(X.copy(), columns=cols)
    else:
        raise TypeError("X must be a pandas DataFrame or a numpy ndarray")


def _to_original_type(X_like: Union[pd.DataFrame, np.ndarray], df: pd.DataFrame):
    if isinstance(X_like, pd.DataFrame):
        # align to original column order just in case
        common = [c for c in X_like.columns if c in df.columns]
        return df[common]
    else:
        return df.to_numpy()


# UPDATED: apply_missingness signature and logic


def apply_missingness(
    X: Union[pd.DataFrame, np.ndarray],
    fraction: float,
    mechanism: str,
    columns: Optional[Sequence[Union[str, int]]] = None,
    per_column: bool = False,  # default now False => one column by default
    random_state: Optional[int] = None,
    na_value=np.nan,
    *,
    y: Optional[Union[pd.Series, np.ndarray]] = None,
    target_col: Optional[str] = None,
    task: Optional[str] = None,
) -> Union[pd.DataFrame, np.ndarray]:
    """Apply MCAR / MAR / MNAR missingness using **jenga**'s MissingValues.

    Parameters
    ----------
    X
        Data to corrupt. Accepts DataFrame or ndarray.
    fraction
        Fraction of entries per selected column to set missing (0 < f < 1).
    mechanism
        One of {"MCAR","MAR","MNAR"} (case-insensitive).
    columns
        Subset of columns to corrupt. If None, a single column is auto-picked
        according to the mechanism (MCAR=random; MAR=2nd MI; MNAR=top MI).
        If ndarray was provided and columns are ints, they refer to positions.
    per_column
        If True, corrupt each selected column independently with `fraction`.
        Default False; with columns=None, exactly one column is corrupted.
    random_state
        Seed for reproducibility.
    na_value
        Value to use for missing entries (default: np.nan).
    y
        Target vector/series used for MI-based selection (optional; required for MAR/MNAR to be meaningful).
    target_col
        Name of target column if y is inside X; excluded from candidates.
    task
        Optional override for task type: "classification" or "regression".

    Returns
    -------
    Data with the same type as the input (DataFrame or ndarray).
    """
    if MissingValues is None:
        raise ImportError(
            "jenga is required for apply_missingness. Please `pip install jenga`. "
            "(We import: from jenga.corruptions.generic import MissingValues)"
        )

    if not (0.0 < float(fraction) < 1.0):
        raise ValueError("fraction must be between 0 and 1 (exclusive)")

    mech = mechanism.upper()
    if mech not in VALID_MECHANISMS:
        raise ValueError(
            f"Unknown mechanism '{mechanism}'. Use one of {sorted(VALID_MECHANISMS)}"
        )

    df = _to_dataframe(X)

    # RNG for reproducibility
    rng = (
        np.random.RandomState(int(random_state))
        if random_state is not None
        else np.random.RandomState()
    )

    # Determine columns to corrupt
    if columns is None:
        # Auto-pick exactly ONE column using the mechanism’s rule
        impacted = _auto_pick_column_for_mech(
            df=df,
            mech=mech,
            y=y,
            target_col=target_col,
            rng=rng,
            task=task,
        )
        cols = [impacted]
    else:
        # map/validate provided columns (be tolerant)
        cols = []
        for c in columns:
            if isinstance(c, int):
                cols.append(df.columns[c])
            else:
                cols.append(str(c))
        cols = [c for c in cols if c in df.columns]

    logger.info(
        "Applying missingness: mechanism=%s, fraction=%.3f, columns=%s, per_column=%s, seed=%s",
        mech,
        float(fraction),
        cols,
        per_column,
        str(random_state),
    )

    # Apply (column-by-column even if it's a single column)
    if per_column:
        targets = cols
    else:
        # Even if multiple provided, follow the same iteration; per_column controls semantics
        targets = cols

    for c in targets:
        corruption = MissingValues(
            column=c,
            fraction=float(fraction),
            missingness=mech,
            na_value=na_value,
        )
        df = corruption.transform(df)
        logger.debug(
            "Applied %s to column '%s' at fraction %.3f", mech, c, float(fraction)
        )

    return _to_original_type(X, df)


# UPDATED: wrappers to pass through y/target_col/task and stop forcing per_column=True


def generate_mcar(
    X: Union[pd.DataFrame, np.ndarray],
    fraction: float,
    columns: Optional[Sequence[Union[str, int]]] = None,
    random_state: Optional[int] = None,
    na_value=np.nan,
    *,
    y: Optional[Union[pd.Series, np.ndarray]] = None,
    target_col: Optional[str] = None,
    task: Optional[str] = None,
) -> Union[pd.DataFrame, np.ndarray]:
    return apply_missingness(
        X=X,
        fraction=fraction,
        mechanism="MCAR",
        columns=columns,
        random_state=random_state,
        na_value=na_value,
        y=y,
        target_col=target_col,
        task=task,
    )


def generate_mar(
    X: Union[pd.DataFrame, np.ndarray],
    fraction: float,
    columns: Optional[Sequence[Union[str, int]]] = None,
    random_state: Optional[int] = None,
    na_value=np.nan,
    *,
    y: Optional[Union[pd.Series, np.ndarray]] = None,
    target_col: Optional[str] = None,
    task: Optional[str] = None,
) -> Union[pd.DataFrame, np.ndarray]:
    return apply_missingness(
        X=X,
        fraction=fraction,
        mechanism="MAR",
        columns=columns,
        random_state=random_state,
        na_value=na_value,
        y=y,
        target_col=target_col,
        task=task,
    )


def generate_mnar(
    X: Union[pd.DataFrame, np.ndarray],
    fraction: float,
    columns: Optional[Sequence[Union[str, int]]] = None,
    random_state: Optional[int] = None,
    na_value=np.nan,
    *,
    y: Optional[Union[pd.Series, np.ndarray]] = None,
    target_col: Optional[str] = None,
    task: Optional[str] = None,
) -> Union[pd.DataFrame, np.ndarray]:
    return apply_missingness(
        X=X,
        fraction=fraction,
        mechanism="MNAR",
        columns=columns,
        random_state=random_state,
        na_value=na_value,
        y=y,
        target_col=target_col,
        task=task,
    )
