
from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

try:
    # jenga provides the MissingValues corruption
    from jenga.corruptions.generic import MissingValues
except Exception as e:
    MissingValues = None  # type: ignore


VALID_MECHANISMS = {"MCAR", "MAR", "MNAR"}


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


def apply_missingness(
    X: Union[pd.DataFrame, np.ndarray],
    fraction: float,
    mechanism: str,
    columns: Optional[Sequence[Union[str, int]]] = None,
    per_column: bool = True,
    random_state: Optional[int] = None,
    na_value=np.nan,
) -> Union[pd.DataFrame, np.ndarray]:
    """Apply MCAR / MAR / MNAR missingness using **jenga**'s MissingValues.

    Parameters
    ----------
    X
        Data to corrupt. Accepts DataFrame or ndarray.
    fraction
        Fraction of entries *per selected column* to set missing (0 < f < 1).
    mechanism
        One of {"MCAR","MAR","MNAR"} (case-insensitive).
    columns
        Subset of columns to corrupt. If None, all columns are considered.
        If ndarray was provided and columns are ints, they refer to positions.
    per_column
        If True (default), corrupt each selected column independently with `fraction`.
        If False, the same call will still iterate the selected columns but you
        can pass a subset via `columns` if you want a single-column corruption.
    random_state
        Seed for reproducibility. We seed NumPy before jenga's transform.
    na_value
        Value to use for missing entries (default: np.nan).

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
        raise ValueError(f"Unknown mechanism '{mechanism}'. Use one of {sorted(VALID_MECHANISMS)}")

    df = _to_dataframe(X)

    # Determine columns to corrupt
    if columns is None:
        cols = list(df.columns)
    else:
        # map integer positions to names if needed
        cols = []
        for c in columns:
            if isinstance(c, int):
                cols.append(df.columns[c])
            else:
                cols.append(str(c))
        # filter out columns not present (be tolerant)
        cols = [c for c in cols if c in df.columns]

    if random_state is not None:
        np.random.seed(int(random_state))

    # Apply column-by-column
    if per_column:
        for c in cols:
            corruption = MissingValues(
                column=c,
                fraction=float(fraction),
                missingness=mech,
                na_value=na_value,
            )
            df = corruption.transform(df)
    else:
        # Still iterate, but gives you control via `columns` to pick a subset.
        for c in cols:
            corruption = MissingValues(
                column=c,
                fraction=float(fraction),
                missingness=mech,
                na_value=na_value,
            )
            df = corruption.transform(df)

    return _to_original_type(X, df)


def generate_mcar(
    X: Union[pd.DataFrame, np.ndarray],
    fraction: float,
    columns: Optional[Sequence[Union[str, int]]] = None,
    random_state: Optional[int] = None,
    na_value=np.nan,
) -> Union[pd.DataFrame, np.ndarray]:
    return apply_missingness(
        X=X,
        fraction=fraction,
        mechanism="MCAR",
        columns=columns,
        per_column=True,
        random_state=random_state,
        na_value=na_value,
    )


def generate_mar(
    X: Union[pd.DataFrame, np.ndarray],
    fraction: float,
    columns: Optional[Sequence[Union[str, int]]] = None,
    random_state: Optional[int] = None,
    na_value=np.nan,
) -> Union[pd.DataFrame, np.ndarray]:
    return apply_missingness(
        X=X,
        fraction=fraction,
        mechanism="MAR",
        columns=columns,
        per_column=True,
        random_state=random_state,
        na_value=na_value,
    )


def generate_mnar(
    X: Union[pd.DataFrame, np.ndarray],
    fraction: float,
    columns: Optional[Sequence[Union[str, int]]] = None,
    random_state: Optional[int] = None,
    na_value=np.nan,
) -> Union[pd.DataFrame, np.ndarray]:
    return apply_missingness(
        X=X,
        fraction=fraction,
        mechanism="MNAR",
        columns=columns,
        per_column=True,
        random_state=random_state,
        na_value=na_value,
    )
