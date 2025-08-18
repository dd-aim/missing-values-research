from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd

from .model import MissingEstimator
from .missingness import apply_missingness

# Default set similar to your OpenML script, now including COMPASS
DEFAULT_IMPUTERS = [
    "zero",
    "mean",
    "knn",
    "iterative",
    "promissing",
    "mpromissing",
    "compass",
]


def _normalize_mechanisms(mechanisms: Sequence[str]) -> List[str]:
    m = [str(x).upper() for x in mechanisms]
    for mech in m:
        if mech not in {"MCAR", "MAR", "MNAR"}:
            raise ValueError(
                f"Unknown mechanism '{mech}'. Must be one of ['MCAR','MAR','MNAR']"
            )
    return m


def _validate_fractions(fracs: Sequence[float]) -> List[float]:
    out = [float(f) for f in fracs]
    for f in out:
        if not (0.0 < f < 1.0):
            raise ValueError("All fractions must be in (0,1)")
    return out


def run_missing_benchmark(
    X_train: Union[pd.DataFrame, np.ndarray],
    y_train: Union[pd.Series, np.ndarray],
    X_test: Union[pd.DataFrame, np.ndarray],
    y_test: Union[pd.Series, np.ndarray],
    fractions: Sequence[float],
    mechanisms: Sequence[str],
    n_runs: int = 5,
    data_augmentation: float = 0.0,
    imputers: Optional[Sequence[str]] = None,
    estimator_kwargs: Optional[Dict] = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Benchmark MissingEstimator across MCAR/MAR/MNAR test corruptions.

    For each run:
      • Optionally augment the *training* data if 0 < data_augmentation < 1
        using the estimator's augmentation wrapper (which delegates to utils).
      • Train one model per imputer.
      • Score on clean test (baseline).
      • For each (mechanism, fraction) pair, corrupt a fresh copy of the test
        set with jenga and score again.

    Returns
    -------
    pd.DataFrame
        Aggregated mean ± std metrics per (imputer, mechanism, fraction).
        Columns are:
          ['imputer','mechanism','fraction', '<metric>_mean','<metric>_std', ...]
    """
    if imputers is None:
        imputers = DEFAULT_IMPUTERS
    fractions = _validate_fractions(fractions)
    mechanisms = _normalize_mechanisms(mechanisms)
    estimator_kwargs = dict(estimator_kwargs or {})

    runs_rows: List[dict] = []

    for run in range(int(n_runs)):
        base_seed = int(random_state) + run

        # Optionally augment the training set once per run
        Xtr_aug, ytr_aug = X_train, y_train
        use_aug = 0.0 < float(data_augmentation) < 1.0
        if use_aug:
            tmp_est = MissingEstimator(
                **{
                    **dict(imputer_name="mean", random_state=base_seed),
                    **estimator_kwargs,
                }
            )
            Xtr_aug, ytr_aug = tmp_est.augment_training_data(
                X=X_train,
                y=y_train,
                augmentation_fraction=float(data_augmentation),
                exclude_columns=None,
                random_state=base_seed,
            )

        for imp in imputers:
            est = MissingEstimator(
                **{**dict(imputer_name=imp, random_state=base_seed), **estimator_kwargs}
            )
            try:
                est.fit(Xtr_aug, ytr_aug)
                # Baseline on clean test
                base_scores = est.score(X_test, y_test)
                runs_rows.append(
                    dict(
                        run=run,
                        imputer=imp,
                        mechanism="BASELINE",
                        fraction=0.0,
                        **{f"metric_{k}": v for k, v in base_scores.items()},
                    )
                )

                # Corrupt-and-score
                for mech in mechanisms:
                    for frac in fractions:
                        seed = base_seed + hash((mech, float(frac), imp)) % 10_000_000
                        X_test_cor = apply_missingness(
                            X_test,
                            fraction=float(frac),
                            mechanism=mech,
                            random_state=seed,
                        )
                        scores = est.score(X_test_cor, y_test)
                        runs_rows.append(
                            dict(
                                run=run,
                                imputer=imp,
                                mechanism=mech,
                                fraction=float(frac),
                                **{f"metric_{k}": v for k, v in scores.items()},
                            )
                        )
            except Exception as e:
                # Skip unsupported configurations (e.g., COMPASS on regression/multiclass)
                runs_rows.append(
                    dict(
                        run=run,
                        imputer=imp,
                        mechanism="ERROR",
                        fraction=np.nan,
                        metric_error=str(e),
                    )
                )
                continue

    runs_df = pd.DataFrame(runs_rows)

    # Identify metric columns
    metric_cols = [c for c in runs_df.columns if c.startswith("metric_")]
    # Filter out error rows for aggregation
    agg_input = runs_df[runs_df["mechanism"] != "ERROR"]

    if agg_input.empty:
        return agg_input

    # Aggregate mean/std by imputer, mechanism, fraction
    agg = agg_input.groupby(["imputer", "mechanism", "fraction"], as_index=False)[
        metric_cols
    ].agg(["mean", "std"])
    # Flatten columns
    agg.columns = [
        "_".join([c for c in col if c]) if isinstance(col, tuple) else col
        for col in agg.columns
    ]
    # Clean names like 'metric_accuracy_mean' -> 'accuracy_mean'
    rename = {c: c.replace("metric_", "") for c in agg.columns}
    agg = agg.rename(columns=rename)

    return agg


if __name__ == "__main__":
    # Minimal debug runs: XOR, one OpenML classification, one OpenML regression
    import logging
    from pathlib import Path
    from sklearn.model_selection import train_test_split

    from .xor import generate_xor
    from .openml import fetch_single_dataset_openml

    logging.basicConfig(level=logging.DEBUG)

    def is_classification_target(y: Union[pd.Series, np.ndarray]) -> bool:
        s = pd.Series(y)
        if pd.api.types.is_numeric_dtype(s.dtype):
            vals = s.dropna().unique()
            return len(vals) <= 20  # heuristic: small unique count => classification
        return True  # non-numeric -> classification

    def split_xy_from_df(
        df: pd.DataFrame,
        target_col: str = "target",
        test_size: float = 0.3,
        seed: int = 0,
    ):
        y = df[target_col].to_numpy()
        X = df.drop(columns=[target_col]).to_numpy()
        strat = y if is_classification_target(y) else None
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=test_size, random_state=seed, stratify=strat
        )
        return X_tr, y_tr, X_te, y_te

    def split_xy_from_arrays(
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        test_size: float = 0.3,
        seed: int = 0,
    ):
        y_arr = pd.Series(y).to_numpy()
        strat = y_arr if is_classification_target(y_arr) else None
        X_tr, X_te, y_tr, y_te = train_test_split(
            np.asarray(X), y_arr, test_size=test_size, random_state=seed, stratify=strat
        )
        return X_tr, y_tr, X_te, y_te

    # Common benchmark settings (fast)
    fractions = [0.1, 0.3, 0.5]
    mechanisms = ["MCAR", "MAR", "MNAR"]
    estimator_kwargs = dict(epochs=5, batch_size=32, hidden_dims=(8,), verbose=False)
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1) XOR dataset (binary classification)
    xor_df = generate_xor(n_samples=2000, noise_var=0.25, random_state=0)
    X_tr, y_tr, X_te, y_te = split_xy_from_df(
        xor_df, target_col="target", test_size=0.3, seed=0
    )
    xor_res = run_missing_benchmark(
        X_train=X_tr,
        y_train=y_tr,
        X_test=X_te,
        y_test=y_te,
        fractions=fractions,
        mechanisms=mechanisms,
        n_runs=1,
        data_augmentation=0.0,
        imputers=["mean"],  # keep it simple for debugging
        estimator_kwargs=estimator_kwargs,
        random_state=0,
    )
    print("XOR benchmark results:\n", xor_res.head())
    xor_res.to_csv(results_dir / "debug_xor_results.csv", index=False)

    # 2) OpenML classification dataset (e.g., 'phoneme')
    ds_cls = fetch_single_dataset_openml(
        name="phoneme", cache=True, cache_dir="./openml_cache"
    )
    print(ds_cls)
    Xc, yc = ds_cls.data, ds_cls.target
    X_tr, X_te, y_tr, y_te = train_test_split(
        Xc, yc, test_size=0.3, random_state=1, stratify=yc
    )
    cls_res = run_missing_benchmark(
        X_train=X_tr,
        y_train=y_tr,
        X_test=X_te,
        y_test=y_te,
        fractions=fractions,
        mechanisms=mechanisms,
        n_runs=1,
        data_augmentation=0.0,
        imputers=["mean"],
        estimator_kwargs=estimator_kwargs,
        random_state=1,
    )
    print("OpenML classification ('phoneme') results:\n", cls_res.head())
    cls_res.to_csv(results_dir / "debug_openml_cls_results.csv", index=False)

    # 3) OpenML regression dataset (e.g., 'kin8nm')
    ds_reg = fetch_single_dataset_openml(
        name="kin8nm", cache=True, cache_dir="./openml_cache"
    )
    Xr, yr = ds_reg.data, ds_reg.target
    X_tr, X_te, y_tr, y_te = train_test_split(Xr, yr, test_size=0.3, random_state=2)
    reg_res = run_missing_benchmark(
        X_train=X_tr,
        y_train=y_tr,
        X_test=X_te,
        y_test=y_te,
        fractions=fractions,
        mechanisms=mechanisms,
        n_runs=1,
        data_augmentation=0.0,
        imputers=["mean"],
        estimator_kwargs=estimator_kwargs,
        random_state=2,
    )
    print("OpenML regression ('kin8nm') results:\n", reg_res.head())
    reg_res.to_csv(results_dir / "debug_openml_reg_results.csv", index=False)
