from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Union
import os
import numpy as np
import pandas as pd
import logging
from tqdm import tqdm

from .model import MissingEstimator
from .missingness import apply_missingness

# Module logger
logger = logging.getLogger(__name__)

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


def _flatten_agg_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten MultiIndex columns produced by .agg(['mean','std']) and
    strip the 'metric_' prefix."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            "_".join([c for c in col if c]) if isinstance(col, tuple) else str(col)
            for col in df.columns
        ]
    # Clean names like 'metric_accuracy_mean' -> 'accuracy_mean'
    rename = {c: c.replace("metric_", "") for c in df.columns}
    return df.rename(columns=rename)


def run_missing_benchmark(
    X_train: Union[pd.DataFrame, np.ndarray],
    y_train: Union[pd.Series, np.ndarray],
    X_test: Union[pd.DataFrame, np.ndarray],
    y_test: Union[pd.Series, np.ndarray],
    fractions: Sequence[float],
    mechanisms: Sequence[str],
    n_runs: int = 5,
    imputers: Optional[Sequence[str]] = None,
    estimator_kwargs: Optional[Dict] = None,
    random_state: int = 42,
    *,
    dataset_name: str,
    results_root: str = "results",
) -> pd.DataFrame:
    """Benchmark MissingEstimator across MCAR/MAR/MNAR corruptions with incremental saves.

    Loop order: mechanism → fraction → imputer → run

    For each (mechanism, fraction), we run all imputers across 'n_runs',
    aggregate mean/std per metric per imputer, and immediately write a CSV:
        results/{dataset_name}/{mechanism}/{dataset_name}_{mechanism}_{fraction:.2f}.csv

    Additionally, we compute a dataset-level BASELINE on clean data (train on
    clean X_train, score on clean X_test), aggregated once per imputer across runs,
    and include those baseline rows in EVERY per-fraction CSV. The returned DataFrame
    includes baseline rows ONCE (mechanism='BASELINE', fraction=0.0), plus one row
    per imputer for each (mechanism, fraction).

    Returns
    -------
    pd.DataFrame
        Aggregated mean ± std metrics with columns:
          ['imputer','mechanism','fraction', '<metric>_mean','<metric>_std', ...]
        Includes one baseline row per imputer (mechanism='BASELINE', fraction=0.0).
    """
    if imputers is None:
        imputers = DEFAULT_IMPUTERS
    fractions = _validate_fractions(fractions)
    mechanisms = _normalize_mechanisms(mechanisms)
    estimator_kwargs = dict(estimator_kwargs or {})

    logger.info(
        "Starting benchmark: runs=%d, imputers=%s, mechanisms=%s, fractions=%s",
        n_runs,
        list(imputers),
        list(mechanisms),
        list(map(float, fractions)),
    )

    # Prepare output directories
    root_dir = os.path.join(results_root, str(dataset_name))
    os.makedirs(root_dir, exist_ok=True)
    for mech in mechanisms:
        os.makedirs(os.path.join(root_dir, mech), exist_ok=True)

    # ---------- 1) Compute dataset-level BASELINE once per imputer across runs ----------
    baseline_rows: List[dict] = []
    metric_cols: Optional[List[str]] = None

    logger.info("Computing dataset-level baseline for imputers: %s", list(imputers))
    with tqdm(range(int(n_runs)), desc="Benchmarking base model", unit="run") as base_pbar:
        for run in base_pbar:
            base_seed = int(random_state) + run
            logger.debug("Baseline run %d with base_seed=%d", run, base_seed)

            est = MissingEstimator(
                **{**dict(imputer_name="zero", random_state=base_seed), **estimator_kwargs}
            )
            try:
                est.fit(X_train, y_train)
                base_scores = est.score(X_test, y_test)
                row = dict(
                    run=run,
                    imputer="BASELINE",
                    mechanism="BASELINE",
                    fraction=0.0,
                    **{f"metric_{k}": v for k, v in base_scores.items()},
                )
                baseline_rows.append(row)

                if metric_cols is None:
                    metric_cols = [f"metric_{k}" for k in base_scores.keys()]
                    logger.debug("Detected metric columns: %s", metric_cols)

            except Exception as e:
                logger.warning(
                    "Skipping baseline for imputer='%s' on run=%d due to error: %s",
                    imp,
                    run,
                    e,
                )
                # Record error row; will be excluded from aggregation
                baseline_rows.append(
                    dict(
                        run=run,
                        imputer="BASELINE",
                        mechanism="ERROR",
                        fraction=np.nan,
                        metric_error=str(e),
                    )
                )
    
    baseline_df = pd.DataFrame(baseline_rows)
    if baseline_df.empty or (metric_cols is None):
        logger.warning(
            "No successful baseline runs detected; continuing without baselines."
        )
        baseline_agg = pd.DataFrame(columns=["imputer", "mechanism", "fraction"])
    else:
        # Filter out errors and aggregate by imputer
        base_ok = baseline_df[baseline_df["mechanism"] != "ERROR"]
        baseline_agg = (
            base_ok.groupby(["imputer"], as_index=False)[metric_cols]
            .agg(["mean", "std"])
            .pipe(_flatten_agg_columns)
        )
        baseline_agg.insert(1, "mechanism", "BASELINE")
        baseline_agg.insert(2, "fraction", 0.0)

    # Keep a copy to reuse in each CSV
    baseline_for_csv = baseline_agg.copy()

    # ---------- 2) For each (mechanism, fraction): run, aggregate, and SAVE ----------
    all_agg_parts: List[pd.DataFrame] = []
    if not baseline_agg.empty:
        # Only add baseline once to the *returned* DataFrame
        all_agg_parts.append(baseline_agg)

    # Progress bar: count only scenario model trainings (exclude baseline as per requested total)
    total_models = len(mechanisms) * len(fractions) * int(n_runs) * len(list(imputers))
    with tqdm(
        total=total_models, desc=f"Benchmark {dataset_name}", unit="model"
    ) as pbar:
        for mech in mechanisms:
            for frac in fractions:
                logger.info("Processing mechanism=%s, fraction=%.2f", mech, float(frac))

                scenario_rows: List[dict] = []
                for imp in imputers:
                    for run in range(int(n_runs)):
                        base_seed = int(random_state) + run
                        seed = base_seed + (hash((mech, float(frac), imp)) % 10_000_000)
                        logger.debug(
                            "Run=%d, imputer=%s, mech=%s, frac=%.2f, seed=%d",
                            run,
                            imp,
                            mech,
                            float(frac),
                            seed,
                        )
                        try:
                            # Corrupt test
                            X_test_cor = apply_missingness(
                                X_test,
                                fraction=float(frac),
                                mechanism=mech,
                                random_state=seed,
                                y=y_test,  # for MAR/MNAR helpers if needed
                            )
                            # Corrupt training (different seed)
                            X_train_cor = apply_missingness(
                                X_train,
                                fraction=float(frac),
                                mechanism=mech,
                                random_state=seed + 1,
                                y=y_train,
                            )
                            # Refit estimator on corrupted training
                            est_refit = MissingEstimator(
                                **{
                                    **dict(imputer_name=imp, random_state=seed),
                                    **estimator_kwargs,
                                }
                            )
                            est_refit.fit(X_train_cor, y_train)
                            scores = est_refit.score(X_test_cor, y_test)

                            if metric_cols is None:
                                metric_cols = [f"metric_{k}" for k in scores.keys()]
                                logger.debug("Detected metric columns: %s", metric_cols)

                            scenario_rows.append(
                                dict(
                                    run=run,
                                    imputer=imp,
                                    mechanism=mech,
                                    fraction=float(frac),
                                    **{f"metric_{k}": v for k, v in scores.items()},
                                )
                            )
                        except Exception as ex:
                            logger.warning(
                                "Failed scenario for imputer='%s' on run=%d (mech=%s, frac=%.2f): %s",
                                imp,
                                run,
                                mech,
                                float(frac),
                                ex,
                            )
                            # Record error row for traceability; excluded from aggregation
                            scenario_rows.append(
                                dict(
                                    run=run,
                                    imputer=imp,
                                    mechanism="ERROR",
                                    fraction=float(frac),
                                    metric_error=str(ex),
                                )
                            )
                        finally:
                            # +1 per model attempt (keeps bar consistent with requested total)
                            pbar.update(1)

                scenario_df = pd.DataFrame(scenario_rows)
                if scenario_df.empty or (metric_cols is None):
                    logger.warning(
                        "No successful runs for mechanism=%s, fraction=%.2f; saving baseline only.",
                        mech,
                        float(frac),
                    )
                    agg_this = pd.DataFrame(
                        columns=["imputer", "mechanism", "fraction"]
                    )
                else:
                    ok = scenario_df[scenario_df["mechanism"] != "ERROR"]
                    if ok.empty:
                        logger.warning(
                            "All runs failed for mechanism=%s, fraction=%.2f; saving baseline only.",
                            mech,
                            float(frac),
                        )
                        agg_this = pd.DataFrame(
                            columns=["imputer", "mechanism", "fraction"]
                        )
                    else:
                        agg_this = (
                            ok.groupby(["imputer"], as_index=False)[metric_cols]
                            .agg(["mean", "std"])
                            .pipe(_flatten_agg_columns)
                        )
                        agg_this.insert(1, "mechanism", mech)
                        agg_this.insert(2, "fraction", float(frac))

                        # Add to overall return (without duplicating baseline here)
                        all_agg_parts.append(agg_this)

                # Build CSV content: BASELINE rows + current (mech, frac) rows
                csv_df = pd.concat(
                    [baseline_for_csv, agg_this],
                    ignore_index=True,
                    sort=False,
                )

                # Order columns: imputer, mechanism, fraction, then metrics alphabetically for stability
                fixed_cols = ["imputer", "mechanism", "fraction"]
                metric_order = sorted(
                    [c for c in csv_df.columns if c not in fixed_cols]
                )
                csv_df = csv_df[fixed_cols + metric_order]

                # Save immediately
                mech_dir = os.path.join(root_dir, mech)
                os.makedirs(mech_dir, exist_ok=True)
                frac_str = f"{float(frac):.2f}"
                filename = f"{dataset_name}_{mech}_{frac_str}.csv"
                out_path = os.path.join(mech_dir, filename)
                csv_df.to_csv(out_path, index=False)
                logger.info("Saved results to %s", out_path)

    # ---------- 3) Build final return DataFrame ----------
    if not all_agg_parts:
        logger.warning("No successful aggregations; returning empty DataFrame.")
        return pd.DataFrame()

    final_df = pd.concat(all_agg_parts, ignore_index=True, sort=False)
    # Normalize column order in the return value as well
    fixed_cols = ["imputer", "mechanism", "fraction"]
    metric_order = sorted([c for c in final_df.columns if c not in fixed_cols])
    final_df = final_df[fixed_cols + metric_order]

    logger.info("Benchmark complete. Aggregated rows: %d", len(final_df))
    return final_df


if __name__ == "__main__":
    # Minimal debug runs: XOR, one OpenML classification, one OpenML regression
    import logging as _logging
    from pathlib import Path
    from sklearn.model_selection import train_test_split

    from .xor import generate_xor
    from .openml import fetch_single_dataset_openml

    # Configure basic logging for manual script execution only
    _logging.basicConfig(level=_logging.INFO)

    def is_classification_target(y: Union[pd.Series, np.ndarray]) -> bool:
        s = pd.Series(y)
        if pd.api.types.is_numeric_dtype(s.dtype):
            vals = s.dropna().unique()
            return len(vals) <= 20  # heuristic: small unique count => classification
        return True  # non-numeric -> classification

    def split_xy_from_df(
        df: pd.DataFrame,
        target_col: str = "target",
        test_size: float = 0.5,
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
        test_size: float = 0.5,
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
    base_estimator_kwargs = dict(
        epochs=5, batch_size=32, hidden_dims=(4,), verbose=False
    )
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1) XOR dataset (binary classification)
    xor_df = generate_xor(n_samples=2000, noise_var=0.25, random_state=0)
    X_tr, y_tr, X_te, y_te = split_xy_from_df(
        xor_df, target_col="target", test_size=0.5, seed=0
    )
    # Explicitly set binary head to avoid 'auto' warnings
    estimator_kwargs_xor = {
        **base_estimator_kwargs,
        "output_activation": "sigmoid",
        "output_dim": 1,
    }
    xor_res = run_missing_benchmark(
        X_train=X_tr,
        y_train=y_tr,
        X_test=X_te,
        y_test=y_te,
        fractions=fractions,
        mechanisms=mechanisms,
        n_runs=5,
        imputers=None,  # keep it simple for debugging
        estimator_kwargs=estimator_kwargs_xor,
        random_state=0,
    )
    logger.info("XOR benchmark results:\n%s", xor_res.head())
    xor_res.to_csv(results_dir / "debug_xor_results.csv", index=False)

    # # 2) OpenML classification dataset (e.g., 'phoneme')
    # ds_cls = fetch_single_dataset_openml(
    #     name="phoneme", cache=True, cache_dir="./openml_cache"
    # )
    # logger.info(
    #     "Fetched OpenML classification dataset: %s",
    #     ds_cls.details.get("name") if hasattr(ds_cls, "details") else "phoneme",
    # )
    # Xc, yc = ds_cls.data, ds_cls.target
    # print("OpenML classification dataset shape:", Xc.shape, yc.shape)
    # print("yc stats:")
    # print("  Type:", type(yc))
    # print("  Shape:", yc.shape)
    # print("  Unique values:", pd.Series(yc).unique())
    # print("  Value counts:\n", pd.Series(yc).value_counts())
    # print("  Missing values:", pd.Series(yc).isnull().sum())
    # X_tr, X_te, y_tr, y_te = train_test_split(
    #     Xc, yc, test_size=0.3, random_state=1, stratify=yc
    # )
    # logger.info(
    #     "OpenML classification dataset 'phoneme': %d train, %d test",
    #     len(X_tr),
    #     len(X_te),
    # )
    # # Explicitly set multiclass head
    # n_classes = int(pd.Series(y_tr).nunique())
    # estimator_kwargs_cls = {
    #     **base_estimator_kwargs,
    #     "output_activation": "softmax",
    #     "output_dim": n_classes,
    # }
    # cls_res = run_missing_benchmark(
    #     X_train=X_tr,
    #     y_train=y_tr,
    #     X_test=X_te,
    #     y_test=y_te,
    #     fractions=fractions,
    #     mechanisms=mechanisms,
    #     n_runs=1,
    #     data_augmentation=0.0,
    #     imputers=None,
    #     estimator_kwargs=estimator_kwargs_cls,
    #     random_state=1,
    # )
    # logger.info("OpenML classification ('phoneme') results:\n%s", cls_res.head())
    # cls_res.to_csv(results_dir / "debug_openml_cls_results.csv", index=False)

    # # 3) OpenML regression dataset (e.g., 'kin8nm')
    # ds_reg = fetch_single_dataset_openml(
    #     name="kin8nm", cache=True, cache_dir="./openml_cache"
    # )
    # Xr, yr = ds_reg.data, ds_reg.target
    # X_tr, X_te, y_tr, y_te = train_test_split(Xr, yr, test_size=0.3, random_state=2)
    # # Keep base kwargs (or set 'linear' if you also want to silence warnings)
    # reg_res = run_missing_benchmark(
    #     X_train=X_tr,
    #     y_train=y_tr,
    #     X_test=X_te,
    #     y_test=y_te,
    #     fractions=fractions,
    #     mechanisms=mechanisms,
    #     n_runs=1,
    #     data_augmentation=0.0,
    #     imputers=None,
    #     estimator_kwargs=base_estimator_kwargs,
    #     random_state=2,
    # )
    # logger.info("OpenML regression ('kin8nm') results:\n%s", reg_res.head())
    # reg_res.to_csv(results_dir / "debug_openml_reg_results.csv", index=False)
