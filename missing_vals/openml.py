import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from tqdm import tqdm
import logging

DATASETS_CLS = {
    "space-ga": 737,
    "pollen": 871,
    "wilt": 40983,
    "analcatdata-supreme": 728,
    "phoneme": 1489,
    "delta-ailerons": 803,
    "visualizing-soil": 923,
    "bank8FM": 725,
    "compas-two-years": 42192,
    "bank-marketing": 1558,
    "mammography": 310,
    "mozilla4": 1046,
    "wind": 847,
    "churn": 40701,
    "sylvine": 41146,
    "ringnorm": 1496,
    "twonorm": 1507,
    "houses": 823,
    "airlines": 42493,
    "eeg-eye-state": 1471,
    "MagicTelescope": 1120,
    "Amazon-employee-access": 4135,
    "BNG(tic-tac-toe)": 137,
    "BNG(breast-w)": 251,
    "Click-prediction-small": 1220,
    "electricity": 151,
    "fried": 901,
    "mv": 881,
    "Run-or-walk-information": 40922,
    "default-of-credit-card-clients": 42477,
    "numerai28.6": 23517,
}

DATASETS_REG = {
    "stock-fardamento02": 42545,
    # "auml-eml-1-d": 42675, # Not available in OpenML
    "delta-elevators": 198,
    "sulfur": 23515,
    "kin8nm": 189,
    "wine-quality": 287,
    "Long": 42636,
    "Brazilian-houses": 42688,
    "dataset-sales": 42183,
    "BNG(echoMonths)": 1199,
    "cpu-act": 197,
    "house-8L": 218,
    "Bike-Sharing-Demand": 42712,
    "BNG(lowbwt)": 1193,
    "elevators": 216,
    "2dplanes": 215,
    "COMET-MC-SAMPLE": 23395,
    "diamonds": 42225,
    "BNG(stock)": 1200,
    "BNG(mv)": 1213,
    # "auml-url-2": 42669, # Not available in OpenML
}


def determine_task_type(dataset) -> str:
    """
    Determine if an OpenML dataset is for classification or regression.

    Rules
    -----
    - Categorical or object target → 'classification'
    - Numeric target:
        * Treat as 'classification' only if ALL are integer-like, the unique
          labels are consecutive (no gaps), and the minimum label is 0 or 1:
              - #classes in [2, 20] → 'classification'
              - Special-case: exactly {0..20} (21 classes) → 'classification'
        * Otherwise → 'regression'
    - Fallback → 'unknown'
    """
    logger = logging.getLogger(__name__)
    target = getattr(dataset, "target", None)
    if target is None:
        logger.warning("No target found on dataset; returning 'unknown'.")
        return "unknown"

    s = pd.Series(target).dropna()
    logger.debug(f"Determining task type for target with dtype: {getattr(s, 'dtype', None)}")

    if s.empty:
        logger.warning("Target is empty after dropping NaNs; returning 'unknown'.")
        return "unknown"

    # 1) Categorical or object → classification
    if pd.api.types.is_categorical_dtype(s.dtype) or pd.api.types.is_object_dtype(s.dtype):
        logger.info("Task type determined: classification (categorical/object)")
        return "classification"

    # 2) Numeric path
    if pd.api.types.is_numeric_dtype(s.dtype):
        arr = s.to_numpy()
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            logger.info("Task type determined: regression (numeric but no finite values)")
            return "regression"

        # Integer-like check (allow float storage of ints)
        integer_like = np.all(np.isclose(arr, np.round(arr), atol=1e-12))
        if integer_like:
            vals = np.unique(np.round(arr).astype(int))
            k = vals.size
            vmin, vmax = int(vals.min()), int(vals.max())

            consecutive = np.array_equal(vals, np.arange(vmin, vmax + 1))
            starts_ok = vmin in (0, 1)
            dense_0_to_20 = (vmin == 0 and vmax == 20 and consecutive)  # special-case

            logger.debug(
                f"integer_like={integer_like}, k={k}, vmin={vmin}, vmax={vmax}, "
                f"consecutive={consecutive}, starts_ok={starts_ok}"
            )

            if consecutive and starts_ok and ((2 <= k <= 20) or dense_0_to_20):
                logger.info("Task type determined: classification (integer, consecutive, start 0/1)")
                return "classification"

        # Anything else numeric → regression
        logger.info("Task type determined: regression (numeric)")
        return "regression"

    # 3) Fallback
    logger.warning("Task type could not be determined, returning 'unknown'.")
    return "unknown"



def fetch_single_dataset_openml(
    name: str, version: int = 1, cache: bool = False, cache_dir: str = None
) -> dict:
    logger = logging.getLogger(__name__)
    """
    Fetch a dataset from OpenML by name and version.

    Args:
        name (str): The name of the dataset.
        version (int): The version of the dataset to fetch.

    Returns:
        pd.DataFrame: The dataset as a pandas DataFrame.
    """
    if cache and cache_dir is None:
        cache_dir = "./openml_cache"

    logger.info(f"Fetching OpenML dataset: {name}, version: {version}, cache: {cache}")
    dataset = fetch_openml(
        name=name, version=version, as_frame=True, cache=cache, data_home=cache_dir
    )
    logger.debug(f"Fetched dataset: {name}")
    return dataset


def fetch_datasets_openml(
    dataset_ids: list, cache: bool = False, cache_dir: str = None
) -> dict:
    logger = logging.getLogger(__name__)
    """
    Fetch multiple datasets from OpenML by their IDs.

    Args:
        dataset_ids (list): List of dataset IDs to fetch.
        cache (bool): Whether to cache the datasets.
        cache_dir (str): Directory to use for caching.

    Returns:
        dict: A dictionary with dataset names as keys and DataFrames as values.
    """
    logger.info(f"Fetching multiple OpenML datasets: {dataset_ids}")
    datasets = {}
    for dataset_id in dataset_ids:
        logger.debug(f"Fetching dataset id: {dataset_id}")
        dataset = fetch_openml(
            data_id=dataset_id, as_frame=True, cache=cache, data_home=cache_dir
        )
        datasets[dataset_id] = dataset
        logger.debug(f"Fetched dataset id: {dataset_id}")
    logger.info(f"Fetched {len(datasets)} datasets.")
    return datasets


if __name__ == "__main__":
    reg_datasets = fetch_datasets_openml(
        list(DATASETS_REG.values()), cache=True, cache_dir="./openml_cache"
    )
    cls_datasets = fetch_datasets_openml(
        list(DATASETS_CLS.values()), cache=True, cache_dir="./openml_cache"
    )
