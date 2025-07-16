import numpy as np
from sklearn.datasets import fetch_openml
from tqdm import tqdm

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

def determine_task_type(dataset):
    """
    Determine if an OpenML dataset is for classification or regression.
    
    Parameters
    ----------
    dataset_name : str, optional
        Name of the OpenML dataset
    data_id : int, optional
        OpenML dataset ID
        
    Returns
    -------
    str : 'classification', 'regression', or 'unknown'
    """
    
    target = dataset.target
    
    # Method 1: Check target data type
    if hasattr(target, 'dtype'):
        if target.dtype == 'object' or target.dtype.name == 'category':
            return 'classification'
        elif np.issubdtype(target.dtype, np.floating):
            return 'regression'
        elif np.issubdtype(target.dtype, np.integer):
            # For integers, check uniqueness ratio
            unique_values = len(np.unique(target))
            total_values = len(target)
            
            if unique_values / total_values < 0.1:
                return 'classification'
            else:
                return 'regression'
    
    return 'unknown'

def fetch_single_dataset_openml(name: str, version: int = 1, cache: bool = False, cache_dir: str = None) -> dict:
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
    
    dataset = fetch_openml(name=name, version=version, as_frame=True, cache=cache, data_home=cache_dir)
    return dataset


def fetch_datasets_openml(dataset_ids: list, cache: bool = False, cache_dir: str = None) -> dict:
    """
    Fetch multiple datasets from OpenML by their IDs.

    Args:
        dataset_ids (list): List of dataset IDs to fetch.
        cache (bool): Whether to cache the datasets.
        cache_dir (str): Directory to use for caching.

    Returns:
        dict: A dictionary with dataset names as keys and DataFrames as values.
    """
    datasets = {}
    for dataset_id in dataset_ids:
        dataset = fetch_openml(data_id=dataset_id, as_frame=True, cache=cache, data_home=cache_dir)
        datasets[dataset_id] = dataset
    
    
    return datasets


if __name__ == "__main__":
    reg_datasets = fetch_datasets_openml(list(DATASETS_REG.values()), cache=True, cache_dir="./openml_cache")
    cls_datasets = fetch_datasets_openml(list(DATASETS_CLS.values()), cache=True, cache_dir="./openml_cache")