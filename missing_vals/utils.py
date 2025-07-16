import random
import warnings
from itertools import combinations

import numpy as np
import pandas as pd

import torch




def set_seed(seed: int):
    """
    Set the random seed for random, numpy, and torch (if available).

    Args:
        seed (int): The seed value to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def augment_with_missing_values(
    data: pd.DataFrame,
    augmentation_fraction: float = 0.3,
    exclude_columns: list = None,
    random_state: int = 42,
    missing_cols: list = [1, 2],
) -> pd.DataFrame:
    """
    Augment a DataFrame by introducing missing values with different patterns.

    Args:
        data (pd.DataFrame): The input DataFrame.
        augmentation_fraction (float): Fraction of rows to augment with missing values.
        exclude_columns (list): Columns to exclude from augmentation.
        random_state (int): Seed for reproducibility.
        missing_cols (list): List specifying number of columns to make missing simultaneously:
            - 1: Single column missing (one column at a time)
            - 2: Every combination of 2 columns missing
            - 3: Every combination of 3 columns missing
            - etc.

    Returns:
        pd.DataFrame: Augmented DataFrame with missing values following specified patterns.
    """
    if exclude_columns is None:
        exclude_columns = []
    elif isinstance(exclude_columns, str):
        exclude_columns = [exclude_columns]
    elif not isinstance(exclude_columns, list):
        raise TypeError("exclude_columns must be a list or a string")

    # Validate inputs
    if not 0 < augmentation_fraction <= 1:
        raise ValueError("augmentation_fraction must be between 0 and 1")

    if data.empty:
        raise ValueError("Input DataFrame is empty")

    # Get eligible columns for augmentation
    eligible_columns = [col for col in data.columns if col not in exclude_columns]

    if not eligible_columns:
        raise ValueError("No eligible columns found for augmentation.")

    data_copy = data.copy()
    augmented_samples = []

    # Calculate number of rows to augment
    n_augment_rows = int(len(data) * augmentation_fraction)

    for missing_idx, num_missing in enumerate(missing_cols):
        # Set different seed for each missing pattern to ensure variety
        pattern_seed = random_state + missing_idx * 1000
        np.random.seed(pattern_seed)

        if num_missing == 1:
            # Single column missing - one column at a time
            for i, column in enumerate(eligible_columns):
                column_seed = pattern_seed + i
                np.random.seed(column_seed)

                # Sample rows for this column
                sample_indices = np.random.choice(
                    data.index, size=min(n_augment_rows, len(data)), replace=False
                )

                aug_sample = data_copy.loc[sample_indices].copy()
                aug_sample[column] = np.nan

                # Only keep rows that have at least one non-missing value in other columns
                other_cols = [col for col in eligible_columns if col != column]
                if other_cols:
                    aug_sample = aug_sample.dropna(subset=other_cols)

                if not aug_sample.empty:
                    augmented_samples.append(aug_sample)

        elif num_missing >= 2 and num_missing <= len(eligible_columns):
            # Generate all combinations of num_missing columns
            column_combinations = list(combinations(eligible_columns, num_missing))

            for combo_idx, missing_column_combo in enumerate(column_combinations):
                combo_seed = pattern_seed + combo_idx
                np.random.seed(combo_seed)

                # Sample rows for this combination
                sample_indices = np.random.choice(
                    data.index, size=min(n_augment_rows, len(data)), replace=False
                )

                aug_sample = data_copy.loc[sample_indices].copy()

                # Set all columns in the combination to NaN
                for col in missing_column_combo:
                    aug_sample[col] = np.nan

                # Only keep rows that have at least one non-missing value in other columns
                other_cols = [
                    col for col in eligible_columns if col not in missing_column_combo
                ]
                if other_cols:
                    aug_sample = aug_sample.dropna(subset=other_cols)

                if not aug_sample.empty:
                    augmented_samples.append(aug_sample)

    # Combine all augmented samples
    if augmented_samples:
        result = pd.concat([data_copy] + augmented_samples, axis=0, ignore_index=True)
        return result.reset_index(drop=True)
    else:
        warnings.warn("No augmented samples created. Returning original data.")
        return data_copy
