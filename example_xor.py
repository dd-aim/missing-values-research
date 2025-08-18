from __future__ import annotations

from pathlib import Path
from typing import List

import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from missing_vals.xor import generate_xor
from missing_vals.benchmark import run_missing_benchmark

# Set logging level to WARNING
logging.basicConfig(level=logging.WARNING)


def run_xor_experiments():
    # Experiment setup (same hyperparameters as before)
    N_SAMPLES = 1000
    NOISE_VAR = 0.25
    TEST_SIZE = 0.5  # 500/500 split
    N_RUNS = 3

    # Model/optimizer
    EPOCHS = 1000
    BATCH_SIZE = 10
    LR = 0.01
    ACTIVATION = "tanh"
    OUTPUT_ACTIVATION = "sigmoid"
    HIDDEN_DIMS = (4,)
    EARLY_STOPPING = 0.1
    PATIENCE = 50

    MECHANISMS: List[str] = ["MCAR", "MAR", "MNAR"]
    FRACTIONS: List[float] = [0.30, 0.50]  # filenames will be 0.30, 0.50

    REQUESTED_IMPUTERS = [
        "zero",
        "mean",
        "knn",
        "iterative",
        "promissing",
        "mpromissing",
        "compass",
    ]

    # Generate a single clean XOR dataset and do a 50/50 split (stratified)
    base_seed = 0
    df = generate_xor(n_samples=N_SAMPLES, noise_var=NOISE_VAR, random_state=base_seed)
    X = df[["x1", "x2"]].to_numpy()
    y = df["target"].to_numpy()

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=base_seed, stratify=y
    )

    # Estimator kwargs (applied across imputers/mechanisms)
    estimator_kwargs = dict(
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LR,
        activation=ACTIVATION,
        hidden_dims=HIDDEN_DIMS,
        output_activation=OUTPUT_ACTIVATION,
        early_stopping=EARLY_STOPPING,
        patience=PATIENCE,
    )

    # Print hyperparameters to ensure alignment
    print("XOR benchmark configuration:")
    print(f"  mechanisms: {MECHANISMS}")
    print(f"  fractions: {FRACTIONS}")
    print(f"  n_runs: {N_RUNS}")
    print("  estimator_kwargs:")
    for k, v in estimator_kwargs.items():
        print(f"    - {k}: {v}")
    print(f"  imputers: {REQUESTED_IMPUTERS}")

    # Run centralized benchmark (handles missingness corruption, runs, aggregation, and CSV saving)
    res = run_missing_benchmark(
        X_train=X_tr,
        y_train=y_tr,
        X_test=X_te,
        y_test=y_te,
        fractions=FRACTIONS,
        mechanisms=MECHANISMS,
        n_runs=N_RUNS,
        imputers=REQUESTED_IMPUTERS,
        estimator_kwargs=estimator_kwargs,
        random_state=base_seed,
        dataset_name="xor",
        results_root="results",
    )

    # Optional: write a debug CSV summary of the aggregated table
    Path("results").mkdir(parents=True, exist_ok=True)
    res.to_csv(Path("results") / "debug_xor_results.csv", index=False)
    print("Saved aggregated XOR summary to results/debug_xor_results.csv")


if __name__ == "__main__":
    run_xor_experiments()
