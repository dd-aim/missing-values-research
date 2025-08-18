from __future__ import annotations

import os
import random
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
from jenga.corruptions.generic import MissingValues
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_X_y, check_array
from sklearn.utils.validation import check_is_fitted
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer

from .utils import set_seed
from .promissing import PromissingLinear, mPromissingLinear
from .compass_net import COMPASSNet

logger = logging.getLogger(__name__)


def generate_xor(n_samples=1_000, noise_var=0.25, random_state=42):
    """
    Reproduces the dataset in Sec. 3.1 of the PROMISSING paper.
    Each raw feature is uniform on [-1, 1] with additive N(0, noise_var) noise.
    The label is the XOR of the *signs* of the two coordinates.
    """
    rng = np.random.RandomState(random_state)
    X = rng.uniform(-1.0, 1.0, size=(n_samples, 2))
    X += rng.normal(scale=np.sqrt(noise_var), size=X.shape)
    y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int)  # Bayes-optimal rule
    df = pd.DataFrame(X, columns=["x1", "x2"])
    df["target"] = y
    logger.info(
        "Generated XOR dataset: n=%d, noise_var=%.3f, seed=%s",
        n_samples,
        noise_var,
        str(random_state),
    )
    return df


def create_benchmark_datasets(
    n_samples=1_000,
    noise_var=0.25,
    random_state=42,
    missing_fractions=[0.30, 0.50],
    missingness_types=["MCAR", "MAR", "MNAR"],
    save_to_disk=True,
    low_memory=False,
    output_dir="data",
):
    """
    Creates a benchmark dataset for the XOR problem.
    Returns a dictionary with the dataset and metadata.
    """
    # Decide which feature we attack: choose at random each time
    set_seed(random_state)
    xor_df = generate_xor(
        n_samples=n_samples, noise_var=noise_var, random_state=random_state
    )

    # Container for later inspection
    if save_to_disk:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "xor_datasets"), exist_ok=True)
        xor_df.to_parquet(
            os.path.join(output_dir, "xor_datasets", "xor_clean.parquet"), index=False
        )
        logger.info(
            "Saved clean XOR dataset to %s",
            os.path.join(output_dir, "xor_datasets", "xor_clean.parquet"),
        )
    if not low_memory:
        datasets_with_holes = dict()

    progress_bar = tqdm(
        total=len(missingness_types) * len(missing_fractions),
        desc="Creating datasets with missing values",
        unit="dataset",
    )

    for mech in missingness_types:
        for frac in missing_fractions:
            impacted_col = random.choice(["x1", "x2"])  # Choose between "x1" and "x2"
            logger.debug(
                "Applying missingness: mech=%s, frac=%.2f, column=%s",
                mech,
                frac,
                impacted_col,
            )
            corruption = MissingValues(
                column=impacted_col,
                fraction=frac,
                missingness=mech,
                na_value=np.nan,
            )
            df_corrupted = corruption.transform(xor_df.copy(deep=True))
            key = f"{mech}_{int(frac*100)}"
            if save_to_disk:
                path = os.path.join(output_dir, "xor_datasets", f"xor_{key}.parquet")
                df_corrupted.to_parquet(path, index=False)
                logger.info("Saved corrupted XOR dataset to %s", path)
            if not low_memory:
                datasets_with_holes[key] = df_corrupted

            progress_bar.update(1)

    if not low_memory:
        logger.info(
            "Created %d corrupted XOR datasets in-memory", len(datasets_with_holes)
        )
        return datasets_with_holes


# NOTE: The following classes are commented out because they are not used in the current implementation
# They where adapted now in a generic method in model.py so they are not specific to the XOR problem.

# --------------------------------------------------------------------------------------------------------

# class _XORNet(nn.Module):
#     """Promissing papers binary classifier used for benchmark:
#     2-D → 4 tanh → 1 sigmoid (binary) network."""

#     def __init__(self):
#         super().__init__()
#         self.hidden = nn.Linear(2, 4, bias=True)
#         self.act = nn.Tanh()
#         self.out = nn.Linear(4, 1, bias=True)
#         self.sigm = nn.Sigmoid()

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.act(self.hidden(x))
#         x = self.sigm(self.out(x))  # explicit sigmoid so we can use BCELoss
#         return x


# class _PromissingXORNet(nn.Module):
#     """2-D → 4 tanh → 1 sigmoid network with PROMISSING input layer."""
#     def __init__(self):
#         super().__init__()
#         self.hidden = PromissingLinear(2, 4, bias=True)
#         self.act    = nn.Tanh()
#         self.out    = nn.Linear(4, 1, bias=True)
#         self.sigm   = nn.Sigmoid()

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.act(self.hidden(x))
#         x = self.sigm(self.out(x))
#         return x


# class _mPromissingXORNet(nn.Module):
#     """2-D → 4 tanh → 1 sigmoid network with *compensated* PROMISSING layer."""
#     def __init__(self):
#         super().__init__()
#         self.hidden = mPromissingLinear(2, 4, bias=True)
#         self.act    = nn.Tanh()
#         self.out    = nn.Linear(4, 1, bias=True)
#         self.sigm   = nn.Sigmoid()

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.act(self.hidden(x))
#         x = self.sigm(self.out(x))
#         return x


# class XOREstimator(BaseEstimator, ClassifierMixin):
#     """Replicates the XOR classifier from the PROMISSING paper.

#     Parameters
#     ----------
#     lr : float, default=0.1
#         Learning-rate for SGD.
#     epochs : int, default=100
#         Maximum number of epochs to train.
#     batch_size : int, default=10
#         Mini-batch size.
#     seed : int, default=0
#         Random seed for reproducibility.
#     early_stopping : float, default=0.0
#         Fraction (0 < early_stopping < 1) of the training data to hold out as
#         a validation set for early stopping. If 0, no early stopping.
#     patience : int, default=10
#         Number of consecutive epochs without validation-loss improvement
#         tolerated before training is stopped early.
#     verbose : bool, default=False
#         If True, print training progress.
#     """  # noqa: E501

#     def __init__(
#         self,
#         *,
#         imputer_name: str = "none",
#         custom_model: Optional[nn.Module] = None,
#         lr: float = 0.1,
#         epochs: int = 100,
#         batch_size: int = 10,
#         random_state: int = 0,
#         early_stopping: float = 0.0,
#         patience: int = 10,
#         verbose: bool = False,
#     ) -> None:
#         self.lr = lr
#         self.epochs = epochs
#         self.batch_size = batch_size
#         self.random_state = random_state
#         self.early_stopping = early_stopping
#         self.patience = patience
#         self.verbose = verbose
#         self.imputer_name = imputer_name
#         self.imputer = self._get_imputer()
#         self.custom_model = custom_model

#     # --------------------------------------------------------------------- #
#     # Internal helpers
#     # --------------------------------------------------------------------- #
#     def _set_seed(self) -> None:
#         if self.random_state is not None:
#             set_seed(self.random_state)

#     @staticmethod
#     def _get_device() -> torch.device:
#         return torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     def _get_imputer(self) -> None | SimpleImputer | KNNImputer | IterativeImputer:
#         if self.imputer_name in ["none", "custom", "promissing", "mpromissing"]:
#             return None
#         elif self.imputer_name == "zero":
#             return SimpleImputer(strategy="constant", fill_value=0)
#         elif self.imputer_name == "mean":
#             return SimpleImputer(strategy="mean")
#         elif self.imputer_name == "knn":
#             return KNNImputer()
#         elif self.imputer_name == "iterative":
#             return IterativeImputer()
#         else:
#             raise ValueError(f"Unknown imputer: {self.imputer_name}")

#     def _check_X_y(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
#         """Check and convert input data to the correct format."""
#         if self.imputer_name in ["none", "custom", "promissing", "mpromissing"]:
#             X, y = check_X_y(X, y, ensure_all_finite="allow-nan")
#         else:
#             X, y = check_X_y(X, y, ensure_all_finite=True)

#         y = y.astype(np.float32).reshape(-1, 1)
#         return X, y

#     # --------------------------------------------------------------------- #
#     # Scikit-learn API
#     # --------------------------------------------------------------------- #
#     def fit(self, X: np.ndarray, y: np.ndarray):
#         self._set_seed()
#         device = self._get_device()

#         # Build network
#         if self.imputer_name == "custom" and self.custom_model is not None:
#             if not isinstance(self.custom_model, nn.Module):
#                 raise ValueError("Custom model must be a subclass of nn.Module.")
#             self._model_ = self.custom_model.to(device)
#         elif self.imputer_name == "custom" and self.custom_model is None:
#             raise ValueError(
#             "Custom model must be provided when imputer_name is 'custom'."
#             )
#         else:
#             if self.imputer_name == "promissing":
#                 self._model_ = _PromissingXORNet().to(device)
#             elif self.imputer_name == "mpromissing":
#                 self._model_ = _mPromissingXORNet().to(device)
#             else:
#                 self._model_ = _XORNet().to(device)

#         criterion = nn.BCELoss()
#         optimiser = optim.SGD(self._model_.parameters(), lr=self.lr)

#         # Decide whether to split validation
#         if self.early_stopping and 0.0 < self.early_stopping < 1.0:
#             # Shuffle X and y together
#             indices = np.arange(len(X))
#             rng = np.random.default_rng(self.random_state)
#             rng.shuffle(indices)
#             X, y = X[indices], y[indices]
#             n_val = max(1, int(len(X) * self.early_stopping))
#             X_val, y_val = X[:n_val], y[:n_val]
#             X_train, y_train = X[n_val:], y[n_val:]
#             has_val = True
#             if self.verbose:
#                 print(f"Early-stopping enabled: {n_val} samples for validation.")
#         else:
#             X_train, y_train = X, y
#             has_val = False

#         # Fit imputer if needed
#         if self.imputer is not None:
#             self.imputer.fit(X_train)
#             X_train = self.imputer.transform(X_train)
#             if has_val:
#                 X_val = self.imputer.transform(X_val)

#         # Check and convert input data
#         X_train, y_train = self._check_X_y(X_train, y_train)
#         if has_val:
#             X_val, y_val = self._check_X_y(X_val, y_val)

#         # Build data loaders
#         train_ds = torch.utils.data.TensorDataset(
#             torch.tensor(X_train, dtype=torch.float32),
#             torch.tensor(y_train, dtype=torch.float32),
#         )
#         train_loader = torch.utils.data.DataLoader(
#             train_ds, batch_size=self.batch_size, shuffle=True
#         )

#         if has_val:
#             val_ds = torch.utils.data.TensorDataset(
#                 torch.tensor(X_val, dtype=torch.float32),
#                 torch.tensor(y_val, dtype=torch.float32),
#             )
#             val_loader = torch.utils.data.DataLoader(
#                 val_ds, batch_size=self.batch_size, shuffle=False
#             )
#             best_val_loss = float("inf")
#             epochs_no_improve = 0
#             best_state: Optional[dict] = None

#         # Training loop
#         self._model_.train()
#         for epoch in range(self.epochs):
#             epoch_loss = 0.0
#             for xb, yb in train_loader:
#                 xb, yb = xb.to(device), yb.to(device)
#                 optimiser.zero_grad()
#                 preds = self._model_(xb)
#                 loss = criterion(preds, yb)
#                 loss.backward()
#                 optimiser.step()
#                 epoch_loss += loss.item() * len(xb)

#             if self.verbose:
#                 print(
#                     f"Epoch {epoch+1:03d}, train loss = {epoch_loss/len(train_ds):.4f}"
#                 )

#             # Early-stopping evaluation
#             if has_val:
#                 val_loss = self._evaluate_loss(val_loader, criterion, device)
#                 if self.verbose:
#                     print(f"            val   loss = {val_loss:.4f}")
#                 # Check improvement
#                 if val_loss < best_val_loss - 1e-6:
#                     best_val_loss = val_loss
#                     epochs_no_improve = 0
#                     best_state = self._model_.state_dict()
#                 else:
#                     epochs_no_improve += 1
#                     if epochs_no_improve >= self.patience:
#                         if self.verbose:
#                             print(
#                                 f"Early stopping triggered after {epoch+1} epochs. "
#                                 f"Best val loss: {best_val_loss:.4f}"
#                             )
#                         break

#         # Restore best model if early stopping was used
#         if has_val and best_state is not None:
#             self._model_.load_state_dict(best_state)

#         # scikit-learn attributes
#         self.classes_ = np.array([0, 1], dtype=int)
#         self.n_features_in_ = X.shape[1]
#         return self

#     @torch.no_grad()
#     def _evaluate_loss(
#         self,
#         loader: torch.utils.data.DataLoader,  # type: ignore
#         criterion: nn.Module,
#         device: torch.device,
#     ) -> float:
#         """Compute average loss over `loader`."""
#         self._model_.eval()
#         running = 0.0
#         total = 0
#         for xb, yb in loader:
#             xb, yb = xb.to(device), yb.to(device)
#             preds = self._model_(xb)
#             loss = criterion(preds, yb)
#             running += loss.item() * len(xb)
#             total += len(xb)
#         self._model_.train()
#         return running / total

#     @torch.no_grad()
#     def predict_proba(self, X: np.ndarray) -> np.ndarray:
#         check_is_fitted(self)
#         device = self._get_device()
#         X = np.asarray(X, dtype=np.float32)
#         if self.imputer is not None:
#             X = self.imputer.transform(X)
#         self._model_.eval()
#         preds = self._model_(torch.tensor(X, dtype=torch.float32).to(device))
#         return np.hstack([1 - preds.cpu().numpy(), preds.cpu().numpy()])

#     def predict(self, X: np.ndarray) -> np.ndarray:
#         probs = self.predict_proba(X)[:, 1]
#         return (probs >= 0.5).astype(int)
