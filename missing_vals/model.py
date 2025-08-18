from __future__ import annotations

import os
import random
import warnings
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
from jenga.corruptions.generic import MissingValues

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_X_y, check_array
from sklearn.utils.validation import check_is_fitted
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    r2_score,
    mean_squared_error,
    mean_absolute_error,
)

from .utils import set_seed
from .promissing import PromissingLinear, mPromissingLinear
from .compass_net import COMPASSNet, compass_expand_patterns
import json
import logging


def _get_activation(name: str) -> nn.Module:
    """Get an activation function by name (hidden layers only)."""
    activations = {
        "tanh": nn.Tanh(),
        "relu": nn.ReLU(),
        "sigmoid": nn.Sigmoid(),
        "leaky_relu": nn.LeakyReLU(),
        "elu": nn.ELU(),
        "selu": nn.SELU(),
        "softmax": nn.Softmax(dim=-1),
    }
    key = name.strip().lower() if isinstance(name, str) else name
    if key == "linear":
        return None
    if key not in activations:
        raise ValueError(f"Unknown activation function: {name}")
    return activations[key]


class _Net(nn.Module):
    """MLP network returning logits; no final activation in forward."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Tuple[int, ...] = (4,),
        output_dim: int = 1,
        activation: str = "relu",
        output_activation: str = "sigmoid",
    ) -> None:
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.logger.debug(
            f"Net init: Input dimension: {input_dim}, Hidden dimensions: {hidden_dims}"
        )
        self.act = _get_activation(activation)

        self.hidden_layers = nn.ModuleList()
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            self.hidden_layers.append(nn.Linear(prev_dim, hidden_dim))
            prev_dim = hidden_dim

        self.out = nn.Linear(prev_dim, output_dim)
        # Store requested output activation for inference only
        self.out_activation_name = output_activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.logger.debug(f"Net forward: Input tensor shape: {x.shape}")
        if torch.isnan(x).any() or torch.isinf(x).any():
            self.logger.warning("Input tensor contains NaNs or infinite values!")
            raise ValueError(
                "Input tensor contains NaNs or infinite values, which may cause the hidden layer to return NaNs"
            )
        # Forward through hidden layers
        for i, hidden_layer in enumerate(self.hidden_layers):
            x = hidden_layer(x)
            if i == 0 and self.act is not None:
                x = self.act(x)
        # Return logits (no activation); losses expect logits
        out = self.out(x)
        return out


class _PromissingNet(nn.Module):
    """MLP with PROMISSING first layer; returns logits in forward."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Tuple[int, ...] = (4,),
        output_dim: int = 1,
        activation: str = "tanh",
        output_activation: str = "sigmoid",
    ) -> None:
        super().__init__()
        self.hidden = PromissingLinear(input_dim, hidden_dims[0], bias=True)
        self.act = _get_activation(activation)
        self.additional_hidden_layers = nn.ModuleList()
        prev_dim = hidden_dims[0]
        for hidden_dim in hidden_dims[1:]:
            self.additional_hidden_layers.append(nn.Linear(prev_dim, hidden_dim))
            prev_dim = hidden_dim
        self.out = nn.Linear(prev_dim, output_dim, bias=True)
        self.out_activation_name = output_activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.hidden(x)
        if self.act is not None:
            x = self.act(x)
        for i, hidden_layer in enumerate(self.additional_hidden_layers):
            x = hidden_layer(x)
            if i == 0 and self.act is not None:
                x = self.act(x)
        return self.out(x)  # logits


class _mPromissingNet(nn.Module):
    """MLP with mPROMISSING first layer; returns logits in forward."""

    def __init(
        self,
        input_dim: int,
        hidden_dims: Tuple[int, ...] = (4,),
        output_dim: int = 1,
        activation: str = "tanh",
        output_activation: str = "sigmoid",
    ) -> None:
        super().__init__()
        self.hidden = mPromissingLinear(input_dim, hidden_dims[0], bias=True)
        self.act = _get_activation(activation)
        self.additional_hidden_layers = nn.ModuleList()
        prev_dim = hidden_dims[0]
        for hidden_dim in hidden_dims[1:]:
            self.additional_hidden_layers.append(nn.Linear(prev_dim, hidden_dim))
            prev_dim = hidden_dim
        self.out = nn.Linear(prev_dim, output_dim, bias=True)
        self.out_activation_name = output_activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.hidden(x)
        if self.act is not None:
            x = self.act(x)
        for i, hidden_layer in enumerate(self.additional_hidden_layers):
            x = hidden_layer(x)
            if i == 0 and self.act is not None:
                x = self.act(x)
        return self.out(x)  # logits


class MissingEstimator(BaseEstimator):
    """Estimator supporting different missing-value strategies and neural heads.

    Training uses logits-only heads:
      - Binary: BCEWithLogitsLoss
      - Multiclass: CrossEntropyLoss
      - Regression: MSELoss

    Inference applies activations in predict_proba only.
    """

    def __init__(
        self,
        *,
        imputer_name: str = "none",
        custom_model: Optional[nn.Module] = None,
        hidden_dims: Tuple[int, ...] = (4,),
        output_dim: int = 1,
        activation: str = "relu",
        output_activation: str = "auto",
        lr: float = 0.1,
        epochs: int = 100,
        batch_size: int = 10,
        random_state: int = 0,
        early_stopping: float = 0.0,
        patience: int = 10,
        verbose: bool = False,
    ) -> None:
        self.lr = lr
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.activation = activation
        self.output_activation = output_activation
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.early_stopping = early_stopping
        self.patience = patience
        self.verbose = verbose
        self.imputer_name = imputer_name
        self.imputer = self._get_imputer()
        self.custom_model = custom_model
        self.logger = logging.getLogger(__name__)

    # --------------------------------------------------------------------- #
    # Internal helpers
    # --------------------------------------------------------------------- #
    def _set_seed(self) -> None:
        if self.random_state is not None:
            set_seed(self.random_state)

    @staticmethod
    def _get_device() -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _get_imputer(self) -> None | SimpleImputer | KNNImputer | IterativeImputer:
        if self.imputer_name in [
            "none",
            "custom",
            "promissing",
            "mpromissing",
            "compass",
        ]:
            return None
        elif self.imputer_name == "zero":
            return SimpleImputer(strategy="constant", fill_value=0)
        elif self.imputer_name == "mean":
            return SimpleImputer(strategy="mean")
        elif self.imputer_name == "knn":
            return KNNImputer()
        elif self.imputer_name == "iterative":
            return IterativeImputer()
        else:
            raise ValueError(f"Unknown imputer: {self.imputer_name}")

    def _check_X_y(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Check and convert input data to the correct format."""
        if self.imputer_name in [
            "none",
            "custom",
            "promissing",
            "mpromissing",
            "compass",
        ]:
            X, y = check_X_y(X, y, ensure_all_finite="allow-nan")
        else:
            X, y = check_X_y(X, y, ensure_all_finite=True)

        # Format labels for internal torch tensors; actual encoding handled elsewhere
        if hasattr(self, "output_activation"):
            if self.output_activation == "softmax":
                y = y.astype(np.int64).flatten()
            else:
                y = y.astype(np.float32).reshape(-1, 1)
        else:
            y = y.astype(np.float32).reshape(-1, 1)
        return X, y

    def _ensure_binary_01(self, y: pd.Series | np.ndarray) -> np.ndarray:
        s = pd.Series(y).copy()
        if s.dropna().nunique() != 2:
            return np.asarray(y)
        if pd.api.types.is_categorical_dtype(s.dtype):
            return s.cat.codes.to_numpy()
        non_null = s.dropna()
        try:
            vals = np.sort(pd.unique(non_null).astype(float))
            if vals.size == 2 and np.allclose(vals, [0.0, 1.0]):
                return s.to_numpy()
        except Exception:
            pass
        uniq = pd.unique(non_null)
        try:
            ordered = np.sort(uniq)
        except Exception:
            ordered = sorted(list(uniq), key=lambda x: str(x))
        mapping = {ordered[0]: 0, ordered[1]: 1}
        return s.map(mapping).to_numpy()

    @staticmethod
    def _get_task_type_of_data(y: np.ndarray | pd.Series) -> str:
        """
        Decide between 'binary_classification', 'multi_class_classification', 'regression'
        under these numeric rules:
        • For numeric labels: must be integer-like, consecutive, and start at 0 or 1.
            - {0,1} or {1,2}                → binary_classification
            - {0..K} or {1..K} (no gaps)    → multi_class_classification if 3 ≤ #classes ≤ 20
        • Special-case: {0..20} (21 classes) → multi_class_classification
        • Any gaps (e.g., {0,4,5,6}) → regression
        • Categorical dtype → classification based on number of categories.
        • Otherwise → regression.
        """
        s = pd.Series(y).dropna()
        if s.empty:
            return "regression"

        # 1) Categorical labels: trust categories
        if pd.api.types.is_categorical_dtype(s.dtype):
            k = len(s.dtype.categories)
            if k == 2:
                return "binary_classification"
            return "multi_class_classification" if k >= 3 else "regression"

        # 2) Numeric labels
        if pd.api.types.is_numeric_dtype(s.dtype):
            arr = s.to_numpy()
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                return "regression"

            # Integer-like check (allow float storage of ints)
            if np.all(np.isclose(arr, np.round(arr), atol=1e-12)):
                vals = np.unique(np.round(arr).astype(int))
                k = vals.size
                vmin, vmax = int(vals.min()), int(vals.max())

                consecutive = np.array_equal(vals, np.arange(vmin, vmax + 1))
                starts_ok = vmin in (0, 1)

                # Special-case dense 0..20 inclusive (21 classes)
                dense_0_to_20 = vmin == 0 and vmax == 20 and consecutive

                if consecutive and starts_ok:
                    if k == 2:
                        return "binary_classification"
                    if 3 <= k <= 20 or dense_0_to_20:
                        return "multi_class_classification"

            # Not integer-like, or gaps, or wrong start
            return "regression"

        # 3) Fallback
        return "regression"

    # --------------------------------------------------------------------- #
    # Scikit-learn API
    # --------------------------------------------------------------------- #
    def fit(
        self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray
    ) -> MissingEstimator:
        self._set_seed()
        device = self._get_device()
        self.logger.info(
            "Starting fit: device=%s, imputer=%s, epochs=%d, batch=%d",
            device,
            self.imputer_name,
            self.epochs,
            self.batch_size,
        )

        # Decide task/activation
        if self.output_activation == "auto":
            task = self._get_task_type_of_data(y)
            self.logger.info(f"Determined task type: {task}")
            if task == "binary_classification":
                self.output_activation = "sigmoid"
            elif task == "regression":
                self.output_activation = "linear"
            elif task == "multi_class_classification":
                self.output_activation = "softmax"

        # Early stopping split
        if self.early_stopping and 0.0 < self.early_stopping < 1.0:
            indices = np.arange(len(X))
            rng = np.random.default_rng(self.random_state)
            rng.shuffle(indices)
            if isinstance(X, (pd.DataFrame, pd.Series)):
                X, y = X.iloc[indices], y.iloc[indices]
                n_val = max(1, int(len(X) * self.early_stopping))
                X_val, y_val = X.iloc[:n_val], y.iloc[:n_val]
                X_train, y_train = X.iloc[n_val:], y.iloc[n_val:]
            else:
                X, y = X[indices], y[indices]
                n_val = max(1, int(len(X) * self.early_stopping))
                X_val, y_val = X[:n_val], y[:n_val]
                X_train, y_train = X[n_val:], y[n_val:]
            has_val = True
        else:
            X_train, y_train = X, y
            has_val = False

        # Label handling
        self._classes_encoder = None
        if self.output_activation == "sigmoid":
            y_train = self._ensure_binary_01(y_train)
            if has_val:
                y_val = self._ensure_binary_01(y_val)
            self._unique_classes = np.array([0, 1], dtype=int)
        elif self.output_activation == "softmax":
            # Map labels to 0..C-1
            y_series = pd.Series(y_train)
            classes_sorted = np.array(
                sorted(pd.unique(y_series.dropna())), dtype=object
            )
            class_to_index = {cls: i for i, cls in enumerate(classes_sorted)}
            y_train = y_series.map(class_to_index).to_numpy()
            if has_val:
                y_val = pd.Series(y_val).map(class_to_index).to_numpy()
            self._classes_encoder = {
                "classes_": classes_sorted,
                "to_index": class_to_index,
            }
            self._unique_classes = np.arange(len(classes_sorted))
            # Adjust output_dim to #classes if not set
            self.output_dim = int(len(classes_sorted))

        # Impute if needed
        if self.imputer is not None:
            self.imputer.fit(X_train)
            X_train = self.imputer.transform(X_train)
            if has_val:
                X_val = self.imputer.transform(X_val)

        input_dim = X_train.shape[1]

        # If using COMPASS, expand the training (and validation) sets to include
        # mask patterns so every sub-network is trained. This operates on numpy arrays.
        if self.imputer_name == "compass":
            X_train, y_train = compass_expand_patterns(X_train, y_train)
            if has_val:
                X_val, y_val = compass_expand_patterns(X_val, y_val)

        # Build model (logits-only forward)
        if self.imputer_name == "custom" and self.custom_model is not None:
            if not isinstance(self.custom_model, nn.Module):
                raise ValueError("Custom model must be a subclass of nn.Module.")
            self._model_ = self.custom_model.to(device)
        elif self.imputer_name == "custom" and self.custom_model is None:
            raise ValueError(
                "Custom model must be provided when imputer_name is 'custom'."
            )
        else:
            if self.imputer_name == "promissing":
                self._model_ = _PromissingNet(
                    input_dim,
                    self.hidden_dims,
                    self.output_dim,
                    self.activation,
                    self.output_activation,
                ).to(device)
            elif self.imputer_name == "mpromissing":
                self._model_ = _mPromissingNet(
                    input_dim,
                    self.hidden_dims,
                    self.output_dim,
                    self.activation,
                    self.output_activation,
                ).to(device)
            elif self.imputer_name == "compass":
                # Build COMPASS with same hidden_dims; heads return logits
                task = (
                    "multiclass"
                    if self.output_activation == "softmax"
                    else (
                        "binary"
                        if self.output_activation == "sigmoid"
                        else "regression"
                    )
                )
                n_classes = self.output_dim if task == "multiclass" else 2
                self._model_ = COMPASSNet(
                    in_features=input_dim,
                    hidden_dims=self.hidden_dims,
                    task=task,
                    n_classes=n_classes,
                    output_activation=None,  # logits only during training
                    linear_cls=nn.Linear,
                ).to(device)
            else:
                self._model_ = _Net(
                    input_dim,
                    self.hidden_dims,
                    self.output_dim,
                    self.activation,
                    self.output_activation,
                ).to(device)

        # Select loss (expects logits)
        if self.output_activation == "sigmoid":
            criterion = nn.BCEWithLogitsLoss()
        elif self.output_activation == "softmax":
            criterion = nn.CrossEntropyLoss()
        else:  # linear/regression
            criterion = nn.MSELoss()

        optimiser = optim.SGD(self._model_.parameters(), lr=self.lr)

        # Check and convert to tensors
        X_train, y_train = self._check_X_y(X_train, y_train)
        if has_val:
            X_val, y_val = self._check_X_y(X_val, y_val)

        # Data loaders
        if self.output_activation == "softmax":
            train_ds = torch.utils.data.TensorDataset(
                torch.tensor(X_train, dtype=torch.float32),
                torch.tensor(y_train, dtype=torch.long),
            )
        else:
            train_ds = torch.utils.data.TensorDataset(
                torch.tensor(X_train, dtype=torch.float32),
                torch.tensor(y_train, dtype=torch.float32),
            )
        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=self.batch_size, shuffle=True
        )

        if has_val:
            if self.output_activation == "softmax":
                val_ds = torch.utils.data.TensorDataset(
                    torch.tensor(X_val, dtype=torch.float32),
                    torch.tensor(y_val, dtype=torch.long),
                )
            else:
                val_ds = torch.utils.data.TensorDataset(
                    torch.tensor(X_val, dtype=torch.float32),
                    torch.tensor(y_val, dtype=torch.float32),
                )
            val_loader = torch.utils.data.DataLoader(
                val_ds, batch_size=self.batch_size, shuffle=False
            )
            best_val_loss = float("inf")
            epochs_no_improve = 0
            best_state: Optional[dict] = None

        # Training loop
        self._model_.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimiser.zero_grad()
                logits = self._model_(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimiser.step()
                epoch_loss += loss.item() * len(xb)

            if self.verbose:
                self.logger.info(
                    f"Epoch {epoch+1:03d}, train loss = {epoch_loss/len(train_ds):.4f}"
                )

            # Early stopping
            if has_val:
                val_loss = self._evaluate_loss(val_loader, criterion, device)
                if self.verbose:
                    self.logger.info(f"            val   loss = {val_loss:.4f}")
                if val_loss < best_val_loss - 1e-6:
                    best_val_loss = val_loss
                    epochs_no_improve = 0
                    best_state = self._model_.state_dict()
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= self.patience:
                        if self.verbose:
                            self.logger.info(
                                f"Early stopping triggered after {epoch+1} epochs. "
                                f"Best val loss: {best_val_loss:.4f}"
                            )
                        break

        if has_val and best_state is not None:
            self._model_.load_state_dict(best_state)

        # scikit-learn attributes
        if self.output_activation in ["sigmoid", "softmax"]:
            self.classes_ = self._unique_classes
        self.n_features_in_ = int(input_dim)
        self.logger.info("Fit complete.")
        return self

    @torch.no_grad()
    def _evaluate_loss(
        self,
        loader: torch.utils.data.DataLoader,  # type: ignore
        criterion: nn.Module,
        device: torch.device,
    ) -> float:
        self._model_.eval()
        running = 0.0
        total = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = self._model_(xb)
            loss = criterion(logits, yb)
            running += loss.item() * len(xb)
            total += len(xb)
        self._model_.train()
        return running / total

    @torch.no_grad()
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self)
        if self.output_activation not in ["sigmoid", "softmax"]:
            raise ValueError(
                "predict_proba is only available for classification tasks with sigmoid or softmax output activation."
            )
        device = self._get_device()
        X = np.asarray(X, dtype=np.float32)
        if self.imputer is not None:
            X = self.imputer.transform(X)
        self._model_.eval()
        logits = self._model_(torch.tensor(X, dtype=torch.float32).to(device))
        if self.output_activation == "sigmoid":
            probs_pos = torch.sigmoid(logits).cpu().numpy()
            return np.hstack([1 - probs_pos, probs_pos])
        else:
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            return probs

    def _predict_regression(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self)
        device = self._get_device()
        X = np.asarray(X, dtype=np.float32)
        if self.imputer is not None:
            X = self.imputer.transform(X)
        self._model_.eval()
        preds = self._model_(torch.tensor(X, dtype=torch.float32).to(device))
        return preds.cpu().numpy().flatten()

    def _predict_multi_class(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self)
        device = self._get_device()
        X = np.asarray(X, dtype=np.float32)
        if self.imputer is not None:
            X = self.imputer.transform(X)
        self._model_.eval()
        with torch.no_grad():
            logits = self._model_(torch.tensor(X, dtype=torch.float32).to(device))
            preds_idx = torch.argmax(logits, dim=1).cpu().numpy()
            # Map back to original labels if encoder exists
            if getattr(self, "_classes_encoder", None) is not None:
                classes = self._classes_encoder["classes_"]
                return classes[preds_idx]
            return preds_idx

    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self)
        if self.output_activation == "sigmoid":
            probs = self.predict_proba(X)[:, 1]
            return (probs >= 0.5).astype(int)
        elif self.output_activation == "softmax":
            return self._predict_multi_class(X)
        else:
            return self._predict_regression(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> dict:
        check_is_fitted(self)
        if self.output_activation == "sigmoid":
            y_true = self._ensure_binary_01(y)
            y_pred = self.predict(X)
            y_proba = self.predict_proba(X)[:, 1]
            scores = {
                "accuracy": accuracy_score(y_true, y_pred),
                "precision": precision_score(
                    y_true, y_pred, average="binary", zero_division=0
                ),
                "recall": recall_score(
                    y_true, y_pred, average="binary", zero_division=0
                ),
                "f1": f1_score(y_true, y_pred, average="binary", zero_division=0),
            }
            if len(np.unique(y_true)) > 1:
                scores["roc_auc"] = roc_auc_score(y_true, y_proba)
            else:
                scores["roc_auc"] = 0.0
            return scores
        elif self.output_activation == "softmax":
            y_pred = self.predict(X)
            return {
                "accuracy": accuracy_score(y, y_pred),
                "precision_macro": precision_score(
                    y, y_pred, average="macro", zero_division=0
                ),
                "recall_macro": recall_score(
                    y, y_pred, average="macro", zero_division=0
                ),
                "f1_macro": f1_score(y, y_pred, average="macro", zero_division=0),
            }
        else:
            y_pred = self._predict_regression(X)
            return {
                "r2": r2_score(y, y_pred),
                "mse": mean_squared_error(y, y_pred),
                "mae": mean_absolute_error(y, y_pred),
            }
