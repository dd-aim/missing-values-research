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

from .utils import set_seed, augment_with_missing_values
from .promissing import PromissingLinear, mPromissingLinear
from .compass_net import COMPASSNet


def _get_activation(name: str) -> nn.Module:
    """Get an activation function by name."""
    activations = {
        "tanh": nn.Tanh(),
        "relu": nn.ReLU(),
        "sigmoid": nn.Sigmoid(),
        "leaky_relu": nn.LeakyReLU(),
        "elu": nn.ELU(),
        "selu": nn.SELU(),
        "softmax": nn.Softmax(dim=-1),
    }
    if name == "linear":
        return nn.Identity()
    if name not in activations:
        raise ValueError(f"Unknown activation function: {name}")
    return activations[name.strip().lower()]


class _Net(nn.Module):
    """MLP network"""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Tuple[int, ...] = (4,),
        output_dim: int = 1,
        activation: str = "relu",
        output_activation: str = "sigmoid",
    ) -> None:
        super().__init__()
        self.hidden = nn.Linear(input_dim, hidden_dims[0], bias=True)
        self.act = _get_activation(activation)
        self.out = nn.Linear(hidden_dims[0], output_dim, bias=True)
        self.out_act = _get_activation(output_activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.hidden(x))
        x = self.out_act(self.out(x))
        return x


class _PromissingNet(nn.Module):
    """MLP network with PROMISSING input layer."""

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
        self.out = nn.Linear(hidden_dims[0], output_dim, bias=True)
        self.out_act = _get_activation(output_activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.hidden(x))
        x = self.out_act(self.out(x))
        return x


class _mPromissingNet(nn.Module):
    """MLP with *compensated* PROMISSING layer."""

    def __init__(
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
        self.out = nn.Linear(hidden_dims[0], output_dim, bias=True)
        self.out_act = _get_activation(output_activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.hidden(x))
        x = self.out_act(self.out(x))
        return x


class MissingEstimator(BaseEstimator):
    """Implements a missing value estimator.

    Parameters
    ----------
    imputer_name : str, default="none"
        Name of the imputer to use. Options are:
        - "none": No imputation, just a custom model.
        - "custom": Use a custom model provided via `custom_model`.
        - "zero": Fill missing values with zero.
        - "mean": Fill missing values with the mean of each feature.
        - "knn": Use KNN imputation.
        - "iterative": Use iterative imputation.
        - "promissing": Use PROMISSING input layer.
        - "mpromissing": Use mPROMISSING input layer with compensatory weights.
    custom_model : nn.Module, optional
        A custom PyTorch model to use when `imputer_name` is "custom".
        Must be a subclass of `torch.nn.Module`.
        see compass_net.py for an example.
        If `imputer_name` is "custom" and no model is provided, an error will be raised.
    hidden_dims : tuple of int, default=(4,)
        Dimensions of the hidden layers in the network.
        This is only used when `imputer_name` is "custom", "promissing", or "mpromissing".
    output_dim : int, default=1
        Number of output neurons in the network.
    activation : str, default="relu"
        Activation function to use in the hidden layers.
    output_activation : str, default="auto"
        Output activation function to use.
        If "auto", the output activation function will be determined based on the task.
    lr : float, default=0.1
        Learning-rate for SGD.
    epochs : int, default=100
        Maximum number of epochs to train.
    batch_size : int, default=10
        Mini-batch size.
    seed : int, default=0
        Random seed for reproducibility.
    early_stopping : float, default=0.0
        Fraction (0 < early_stopping < 1) of the training data to hold out as
        a validation set for early stopping. If 0, no early stopping.
    patience : int, default=10
        Number of consecutive epochs without validation-loss improvement
        tolerated before training is stopped early.
    verbose : bool, default=False
        If True, print training progress.
    """  # noqa: E501

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
        if self.imputer_name in ["none", "custom", "promissing", "mpromissing"]:
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
        if self.imputer_name in ["none", "custom", "promissing", "mpromissing"]:
            X, y = check_X_y(X, y, ensure_all_finite="allow-nan")
        else:
            X, y = check_X_y(X, y, ensure_all_finite=True)

        # Format labels based on task type
        if hasattr(self, 'output_activation'):
            if self.output_activation == "sigmoid":
                # Binary classification: keep as (n_samples, 1) for BCELoss
                y = y.astype(np.float32).reshape(-1, 1)
            elif self.output_activation == "softmax":
                # Multi-class: keep as 1D for CrossEntropyLoss
                y = y.astype(np.int64).flatten()
            else:
                # Regression: keep as (n_samples, 1) for MSELoss
                y = y.astype(np.float32).reshape(-1, 1)
        else:
            # Default behavior during initialization
            y = y.astype(np.float32).reshape(-1, 1)
            
        return X, y

    @staticmethod
    def _get_task_type_of_data(y: np.ndarray) -> str:
        """
        Helper method to check the given label's task type (classification/multi-class/regression).

        Returns:
            str: Task type string
        """

        if pd.api.types.is_numeric_dtype(y):
            # Check if it's actually discrete values (classification) or continuous (regression)
            unique_vals = np.unique(y)
            if len(unique_vals) <= 20 and np.all(np.equal(np.mod(y, 1), 0)):
                # Discrete integer values, likely classification
                if len(unique_vals) == 2:
                    return "binary_classification"
                else:
                    return "multi_class_classification"
            else:
                return "regression"

        elif pd.api.types.is_categorical_dtype(y):
            num_classes = len(y.dtype.categories)

            if num_classes == 2:
                return "binary_classification"
            elif num_classes > 2:
                return "multi_class_classification"

        raise ValueError(
            "Task type could not be determined. "
            "Please ensure that the labels are either numeric or categorical using the correct dtype with pandas."
            "Example: if its categorical, use pd.Categorical to convert the labels or .astype('category')."
        )

    # --------------------------------------------------------------------- #
    # Scikit-learn API
    # --------------------------------------------------------------------- #
    def fit(
        self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray
    ) -> MissingEstimator:
        self._set_seed()
        device = self._get_device()

        # Check input data
        if self.output_activation == "auto":
            if isinstance(y, np.ndarray):
                warnings.warn(
                    "output activation set to 'auto' but y is a numpy array. "
                    "Assuming regression task."
                )
                self.output_activation = "linear"

            task = self._get_task_type_of_data(y)
            if task == "binary_classification":
                self.output_activation = "sigmoid"
            elif task == "regression":
                self.output_activation = "linear"
            elif task == "multi_class_classification":
                self.output_activation = "softmax"

        # Decide whether to split validation first
        if self.early_stopping and 0.0 < self.early_stopping < 1.0:
            # Shuffle X and y together
            indices = np.arange(len(X))
            rng = np.random.default_rng(self.random_state)
            rng.shuffle(indices)
            if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
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
            if self.verbose:
                print(f"Early-stopping enabled: {n_val} samples for validation.")
        else:
            X_train, y_train = X, y
            has_val = False

        # Fit imputer if needed
        if self.imputer is not None:
            self.imputer.fit(X_train)
            X_train = self.imputer.transform(X_train)
            if has_val:
                X_val = self.imputer.transform(X_val)

        # Get input dimension after potential imputation
        input_dim = X_train.shape[1]

        # Build network
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
            else:
                self._model_ = _Net(
                    input_dim,
                    self.hidden_dims,
                    self.output_dim,
                    self.activation,
                    self.output_activation,
                ).to(device)

        # Select appropriate loss function based on task type
        if self.output_activation == "sigmoid":
            criterion = nn.BCELoss()
        elif self.output_activation == "softmax":
            criterion = nn.CrossEntropyLoss()
        else:  # linear/regression
            criterion = nn.MSELoss()
            
        optimiser = optim.SGD(self._model_.parameters(), lr=self.lr)

        # Check and convert input data
        X_train, y_train = self._check_X_y(X_train, y_train)
        if has_val:
            X_val, y_val = self._check_X_y(X_val, y_val)
        
        # Store original labels for classes_ attribute (before tensor conversion)
        if self.output_activation == "softmax":
            self._unique_classes = np.unique(y_train).astype(int)
        elif self.output_activation == "sigmoid":
            self._unique_classes = np.array([0, 1], dtype=int)

        # Build data loaders
        if self.output_activation == "softmax":
            # Multi-class: y should be 1D LongTensor for CrossEntropyLoss
            train_ds = torch.utils.data.TensorDataset(
                torch.tensor(X_train, dtype=torch.float32),
                torch.tensor(y_train, dtype=torch.long),
            )
        else:
            # Binary classification or regression: y should be FloatTensor
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
                preds = self._model_(xb)
                loss = criterion(preds, yb)
                loss.backward()
                optimiser.step()
                epoch_loss += loss.item() * len(xb)

            if self.verbose:
                print(
                    f"Epoch {epoch+1:03d}, train loss = {epoch_loss/len(train_ds):.4f}"
                )

            # Early-stopping evaluation
            if has_val:
                val_loss = self._evaluate_loss(val_loader, criterion, device)
                if self.verbose:
                    print(f"            val   loss = {val_loss:.4f}")
                # Check improvement
                if val_loss < best_val_loss - 1e-6:
                    best_val_loss = val_loss
                    epochs_no_improve = 0
                    best_state = self._model_.state_dict()
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= self.patience:
                        if self.verbose:
                            print(
                                f"Early stopping triggered after {epoch+1} epochs. "
                                f"Best val loss: {best_val_loss:.4f}"
                            )
                        break

        # Restore best model if early stopping was used
        if has_val and best_state is not None:
            self._model_.load_state_dict(best_state)

        # Set scikit-learn attributes based on task type
        if self.output_activation in ["sigmoid", "softmax"]:
            self.classes_ = self._unique_classes
        # No classes_ attribute for regression
        
        self.n_features_in_ = X.shape[1]
        return self

    @torch.no_grad()
    def _evaluate_loss(
        self,
        loader: torch.utils.data.DataLoader,  # type: ignore
        criterion: nn.Module,
        device: torch.device,
    ) -> float:
        """Compute average loss over `loader`."""
        self._model_.eval()
        running = 0.0
        total = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = self._model_(xb)
            loss = criterion(preds, yb)
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
        preds = self._model_(torch.tensor(X, dtype=torch.float32).to(device))

        if self.output_activation == "sigmoid":
            # Binary classification: return probabilities for [class 0, class 1]
            return np.hstack([1 - preds.cpu().numpy(), preds.cpu().numpy()])
        else:  # softmax
            # Multi-class classification: return probabilities for all classes
            return preds.cpu().numpy()

    def _predict_binary(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)[:, 1]
        return (probs >= 0.5).astype(int)

    @torch.no_grad()
    def _predict_regression(self, X: np.ndarray) -> np.ndarray:
        """Predict for regression task."""
        check_is_fitted(self)
        device = self._get_device()
        X = np.asarray(X, dtype=np.float32)
        if self.imputer is not None:
            X = self.imputer.transform(X)
        self._model_.eval()
        preds = self._model_(torch.tensor(X, dtype=torch.float32).to(device))
        return preds.cpu().numpy().flatten()

    def _predict_multi_class(self, X: np.ndarray) -> np.ndarray:
        """Predict for multi-class classification task."""
        check_is_fitted(self)
        device = self._get_device()
        X = np.asarray(X, dtype=np.float32)
        if self.imputer is not None:
            X = self.imputer.transform(X)
        self._model_.eval()
        with torch.no_grad():
            preds = self._model_(torch.tensor(X, dtype=torch.float32).to(device))
            return torch.argmax(preds, dim=1).cpu().numpy()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions based on the task type.

        Parameters
        ----------
        X : np.ndarray
            Input features.

        Returns
        -------
        np.ndarray
            Predictions. For classification tasks, returns class labels.
            For regression tasks, returns continuous values.
        """
        check_is_fitted(self)

        if self.output_activation == "sigmoid":
            # Binary classification
            probs = self.predict_proba(X)[:, 1]
            return (probs >= 0.5).astype(int)
        elif self.output_activation == "softmax":
            # Multi-class classification
            probs = self.predict_proba(X)
            return np.argmax(probs, axis=1)
        else:
            # Regression
            return self._predict_regression(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Compute scores for the model based on the task type.

        Parameters
        ----------
        X : np.ndarray
            Input features.
        y : np.ndarray
            True labels.

        Returns
        -------
        dict
            Dictionary containing scores for each scoring method based on task type.
            For binary classification: accuracy, precision, recall, f1, roc_auc
            For multi-class classification: accuracy, precision_macro, recall_macro, f1_macro
            For regression: r2, mse, mae
        """

        check_is_fitted(self)

        # Determine task type
        task_type = "regression"  # default
        if hasattr(self, "output_activation"):
            if self.output_activation == "sigmoid":
                task_type = "binary_classification"
            elif self.output_activation == "softmax":
                task_type = "multi_class_classification"
            elif self.output_activation in ["linear", "relu", "identity"]:
                task_type = "regression"

        if task_type == "binary_classification":
            # Get predictions and probabilities
            y_pred = self.predict(X)
            y_proba = self.predict_proba(X)[:, 1]  # Probability of positive class

            scores = {
                "accuracy": accuracy_score(y, y_pred),
                "precision": precision_score(
                    y, y_pred, average="binary", zero_division=0
                ),
                "recall": recall_score(y, y_pred, average="binary", zero_division=0),
                "f1": f1_score(y, y_pred, average="binary", zero_division=0),
            }

            # Add ROC AUC if we have both classes in y
            if len(np.unique(y)) > 1:
                scores["roc_auc"] = roc_auc_score(y, y_proba)
            else:
                scores["roc_auc"] = 0.0

        elif task_type == "multi_class_classification":
            y_pred = self.predict(X)

            scores = {
                "accuracy": accuracy_score(y, y_pred),
                "precision_macro": precision_score(
                    y, y_pred, average="macro", zero_division=0
                ),
                "recall_macro": recall_score(
                    y, y_pred, average="macro", zero_division=0
                ),
                "f1_macro": f1_score(y, y_pred, average="macro", zero_division=0),
            }

        else:  # regression
            y_pred = self.predict(X)

            scores = {
                "r2": r2_score(y, y_pred),
                "mse": mean_squared_error(y, y_pred),
                "mae": mean_absolute_error(y, y_pred),
            }

        return scores
