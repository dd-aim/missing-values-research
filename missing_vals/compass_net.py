# compass_net.py
"""
COMPASS-Net (COMbinatorial PAttern Sub-modelS Network)

- Task-aware head: supports binary, multiclass, and regression
- Variable-depth sub-models: hidden_dims=(...) builds stacked hidden layers
- max_missing controls the largest missing-feature mask handled at inference
"""
from __future__ import annotations

import itertools
from itertools import combinations
import logging
from typing import Dict, Iterable, Tuple, Type, Sequence, Optional

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------- #
# Data Preprocessing for Compass training
# ----------------------------------------------------------------------------- #





def compass_expand_patterns(
    data: pd.DataFrame | np.ndarray,
    y: Optional[pd.Series | pd.DataFrame | np.ndarray] = None,
    missing_cols: Iterable[int] = (1, 2),
):
    """
    Augment X by introducing missing values with different patterns, and replicate y accordingly.

    Supports both pandas and NumPy inputs. If `y` is provided, returns a tuple (X_aug, y_aug).
    If `y` is None, returns only X_aug.

    Args:
        data: Feature matrix X as a pandas DataFrame or NumPy array. If array, it will be
              converted to a DataFrame internally for processing.
        y: Target vector/array as pandas Series/DataFrame or NumPy array. If provided, it will be
           replicated and filtered to stay aligned with augmented samples.
        missing_cols: Iterable specifying how many columns to set missing simultaneously:
            - 1: Single column missing (one column at a time)
            - 2: Every combination of 2 columns missing
            - 3: Every combination of 3 columns missing
            - etc.

    Returns:
        If y is provided: (X_aug, y_aug) with the same types as the inputs (DataFrame/Series or ndarray).
        If y is None: X_aug with the same type as the input `data`.
    """
    # Normalize X to DataFrame for processing while tracking original types
    x_was_numpy = isinstance(data, np.ndarray)
    if x_was_numpy:
        X_df = pd.DataFrame(data)
    elif isinstance(data, pd.DataFrame):
        X_df = data.copy()
    else:
        raise TypeError("Input X must be a pandas DataFrame or NumPy array")

    if X_df.empty:
        raise ValueError("Input X is empty")

    # Normalize y to pandas object if provided and track shape/type
    y_provided = y is not None
    if y_provided:
        y_was_numpy = isinstance(y, np.ndarray)
        if y_was_numpy:
            if y.ndim == 1:
                y_pd: pd.Series | pd.DataFrame = pd.Series(y)
                y_is_1d = True
            elif y.ndim == 2 and y.shape[1] == 1:
                y_pd = pd.DataFrame(y)
                y_is_1d = False
            else:
                # multi-output not supported in this helper
                raise ValueError("y with shape != (n,) or (n,1) is not supported")
        elif isinstance(y, (pd.Series, pd.DataFrame)):
            y_pd = y.copy()
            y_is_1d = isinstance(y_pd, pd.Series)
        else:
            raise TypeError("y must be a pandas Series/DataFrame or NumPy array")

        if len(y_pd) != len(X_df):
            raise ValueError(
                f"Length mismatch: X has {len(X_df)} rows but y has {len(y_pd)}"
            )
        # Align indices
        y_pd.index = X_df.index

    # Validate missing sizes (k): allow 1..n_cols-1
    n_cols = X_df.shape[1]
    try:
        k_values = sorted({int(k) for k in missing_cols})
    except Exception as e:
        raise TypeError("missing_cols must be an iterable of integers") from e

    valid_k = [k for k in k_values if 1 <= k < n_cols]
    if not valid_k:
        logger.warning(
            "No valid missing sizes in missing_cols=%s for %d columns. Returning original data.",
            list(k_values),
            n_cols,
        )
        if y_provided:
            return (data if x_was_numpy else X_df, y)
        return data if x_was_numpy else X_df

    columns = list(X_df.columns)
    augmented_X: list[pd.DataFrame] = []
    augmented_y: list[pd.Series | pd.DataFrame] = [] if y_provided else []

    def _append_aug(aug_sample: pd.DataFrame):
        if not aug_sample.empty:
            augmented_X.append(aug_sample)
            if y_provided:
                augmented_y.append(y_pd.loc[aug_sample.index])  # type: ignore[name-defined]

    # Generate augmentations
    for num_missing in valid_k:
        if num_missing == 1:
            for col in columns:
                mask = X_df[col].notna()
                if not mask.any():
                    continue
                aug_sample = X_df.loc[mask].copy()
                aug_sample[col] = np.nan
                other_cols = [c for c in columns if c != col]
                if other_cols:
                    aug_sample.dropna(subset=other_cols, inplace=True)
                _append_aug(aug_sample)
        else:
            for combo in combinations(columns, num_missing):
                combo = list(combo)
                mask = X_df[combo].notna().all(axis=1)
                if not mask.any():
                    continue
                aug_sample = X_df.loc[mask].copy()
                for c in combo:
                    aug_sample[c] = np.nan
                other_cols = [c for c in columns if c not in combo]
                if other_cols:
                    aug_sample.dropna(subset=other_cols, inplace=True)
                _append_aug(aug_sample)

    # Stack originals + augmentations (if any)
    if augmented_X:
        logger.info("Created %d augmented samples.", len(augmented_X))
        X_all = pd.concat([X_df] + augmented_X, axis=0, ignore_index=False)
        if y_provided:
            y_all = pd.concat([y_pd] + augmented_y, axis=0, ignore_index=False)  # type: ignore[name-defined]
    else:
        logger.warning("No augmented samples created. Returning original data.")
        X_all = X_df
        if y_provided:
            y_all = y_pd  # type: ignore[name-defined]

    # Clean indices
    X_all = X_all.reset_index(drop=True)
    if y_provided:
        y_all = y_all.reset_index(drop=True)  # type: ignore[name-defined]

    # Preserve original types on return
    if y_provided:
        X_ret = X_all.to_numpy() if x_was_numpy else X_all
        if y_was_numpy:  # type: ignore[name-defined]
            y_np = y_all.to_numpy()  # type: ignore[name-defined]
            # Restore original ndim: (n,) or (n,1)
            if isinstance(y, np.ndarray) and y.ndim == 1 and y_np.ndim != 1:
                y_np = y_np.reshape(-1)
            if (
                isinstance(y, np.ndarray)
                and y.ndim == 2
                and y.shape[1] == 1
                and y_np.ndim == 1
            ):
                y_np = y_np.reshape(-1, 1)
            return X_ret, y_np
        else:
            return X_ret, y_all  # type: ignore[return-value]
    else:
        return X_all.to_numpy() if x_was_numpy else X_all


# ----------------------------------------------------------------------------- #
# Utilities
# ----------------------------------------------------------------------------- #


def _resolve_activation(task: str, output_activation: str | None) -> Optional[str]:
    """Resolve a concrete output activation from (task, output_activation).

    Returns one of: 'sigmoid', 'softmax', 'linear', or None.
    """
    task_l = (task or "binary").lower()
    act = (output_activation or "auto").lower()

    if act == "auto":
        if task_l in ("binary", "classification_binary", "cls_binary"):
            return "sigmoid"
        elif task_l in ("multiclass", "classification", "cls_multiclass"):
            return "softmax"
        elif task_l in ("regression", "reg"):
            return "linear"
        else:
            return "linear"  # default
    return act


# ----------------------------------------------------------------------------- #
# Sub-network
# ----------------------------------------------------------------------------- #


class _CompassSubNet(nn.Module):
    """MLP with variable depth and task-aware head.

    Hidden: as many Linear(+tanh) layers as provided in `hidden_dims`.
    Head:
        - regression: Linear -> (no activation) or 'linear'
        - binary:     Linear(->1) -> Sigmoid if required
        - multiclass: Linear(->n_classes) -> Softmax if required
    """

    def __init__(
        self,
        in_features: int,
        hidden_dims: Sequence[int] = (4,),
        linear_cls: Type[nn.Linear] = nn.Linear,
        task: str = "binary",
        n_classes: int = 2,
        output_activation: str | None = "auto",
    ) -> None:
        super().__init__()
        # Hidden stack
        self.hidden_layers = nn.ModuleList()
        prev = in_features
        for h in hidden_dims:
            self.hidden_layers.append(linear_cls(prev, int(h), bias=True))
            prev = int(h)

        # Head
        task_l = (task or "binary").lower()
        if task_l in ("regression", "reg"):
            out_dim = 1
        elif task_l in ("binary", "classification_binary", "cls_binary"):
            out_dim = 1
        elif task_l in ("multiclass", "classification", "cls_multiclass"):
            if int(n_classes) < 2:
                raise ValueError("For multiclass, n_classes must be >= 2")
            out_dim = int(n_classes)
        else:
            out_dim = 1

        self.out = nn.Linear(prev, out_dim, bias=True)
        self._final_act = _resolve_activation(task_l, output_activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        for layer in self.hidden_layers:
            x = torch.tanh(layer(x))
        x = self.out(x)
        # Apply head activation if requested
        if self._final_act == "sigmoid":
            return torch.sigmoid(x)
        elif self._final_act == "softmax":
            return torch.softmax(x, dim=-1)
        elif self._final_act in ("linear", None):
            return x
        else:
            # Fallback: no activation
            return x


# ----------------------------------------------------------------------------- #
# Router mixin
# ----------------------------------------------------------------------------- #


class _RouterMixin:
    """Routing logic shared by all COMPASS variants."""

    def _mask_to_key(self, mask: torch.Tensor) -> Tuple[int, ...]:
        """Convert boolean mask vector to an *ordered* tuple of missing indices."""
        return tuple(torch.nonzero(mask, as_tuple=False).squeeze(1).tolist())

    def _build_subnets(
        self,
        in_features: int,
        linear_cls: Type[nn.Linear],
        hidden_dims: Sequence[int],
        task: str,
        n_classes: int,
        output_activation: str | None,
        pattern_sizes: Iterable[int] = (0, 1, 2),
    ) -> nn.ModuleDict:
        patterns: Dict[Tuple[int, ...], _CompassSubNet] = {}
        for k in pattern_sizes:
            for comb in itertools.combinations(range(in_features), k):
                sub_in_dim = in_features - k
                patterns[comb] = _CompassSubNet(
                    in_features=sub_in_dim,
                    hidden_dims=hidden_dims,
                    linear_cls=linear_cls,
                    task=task,
                    n_classes=n_classes,
                    output_activation=output_activation,
                )
        logger.debug(
            "Built %d subnets for patterns up to size %s",
            len(patterns),
            max(pattern_sizes) if pattern_sizes else 0,
        )
        # store as string keys for ModuleDict
        return nn.ModuleDict({str(k): v for k, v in patterns.items()})  # type: ignore

    def _route_batch(
        self, x: torch.Tensor, pattern_key: Tuple[int, ...]
    ) -> tuple[_CompassSubNet, torch.Tensor]:
        """Return the sub-network and observed features for a batch with the same missing pattern."""
        if x.dim() != 2:
            raise ValueError("Batch routing expects a 2-D tensor (batch, features).")

        # Get the subnet for this pattern
        subnet = self._subnets[str(pattern_key)]

        # Create mask for observed features (inverse of missing pattern)
        observed_mask = torch.ones(x.shape[1], dtype=torch.bool, device=x.device)
        if pattern_key:  # If there are missing features
            observed_mask[list(pattern_key)] = False

        # Extract observed features
        observed_x = x[:, observed_mask]
        logger.debug(
            "Routing batch of size %d with pattern %s to subnet having in_dim=%d",
            x.shape[0],
            pattern_key,
            observed_x.shape[1],
        )
        return subnet, observed_x


# ----------------------------------------------------------------------------- #
# Public model
# ----------------------------------------------------------------------------- #


class COMPASSNet(nn.Module, _RouterMixin):
    """COMPASS-Net with variable-depth subnets and task-aware heads.

    Parameters
    ----------
    in_features : int
        Number of input features (full vector).
    hidden_dims : Sequence[int], default=(4,)
        Width of each hidden layer (shared across all sub-models).
    task : {'binary','multiclass','regression'}, default='binary'
        Determines the output head size and (if output_activation='auto') the head activation.
    n_classes : int, default=2
        Used only when task='multiclass'.
    output_activation : {'auto','sigmoid','softmax','linear', None}, default='auto'
        Final activation applied by each sub-model.
    linear_cls : Type[nn.Linear], default=nn.Linear
        Allows substituting the first linear with PROMISSING variants if desired.
    max_missing : int, default=2
        Maximum number of missing inputs supported (builds subnets for sizes 0..max_missing).
    """

    def __init__(
        self,
        in_features: int,
        hidden_dims: Sequence[int] = (4,),
        task: str = "binary",
        n_classes: int = 2,
        output_activation: str | None = "auto",
        linear_cls: Type[nn.Linear] = nn.Linear,
        max_missing: int = 2,
    ):
        super().__init__()
        self.in_features = int(in_features)
        self.task = task
        self.n_classes = int(n_classes)
        self.output_activation = output_activation
        # Clamp max_missing to avoid building an all-missing subnet (sub_in_dim == 0)
        requested_max_missing = int(max_missing)
        max_allowed = max(0, self.in_features - 1)
        if requested_max_missing > max_allowed:
            logger.info(
                "Clamping max_missing from %d to %d (in_features=%d) to avoid all-missing subnet",
                requested_max_missing,
                max_allowed,
                self.in_features,
            )
        self.max_missing = min(requested_max_missing, max_allowed)

        self._subnets = self._build_subnets(
            in_features=self.in_features,
            linear_cls=linear_cls,
            hidden_dims=hidden_dims,
            task=task,
            n_classes=n_classes,
            output_activation=output_activation,
            pattern_sizes=tuple(range(0, self.max_missing + 1)),  # e.g. (0,1,2)
        )

        # Determine output dimension for allocating batch outputs
        if (task or "binary").lower() in (
            "multiclass",
            "classification",
            "cls_multiclass",
        ):
            self._out_dim = int(n_classes)
        else:
            self._out_dim = 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2:
            raise ValueError("Input must be 2-D (batch, features).")

        batch_size = x.shape[0]
        device = x.device

        # Compute missing masks for all samples at once
        missing_masks = torch.isnan(x)  # (batch_size, in_features)

        # Check for too many missing features
        missing_counts = missing_masks.sum(dim=1)
        if torch.any(missing_counts > self.max_missing):
            raise ValueError(
                f"COMPASS-Net instantiated for up to {self.max_missing} missing inputs."
            )

        # Convert masks to pattern strings and group indices
        pattern_strings = []
        for i in range(batch_size):
            mask = missing_masks[i]
            key = tuple(torch.nonzero(mask, as_tuple=False).squeeze(1).tolist())
            pattern_strings.append(str(key))

        unique_patterns = list(set(pattern_strings))
        pattern_groups = {
            pattern: torch.tensor(
                [i for i, p in enumerate(pattern_strings) if p == pattern],
                device=device,
            )
            for pattern in unique_patterns
        }
        logger.debug(
            "Forward pass: batch=%d, unique_patterns=%s", batch_size, unique_patterns
        )

        # Initialize output tensor
        outputs = torch.zeros((batch_size, self._out_dim), device=device)

        # Process each pattern group in batch
        for pattern_str, indices in pattern_groups.items():
            if len(indices) == 0:
                continue

            # Convert pattern string back to tuple (safe: we control serialization)
            pattern_key = eval(pattern_str)

            # Extract samples for this pattern group
            group_x = x[indices]  # (group_size, in_features)

            # Route to subnet and slice observed features
            subnet, observed_x = self._route_batch(group_x, pattern_key)

            # Forward through subnet and place outputs back
            group_outputs = subnet(observed_x)  # (group_size, out_dim)
            outputs[indices] = group_outputs

        return outputs
