"""
COMPASS-Net (COMbinatorial PAttern Sub-modelS Network) python implementation

Problem addressed:
    Robust inference in fully-connected, feed-forward neural networks (FC-FF NNs) when 1 or 2 input features are missing at test time, without resorting to imputation.
Key idea:
    Train a separate sub-network for every missing-feature pattern of size 1 or 2, then at inference time route each sample to the sub-network that matches its mask of missing features.
Number of sub-networks:
    For an input dimension n:
        • 1-feature-missing patterns: n
        • 2-features-missing patterns: C(n, 2) = n(n – 1)/2
      Total = n + n(n – 1)/2 = n(n + 1)/2 sub-models.
      (Extension to larger masks is possible.)
Architecture of each sub-network:
    • Hidden layers: identical depth, width, activations, and initial hyper-parameters as the original “full” model.
    • Input layer: width = n – k, where k ∈ {1, 2}.
    • Output layer: unchanged.
Training data per sub-network:
    Use the same full training set, but delete the corresponding feature(s) in every row so that the network sees a consistent mask during training. No synthetic imputation is performed.
Training procedure:
    1. Start from the architecture/hyper-parameters of a well-performing baseline FC-FF NN.
    2. For each mask pattern:
        a. Delete the masked feature column(s) from every row.
        b. Re-initialize and train the model end-to-end (optimizer, epochs, loss, etc. unchanged from the baseline).
    3. Optionally train the baseline “no-features-missing” model alongside the masked sub-networks.
On-chip deployment:
    All sub-networks are stored simultaneously on the target inference chip; the hardware automatically selects and runs the correct sub-network once it sees which feature(s) are absent in an input vector.
Inference workflow:
    1. Detect which features (if any) are missing in the incoming sample.
    2. Remove those feature slots.
    3. Dispatch the reduced vector to the sub-network whose mask matches.
Evaluation protocol:
    • Create perturbed copies of the hold-out / validation set by randomly dropping 1 feature in X % of rows and 2 features in Y % of rows (e.g. X = 3 %, Y = 1 %).
    • For each row, run the appropriate sub-network.
    • Report pattern-wise and overall accuracy; sweep several (X, Y) pairs for robustness.
Scalability / extensions:
    • If domain knowledge or deployment data indicate >2 missing features are likely, extend the ensemble to larger mask sizes (combinatorial growth is the only constraint).
    • Can incorporate model-compression or parameter-sharing techniques to stay within chip memory if n is large.
"""

from __future__ import annotations

import itertools
from typing import Dict, Iterable, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# COMPASS building blocks ------------------------------------------------------
# -----------------------------------------------------------------------------


class _CompassSubNet(nn.Module):
    """A tiny MLP: (d-in) → 4 tanh → 1 sigmoid.

    The *Linear* class of the first layer can be swapped for PROMISSING
    variants by passing ``linear_cls``.
    """

    def __init__(
        self,
        in_features: int,
        linear_cls: Type[nn.Linear] = nn.Linear,
        hidden_units: int = 4,
    ) -> None:
        super().__init__()
        self.hidden = linear_cls(in_features, hidden_units, bias=True)
        self.act = nn.Tanh()
        self.out = nn.Linear(hidden_units, 1, bias=True)
        self.sigm = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        x = self.act(self.hidden(x))
        return self.sigm(self.out(x))


class _RouterMixin:
    """Mixin that provides routing logic shared by all COMPASS variants."""

    def _mask_to_key(self, mask: torch.Tensor) -> Tuple[int, ...]:
        """Convert boolean mask vector to an *ordered* tuple of missing indices."""
        return tuple(torch.nonzero(mask, as_tuple=False).squeeze(1).tolist())

    def _build_subnets(
        self,
        in_features: int,
        linear_cls: Type[nn.Linear],
        pattern_sizes: Iterable[int] = (0, 1, 2),
    ) -> Dict[Tuple[int, ...], _CompassSubNet]:
        patterns: Dict[Tuple[int, ...], _CompassSubNet] = {}
        for k in pattern_sizes:
            for comb in itertools.combinations(range(in_features), k):
                sub_in_dim = in_features - k
                patterns[comb] = _CompassSubNet(
                    in_features=sub_in_dim,
                    linear_cls=linear_cls,
                )
        return nn.ModuleDict({str(k): v for k, v in patterns.items()})  # type: ignore

    def _route(self, x: torch.Tensor) -> Tuple[_CompassSubNet, torch.Tensor]:
        """Return the sub-network and the *observed* slice of x for one sample."""
        if x.dim() != 1:
            raise ValueError("Routing expects a single sample (1-D tensor).")
        mask = torch.isnan(x)
        missing = mask.sum().item()
        if missing > 2:
            raise ValueError("COMPASS-Net instantiated for up to 2 missing inputs.")
        key = self._mask_to_key(mask)
        sub = self._subnets[str(key)]
        observed_x = x[~mask]
        return sub, observed_x

    def _route_batch(
        self, x: torch.Tensor, pattern_key: Tuple[int, ...]
    ) -> Tuple[_CompassSubNet, torch.Tensor]:
        """Return the sub-network and observed features for a batch of samples with the same missing pattern."""
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

        return subnet, observed_x


# -----------------------------------------------------------------------------
# Public models ----------------------------------------------------------------
# -----------------------------------------------------------------------------


class COMPASSNet(nn.Module, _RouterMixin):
    """COMPASS-Net with *plain* linear layers (expects imputed inputs or none missing)."""

    def __init__(self, in_features: int):
        super().__init__()
        self.in_features = in_features
        self._subnets = self._build_subnets(in_features, nn.Linear)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2:
            raise ValueError("Input must be 2-D (batch, features).")

        batch_size = x.shape[0]
        device = x.device

        # Compute missing masks for all samples at once
        missing_masks = torch.isnan(x)  # (batch_size, in_features)

        # Check for too many missing features
        missing_counts = missing_masks.sum(dim=1)
        if torch.any(missing_counts > 2):
            raise ValueError("COMPASS-Net instantiated for up to 2 missing inputs.")

        # Convert masks to pattern strings for efficient grouping
        # This is faster than the loop-based approach for large batches
        pattern_strings = []
        for i in range(batch_size):
            mask = missing_masks[i]
            key = self._mask_to_key(mask)
            pattern_strings.append(str(key))

        # Group indices by pattern using dictionary comprehension
        unique_patterns = list(set(pattern_strings))
        pattern_groups = {
            pattern: torch.tensor(
                [i for i, p in enumerate(pattern_strings) if p == pattern],
                device=device,
            )
            for pattern in unique_patterns
        }

        # Initialize output tensor
        outputs = torch.zeros((batch_size, 1), device=device)

        # Process each pattern group in batch
        for pattern_str, indices in pattern_groups.items():
            if len(indices) == 0:
                continue

            # Convert pattern string back to tuple
            pattern_key = eval(pattern_str)  # Safe since we control the format

            # Extract samples for this pattern group
            group_x = x[indices]  # (group_size, in_features)

            # Use batch routing to get subnet and observed features
            subnet, observed_x = self._route_batch(group_x, pattern_key)

            # Forward through subnet
            group_outputs = subnet(observed_x)  # (group_size, 1)

            # Place outputs back in correct positions
            outputs[indices] = group_outputs

        return outputs