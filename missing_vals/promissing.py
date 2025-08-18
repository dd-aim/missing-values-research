"""Replicates the custom linear models proposed in PROMISSING paper."""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class PromissingLinear(nn.Module):
    """
    PROMISSING first dense layer.
    Implements Eq. (3) from the paper:
        a = (x_obs @ Wᵀ) + (q / p) * b
    where
        x_obs : input with NaNs zeroed out
        q     : # observed features in each sample
        p     : total # features (constant = in_features)
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features  # == p
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = torch.isnan(x)  # True where value is missing
        x_filled = torch.where(mask, torch.zeros_like(x), x)
        q = (~mask).sum(dim=1, keepdim=True).type_as(x)  # observed-feature count
        linear_out = F.linear(x_filled, self.weight, bias=None)  # Σ xᵢ wᵢ
        if self.bias is not None:
            bias_term = (q / self.in_features) * self.bias  # (q/p)*b  (broadcast)
            linear_out = linear_out + bias_term
        return linear_out


class mPromissingLinear(PromissingLinear):
    """
    mPROMISSING variant (Eq. 5):
        a = (x_obs @ Wᵀ) + (q/p)*b + (r/p)*w_c
    Adds one learnable compensatory weight per neuron.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__(in_features, out_features, bias)
        # one extra scalar per output neuron
        self.w_c = nn.Parameter(torch.zeros(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = torch.isnan(x)
        x_filled = torch.where(mask, torch.zeros_like(x), x)
        q = (~mask).sum(dim=1, keepdim=True).type_as(x)  # observed
        r = self.in_features - q  # missing
        linear_out = F.linear(x_filled, self.weight, bias=None)
        if self.bias is not None:
            linear_out = (
                linear_out
                + (q / self.in_features) * self.bias
                + (r / self.in_features) * self.w_c
            )
        return linear_out
