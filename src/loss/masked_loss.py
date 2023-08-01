"""Masked MSE loss"""

from torch import Tensor
from torch import abs as torch_abs
from torch import mean, square


def masked_mse_loss(input_1: Tensor, input_2: Tensor, mask: Tensor) -> Tensor:
    """MSE with masking"""
    return mean(mask * square(input_1 - input_2))


def masked_mae_loss(input_1: Tensor, input_2: Tensor, mask: Tensor) -> Tensor:
    """MAE with masking"""
    return mean(mask * torch_abs(input_1 - input_2))
