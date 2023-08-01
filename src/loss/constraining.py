"""Constrain some value to be smaller than threshold"""

from torch import Tensor, mean
from torch import sum as torch_sum
from torch import square

def squared_constraint(value: Tensor, max_value: float) -> Tensor:
    """Squared penalty is added to values above max_value"""
    shifted_value = value - max_value
    mask = (shifted_value > 0).type(value.dtype)
    return torch_sum(mean(square(shifted_value) * mask, dim=0))
