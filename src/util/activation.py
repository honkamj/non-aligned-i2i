"""Activation function related utilities"""

from typing import Callable

from torch import Tensor
from torch.nn import Softplus


def str_to_activation(activation: str) -> Callable[[Tensor], Tensor]:
    """Convert string to activation function"""
    if activation == 'linear':
        return lambda x: x
    if activation == 'softplus':
        return Softplus()
    raise ValueError(f"Undefined activation function {activation}")
