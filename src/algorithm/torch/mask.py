"""Mask related algorithms"""

from typing import Sequence
from torch import Tensor, mul
from torch import any as torch_any
from torch import argmax, flip
from torch import max as torch_max
from torch import min as torch_min
from torch import ones_like
from torch import sum as torch_sum
from torch import uint8


def mask_to_bounding_box_mask(mask: Tensor) -> Tensor:
    """Convert mask to bounding box mask

    Args:
        mask: Tensor with shape (batch_size, 1, dim_1, ..., dim_{n_dims})

    Returns:
        Tensor with shape (batch_size, 1, dim_1, ..., dim_{n_dims})
    """
    batch_size = mask.size(0)
    new_mask = ones_like(mask)
    for dim in range(2, mask.ndim):
        other_dims = tuple(
            iter_dim for iter_dim in range(2, mask.ndim)
            if iter_dim != dim)
        reduced_mask = (torch_sum(mask, dim=other_dims) > 0).type(uint8)
        first_indices = argmax(reduced_mask, dim=2)
        last_indices = reduced_mask.size(2) - argmax(flip(reduced_mask, dims=(2,)), dim=2)
        min_indices, _ = torch_min(first_indices.view(batch_size, -1), dim=1)
        max_indices, _ = torch_max(last_indices.view(batch_size, -1), dim=1)
        for batch_index in range(batch_size):
            if torch_any(reduced_mask[batch_index]):
                min_slice_tuple = (
                    (slice(None),) * (dim - 1) +
                    (slice(0, int(min_indices[batch_index])),)
                )
                max_slice_tuple = (
                    (slice(None),) * (dim - 1) +
                    (slice(int(max_indices[batch_index]), None),)
                )
                new_mask[batch_index][min_slice_tuple] = 0
                new_mask[batch_index][max_slice_tuple] = 0
            else:
                new_mask[batch_index][:] = 0

    return new_mask


def mask_and(masks: Sequence[Tensor]) -> Tensor:
    """Mask and operator, works with floating point values"""
    intersection = masks[0]
    for mask in masks[1:]:
        intersection = mul(intersection, mask)
    return intersection


def discretize_mask(mask: Tensor, error_threshold=1e-5) -> Tensor:
    """Make mask values discrete"""
    thresholded_mask = (mask >= 1 - error_threshold).type(mask.dtype)
    return thresholded_mask
