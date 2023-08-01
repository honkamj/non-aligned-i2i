"""Normalized mutual information"""

from math import floor, isnan
from typing import Tuple

from numpy import concatenate, ndarray, sqrt
from numpy import sum as np_sum
from skimage.metrics import (  # type: ignore
    normalized_mutual_information as
    skimage_normalized_mutual_information)


def normalized_mutual_information(
        label: ndarray,
        output: ndarray,
        mask: ndarray,
        bins=100
    ) -> Tuple[float, float]:
    """Masked structural similarity index

    Args:
        label: ndarray with shape (batch_size, n_channels, dim_1, ..., dim_{n_dims})
        output: ndarray with shape (batch_size, n_channels, dim_1, ..., dim_{n_dims})
        mask: ndarray with shape (batch_size, 1, dim_1, ..., dim_{n_dims})

    Returns:
        nmi_sum: Sum of NMI values over samples
        averaging_mass: Basically batch size
    """
    batch_size = label.shape[0]
    n_channels = label.shape[1]
    nmi_sum = 0.0
    for batch_index in range(batch_size):
        n_voxels = float(np_sum(mask[batch_index]))
        if n_voxels == 0:
            continue
        bins = max(
            int(floor(sqrt(np_sum(mask[batch_index]) / 5))),
            2
        )
        flattened_label = label[batch_index].flatten()
        flattened_output = output[batch_index].flatten()
        stacked_mask = concatenate(
            (mask[batch_index],) * n_channels,
            axis=0
        )
        flattened_mask = stacked_mask.flatten().astype(bool)
        nmi = skimage_normalized_mutual_information(
            image0=flattened_label[flattened_mask],
            image1=flattened_output[flattened_mask],
            bins=bins
        )
        if isnan(nmi):
            continue
        nmi_sum += nmi * n_voxels
    return nmi_sum, float(np_sum(mask))
