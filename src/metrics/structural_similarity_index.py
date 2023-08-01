"""Masked structural similarity index metric"""

from typing import Tuple

from numpy import ndarray, ones
from numpy import sum as np_sum
from scipy.ndimage import binary_erosion  # type: ignore
from skimage.metrics import structural_similarity  # type: ignore


def structural_similarity_index(
        label: ndarray,
        output: ndarray,
        content_mask: ndarray,
        evaluation_mask: ndarray,
        data_range: float
    ) -> Tuple[float, float]:
    """Masked structural similarity index

    Args:
        label: ndarray with shape (batch_size, n_channels, dim_1, ..., dim_{n_dims})
        output: ndarray with shape (batch_size, n_channels, dim_1, ..., dim_{n_dims})
        content_mask: ndarray with shape (batch_size, 1, dim_1, ..., dim_{n_dims}),
            marks valid regions in the image
        evaluation_mask: ndarray with shape (batch_size, 1, dim_1, ..., dim_{n_dims}),
            areas to take into account in evaluation

    Returns:
        structural_similarity_sum: Sum of structural similarity values over valid pixels
        averaging_mass: Divide by this number to get mean
    """
    batch_size = label.shape[0]
    n_dims = label.ndim - 2
    n_channels = label.shape[1]
    win_size = 7
    averaging_mass = 0.0
    structural_similarity_sum = 0.0
    for batch_index in range(batch_size):
        _, structural_similarity_volume = structural_similarity(
            label[batch_index],
            output[batch_index],
            data_range=data_range,
            channel_axis=0,
            gaussian_weights=False,
            full=True,
            win_size=win_size)
        pad = (win_size - 1) // 2
        structure = ones((3,) * n_dims)
        eroded_content_mask = binary_erosion(
            content_mask[batch_index][0],
            structure=structure,
            iterations=pad
        )[None]
        structural_similarity_sum += np_sum(
            structural_similarity_volume * eroded_content_mask * evaluation_mask[batch_index])
        averaging_mass += np_sum(eroded_content_mask * evaluation_mask[batch_index]) * n_channels
    return structural_similarity_sum, averaging_mass
