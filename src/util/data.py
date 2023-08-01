"""Data processing utility functions"""

from typing import Optional

from numpy import asarray, clip, ndarray
from numpy.typing import ArrayLike
from torch import Tensor, tensor

from algorithm.torch.deformation.masked import MaskedVolume


def optionally_as_ndarray(array: Optional[ArrayLike]) -> Optional[ndarray]:
    """Conver optionally to ndarray"""
    if array is None:
        return None
    return asarray(array)


def normalize(
    array: ndarray, mean_and_std: ndarray, min_and_max: Optional[ndarray] = None
) -> ndarray:
    """Normalize array by mean and std

    Channel dimension is assumed to be the last.
    """
    if min_and_max is not None:
        array = clip(array, min_and_max[0], min_and_max[1])
    return (array - mean_and_std[0]) / mean_and_std[1]


def denormalize_tensor(
    image: Tensor, mean_and_std: ArrayLike, mask: Optional[ArrayLike] = None
) -> Tensor:
    """Denormalize tensor by mean and std

    Channel dimension is assumed to be the first.
    """
    mean_and_std = tensor(mean_and_std, device=image.device)
    denormalized = image.swapaxes(1, -1) * mean_and_std[1] + mean_and_std[0]
    denormalized_scaled = denormalized.swapaxes(1, -1)
    if mask is not None:
        denormalized_scaled *= mask
    return denormalized_scaled


def denormalize_masked_volume(masked_volume: MaskedVolume, mean_and_std: ArrayLike) -> MaskedVolume:
    """Denoramlize masked volume by mean and std

    Channel dimension is assumed to be the first.
    """
    mean_and_std = tensor(mean_and_std, device=masked_volume.volume.device)
    denormalized = masked_volume.volume.swapaxes(1, -1) * mean_and_std[1] + mean_and_std[0]
    denormalized_scaled = denormalized.swapaxes(1, -1)
    if masked_volume.mask is not None:
        denormalized_scaled *= masked_volume.mask
    return MaskedVolume(denormalized_scaled, masked_volume.voxel_size, masked_volume.mask)
