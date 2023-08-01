"""Augmented style transform functions"""

from numpy import ndarray, roll


def swap_channels(image: ndarray, channel_axis: int=-1) -> ndarray:
    """Swaps image channels

    This is perfectly invariant to diffeomorphic deformations.
    """
    return roll(image, shift=1, axis=channel_axis)
