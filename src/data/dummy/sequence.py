"""Synthetic dataset sequences"""


from typing import Sequence, Tuple

from numpy.random import RandomState
from torch import Tensor, float32, ones

from data.interface import ITrainingArraySequence


class TrainingDummySequence(ITrainingArraySequence):
    """Dummy sequence of all ones"""
    def __init__(self, patch_size: Tuple[int, ...], length: int) -> None:
        self._patch_size = patch_size
        self._length = length

    @property
    def item_shape(self) -> Sequence[int]:
        return self._patch_size

    def generate(self, _random_state: RandomState) -> None:
        """Generate the sequence"""

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Generate dummy sequence"""
        return (
            ones((3,) + self._patch_size, dtype=float32),
            ones((3,) + self._patch_size, dtype=float32),
            ones((1,) + self._patch_size, dtype=float32),
            ones((1,) + self._patch_size, dtype=float32)
        )
