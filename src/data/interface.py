"""Data related interfaces"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple

from numpy.random import RandomState
from torch import Tensor
from torch.utils.data import DataLoader

from util.import_util import import_object

from .patch_sequence import Patch


class ITrainingArraySequence(ABC):
    """Interface representing random array sequence"""

    @property
    @abstractmethod
    def item_shape(self) -> Sequence[int]:
        """Shape of the sequence items"""

    @abstractmethod
    def generate(self, random_state: RandomState) -> None:
        """Generate the sequence

        Usually for example shuffles the items.
        """

    @abstractmethod
    def __len__(self) -> int:
        """Length of the sequence"""

    @abstractmethod
    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Sequence item

        Returns:
            Tuple of (input, label, input_mask, label_mask)
        """


class IInferenceArrayPatchSequence(ABC):
    """Interface representing inference array sequence"""

    @property
    @abstractmethod
    def item_shape(self) -> Sequence[int]:
        """Shape of the sequence items"""

    @property
    @abstractmethod
    def stride(self) -> Sequence[int]:
        """Stride of the moving window"""

    @abstractmethod
    def __len__(self) -> int:
        """Length of the sequence"""

    @abstractmethod
    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor, Patch, Tensor]:
        """Sequence item

        Returns:
            Tuple of (input, mask, Patch in the larger volume, patch mask)
        """


class IEvaluationArraySequence(ABC):
    """Interface representing evaluation array sequence"""

    @property
    @abstractmethod
    def item_shape(self) -> Sequence[int]:
        """Shape of the sequence items"""

    @abstractmethod
    def __len__(self) -> int:
        """Length of the sequence"""

    @abstractmethod
    def __getitem__(
        self, index: int
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Sequence item

        Returns:
            Tuple of (
                input,
                aligned_label,
                training_label,
                input_mask,
                aligned_label_mask,
                training_label_mask,
                ground_truth_deformation,
                patch_mask,
                evaluation_mask)
        """


def init_generic_training_data_loader(
    data_config: Mapping[str, Any],
    data_loader_config: Mapping[str, Any],
    seed: int,
    data_set: str,
    num_workers: int,
) -> Tuple[DataLoader, Callable[[], None]]:
    """Initialize training data loader from config"""
    init_func = import_object(data_loader_config["factory_function_training"])
    data_loader, generate_new_variant = init_func(
        data_config, data_loader_config, seed, data_set, num_workers
    )
    return data_loader, generate_new_variant


def init_generic_evaluation_data_loader(
    data_config: Mapping[str, Any],
    data_loader_config: Mapping[str, Any],
    shuffle: bool,
    data_set: str,
    seed: int,
    sequence_item: Optional[int] = None,
) -> DataLoader:
    """Initialize evaluation data loader from config"""
    init_func = import_object(data_loader_config["factory_function_evaluation"])
    data_loader = init_func(data_config, data_loader_config, shuffle, data_set, seed, sequence_item)
    return data_loader


def init_generic_inference_data_loader(
    sample: str, data_config: Mapping[str, Any], data_loader_config: Mapping[str, Any]
) -> DataLoader:
    """Initialize inference data loader from config"""
    init_func = import_object(data_loader_config["factory_function_inference"])
    return init_func(sample, data_config, data_loader_config)
