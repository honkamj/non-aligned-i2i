"""Base implementations for datasets"""


from multiprocessing.context import BaseContext
from numbers import Number
from typing import Optional, Sequence, Tuple, Union

from numpy import eye, ndarray, stack
from numpy.random import RandomState
from scipy.special import binom  # type: ignore
from torch import Tensor, from_numpy
from torch.utils.data import Dataset

from .interface import (
    IEvaluationArraySequence,
    IInferenceArrayPatchSequence,
    ITrainingArraySequence,
)
from .rigid_transformation import generate_simulated_affine_transformations


def training_worker_init_fn(worker_id: int):
    """Worker init function for using TrainingDataset in multiprocessing"""
    TrainingDataset.set_worker_id(worker_id)


class TrainingDataset(Dataset):
    """Dataset for training DI algorithm"""

    WORKER_ID: Optional[int] = None

    def __init__(
        self,
        sequence: ITrainingArraySequence,
        rotation_degree_range: Optional[Sequence[Tuple[float, float]]],
        log_scale_scale: Optional[Union[Sequence[float], float]],
        log_shear_scale: Optional[Union[Sequence[float], float]],
        translation_range: Optional[Sequence[Tuple[float, float]]],
        n_random_deformations: Optional[int],
        zero_random_deformation_prob: float,
        random_state: RandomState,
        voxel_size: Sequence[float],
        generate_orthogonal_rotations: bool,
        generate_flips: bool,
        input_noise_amplitude_range: Optional[Tuple[float, float]],
        label_noise_amplitude_range: Optional[Tuple[float, float]],
        multiprocessing_context: BaseContext,
    ) -> None:
        n_dims = len(voxel_size)
        n_rotation_axes = int(binom(n_dims, 2))
        self._sequence = sequence
        if log_scale_scale is None:
            self._log_scale_scale: Sequence[float] = [0.0] * n_dims
        elif isinstance(log_scale_scale, Sequence):
            self._log_scale_scale = log_scale_scale
        else:
            self._log_scale_scale = [log_scale_scale] * n_dims
        if log_shear_scale is None:
            self._log_shear_scale: Sequence[float] = [0.0] * int(n_dims * (n_dims - 1) / 2)
        elif isinstance(log_shear_scale, Sequence):
            self._log_shear_scale = log_shear_scale
        else:
            self._log_shear_scale = [log_shear_scale] * int(n_dims * (n_dims - 1) / 2)
        if translation_range is None:
            self._translation_range: Sequence[tuple[float, float]] = [(0.0, 0.0)] * n_dims
        elif isinstance(translation_range[0], Number):
            self._translation_range = [translation_range] * n_dims
        else:
            self._translation_range = translation_range
        if rotation_degree_range is None:
            self._rotation_degree_range: Sequence[tuple[float, float]] = [
                (0.0, 0.0)
            ] * n_rotation_axes
        elif isinstance(rotation_degree_range[0], Number):
            self._rotation_degree_range = [rotation_degree_range] * n_rotation_axes
        else:
            self._rotation_degree_range = rotation_degree_range
        self._generate_orthogonal_rotations = generate_orthogonal_rotations
        self._generate_flips = generate_flips
        self._zero_random_deformation_prob = zero_random_deformation_prob
        self._input_noise_amplitude_range = input_noise_amplitude_range
        self._label_noise_amplitude_range = label_noise_amplitude_range
        self._n_random_deformations = n_random_deformations
        self._voxel_size = voxel_size
        self._base_random_state = random_state
        self._base_seed = self._sample_base_seed()
        self._sequence.generate(self._base_random_state)
        self._process_random_state: Optional[RandomState] = None
        self._shared_generation = multiprocessing_context.Value("i", 0)
        self._local_generation = 0

    def _sample_base_seed(self) -> int:
        return self._base_random_state.randint(2**32, dtype="uint32")

    @staticmethod
    def _generate_noise(
        shape: Sequence[int], random_state: RandomState, amplitude_range: Tuple[float, float]
    ) -> ndarray:
        noise_level = random_state.uniform(amplitude_range[0], amplitude_range[1])
        return random_state.normal(0, noise_level, size=tuple(shape)).astype("float32")

    def _get_augmentation_random_state(self) -> RandomState:
        if self.WORKER_ID is None:
            return self._base_random_state
        if self._process_random_state is None:
            self._process_random_state = RandomState(self._base_seed + self.WORKER_ID)
        return self._process_random_state

    @classmethod
    def set_worker_id(cls, worker_id: int) -> None:
        """Set ID of the process where the dataset is running"""
        cls.WORKER_ID = worker_id

    def _sync_generation(self) -> None:
        while self._local_generation < self._shared_generation.value:  # type: ignore
            self._base_seed = self._sample_base_seed()
            self._sequence.generate(self._base_random_state)
            self._process_random_state = None
            self._local_generation += 1

    def generate_new_variant(self) -> None:
        """Generate new variant of the dataset"""
        with self._shared_generation.get_lock():
            self._shared_generation.value += 1  # type: ignore

    def __len__(self) -> int:
        self._sync_generation()
        return len(self._sequence)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        self._sync_generation()
        random_state = self._get_augmentation_random_state()
        input_patch, label_patch, input_mask_patch, label_mask_patch = self._sequence[index]
        n_dims = len(self._sequence.item_shape)
        if random_state.uniform() < self._zero_random_deformation_prob:
            random_deformation = eye(n_dims + 1, dtype="float32")
            if self._n_random_deformations is not None:
                random_deformation = stack(
                    [random_deformation] * self._n_random_deformations, axis=0
                )
        else:
            random_deformation = (
                generate_simulated_affine_transformations(
                    batch_size=1
                    if self._n_random_deformations is None
                    else self._n_random_deformations,
                    degree_ranges=self._rotation_degree_range,
                    log_scale_scales=self._log_scale_scale,
                    log_shear_scales=self._log_shear_scale,
                    translation_ranges=self._translation_range,
                    generate_orthogonal_rotations=self._generate_orthogonal_rotations,
                    generate_flips=self._generate_flips,
                    shape=self._sequence.item_shape,
                    random_state=random_state,
                    voxel_size=self._voxel_size,
                )
            ).astype("float32")
            if self._n_random_deformations is None:
                random_deformation = random_deformation[0]
        random_deformation_torch = from_numpy(random_deformation)
        if self._input_noise_amplitude_range is None:
            augmented_input_patch = input_patch
        else:
            augmented_input_patch = input_patch + from_numpy(
                self._generate_noise(
                    shape=input_patch.shape,
                    random_state=random_state,
                    amplitude_range=self._input_noise_amplitude_range,
                )
            )
        if self._label_noise_amplitude_range is None:
            augmented_label_patch = label_patch
        else:
            augmented_label_patch = label_patch + from_numpy(
                self._generate_noise(
                    shape=label_patch.shape,
                    random_state=random_state,
                    amplitude_range=self._label_noise_amplitude_range,
                )
            )
        return (
            augmented_input_patch,
            augmented_label_patch,
            input_mask_patch,
            label_mask_patch,
            random_deformation_torch,
        )


class InferenceDataset(Dataset):
    """Dataset for patched inference"""

    def __init__(self, sequence: IInferenceArrayPatchSequence) -> None:
        self._sequence = sequence

    def __len__(self) -> int:
        return len(self._sequence)

    def __getitem__(self, index) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        input_patch, mask_patch, patch, patch_mask = self._sequence[index]
        return (
            input_patch,
            mask_patch,
            patch_mask,
            from_numpy(patch.export_to_numpy()),
        )


class EvaluationDataset(Dataset):
    """Dataset for evaluation"""

    def __init__(self, sequence: IEvaluationArraySequence) -> None:
        self._sequence = sequence

    def __len__(self) -> int:
        return len(self._sequence)

    def __getitem__(
        self, index
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        return self._sequence[index]


class SequenceDataset(Dataset):
    """Dataset based on a sequence"""

    def __init__(self, sequence: Sequence) -> None:
        self._sequence = sequence

    def __len__(self) -> int:
        return len(self._sequence)

    def __getitem__(self, index) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        return self._sequence[index]
