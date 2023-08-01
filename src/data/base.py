"""Base implementation for sequence iterating multiple arrays as patches"""

from abc import abstractmethod
from os import environ
from os.path import isfile, join
from shutil import copy
from typing import Optional, Sequence, Tuple

from numpy import any as np_any, zeros_like
from numpy import ndarray
from numpy import prod as np_prod
from numpy import sum as np_sum
from numpy import zeros
from numpy.random import RandomState
from torch import Tensor, from_numpy

from util.data import normalize
from util.import_util import import_object

from .interface import (
    IEvaluationArraySequence,
    IInferenceArrayPatchSequence,
    ITrainingArraySequence,
)
from .patch_sequence import InferenceIndexSequence, Patch, PatchCounter, RandomIndexSequence


def _obtain_shapes(arrays: Sequence[ndarray], ndims: int) -> Sequence[Sequence[int]]:
    return [array.shape[:ndims] for array in arrays]


def _to_float32_tensor(array: ndarray) -> Tensor:
    return from_numpy(array.astype("float32"))


def _channel_first(array: Tensor, item_shape: Sequence[int]) -> Tensor:
    if array.ndim == len(item_shape):
        assert list(array.shape) == item_shape
        return array[None]
    assert list(array.shape[:-1]) == item_shape
    return array.permute(-1, *range(0, array.ndim - 1))


class BaseTrainingArrayPatchSequence(ITrainingArraySequence):
    """Samples patches from multiple files"""

    def __init__(
        self,
        input_paths: Sequence[str],
        label_paths: Sequence[str],
        input_mask_paths: Sequence[str],
        label_mask_paths: Sequence[str],
        input_mean_and_std: ndarray,
        label_mean_and_std: ndarray,
        input_min_and_max: Optional[ndarray],
        label_min_and_max: Optional[ndarray],
        stride: Sequence[int],
        patch_size: Sequence[int],
        min_input_mask_ratio: float,
        min_label_mask_ratio: float,
        shuffling_cluster_size: Optional[int] = None,
        mask_threshold: Optional[float] = None,
        paired: bool = True,
    ) -> None:
        # Postpone creation of the memmaps until the worker processes have been
        # started, since otherwise the whole arrays are pickled.
        self._input_arrays: Optional[list[ndarray]] = None
        self._label_arrays: Optional[list[ndarray]] = None
        self._input_mask_arrays: Optional[list[ndarray]] = None
        self._label_mask_arrays: Optional[list[ndarray]] = None

        self._input_paths = input_paths
        self._label_paths = label_paths
        self._input_mask_paths = input_mask_paths
        self._label_mask_paths = label_mask_paths

        self._input_mean_and_std = input_mean_and_std
        self._label_mean_and_std = label_mean_and_std
        self._input_min_and_max = input_min_and_max
        self._label_min_and_max = label_min_and_max
        self._patch_size = patch_size
        self._random_index_sequence = RandomIndexSequence(
            array_shapes=_obtain_shapes(
                list(map(self._to_array, input_paths)), ndims=len(patch_size)
            ),
            patch_counter=PatchCounter(stride, patch_size, include_last=False),
            shuffling_cluster_size=shuffling_cluster_size,
        )
        if paired:
            self._label_random_index_sequence = None
        else:
            self._label_random_index_sequence = RandomIndexSequence(
                array_shapes=_obtain_shapes(
                    list(map(self._to_array, label_paths)), ndims=len(patch_size)
                ),
                patch_counter=PatchCounter(stride, patch_size, include_last=False),
                shuffling_cluster_size=shuffling_cluster_size,
            )

        self._min_input_mask_ratio = min_input_mask_ratio
        self._min_label_mask_ratio = min_label_mask_ratio
        self._mask_threshold = mask_threshold

    def _get_arrays(self) -> tuple[list[ndarray], list[ndarray], list[ndarray], list[ndarray]]:
        if (
            self._input_arrays is None
            or self._label_arrays is None
            or self._input_mask_arrays is None
            or self._label_mask_arrays is None
        ):
            self._input_arrays = list(map(self._to_array, self._input_paths))
            self._label_arrays = list(map(self._to_array, self._label_paths))
            self._input_mask_arrays = list(map(self._to_array, self._input_mask_paths))
            self._label_mask_arrays = list(map(self._to_array, self._label_mask_paths))
        return (
            self._input_arrays,
            self._label_arrays,
            self._input_mask_arrays,
            self._label_mask_arrays,
        )

    @property
    def item_shape(self) -> Sequence[int]:
        return self._patch_size

    def generate(self, random_state: RandomState) -> None:
        """Generate the sequence"""
        self._random_index_sequence.generate(random_state)
        if self._label_random_index_sequence is not None:
            self._label_random_index_sequence.generate(random_state)

    @abstractmethod
    def _to_array(self, path: str) -> ndarray:
        """Generate array from path"""

    def __len__(self) -> int:
        return len(self._random_index_sequence)

    def _get_random_patch(
        self, random_index_sequence: RandomIndexSequence, index: int, version: int
    ) -> tuple[Patch, int]:
        random_patch, array_index, _patch_mask = random_index_sequence[index, version]
        return random_patch, array_index

    @staticmethod
    def _is_suitable_patch(
        mask_arrays: Sequence[ndarray],
        array_index: int,
        patch: Patch,
        threshold: float,
    ) -> bool:
        mask = patch.extract(mask_arrays[array_index]) > 0
        n_pixels = np_prod(mask.shape)
        ratio = np_sum(mask) / n_pixels
        return bool(ratio >= threshold)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Sample random patch from given files"""
        (
            input_arrays,
            label_arrays,
            input_mask_arrays,
            label_mask_arrays,
        ) = self._get_arrays()
        input_version = 0
        while True:
            input_random_patch, input_array_index = self._get_random_patch(
                random_index_sequence=self._random_index_sequence,
                index=index,
                version=input_version,
            )
            if self._is_suitable_patch(
                mask_arrays=input_mask_arrays,
                array_index=input_array_index,
                patch=input_random_patch,
                threshold=self._min_input_mask_ratio,
            ) and (
                self._label_random_index_sequence is not None
                or self._is_suitable_patch(
                    mask_arrays=label_mask_arrays,
                    array_index=input_array_index,
                    patch=input_random_patch,
                    threshold=self._min_label_mask_ratio,
                )
            ):
                break
            input_version += 1
        if self._label_random_index_sequence is None:
            label_random_patch = input_random_patch
            label_array_index = input_array_index
        else:
            label_version = 0
            while True:
                label_random_patch, label_array_index = self._get_random_patch(
                    random_index_sequence=self._label_random_index_sequence,
                    index=index,
                    version=label_version,
                )
                if self._is_suitable_patch(
                    mask_arrays=label_mask_arrays,
                    array_index=label_array_index,
                    patch=label_random_patch,
                    threshold=self._min_label_mask_ratio,
                ):
                    break
                label_version += 1
        input_patch = input_random_patch.extract(input_arrays[input_array_index])
        label_patch = label_random_patch.extract(label_arrays[label_array_index])
        input_mask_patch = input_random_patch.extract(input_mask_arrays[input_array_index])
        label_mask_patch = label_random_patch.extract(label_mask_arrays[label_array_index])
        if self._mask_threshold is not None:
            input_mask_patch = (
                np_any(input_patch > self._mask_threshold, axis=-1) & input_mask_patch
            )
            label_mask_patch = (
                np_any(label_patch > self._mask_threshold, axis=-1) & label_mask_patch
            )
        input_patch = normalize(
            input_patch.astype("float32"), self._input_mean_and_std, self._input_min_and_max
        )
        label_patch = normalize(
            label_patch.astype("float32"), self._label_mean_and_std, self._label_min_and_max
        )
        input_mask_torch = _channel_first(
            _to_float32_tensor(input_mask_patch), item_shape=self.item_shape
        )
        label_mask_torch = _channel_first(
            _to_float32_tensor(label_mask_patch), item_shape=self.item_shape
        )
        return (
            _channel_first(_to_float32_tensor(input_patch), item_shape=self.item_shape)
            * input_mask_torch,
            _channel_first(_to_float32_tensor(label_patch), item_shape=self.item_shape)
            * label_mask_torch,
            input_mask_torch,
            label_mask_torch,
        )


class BaseInferenceArrayPatchSequence(IInferenceArrayPatchSequence):
    """Obtains patches from arrays for inference"""

    def __init__(
        self,
        input_paths: Sequence[str],
        mask_paths: Sequence[str],
        input_mean_and_std: ndarray,
        input_min_and_max: Optional[ndarray],
        stride: Sequence[int],
        patch_size: Sequence[int],
        fusing_mask_smoothing: float,
        mask_threshold: Optional[float] = None,
    ) -> None:
        # Postpone creation of the memmaps until the worker processes have been
        # started, since otherwise the whole arrays are pickled.
        self._input_arrays: Optional[list[ndarray]] = None
        self._mask_arrays: Optional[list[ndarray]] = None
        self._input_paths = input_paths
        self._mask_paths = mask_paths
        self._input_mean_and_std = input_mean_and_std
        self._input_min_and_max = input_min_and_max
        self._patch_size = patch_size
        self._stride = stride
        self._inference_index_sequence = InferenceIndexSequence(
            array_shapes=_obtain_shapes(
                list(map(self._to_array, input_paths)), ndims=len(patch_size)
            ),
            patch_counter=PatchCounter(stride, patch_size, include_last=True),
            fusing_mask_smoothing=fusing_mask_smoothing,
        )
        self._mask_threshold = mask_threshold

    def _get_arrays(self) -> tuple[list[ndarray], list[ndarray]]:
        if self._input_arrays is None or self._mask_arrays is None:
            self._input_arrays = list(map(self._to_array, self._input_paths))
            self._mask_arrays = list(map(self._to_array, self._mask_paths))
        return (
            self._input_arrays,
            self._mask_arrays,
        )

    @property
    def item_shape(self) -> Sequence[int]:
        return self._patch_size

    @property
    def stride(self) -> Sequence[int]:
        return self._stride

    @abstractmethod
    def _to_array(self, path: str) -> ndarray:
        """Generate array from path"""

    def __len__(self) -> int:
        return len(self._inference_index_sequence)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor, Patch, Tensor]:
        """Get sequential patch from given files"""
        input_arrays, mask_arrays = self._get_arrays()
        patch, array_index, patch_mask = self._inference_index_sequence[index, 0]
        if patch_mask is None:
            raise ValueError("Patch mask required for inference")
        mask_patch = patch.extract(mask_arrays[array_index]) > 0
        input_patch = patch.extract(input_arrays[array_index])
        if self._mask_threshold is not None:
            mask_patch = np_any(input_patch > self._mask_threshold, axis=-1) & mask_patch
        input_patch = normalize(
            input_patch.astype("float32"), self._input_mean_and_std, self._input_min_and_max
        )
        mask_torch = _channel_first(_to_float32_tensor(mask_patch), item_shape=self.item_shape)
        patch_mask_torch = _channel_first(
            _to_float32_tensor(patch_mask), item_shape=self.item_shape
        )
        return (
            _channel_first(_to_float32_tensor(input_patch), item_shape=self.item_shape)
            * mask_torch,
            mask_torch,
            patch,
            patch_mask_torch,
        )


class BaseEvaluationArrayPatchSequence(IEvaluationArraySequence):
    """Obtains patches from arrays for evaluation"""

    def __init__(
        self,
        input_paths: Sequence[str],
        aligned_label_paths: Sequence[str],
        non_aligned_label_paths: Sequence[str],
        training_label_paths: Sequence[str],
        input_mask_paths: Sequence[str],
        aligned_label_mask_paths: Sequence[str],
        non_aligned_label_mask_paths: Sequence[str],
        training_label_mask_paths: Sequence[str],
        evaluation_mask_paths: Sequence[str],
        ground_truth_deformation_paths: Optional[Sequence[str]],
        input_mean_and_std: ndarray,
        label_mean_and_std: ndarray,
        input_min_and_max: Optional[ndarray],
        label_min_and_max: Optional[ndarray],
        stride: Sequence[int],
        patch_size: Sequence[int],
        voxel_size: Sequence[float],
        mask_threshold: Optional[float] = None,
        use_affinely_registered_non_aligned_label_as_ground_truth: bool = False,
        affine_transformation_cache_path: Optional[str] = None,
        affine_registration_seed: Optional[int] = None,
    ) -> None:
        self._patch_size = patch_size
        # Postpone creation of the memmaps until the worker processes have been
        # started, since otherwise the whole arrays are pickled.
        self._input_arrays: Optional[list[ndarray]] = None
        self._aligned_label_arrays: Optional[list[ndarray]] = None
        self._non_aligned_label_arrays: Optional[list[ndarray]] = None
        self._training_label_arrays: Optional[list[ndarray]] = None
        self._input_mask_arrays: Optional[list[ndarray]] = None
        self._aligned_label_mask_arrays: Optional[list[ndarray]] = None
        self._non_aligned_label_mask_arrays: Optional[list[ndarray]] = None
        self._training_label_mask_arrays: Optional[list[ndarray]] = None
        self._evaluation_mask_arrays: Optional[list[ndarray]] = None
        self._ground_truth_deformation_arrays: Optional[list[ndarray]] = None

        self._input_paths = input_paths
        self._aligned_label_paths = aligned_label_paths
        self._non_aligned_label_paths = non_aligned_label_paths
        self._training_label_paths = training_label_paths
        self._input_mask_paths = input_mask_paths
        self._aligned_label_mask_paths = aligned_label_mask_paths
        self._non_aligned_label_mask_paths = non_aligned_label_mask_paths
        self._training_label_mask_paths = training_label_mask_paths
        self._evaluation_mask_paths = evaluation_mask_paths
        self._ground_truth_deformation_paths = ground_truth_deformation_paths

        self._input_mean_and_std = input_mean_and_std
        self._label_mean_and_std = label_mean_and_std
        self._input_min_and_max = input_min_and_max
        self._label_min_and_max = label_min_and_max
        self._inference_index_sequence = InferenceIndexSequence(
            array_shapes=_obtain_shapes(
                list(map(self._to_array, input_paths)), ndims=len(patch_size)
            ),
            patch_counter=PatchCounter(stride, patch_size, include_last=True),
            fusing_mask_smoothing=0.0,
        )
        self._voxel_size = tuple(voxel_size)
        self._mask_threshold = mask_threshold
        self._use_affinely_registered_non_aligned_label_as_ground_truth = (
            use_affinely_registered_non_aligned_label_as_ground_truth
        )
        self._affine_transformation_cache_path = affine_transformation_cache_path
        self._affine_registration_seed = affine_registration_seed
        # Import dynamically to save process memory during training
        if use_affinely_registered_non_aligned_label_as_ground_truth:
            environ[
                "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"
            ] = "1"  # ANTs registration is not deterministic with multiple threads
            self._ants_from_numpy = import_object("ants.from_numpy")
            self._ants_register = import_object("ants.registration")
            self._ants_apply_transforms = import_object("ants.apply_transforms")

    def _get_arrays(
        self,
    ) -> tuple[
        list[ndarray],
        list[ndarray],
        list[ndarray],
        list[ndarray],
        list[ndarray],
        list[ndarray],
        list[ndarray],
        list[ndarray],
        list[ndarray],
        Optional[list[ndarray]],
    ]:
        if (
            self._input_arrays is None
            or self._training_label_arrays is None
            or self._aligned_label_arrays is None
            or self._non_aligned_label_arrays is None
            or self._input_mask_arrays is None
            or self._aligned_label_mask_arrays is None
            or self._non_aligned_label_mask_arrays is None
            or self._training_label_mask_arrays is None
            or self._evaluation_mask_arrays is None
            or (
                self._ground_truth_deformation_arrays is None
                and self._ground_truth_deformation_paths is not None
            )
        ):
            self._input_arrays = list(map(self._to_array, self._input_paths))
            self._aligned_label_arrays = list(map(self._to_array, self._aligned_label_paths))
            self._non_aligned_label_arrays = list(
                map(self._to_array, self._non_aligned_label_paths)
            )
            self._training_label_arrays = list(map(self._to_array, self._training_label_paths))
            self._input_mask_arrays = list(map(self._to_array, self._input_mask_paths))
            self._training_label_mask_arrays = list(
                map(self._to_array, self._training_label_mask_paths)
            )
            self._aligned_label_mask_arrays = list(
                map(self._to_array, self._aligned_label_mask_paths)
            )
            self._non_aligned_label_mask_arrays = list(
                map(self._to_array, self._non_aligned_label_mask_paths)
            )
            self._evaluation_mask_arrays = list(map(self._to_array, self._evaluation_mask_paths))
            self._ground_truth_deformation_arrays = (
                None
                if self._ground_truth_deformation_paths is None
                else list(map(self._deformation_to_array, self._ground_truth_deformation_paths))
            )
        return (
            self._input_arrays,
            self._aligned_label_arrays,
            self._non_aligned_label_arrays,
            self._training_label_arrays,
            self._input_mask_arrays,
            self._aligned_label_mask_arrays,
            self._non_aligned_label_mask_arrays,
            self._training_label_mask_arrays,
            self._evaluation_mask_arrays,
            self._ground_truth_deformation_arrays,
        )

    @property
    def item_shape(self) -> Sequence[int]:
        return self._patch_size

    @abstractmethod
    def _to_array(self, path: str) -> ndarray:
        """Generate array from path"""

    @abstractmethod
    def _deformation_to_array(self, path: str) -> ndarray:
        """Generate array from path to deformation"""

    def __len__(self) -> int:
        return len(self._inference_index_sequence)

    def _register_volumes(
        self,
        moving_volume: ndarray,
        moving_mask: ndarray,
        fixed_volume: ndarray,
        fixed_mask: ndarray,
        index: int,
    ) -> tuple[ndarray, ndarray]:
        # Due to a limitation in Python interface of ANTs, we can not set a mask
        # for the moving image Hence we simply skip images for which the movig
        # volume is not fully within mask. Additionally we skip images with very
        # little content.
        n_voxels = np_prod(fixed_mask.shape)
        skip_file_path = (
            join(
                self._affine_transformation_cache_path,
                f"{index}_{self._affine_registration_seed}_skip.info",
            )
            if self._affine_transformation_cache_path is not None
            and self._affine_registration_seed is not None
            else None
        )
        if (
            np_sum(fixed_mask) / n_voxels < 0.05
            or np_sum(moving_mask > 0) < n_voxels - 1e-5
            or (skip_file_path is not None and isfile(skip_file_path))
        ):
            return moving_volume, zeros_like(moving_mask)
        moving_projected_ants = self._ants_from_numpy(
            (moving_volume.astype("float32") / 255).mean(axis=-1),
            spacing=self._voxel_size,
        )
        fixed_projected_ants = self._ants_from_numpy(
            (fixed_volume.astype("float32") / 255).mean(axis=-1),
            spacing=self._voxel_size,
        )
        moving_mask_ants = self._ants_from_numpy(
            (moving_mask > 0).astype("float32"), spacing=self._voxel_size
        )
        fixed_mask_ants = self._ants_from_numpy(
            (fixed_mask > 0).astype("float32"), spacing=self._voxel_size
        )
        transformation_file_path = (
            join(
                self._affine_transformation_cache_path,
                f"{index}_{self._affine_registration_seed}.mat",
            )
            if self._affine_transformation_cache_path is not None
            and self._affine_registration_seed is not None
            else None
        )
        output_transformation = None
        if transformation_file_path is None or not isfile(transformation_file_path):
            try:
                registration_output = self._ants_register(
                    fixed=fixed_projected_ants,
                    moving=moving_projected_ants,
                    mask=fixed_mask_ants if fixed_mask_ants.numpy().min() == 0 else None,
                    type_of_transform="Affine",
                    initial_transform=None,
                    aff_random_sampling_rate=0.2,
                    aff_iterations=(2100, 1200, 1200, 100),
                    random_seed=self._affine_registration_seed,
                    verbose=False,
                )
            except RuntimeError:
                if skip_file_path is not None:
                    with open(
                        skip_file_path,
                        encoding="utf-8",
                        mode="w",
                    ) as _skip_file:
                        pass
                return moving_volume, zeros_like(moving_mask)
            output_transformation = registration_output["fwdtransforms"][0]
            if transformation_file_path is not None:
                copy(
                    registration_output["fwdtransforms"][0],
                    transformation_file_path,
                )
        else:
            output_transformation = transformation_file_path
        moved_volume = zeros(moving_volume.shape, dtype="float32")
        # Apply the transform to image separately for each channel since
        # otherwise ANTsPy crashed
        for channel in range(moving_volume.shape[-1]):
            moving_ants_single_channel = self._ants_from_numpy(
                moving_volume[..., channel].astype("float32"), spacing=self._voxel_size
            )
            registered_moving_single_channel_ants = self._ants_apply_transforms(
                fixed=fixed_projected_ants,
                moving=moving_ants_single_channel,
                transformlist=[output_transformation],
                verbose=False,
            )
            moved_volume[..., channel] = registered_moving_single_channel_ants.numpy()
        moved_mask_ants = self._ants_apply_transforms(
            fixed=fixed_projected_ants,
            moving=moving_mask_ants,
            transformlist=[output_transformation],
            verbose=False,
        )
        return moved_volume, moved_mask_ants.numpy() > 1 - 1e-5

    def __getitem__(
        self, index: int
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        (
            input_arrays,
            aligned_label_arrays,
            non_aligned_label_arrays,
            training_label_arrays,
            input_mask_arrays,
            aligned_label_mask_arrays,
            non_aligned_label_mask_arrays,
            training_label_mask_arrays,
            evaluation_mask_arrays,
            ground_truth_deformation_arrays,
        ) = self._get_arrays()
        patch, array_index, patch_mask = self._inference_index_sequence[index, 0]
        if patch_mask is None:
            raise ValueError("Patch mask required for evaluation")
        input_patch = patch.extract(input_arrays[array_index])
        training_label_patch = patch.extract(training_label_arrays[array_index])
        input_mask_patch = patch.extract(input_mask_arrays[array_index]) > 0
        if self._use_affinely_registered_non_aligned_label_as_ground_truth:
            aligned_label_patch, aligned_label_mask_patch = self._register_volumes(
                moving_volume=patch.extract(non_aligned_label_arrays[array_index]),
                moving_mask=patch.extract(non_aligned_label_mask_arrays[array_index]),
                fixed_volume=input_patch,
                fixed_mask=input_mask_patch,
                index=index,
            )
        else:
            aligned_label_patch = patch.extract(aligned_label_arrays[array_index])
            aligned_label_mask_patch = patch.extract(aligned_label_mask_arrays[array_index]) > 0
        training_label_mask_patch = patch.extract(training_label_mask_arrays[array_index]) > 0
        evaluation_mask_patch = patch.extract(evaluation_mask_arrays[array_index]) > 0
        if self._mask_threshold is not None:
            input_mask_patch = (
                np_any(input_patch > self._mask_threshold, axis=-1) & input_mask_patch
            )
            aligned_label_mask_patch = (
                np_any(aligned_label_patch > self._mask_threshold, axis=-1)
                & aligned_label_mask_patch
            )
            training_label_mask_patch = (
                np_any(training_label_patch > self._mask_threshold, axis=-1)
                & training_label_mask_patch
            )
        if ground_truth_deformation_arrays is None:
            ground_truth_deformation_patch = zeros(
                tuple(self.item_shape) + (len(self.item_shape),), dtype="float32"
            )
        else:
            ground_truth_deformation_patch = patch.extract(
                ground_truth_deformation_arrays[array_index]
            )
        input_patch = normalize(
            input_patch.astype("float32"), self._input_mean_and_std, self._input_min_and_max
        )
        aligned_label_patch = normalize(
            aligned_label_patch.astype("float32"), self._label_mean_and_std, self._label_min_and_max
        )
        training_label_patch = normalize(
            training_label_patch.astype("float32"),
            self._label_mean_and_std,
            self._label_min_and_max,
        )
        input_mask_torch = _channel_first(
            _to_float32_tensor(input_mask_patch), item_shape=self.item_shape
        )
        aligned_label_mask_torch = _channel_first(
            _to_float32_tensor(aligned_label_mask_patch), item_shape=self.item_shape
        )
        training_label_mask_torch = _channel_first(
            _to_float32_tensor(training_label_mask_patch), item_shape=self.item_shape
        )
        evaluation_mask_torch = _channel_first(
            _to_float32_tensor(evaluation_mask_patch), item_shape=self.item_shape
        )
        patch_mask_torch = _channel_first(
            _to_float32_tensor(patch_mask), item_shape=self.item_shape
        )

        return (
            _channel_first(_to_float32_tensor(input_patch), item_shape=self.item_shape)
            * input_mask_torch,
            _channel_first(_to_float32_tensor(aligned_label_patch), item_shape=self.item_shape)
            * aligned_label_mask_torch,
            _channel_first(_to_float32_tensor(training_label_patch), item_shape=self.item_shape)
            * training_label_mask_torch,
            input_mask_torch,
            aligned_label_mask_torch,
            training_label_mask_torch,
            _channel_first(
                _to_float32_tensor(ground_truth_deformation_patch), item_shape=self.item_shape
            ),
            patch_mask_torch,
            evaluation_mask_torch,
        )
