"""Data indexing sequences"""

from abc import ABC
from typing import Callable, Iterable, Optional, Sequence, Tuple

from numpy import (  # pylint: disable=no-name-in-module
    arange,
    argmax,
    array,
    asarray,
    ceil,
    concatenate,
    cumsum,
    empty,
    flip,
    floor,
    minimum,
    ndarray,
)
from numpy import prod as np_prod
from numpy import repeat, stack  # pylint: disable=no-name-in-module
from numpy import sum as np_sum
from numpy import unravel_index, zeros  # pylint: disable=no-name-in-module
from numpy.random import RandomState
from scipy.ndimage import gaussian_filter  # type: ignore

from util.typing import Number


class Patch:
    """Class representing one patch within an array"""

    def __init__(self, start_indices: Iterable[int], end_indices: Iterable[int]) -> None:
        self._start_indices = start_indices
        self._end_indices = end_indices
        self._array_slice = self._indices_to_slice(start_indices, end_indices)

    def extract(self, array_to_slice: ndarray) -> ndarray:
        """Extract patch from given array"""
        return array_to_slice[self._array_slice]

    def get_slice(self) -> Tuple[slice, ...]:
        """Select patch from an array"""
        return self._array_slice

    def export_to_numpy(self) -> ndarray:
        """Export patch information to numpy array"""
        return stack([asarray(self._start_indices), asarray(self._end_indices)], axis=0)

    @classmethod
    def from_numpy(cls, exported_array: ndarray) -> "Patch":
        """Factory method for creating patch from numpy export"""
        return cls(exported_array[0], exported_array[1])

    def __repr__(self) -> str:
        return f"Patch(start_indices={self._start_indices}, " f"end_indices={self._end_indices})"

    @staticmethod
    def _indices_to_slice(
        start_indices: Iterable[int],
        end_indices: Iterable[int],
    ) -> Tuple[slice, ...]:
        return tuple(slice(start, end) for (start, end) in zip(start_indices, end_indices))


class PatchCounter:
    """Calculates amount of patches in an array"""

    def __init__(
        self,
        stride: Sequence[Number],
        patch_size: Sequence[Number],
        include_last: bool,
    ) -> None:
        self._stride = asarray(stride)
        self._patch_size = asarray(patch_size)
        self._n_dims = len(self._patch_size)
        assert self._n_dims == len(self._stride)
        self._include_last = include_last

    def num_patches(self, shape: Sequence[Number]) -> int:
        """Calculate amount of patches"""
        return np_prod(self.num_patches_per_dimension(shape))

    def num_patches_per_dimension(self, shape: Sequence[Number]) -> ndarray:
        """Calculate amount of patches per dimension"""
        if self._include_last:
            rounding_function: Callable[[ndarray], ndarray] = ceil
        else:
            rounding_function = floor
        return (rounding_function((asarray(shape) - self._patch_size) / self._stride) + 1).astype(
            int
        )

    def _get_patch_start_indices_with_multidimensional_index(
        self, multidimensional_index: ndarray, shape: Sequence[Number]
    ) -> ndarray:
        shape_array = asarray(shape)
        max_start_indices = shape_array - self._patch_size
        return minimum(multidimensional_index * self._stride, max_start_indices)

    def _get_patch_with_multidimensional_index(
        self, multidimensional_index: ndarray, shape: Sequence[Number]
    ) -> Patch:
        start_indices = self._get_patch_start_indices_with_multidimensional_index(
            multidimensional_index, shape
        )
        return Patch(start_indices, start_indices + self._patch_size)

    def _get_multidimensional_index(
        self, one_dimensional_index: int, shape: Sequence[Number]
    ) -> ndarray:
        patches_per_dimensions = self.num_patches_per_dimension(shape)
        multidimensional_index = asarray(
            unravel_index(one_dimensional_index, patches_per_dimensions)
        )
        return multidimensional_index

    def _one_hot(self, n_dims: int, dim: int) -> ndarray:
        vector = zeros(n_dims, dtype=int)
        vector[dim] = 1
        return vector

    def get_patch(self, one_dimensional_index: int, shape: Sequence[Number]) -> Patch:
        """Return patch inside a volume based on 1d index"""
        return self._get_patch_with_multidimensional_index(
            self._get_multidimensional_index(one_dimensional_index, shape), shape
        )

    def get_patch_mask(self, one_dimensional_index: int, shape: Sequence[Number]) -> ndarray:
        """Return mask indicating which voxels will not be considered by other patches"""
        num_patches_per_dimension = self.num_patches_per_dimension(shape)
        multidimensional_index = self._get_multidimensional_index(one_dimensional_index, shape)
        current_start_indices = self._get_patch_start_indices_with_multidimensional_index(
            multidimensional_index, shape
        )
        mask = zeros(self._patch_size, dtype=bool)
        slices = []
        for dim in range(self._n_dims):
            if multidimensional_index[dim] == 0:
                lower_margin = 0
            else:
                neighbour_start_indices_lower = (
                    self._get_patch_start_indices_with_multidimensional_index(
                        multidimensional_index - self._one_hot(self._n_dims, dim), shape=shape
                    )
                )
                overlap_lower = (
                    neighbour_start_indices_lower[dim]
                    + self._patch_size[dim]
                    - current_start_indices[dim]
                )
                lower_margin = int(floor(max(overlap_lower, 0) / 2))
            if multidimensional_index[dim] == num_patches_per_dimension[dim] - 1:
                upper_index = None
            else:
                neighbour_start_indices_upper = (
                    self._get_patch_start_indices_with_multidimensional_index(
                        multidimensional_index + self._one_hot(self._n_dims, dim), shape=shape
                    )
                )
                overlap_upper = (
                    current_start_indices[dim]
                    + self._patch_size[dim]
                    - neighbour_start_indices_upper[dim]
                )
                upper_margin = int(ceil(max(overlap_upper, 0) / 2))
                if upper_margin == 0:
                    upper_index = None
                else:
                    upper_index = -upper_margin
            slices.append(slice(lower_margin, upper_index))
        mask[tuple(slices)] = 1.0
        return mask

    @property
    def stride(self) -> ndarray:
        """Return stride of the patches"""
        return self._stride

    @property
    def patch_size(self) -> ndarray:
        """Return size of the patches"""
        return self._patch_size


class IIndexSequence(ABC):
    """Represents sequence of slices from multiple multidimensional arrays"""

    def __getitem__(
        self, index_and_version: Tuple[int, int]
    ) -> Tuple[Patch, int, Optional[ndarray]]:
        """Returns patch, array index and optionally patch mask

        Args:
            index: First one is the sequence item index and second
                one is version of that sequence item.
        """

    def __len__(self) -> int:
        """Length of the sequence"""

    def generate(self, random_state: Optional[RandomState] = None) -> bool:
        """Generate the sequence"""


class BaseIndexSequence(IIndexSequence):
    """Base class for index sequence"""

    def __init__(self, array_shapes: Sequence[Sequence[int]], patch_counter: PatchCounter) -> None:
        self._array_shapes = array_shapes
        self._patch_counter = patch_counter
        self._patches_per_array = self._calculate_patches_per_array()
        self._length = self._calculate_length()

    def __getitem__(
        self, index_and_version: Tuple[int, int]
    ) -> Tuple[Patch, int, Optional[ndarray]]:
        index, version = index_and_version
        self._ensure_valid_index(index, version)
        array_index = self._get_array_index(index, version)
        patch, patch_mask = self._get_patch(index, version, array_index)
        return patch, array_index, patch_mask

    def __len__(self) -> int:
        return self._length

    def _calculate_patches_per_array(self) -> ndarray:
        return array(
            [self._patch_counter.num_patches(array_shape) for array_shape in self._array_shapes]
        )

    def _calculate_length(self) -> int:
        return int(np_sum(self._patches_per_array))

    def _ensure_valid_index(self, index: int, version: int) -> None:
        if index < 0 or index >= len(self):
            raise IndexError("Index out of range!")
        if version < 0:
            raise IndexError("Version out of range!")

    def _get_array_index(self, index: int, version: int) -> int:
        raise NotImplementedError

    def _get_patch(
        self, index: int, version: int, array_index: int
    ) -> Tuple[Patch, Optional[ndarray]]:
        raise NotImplementedError


class InferenceIndexSequence(BaseIndexSequence):
    """Random index sequence"""

    def __init__(
        self,
        array_shapes: Sequence[Sequence[int]],
        patch_counter: PatchCounter,
        fusing_mask_smoothing: float,
    ) -> None:
        super().__init__(array_shapes, patch_counter)
        self._array_start_indices = self._calculate_array_start_indices()
        self._fusing_mask_smoothing = fusing_mask_smoothing

    def generate(self, random_state: Optional[RandomState] = None) -> bool:
        return False

    def _calculate_array_start_indices(self) -> ndarray:
        cumulative_sum = cumsum(self._patches_per_array[:-1])
        start_indices = concatenate((array([0]), cumulative_sum))
        return start_indices

    def _get_array_index(self, index: int, _version: int) -> int:
        return (
            len(self._array_start_indices)
            - int(argmax(flip(index >= self._array_start_indices)))
            - 1
        )

    def _get_patch(
        self, index: int, _version: int, array_index: int
    ) -> Tuple[Patch, Optional[ndarray]]:
        array_shape = self._array_shapes[array_index]
        index_within_volume = index - self._array_start_indices[array_index]
        patch = self._patch_counter.get_patch(index_within_volume, array_shape)
        patch_mask = self._patch_counter.get_patch_mask(index_within_volume, array_shape).astype(
            "float32"
        )
        if self._fusing_mask_smoothing > 0:
            patch_mask = gaussian_filter(
                patch_mask, sigma=self._fusing_mask_smoothing, mode="nearest"
            )
        return patch, patch_mask


class RandomIndexSequence(BaseIndexSequence):
    """Random index sequence

    Args:
        array_shapes: Shapes of each array from which patches are extracted
        patch_counter: Instance of patch counter used for counting amount of
            patches in each array
        random_state: Random state used for generating random numbers
        shuffling_cluster_size: From how many arrays data is shuffled at a time. If 1,
            corresponds to reading data from one array at a time, if None, samples
            from all arrays are shuffled together.
    """

    def __init__(
        self,
        array_shapes: Sequence[Sequence[int]],
        patch_counter: PatchCounter,
        shuffling_cluster_size: Optional[int] = None,
    ) -> None:
        super().__init__(array_shapes, patch_counter)
        self._random_state_index = -1
        self._shuffling_cluster_size = shuffling_cluster_size
        self._base_seed: Optional[int] = None
        self._array_indices_by_index: Optional[ndarray] = None

    def generate(self, random_state: Optional[RandomState] = None) -> bool:
        if random_state is None:
            raise RuntimeError("Random state needed!")
        self._array_indices_by_index = self._generate_array_indices(random_state)
        self._base_seed = random_state.randint(2**32)
        return True

    def _generate_array_indices(self, random_state: RandomState) -> ndarray:
        n_arrays = len(self._array_shapes)
        random_array_permutation = random_state.permutation(n_arrays)
        shuffled_array_indices = arange(n_arrays, dtype="uint8")[random_array_permutation]
        shuffled_patches_per_array = self._patches_per_array[random_array_permutation]
        array_indices_by_index = empty(len(self), dtype="uint8")
        array_start_index = 0
        start_index = 0
        while array_start_index < n_arrays:
            if self._shuffling_cluster_size is not None:
                array_end_index = min(array_start_index + self._shuffling_cluster_size, n_arrays)
            else:
                array_end_index = n_arrays
            repeated_array_indices = repeat(
                shuffled_array_indices[array_start_index:array_end_index],
                repeats=shuffled_patches_per_array[array_start_index:array_end_index],
            )
            random_state.shuffle(repeated_array_indices)
            array_indices_by_index[
                start_index : start_index + len(repeated_array_indices)
            ] = repeated_array_indices
            start_index += len(repeated_array_indices)
            array_start_index = array_end_index
        assert start_index == len(self)
        assert array_start_index == n_arrays
        return array_indices_by_index

    def _get_array_index(self, index: int, _version: int) -> int:
        if self._array_indices_by_index is None:
            raise RuntimeError("Generate the sequence first by calling generate method")
        return self._array_indices_by_index[index]

    def _get_patch(
        self, index: int, version: int, array_index: int
    ) -> Tuple[Patch, Optional[ndarray]]:
        if self._base_seed is None:
            raise RuntimeError("Generate the sequence first by calling generate method")
        seed = self._base_seed ^ index
        random_state = RandomState(seed)
        for _ in range(version + 1):
            patch = self._generate_patch(array_index, random_state)
        return patch, None

    def _generate_patch(self, array_index: int, random_state: RandomState) -> Patch:
        start_indices = random_state.randint(
            self._array_shapes[array_index] - self._patch_counter.patch_size + 1
        )
        return Patch(
            start_indices=start_indices, end_indices=start_indices + self._patch_counter.patch_size
        )
