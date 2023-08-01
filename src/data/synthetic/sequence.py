"""Synthetic dataset sequences"""


from typing import Optional, Sequence, Tuple, cast

from numpy import arange, ndarray, transpose
from numpy.random import RandomState
from skimage.io import imread  # type: ignore
from torch import Tensor, float32, from_numpy, ones_like
from torch import zeros as torch_zeros

from algorithm.augmented_style_transform import swap_channels
from algorithm.largest_interior_rectangle import largest_interior_rectangle
from algorithm.torch.deformation.primitive import deform_volume
from algorithm.torch.simulated_elastic_deformation import (
    GaussianElasticDeformationDefinition,
    generate_gaussian_ddf,
)
from data.interface import IEvaluationArraySequence, ITrainingArraySequence
from util.data import normalize
from util.image import gray_to_color_if_needed


class TrainingSyntheticSequence(ITrainingArraySequence):
    """Samples random images, label is deformed version of
    the image with channels swapped"""

    def __init__(
        self,
        input_paths: Sequence[str],
        deformations: Sequence[GaussianElasticDeformationDefinition],
        input_mean_and_std: ndarray,
        label_mean_and_std: ndarray,
        patch_size: Tuple[int, int],
        paired: bool = True,
    ) -> None:
        self._input_paths = input_paths
        self._input_mean_and_std = input_mean_and_std
        self._label_mean_and_std = label_mean_and_std
        self._order_indices: Optional[Sequence[int]] = None
        self._label_order_indices: Optional[Sequence[int]] = None
        self._patch_size = patch_size
        self._deformations = deformations
        self._paired = paired

    @property
    def item_shape(self) -> Sequence[int]:
        return self._patch_size

    def _generate_order_indices(self, random_state: RandomState) -> Sequence[int]:
        order_indices = arange(len(self._input_paths))
        random_state.shuffle(order_indices)
        return cast(Sequence[int], order_indices)

    def generate(self, random_state: RandomState) -> None:
        """Generate the sequence"""
        self._order_indices = self._generate_order_indices(random_state)
        if self._paired:
            self._label_order_indices = self._order_indices
        else:
            self._label_order_indices = self._generate_order_indices(random_state)

    def __len__(self) -> int:
        return len(self._input_paths)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Sample random patch from given tiff-files"""
        if self._order_indices is None:
            raise RuntimeError("Generate the sequence first by calling generate method.")
        assert self._label_order_indices is not None
        image_path = self._input_paths[self._order_indices[index]]
        raw_input_image = gray_to_color_if_needed(imread(image_path).astype("float32"))
        normalized_input_image = normalize(
            raw_input_image, self._input_mean_and_std.astype("float32")
        )
        if self._paired:
            raw_label_image = swap_channels(raw_input_image).astype("float32")
        else:
            label_image_path = self._input_paths[self._label_order_indices[index]]
            raw_label_image = swap_channels(
                gray_to_color_if_needed(imread(label_image_path).astype("float32"))
            ).astype("float32")
        normalized_label_image = normalize(
            raw_label_image, self._label_mean_and_std.astype("float32")
        )
        normalized_input_image_torch = from_numpy(transpose(normalized_input_image, (2, 0, 1)))
        normalized_label_image_torch = from_numpy(transpose(normalized_label_image, axes=(2, 0, 1)))
        random_deformation = generate_gaussian_ddf(
            shape=self._patch_size,
            deformation_definition=self._deformations[self._label_order_indices[index]],
        )
        crop = torch_zeros(1).float().expand((2,) + self._patch_size)
        cropped_input_image, input_mask = deform_volume(
            deformation=crop[None, ...],
            volume=normalized_input_image_torch[None, ...],
            return_mask=True,
        )
        deformed_label_image, label_mask = deform_volume(
            deformation=random_deformation[None, ...],
            volume=normalized_label_image_torch[None, ...],
            return_mask=True,
        )
        input_mask = cast(Tensor, input_mask).type(float32)[0]
        label_mask = cast(Tensor, label_mask)
        label_rectangle_mask = from_numpy(
            largest_interior_rectangle(label_mask[0, 0].numpy(), search_space_per_step=4)[None]
        ).type(float32)

        return (
            cropped_input_image[0] * input_mask,
            deformed_label_image[0] * label_rectangle_mask,
            input_mask,
            label_rectangle_mask,
        )


class EvaluationSyntheticSequence(IEvaluationArraySequence):
    """Inference sequence for swap channel data"""

    def __init__(
        self,
        input_paths: Sequence[str],
        deformations: Sequence[GaussianElasticDeformationDefinition],
        input_mean_and_std: ndarray,
        label_mean_and_std: ndarray,
        patch_size: Tuple[int, int],
    ) -> None:
        self._input_paths = input_paths
        self._input_mean_and_std = input_mean_and_std
        self._label_mean_and_std = label_mean_and_std
        self._order_indices: Optional[Sequence[int]] = None
        self._patch_size = patch_size
        self._deformations = deformations

    @property
    def item_shape(self) -> Sequence[int]:
        return self._patch_size

    def __len__(self) -> int:
        return len(self._input_paths)

    def __getitem__(
        self, index: int
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Get inference item with index"""
        image_path = self._input_paths[index]
        raw_input_image = gray_to_color_if_needed(imread(image_path).astype("float32"))
        normalized_input_image = normalize(
            raw_input_image, self._input_mean_and_std.astype("float32")
        )
        raw_label_image = swap_channels(raw_input_image).astype("float32")
        normalized_label_image = normalize(
            raw_label_image, self._label_mean_and_std.astype("float32")
        )
        normalized_input_image_torch = from_numpy(transpose(normalized_input_image, (2, 0, 1)))
        normalized_label_image_torch = from_numpy(transpose(normalized_label_image, axes=(2, 0, 1)))
        crop = torch_zeros((2,) + self._patch_size).float()
        random_deformation = generate_gaussian_ddf(
            shape=self._patch_size, deformation_definition=self._deformations[index]
        )
        # inverse_random_deformation = generate_gaussian_ddf(
        #    shape=self._patch_size,
        #    deformation_definition=self._deformations[index],
        #    invert=True
        # )
        cropped_input_image, input_mask = deform_volume(
            deformation=crop[None, ...],
            volume=normalized_input_image_torch[None, ...],
            return_mask=True,
        )
        cropped_label_image, _ = deform_volume(
            deformation=crop[None, ...],
            volume=normalized_label_image_torch[None, ...],
            return_mask=False,
        )
        deformed_label_image, label_mask = deform_volume(
            deformation=random_deformation[None, ...],
            volume=normalized_label_image_torch[None, ...],
            return_mask=True,
        )
        input_mask = cast(Tensor, input_mask).type(float32)[0]
        label_mask = cast(Tensor, label_mask)
        label_rectangle_mask = from_numpy(
            largest_interior_rectangle(label_mask[0, 0].numpy(), search_space_per_step=4)[None]
        ).type(float32)
        return (
            cropped_input_image[0] * input_mask,
            cropped_label_image[0] * input_mask,
            deformed_label_image[0] * label_rectangle_mask,
            input_mask,
            label_rectangle_mask,
            random_deformation,
            ones_like(label_rectangle_mask),
            ones_like(label_rectangle_mask),
        )
