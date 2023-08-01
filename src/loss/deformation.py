"""Deformation related loss functions"""

from abc import ABC
from itertools import combinations_with_replacement
from typing import Optional, Sequence, Tuple, TypeVar, Union, cast

from numpy import argsort as np_argsort
from numpy import array as np_array
from numpy import transpose as np_transpose
from numpy import zeros as np_zeros
from torch import Tensor
from torch import abs as torch_abs
from torch import device
from torch import dtype
from torch import eye, from_numpy, matmul, mean, square
from torch import sum as torch_sum
from torch import tensor
from torch.nn import Module
from torch.nn.functional import conv1d, conv2d, conv3d

from algorithm.torch.deformation.masked import MaskedVolume
from algorithm.torch.deformation.primitive import (
    calculate_derivative,
    calculate_jacobian_determinants,
    calculate_jacobian_matrices,
)
from algorithm.torch.interpolation_1d import interpolate_1d


def absolute_value_loss(deformation_svf: Tensor) -> Tensor:
    """Penalize absolute value of deformation svf"""
    return mean(torch_abs(deformation_svf))


class LaplaceRegularization:
    r"""Operator of the form - \alpha \Delta + \gamma \operatorname{Id}

    Can be applied to regularize velocity fields. Taken from
    Beg, M. Faisal, et al. "Computing large deformation metric
    mappings via geodesic flows of diffeomorphisms."(2005)
    """

    def __init__(
        self,
        torch_device: device,
        torch_dtype: dtype,
        voxel_size: Tuple[float, ...],
        alpha: Optional[float],
        gamma: Optional[float],
    ) -> None:
        self._torch_device = torch_device
        self._torch_dtype = torch_dtype
        self._n_dims = len(voxel_size)
        self._voxel_size = np_array(voxel_size)
        self._alpha = alpha
        self._gamma = gamma
        self._convolution_operator = (conv1d, conv2d, conv3d)[self._n_dims - 1]
        self._kernel = self._generate_conv_kernel()

    def _generate_conv_kernel(self) -> Tensor:
        kernel = np_zeros((3,) * self._n_dims, dtype="float32")
        if self._alpha is not None:
            for current_dim in range(self._n_dims):
                current_first_permutation = (current_dim,) + tuple(
                    dim for dim in range(self._n_dims) if dim != current_dim
                )
                kernel = np_transpose(kernel, axes=current_first_permutation)
                kernel[(0,) + (1,) * (self._n_dims - 1)] = (
                    -self._alpha / self._voxel_size[current_dim] ** 2
                )
                kernel[(2,) + (1,) * (self._n_dims - 1)] = (
                    -self._alpha / self._voxel_size[current_dim] ** 2
                )
                kernel[(1,) * self._n_dims] += 2 * self._alpha / self._voxel_size[current_dim] ** 2
                inverse_current_first_permutation = np_argsort(current_first_permutation)
                kernel = np_transpose(kernel, axes=inverse_current_first_permutation)
        if self._gamma is not None:
            kernel[(1,) * self._n_dims] += self._gamma
        return from_numpy(kernel[None, None, ...]).type(self._torch_dtype).to(self._torch_device)

    def __call__(self, flow_field: Tensor) -> Union[float, Tensor]:
        """Calculate loss

        Args:
            flow_field: Tensor with shape (batch_size, n_dims, dim_1, ..., dim_{n_dims})
        """
        if self._alpha is None and self._gamma is None:
            return 0.0
        batch_size = flow_field.size(0)
        operated_flow_field = self._convolution_operator(
            input=flow_field.view(*((batch_size * self._n_dims, 1) + flow_field.shape[2:])),
            weight=self._kernel,
        )
        operated_flow_field = operated_flow_field.view(
            *((batch_size, self._n_dims) + operated_flow_field.shape[2:])
        )
        riemmanian_distance = mean(torch_sum(square(operated_flow_field), dim=1))

        return riemmanian_distance


def affinity(jacobian_matrices: Tensor, central: bool, scale: Tensor) -> Tensor:
    """Affinity (bending energy) penalty for deformation fields penalizing non-affine deformations

    Args:
        jacobian_matrices: Local Jacobian matrices of a ddf representing the deformation,
            Tensor with shape (batch_size, n_dims, n_dims, dim_1, ..., dim_N}

    Returns:
        Tensor with shape (batch_size, dim_1 - 2, ..., dim_N - 2)
    """
    n_dims = jacobian_matrices.size(1)
    torch_device = jacobian_matrices.device
    loss = tensor(0.0, device=torch_device)
    for i, j in combinations_with_replacement(range(n_dims), 2):
        gradient_volume = calculate_derivative(
            jacobian_matrices[:, :, i],
            dim=j,
            scale=float(scale[j]),
            central=central,
            same_shape=True,
        )
        if i == j:
            loss = loss + square(gradient_volume)
        else:
            loss = loss + 2 * square(gradient_volume)
    return torch_sum(loss, dim=1)


def properness(jacobian_determinant: Tensor) -> Tensor:
    """Properness penalty for deformation fields penalizing locally volume changing deformations

    Args:
        jacobian_determinant: Tensor with shape (*any_shape)

    Returns:
        Tensor with shape (*any_shape)
    """
    return square(jacobian_determinant - 1)


def orthonormality(jacobian_matrices: Tensor) -> Tensor:
    """Orthonormality penalty for deformation field penalizing locally non-orthonormal deformations

    Args:
        jacobian_matrices: Local Jacobian matrices of a ddf representing the deformation,
            Tensor with shape (batch_size, n_dims, n_dims, *any_shape)

    Returns:
        Tensor with shape (batch_size, *any_shape)
    """
    n_dims = jacobian_matrices.size(1)
    n_spatial_dims = jacobian_matrices.ndim - 3
    identity_matrix = eye(n_dims, device=jacobian_matrices.device)
    channels_last_jacobian_matrices = jacobian_matrices.permute(
        0, *range(3, 3 + n_spatial_dims), 1, 2
    )
    jacobian_matrices_of_transformation = channels_last_jacobian_matrices + identity_matrix
    orthonormality_product = (
        matmul(
            jacobian_matrices_of_transformation,
            jacobian_matrices_of_transformation.transpose(-1, -2),
        )
        - identity_matrix
    )
    return torch_sum(square(orthonormality_product), dim=(-1, -2))


class RigidityLoss(Module):
    """Rigidity penalty loss

    Staring, Marius, Stefan Klein, and Josien PW Pluim. "A rigidity penalty term
    for nonrigid registration." (2007)

    Arguments:
        voxel_size: Voxel sizes along each dimension
        orthonormality_weight: Weight of the orthonormality factor
        properness_weight: Weight of the properness factor
        affinity_weight: Weight of the affinity factor
        first_derivatives_central: Use central difference (f(x - 1) + f(x + 1)) / 2
            for calculating first order derivatives, default is difference
            between neighboring values f(x + 1) - f(x).
        second_derivatives_central: Use central difference (f(x - 1) + f(x + 1)) / 2
            for calculating second order derivatives, default is difference
            between neighboring values f(x + 1) - f(x).
    """

    def __init__(
        self,
        voxel_size: Sequence[float],
        orthonormality_weight: Optional[float] = 1e-2,
        properness_weight: Optional[float] = 1e-1,
        affinity_weight: Optional[float] = 1.0,
        first_derivatives_central: bool = False,
        second_derivatives_central: bool = False,
    ) -> None:
        super().__init__()
        self.register_buffer("_scale", tensor(voxel_size), persistent=False)
        self._orthonormality_weight = orthonormality_weight
        self._properness_weight = properness_weight
        self._affinity_weight = affinity_weight
        self._first_derivatives_central = first_derivatives_central
        self._second_derivatives_central = second_derivatives_central

    @staticmethod
    def _crop_spatial(volume: Tensor) -> Tensor:
        n_spatial_dims = volume.ndim - 1
        return volume[(slice(None),) + (slice(1, -1),) * n_spatial_dims]

    @staticmethod
    def _average_consequtive_spatial(weighting_volume: Tensor) -> Tensor:
        n_spatial_dims = weighting_volume.ndim - 1
        averaged_weighing_volume = weighting_volume
        for crop_dim in range(n_spatial_dims):
            front_crop_slice = (slice(None),) + tuple(
                slice(None) if dim != crop_dim else slice(1, None) for dim in range(n_spatial_dims)
            )
            back_crop_slice = (slice(None),) + tuple(
                slice(None) if dim != crop_dim else slice(None, -1) for dim in range(n_spatial_dims)
            )
            averaged_weighing_volume = (
                averaged_weighing_volume[front_crop_slice]
                + averaged_weighing_volume[back_crop_slice]
            ) / 2
        return averaged_weighing_volume

    def _rigidity_coefficient_for_first_derivative(self, rigidity_coefficient: Tensor) -> Tensor:
        """Modify rigidity coefficient volume for derivative volume

        Calculating jacobian matrices modifies the volume spatial shape. This
        method modifies the rigidity coefficient volume such that the shapes
        match."""
        if self._first_derivatives_central:
            return self._crop_spatial(rigidity_coefficient)
        return self._average_consequtive_spatial(rigidity_coefficient)

    def _rigidity_coefficient_for_second_derivative(
        self,
        rigidity_coefficient: Tensor,
        first_derivative_rigidity_coefficient: Tensor,
    ) -> Tensor:
        """Modify rigidity coefficient volume for derivative volume

        Calculating jacobian matrices modifies the volume spatial shape. This
        method modifies the rigidity coefficient volume such that the shapes
        match."""
        if self._second_derivatives_central:
            return self._crop_spatial(first_derivative_rigidity_coefficient)
        if self._first_derivatives_central:
            return self._average_consequtive_spatial(first_derivative_rigidity_coefficient)
        return self._crop_spatial(rigidity_coefficient)

    def forward(
        self,
        deformation: Tensor,
        rigidity_coefficient: Optional[Tensor] = None,
    ) -> Union[float, Tensor]:
        """Calculate rigidity loss

        Args:
            deformation: Deformation represented as displacement field
                Tensor with shape (batch_size, n_dims, dim_1, ..., dim_{n_dims})
            rigidity_coefficient: Tensor with shape (batch_size, dim_1, ..., dim_{n_dims})
        """
        scale: Tensor = cast(Tensor, self._scale)
        jacobian_matrices = calculate_jacobian_matrices(
            vector_field=deformation, scale=scale, central=self._first_derivatives_central
        )
        first_derivative_rigidity_coefficient = (
            None
            if rigidity_coefficient is None
            else self._rigidity_coefficient_for_first_derivative(rigidity_coefficient)
        )

        loss: Union[float, Tensor] = 0.0
        if self._orthonormality_weight is not None:
            orthonormality_loss = orthonormality(jacobian_matrices)
            if first_derivative_rigidity_coefficient is not None:
                orthonormality_loss = orthonormality_loss * first_derivative_rigidity_coefficient
            loss += self._orthonormality_weight * mean(orthonormality_loss)
        if self._properness_weight is not None:
            jacobian_determinants = calculate_jacobian_determinants(jacobian_matrices)
            properness_loss = properness(jacobian_determinants)
            if first_derivative_rigidity_coefficient is not None:
                properness_loss = properness_loss * first_derivative_rigidity_coefficient
            loss += self._properness_weight * mean(properness_loss)
        if self._affinity_weight is not None:
            affinity_loss = affinity(
                jacobian_matrices, central=self._second_derivatives_central, scale=scale
            )
            if rigidity_coefficient is not None:
                assert first_derivative_rigidity_coefficient is not None
                second_derivative_rigidity_coefficient = (
                    self._rigidity_coefficient_for_second_derivative(
                        rigidity_coefficient=rigidity_coefficient,
                        first_derivative_rigidity_coefficient=first_derivative_rigidity_coefficient,
                    )
                )
                affinity_loss = affinity_loss * second_derivative_rigidity_coefficient
            loss += self._affinity_weight * mean(affinity_loss)
        return loss


T = TypeVar("T", bound="IRigidityCoefficientGenerator")


class IRigidityCoefficientGenerator(ABC):
    """Interface for classes generating rigidity coefficient volumes"""

    def generate_input_space_volume(
        self,
        input_volume: MaskedVolume,
    ) -> Optional[Tensor]:
        """Generate rigidity coefficient volume in input space"""

    def generate_target_space_volume(
        self,
        target_volume: MaskedVolume,
    ) -> Optional[Tensor]:
        """Generate rigidity coefficient volume in target space"""

    def to(self: T, target: Union[device, dtype]) -> T:  # pylint: disable=invalid-name
        """Put to device"""


class LinearTargetVolumeRigidityCoefficientGenerator(IRigidityCoefficientGenerator):
    """Rigidity coefficient is generated by linearly inteporlating based on
    defined target volume values and coefficient values"""

    def __init__(self, ct_values: Sequence[float], coefficient_values: Sequence[float]) -> None:
        super().__init__()
        self._ct_values = tensor(ct_values)
        self._coefficient_values = tensor(coefficient_values)

    def generate_input_space_volume(
        self,
        input_volume: MaskedVolume,
    ) -> Optional[Tensor]:
        raise NotImplementedError("Allows for coefficient volume only in target space")

    def generate_target_space_volume(
        self,
        target_volume: MaskedVolume,
    ) -> Tensor:
        channelwise_mean_volume = target_volume.volume.mean(dim=1)
        mask = target_volume.generate_mask()[:, 0]
        return (
            interpolate_1d(
                interpolation_x=channelwise_mean_volume.flatten(),
                data_x=self._ct_values,
                data_y=self._coefficient_values,
            ).view(channelwise_mean_volume.shape)
            * mask
            + (1 - mask) * self._coefficient_values[0]
        )

    def to(self, target: Union[device, dtype]) -> "LinearTargetVolumeRigidityCoefficientGenerator":
        self._ct_values = self._ct_values.to(target)
        self._coefficient_values = self._coefficient_values.to(target)
        return self


class EmptyRigidityCoefficientGenerator(IRigidityCoefficientGenerator):
    """Corresponds to constant rigidity coefficient of one"""

    def generate_input_space_volume(
        self,
        input_volume: MaskedVolume,
    ) -> Optional[Tensor]:
        return None

    def generate_target_space_volume(
        self,
        target_volume: MaskedVolume,
    ) -> Optional[Tensor]:
        return None

    def to(self, target: Union[device, dtype]) -> "EmptyRigidityCoefficientGenerator":
        return self
