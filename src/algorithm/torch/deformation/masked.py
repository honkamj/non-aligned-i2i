""""Algorithms for applying deformations to masked volumes"""


from abc import ABC, abstractmethod
from typing import Optional, Tuple, cast

from torch import Tensor
from torch import any as torch_any
from torch import cat, inverse, matmul, ones, tensor

from algorithm.torch.mask import discretize_mask, mask_and

from .primitive import (apply_affine_transformation_to_deformation,
                        compose_deformations, deform_volume,
                        generate_affine_ddf)


def _get_comparison_mask(
        mask_1: Optional[Tensor],
        mask_2: Optional[Tensor],
        mask_shape: Tuple[int, ...],
        dtype,
        device
    ) -> Tensor:
    """Get comparison mask"""
    if mask_1 is not None and mask_2 is not None:
        if mask_1.shape != mask_2.shape:
            raise NotImplementedError(
                'Comparison mask generation not implemented for varying shapes')
        return mask_and([mask_1, mask_2]).detach()
    if mask_2 is not None:
        if mask_2.shape != mask_shape:
            raise RuntimeError('Invalid shape')
        return mask_2
    if mask_1 is not None:
        if mask_1.shape != mask_shape:
            raise RuntimeError('Invalid shape')
        return mask_1
    return ones(
        mask_shape,
        dtype=dtype,
        device=device)


class IMaskedVolumeTransformation(ABC):
    """Interface representing volume transformation"""

    @abstractmethod
    def get_deformation(
            self,
            shape: Optional[Tuple[int, ...]] = None,
            voxel_size: Optional[Tensor] = None
        ) -> Tensor:
        """Generate deformation as ddf in voxel coordinates

        Returns: Tensor with shape (batch_size, n_dims, dim_1, ..., dim_{n_dims})
        """

    @abstractmethod
    def get_mask(
            self,
            shape: Optional[Tuple[int, ...]] = None,
            voxel_size: Optional[Tensor] = None
        ) -> Optional[Tensor]:
        """Generate valid region mask of the transformation

        None indicates that all regions are valid

        Returns: Tensor with shape (batch_size, 1, dim_1, ..., dim_{n_dims})
        """

    @abstractmethod
    def detach(self) -> 'IMaskedVolumeTransformation':
        """Returns transformation detached from the computational graph"""

    @abstractmethod
    def to_device(self, device) -> 'IMaskedVolumeTransformation':
        """Move to device"""

    @property
    @abstractmethod
    def n_dims(self) -> int:
        """Number of dimensions"""

    @property
    @abstractmethod
    def batch_size(self) -> int:
        """Batch size"""

    def compose(
            self,
            transformation: 'IMaskedVolumeTransformation',
            update_mask: bool = True
        ) -> 'IMaskedVolumeTransformation':
        """Compose the transformations

        Args:
            transformation: Transformation which is applied first
        """


class BaseMaskedVolumeTransformation(IMaskedVolumeTransformation):
    """Base masked volume transformation"""

    @abstractmethod
    def _get_deformation(
            self,
            shape: Optional[Tuple[int, ...]] = None,
            voxel_size: Optional[Tensor] = None
        ) -> Tensor:
        pass

    def get_deformation(
            self,
            shape: Optional[Tuple[int, ...]] = None,
            voxel_size: Optional[Tensor] = None
        ) -> Tensor:
        self._check_dimensionality(shape, voxel_size)
        return self._get_deformation(shape, voxel_size)

    @abstractmethod
    def _get_mask(
            self,
            shape: Optional[Tuple[int, ...]] = None,
            voxel_size: Optional[Tensor] = None
        ) -> Optional[Tensor]:
        pass

    def get_mask(
            self,
            shape: Optional[Tuple[int, ...]] = None,
            voxel_size: Optional[Tensor] = None
        ) -> Optional[Tensor]:
        self._check_dimensionality(shape, voxel_size)
        return self._get_mask(shape, voxel_size)

    def _check_dimensionality(
            self,
            shape: Optional[Tuple[int, ...]] = None,
            voxel_size: Optional[Tensor] = None
        ) -> None:
        if (
                (shape is not None and len(shape) != self.n_dims) or
                (voxel_size is not None and len(voxel_size) != self.n_dims)
            ):
            raise RuntimeError('Invalid dimensionality')


class AffineTransformation(BaseMaskedVolumeTransformation):
    """Represents affine transformation

    Arguments:
        affine_transformation: Affine transformation in world coordinates,
            Tensor with shape (batch_size, n_dims + 1, n_dims + 1)
    """
    def __init__(
            self,
            affine_transformation: Tensor
        ) -> None:
        self._transformation = affine_transformation
        self._batch_size = affine_transformation.size(0)
        self._ndims = affine_transformation.size(1) - 1
        self._dtype = affine_transformation.dtype
        self._device = affine_transformation.device
        if (
            affine_transformation.size(2) != self._ndims + 1 or
            affine_transformation.ndim != 3):
            raise ValueError('Invalid transformation shape')

    def _get_deformation(
            self,
            shape: Optional[Tuple[int, ...]] = None,
            voxel_size: Optional[Tensor] = None
        ) -> Tensor:
        """Generate deformation as ddf in voxel coordinates"""
        if voxel_size is None or shape is None:
            raise RuntimeError('Shape and voxel size are needed')
        return generate_affine_ddf(
            affine_transformation=self.transformation,
            deformation_shape=tensor(
                shape,
                dtype=self._dtype,
                device=self._device),
            deformation_voxel_size=voxel_size
        ) / voxel_size[(None, ...) + (None,) * self.n_dims]

    def _get_mask(
            self,
            shape: Optional[Tuple[int, ...]] = None,
            voxel_size: Optional[Tensor] = None
        ) -> Optional[Tensor]:
        return None

    def to_device(self, device) -> 'AffineTransformation':
        """Move to device"""
        return AffineTransformation(
            affine_transformation=self._transformation.to(device))

    def detach(self) -> 'AffineTransformation':
        return AffineTransformation(self._transformation.detach())

    @property
    def n_dims(self) -> int:
        """Number of dimensions"""
        return self._ndims

    @property
    def batch_size(self) -> int:
        """Batch size"""
        return self._batch_size

    @property
    def transformation(self) -> Tensor:
        """Affine transformation in world coordinates"""
        return self._transformation

    def invert(self) -> 'AffineTransformation':
        """Invert the transformation"""
        return AffineTransformation(
            inverse(self._transformation)
        )

    def compose(
            self,
            transformation: IMaskedVolumeTransformation,
            update_mask: bool = True
        ) -> IMaskedVolumeTransformation:
        if transformation.batch_size != self.batch_size:
            raise RuntimeError(
                'Batch sizes do not match')
        if transformation.n_dims != self.n_dims:
            raise RuntimeError(
                'Dimensionalities do not match')
        if isinstance(transformation, MaskedDeformation):
            deformation = self.get_deformation(transformation.shape, transformation.voxel_size)
            other_deformation = transformation.get_deformation(
                transformation.shape,
                transformation.voxel_size)
            composed, fov_mask = compose_deformations(
                other_deformation,
                deformation,
                return_mask=update_mask)
            if update_mask:
                fov_mask = cast(Tensor, fov_mask).type(self._transformation.dtype)
                other_mask = transformation.get_mask(
                    transformation.shape,
                    transformation.voxel_size)
                if other_mask is None:
                    updated_mask: Optional[Tensor] = fov_mask
                else:
                    deformed_mask, _ = deform_volume(deformation, other_mask)
                    deformed_mask = discretize_mask(deformed_mask)
                    updated_mask = mask_and([fov_mask, deformed_mask])
            else:
                updated_mask = None
            return MaskedDeformation(
                deformation=composed,
                voxel_size=transformation.voxel_size,
                mask=updated_mask)
        if isinstance(transformation, AffineTransformation):
            return AffineTransformation(matmul(transformation.transformation, self._transformation))
        raise NotImplementedError(f'Composition not implemented with type {type(transformation)}.')


class MaskedDeformation(BaseMaskedVolumeTransformation):
    """Represents masked deformation

    Arguments:
        deformation: Represented as ddf in voxel cooridinates,
            Tensor with shape (batch_size, n_dims, dim_1, ..., dim_{n_dims})
        voxel_size: Voxel size along each dimesion, Tensor with shape (n_dims,)
        mask: Valid values in the deformation, Tensor with shape
            (batch_size, 1, dim_1, ..., dim_{n_dims})
    """
    def __init__(
            self,
            deformation: Tensor,
            voxel_size: Tensor,
            mask: Optional[Tensor] = None,
        ) -> None:
        self._deformation = deformation
        self._voxel_size = voxel_size
        self._mask = None if mask is None else mask.detach()
        self._shape = tuple(self._deformation.shape[2:])
        self._n_dims = len(self._shape)
        self._batch_size = deformation.size(0)
        self._dtype = deformation.dtype
        self._device = deformation.device
        if deformation.size(1) != self._n_dims:
            raise ValueError('Invalid deformation shape')
        if mask is not None and (
                tuple(mask.shape[2:]) != self._shape or
                mask.size(1) != 1 or
                mask.size(0) != self._batch_size):
            raise ValueError('Invalid mask shape')
        if len(voxel_size) != self._n_dims:
            raise ValueError('Invalid voxel size')

    def _get_deformation(
            self,
            shape: Optional[Tuple[int, ...]] = None,
            voxel_size: Optional[Tensor] = None
        ) -> Tensor:
        """Generate deformation as ddf in voxel coordinates"""
        if shape is not None and self._shape != shape:
            raise NotImplementedError(
                'Obtaining deformation with different shape not implemented')
        if voxel_size is not None and torch_any(self._voxel_size != voxel_size):
            raise NotImplementedError(
                'Obtaining deformation with different voxel size not implemented')
        return self._deformation

    def _get_mask(
            self,
            shape: Optional[Tuple[int, ...]] = None,
            voxel_size: Optional[Tensor] = None
        ) -> Optional[Tensor]:
        if shape is not None and self._shape != shape:
            raise NotImplementedError(
                'Obtaining mask with different shape not implemented')
        if voxel_size is not None and torch_any(self._voxel_size != voxel_size):
            raise NotImplementedError(
                'Obtaining mask with different voxel size not implemented')
        return self._mask

    def to_device(self, device) -> 'MaskedDeformation':
        """Move to device"""
        return MaskedDeformation(
            deformation=self._deformation.to(device),
            voxel_size=self._voxel_size.to(device),
            mask=self._mask.to(device) if self._mask is not None else None)

    @property
    def batch_size(self) -> int:
        """Batch size"""
        return self._batch_size

    @property
    def n_dims(self) -> int:
        """Number of dimensions"""
        return self._n_dims

    @property
    def voxel_size(self) -> Tensor:
        """Voxel size"""
        return self._voxel_size

    @property
    def shape(self) -> Tuple[int, ...]:
        """Shape"""
        return self._shape

    def detach(self) -> 'MaskedDeformation':
        return MaskedDeformation(
            self._deformation.detach(),
            self._voxel_size.detach(),
            self._mask.detach() if self._mask is not None else None)

    def compose(
            self,
            transformation: IMaskedVolumeTransformation,
            update_mask: bool = True
        ) -> IMaskedVolumeTransformation:
        if transformation.batch_size != self.batch_size:
            raise RuntimeError(
                'Batch sizes do not match')
        if transformation.n_dims != self.n_dims:
            raise RuntimeError(
                'Dimensionalities do not match')
        if isinstance(transformation, AffineTransformation):
            voxel_size_broadcast = self._voxel_size[(None, ...) + (None,) * self._n_dims]
            composed = apply_affine_transformation_to_deformation(
                affine_transformation=transformation.transformation,
                deformation=self._deformation * voxel_size_broadcast,
                deformation_voxel_size=self._voxel_size
            ) / voxel_size_broadcast
            return MaskedDeformation(
                deformation=composed,
                voxel_size=self._voxel_size,
                mask=self._mask)
        if isinstance(transformation, MaskedDeformation):
            other_deformation = transformation.get_deformation(self._shape, self._voxel_size)
            composed, fov_mask = compose_deformations(
                other_deformation,
                self._deformation,
                return_mask=update_mask)
            if update_mask:
                fov_mask = cast(Tensor, fov_mask).type(self._deformation.dtype)
                if self._mask is not None:
                    fov_mask = mask_and([fov_mask, self._mask])
                other_mask = transformation.get_mask(self._shape, self._voxel_size)
                if other_mask is None:
                    updated_mask: Optional[Tensor] = fov_mask
                else:
                    deformed_mask, _ = deform_volume(self._deformation, other_mask)
                    deformed_mask = discretize_mask(deformed_mask)
                    updated_mask = mask_and([fov_mask, deformed_mask])
            else:
                updated_mask = self._mask
            return MaskedDeformation(
                deformation=composed,
                voxel_size=self._voxel_size,
                mask=updated_mask)
        raise NotImplementedError(f'Composition not implemented with type {type(transformation)}.')

    def get_comparison_mask(
            self,
            deformation: IMaskedVolumeTransformation,
            shape: Optional[Tuple[int, ...]] = None,
            voxel_size: Optional[Tensor] = None
        ) -> Tensor:
        """Get comparison mask between two deformations"""
        if deformation.batch_size != self.batch_size:
            raise RuntimeError('Can not generate mask for varying batch sizes')
        own_mask = self.get_mask(shape, voxel_size)
        actual_shape = self.shape if shape is None else shape
        other_mask = deformation.get_mask(
            actual_shape,
            self.voxel_size if voxel_size is None else voxel_size)
        return _get_comparison_mask(
            own_mask,
            other_mask,
            mask_shape=(self.batch_size, 1) + actual_shape,
            dtype=self._deformation.dtype,
            device=self._deformation.device)


class MaskedVolume:
    """Masked deformable volume

    Arguments:
        volume: Tensor with shape (batch_size, n_channels, dim_1, ..., dim_{n_dims})
        mask: Tensor with shape (batch_size, 1, dim_1, ..., dim_{n_dims})
    """
    def __init__(
            self,
            volume: Tensor,
            voxel_size: Tensor,
            mask: Optional[Tensor] = None
        ) -> None:
        self._volume = volume
        self._voxel_size = voxel_size
        self._mask = None if mask is None else mask.detach()
        self._shape = tuple(volume.shape[2:])
        self._n_dims = len(self._shape)
        self._batch_size = volume.size(0)
        self._dtype = volume.dtype
        self._device = volume.device
        if mask is not None and (
                tuple(mask.shape[2:]) != self._shape or
                mask.size(1) != 1 or
                mask.size(0) != self._batch_size):
            raise ValueError('Invalid mask shape')
        if len(voxel_size) != self._n_dims:
            raise ValueError('Invalid voxel size')

    def generate_mask(
            self,
            deformation: Optional[IMaskedVolumeTransformation] = None
        ) -> Tensor:
        """Mask of the valid regions of the volume

        If deformation is given, mask is deformed by it.
        """
        if self._mask is None:
            mask = ones(
                (self.batch_size,) + (1,) + self.shape,
                dtype=self._dtype,
                device=self._device)
        else:
            mask = self._mask
        if deformation is None:
            return mask
        deformed_mask, fov_mask = deform_volume(
            deformation=deformation.get_deformation(self._shape, self._voxel_size),
            volume=mask,
            return_mask=True)
        deformed_mask = discretize_mask(deformed_mask)
        fov_mask = cast(Tensor, fov_mask).type(self._volume.dtype)
        return mask_and([deformed_mask, fov_mask])

    def to_device(self, device) -> 'MaskedVolume':
        """Move to device"""
        return MaskedVolume(
            volume=self._volume.to(device),
            voxel_size=self._voxel_size.to(device),
            mask=self._mask.to(device) if self._mask is not None else None)

    @property
    def volume(self) -> Tensor:
        """Underlying volume"""
        return self._volume

    @property
    def mask(self) -> Optional[Tensor]:
        """Mask of the valid regions of the volume or None if no mask is given"""
        return self._mask

    @property
    def voxel_size(self) -> Tensor:
        """Voxel size along each dimension"""
        return self._voxel_size

    @property
    def batch_size(self) -> int:
        """Batch size"""
        return self._batch_size

    @property
    def shape(self) -> Tuple[int, ...]:
        """Shape of the deformation volume"""
        return self._shape

    @property
    def n_dims(self) -> int:
        """Number of dimensions"""
        return self._n_dims

    def detach(self) -> 'MaskedVolume':
        """Detach volume from computational graph"""
        return MaskedVolume(
            volume=self._volume.detach(),
            voxel_size=self._voxel_size.detach(),
            mask=self._mask)

    def deform(
            self,
            transformation: IMaskedVolumeTransformation,
            shape: Optional[Tuple[int, ...]] = None,
            voxel_size: Optional[Tensor] = None,
            coordinate_grid: str = 'centered'
        ) -> 'MaskedVolume':
        """Deform volume with transformation, returns new instance

        Args:
            transformation: Transformation with which to transform the volume
            shape: Shape of the target volume
            voxel_size: Voxel size of the target volume
            coordinate_grid: If deformation voxel size or shape differs with the volume
                voxel size or shape, this option determines the location of origin of
                the deformation coordinates, either 'centered' or 'zero_origin'
        """
        if transformation.batch_size != self.batch_size:
            raise RuntimeError(
                'Batch sizes do not match')
        target_shape = self.shape if shape is None else shape
        target_voxel_size = self.voxel_size if voxel_size is None else voxel_size
        deformation = transformation.get_deformation(target_shape, target_voxel_size)
        transformation_mask = transformation.get_mask(target_shape, target_voxel_size)
        if self.mask is not None:
            combined = cat((self.volume, self.mask), dim=1)
        else:
            combined = self.volume
        deformed_combined, fov_mask = deform_volume(
            deformation=deformation,
            volume=combined,
            return_mask=True,
            deformation_voxel_size=target_voxel_size / self.voxel_size,
            coordinate_grid=coordinate_grid)
        fov_mask = cast(Tensor, fov_mask).type(self._volume.dtype)
        if self._mask is not None:
            deformed = deformed_combined[:, :-1]
            deformed_mask = mask_and(
                [fov_mask, discretize_mask(deformed_combined[:, -1:])])
        else:
            deformed = deformed_combined
            deformed_mask = fov_mask
        if transformation_mask is not None:
            deformed_mask = mask_and([deformed_mask, transformation_mask])
            fov_mask = mask_and([fov_mask, transformation_mask])
        return MaskedVolume(
            volume=deformed * fov_mask,
            voxel_size=target_voxel_size,
            mask=deformed_mask)

    @staticmethod
    def get_comparison_mask(volume_1: 'MaskedVolume', volume_2: 'MaskedVolume') -> Tensor:
        """Get comparison mask between two volumes"""
        if torch_any(volume_1.voxel_size != volume_2.voxel_size):
            raise NotImplementedError(
                'Comparison mask generation not implemented for varying voxel sizes')
        return _get_comparison_mask(
            volume_1.mask,
            volume_2.mask,
            mask_shape=(volume_1.batch_size, 1) + volume_1.shape,
            dtype=volume_1.volume.dtype,
            device=volume_1.volume.device)
