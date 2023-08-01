"""Deformation related algorithms"""

from itertools import product
from typing import Optional, Sequence, Tuple

from torch import Tensor, cat, cos, device, dtype, matmul, sin, zeros
from torch import all as torch_all
from torch import (arange, clamp, det, eye, floor, index_select, linspace,
                   meshgrid, ones, prod, stack)
from torch import sum as torch_sum
from torch import tensor
from torch.nn.functional import grid_sample


def interpolate(
        volume: Tensor,
        points: Tensor,
        padding_mode: str,
        mode: str='bilinear',
        return_mask: bool=False
    ) -> Tuple[Tensor, Optional[Tensor]]:
    """Interpolate volume at given points (in voxel coordinates)

    Args:
        volume: Tensor with shape (batch_size, n_channels, dim_1, ..., dim_{n_dims})
        points: Interpolation locations, Tensor with shape (batch_size, n_dims, *target_shape)
        padding_mode: Padding mode of interpolation
        mode: Interpolation mode
        return_mask: Whether to return mask of values outside field of view

    Returns:
        interpolated: Tensor with shape (batch_size, n_channels, *target_shape)
        mask (optional):  Tensor with shape (batch_size, 1, *target_shape)
    """
    batch_size = volume.size(0)
    n_channels = volume.size(1)
    n_dims = points.size(1)
    target_shape = points.shape[2:]
    n_points = int(prod(tensor(target_shape)))

    scale = tensor(
        volume.shape[2:],
        dtype=volume.dtype,
        device=volume.device).view(1, n_dims, *((1,) * len(target_shape))) - 1
    scaled_points = 2 * points / scale - 1 # Scale to [-1, 1] for F.grid_sample
    sampling_grid = scaled_points.view(
        batch_size,
        n_dims,
        *((1,) * (n_dims - 1)),
        n_points
    ).permute(0, -1, *reversed(range(1, 1 + n_dims)))
    interpolated = grid_sample(
        input=volume.permute(0, 1, *reversed(range(2, 2 + n_dims))),
        grid=sampling_grid,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=True).view(batch_size, n_channels, *target_shape)
    mask: Optional[Tensor] = None
    if return_mask:
        mask = torch_all(
            (scaled_points >= -1)  & (scaled_points <= 1),
            dim=1,
            keepdim=True)
    return interpolated, mask


def linear_interpolate(
        volume: Tensor,
        points: Tensor,
        return_interpolated_values: bool = True,
        return_jacobian_matrices: bool = False,
        return_mask: bool = False
    ) -> Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
    """Interpolate volume at given points

    Can also return Jacobian matrices of the volume at the points.

    Args:
        volume: Tensor with shape (batch_size, n_channels, dim_1, ..., dim_{n_dims})
        points: Interpolation locations, Tensor with shape (batch_size, n_dims, *target_shape)
        return_interpolated_values: Whether to return interpolated values at points
        return_jacobian_matrices: Whether to return local Jacobian matrices of the input
            vector field
        return_mask: Whether to return mask of values inside field of view

    Returns:
        interpolated_values (optional): Tensor with shape
            (batch_size, n_dims, n_dims, *target_shape)
        jacobian_matrices (optional):  Tensor with shape
            (batch_size, n_channels, n_dims, *target_shape)
        mask (optional):  Tensor with shape (batch_size, *target_shape)
    """
    torch_device = volume.device
    batch_size = volume.size(0)
    n_dims = points.size(1)
    n_channels = volume.size(1)
    volume_shape = volume.shape[2:]
    target_shape = points.shape[2:]
    n_points = int(prod(tensor(target_shape)))

    scale = tensor(
        volume_shape,
        dtype=volume.dtype,
        device=torch_device).view(1, n_dims, 1) - 1
    points_flattened = points.view(batch_size, n_dims, n_points)
    points_scaled = points_flattened / scale
    points_clamped = clamp(points_scaled, min=0, max=1) * scale

    points_lower = floor(points_clamped)
    points_upper = floor(clamp((points_flattened + 1) / scale, min=0, max=1) * scale)

    corner_indices = [points_lower.int(), points_upper.int()]

    local_coordinates = points_clamped - points_lower

    dim_product_list = [1]
    for dim in range(0, n_dims):
        dim_product_list.append(dim_product_list[dim] * volume_shape[n_dims  - dim - 1])
    dim_product = tensor(list(reversed(dim_product_list)), device=torch_device)
    if return_jacobian_matrices:
        dim_inclusion_matrix = eye(n_dims, device=torch_device).view(1, n_dims, n_dims, 1)
        dim_exclusion_matrix = 1 - dim_inclusion_matrix

    interpolated_values = tensor(0.0, device=torch_device)
    jacobian_matrices = tensor(0.0, device=torch_device)
    for corner_points in product((0, 1), repeat=n_dims):
        volume_indices = stack(
            [corner_indices[corner_points[dim]][:, dim] for dim in range(n_dims)],
            dim=-1)
        volume_indices_1d = torch_sum(volume_indices * dim_product[1:], dim=-1)
        batch_volume_indices_1d = (
            volume_indices_1d.T + arange(
                batch_size,
                device=torch_device) * dim_product[0]).T.reshape(n_points * batch_size)
        volume_1d = volume.transpose(0, 1).reshape(n_channels, -1)
        volume_values = index_select(
            volume_1d,
            dim=1,
            index=batch_volume_indices_1d).view(n_channels, batch_size, n_points).transpose(0, 1)
        corner_points_tensor = tensor(corner_points, device=torch_device).view(1, n_dims, 1)
        weights = (
            (1 - corner_points_tensor) * (1 - local_coordinates) +
            corner_points_tensor * local_coordinates
        )
        if return_interpolated_values:
            interpolated_values += volume_values * prod(
                weights, dim=1).view(batch_size, 1, n_points)
        if return_jacobian_matrices:
            dim_multiplier = 2 * corner_points_tensor.view(1, 1, n_dims, 1) - 1
            weight_products = prod(
                dim_inclusion_matrix + weights.view(
                    batch_size, n_dims, 1, n_points) * dim_exclusion_matrix,
                dim=1,
                keepdim=True) * dim_multiplier
            jacobian_matrices += weight_products * volume_values.view(
                batch_size, n_channels, 1, n_points)

    if return_jacobian_matrices:
        jacobian_matrices = jacobian_matrices.view(
            batch_size, n_channels, n_dims, *target_shape)
    if return_mask:
        mask = torch_all(
            (points_scaled >= 0)  & (points_scaled <= 1),
            dim=1,
            keepdim=False).view(batch_size, *target_shape)

    return (
        interpolated_values if return_interpolated_values else None,
        jacobian_matrices if return_jacobian_matrices else None,
        mask if return_mask else None
    )


def generate_centered_coordinate_grid(
        grid_shape: Tensor,
        volume_shape: Tensor,
        grid_voxel_size: Optional[Tensor]
    ) -> Tensor:
    """Generates grid with located at the center of the volume

    Args:
        deformation_shape: Tensor with shape (n_dims,)
        volume_shape: Tensor with shape (n_dims,)
        grid_voxel_size: Optional Tensor with shape (n_dims,)

    Return:
        Tensor with shape (n_dims, deformation_shape[0], ..., deformation_shape[n_dims])
    """
    torch_device = volume_shape.device
    n_dims = volume_shape.size(0)
    if grid_voxel_size is None:
        grid_voxel_size = ones(n_dims, device=torch_device)
    origin = ((volume_shape - 1) - (grid_shape - 1) * grid_voxel_size) / 2
    axes = [
        linspace(
            start=origin_dim,
            end=origin_dim + (dim_size - 1) * voxel_dim_size,
            steps=int(dim_size),
            device=torch_device)
        for (dim_size, origin_dim, voxel_dim_size) in zip(
            grid_shape,
            origin,
            grid_voxel_size)
    ]
    coordinates = stack(meshgrid(*axes, indexing='ij'), dim=0)
    return coordinates.detach()


def generate_coordinate_grid(
        grid_shape: Tensor,
        grid_voxel_size: Optional[Tensor] = None
    ) -> Tensor:
    """Generates grid with origin located at top-left corner

    Args:
        grid_shape: Tensor with shape (n_dims,)
        grid_voxel_size: Optional Tensor with shape (n_dims,)

    Return:
        Tensor with shape (n_dims, deformation_shape[0], ..., deformation_shape[n_dims])
    """
    torch_device = grid_shape.device
    n_dims = grid_shape.size(0)
    if grid_voxel_size is None:
        grid_voxel_size = ones(n_dims, device=torch_device)
    axes = [
        linspace(
            start=0,
            end=(dim_size - 1) * voxel_dim_size,
            steps=int(dim_size),
            device=torch_device)
        for (dim_size, voxel_dim_size) in zip(
            grid_shape,
            grid_voxel_size)
    ]
    coordinates = stack(meshgrid(*axes, indexing='ij'), dim=0)
    return coordinates.detach()


def deform_volume(
        deformation: Tensor,
        volume: Tensor,
        deformation_voxel_size: Optional[Tensor] = None,
        return_mask: bool = False,
        coordinate_grid: str = 'centered'
    ) -> Tuple[Tensor, Optional[Tensor]]:
    """Deforms volume with deformation (in voxel coordinates)

    Args:
        deformation: Deformation represented as DDF, Tensor with
            shape (batch_size, n_dims, deformation_dim_1, ..., deformation_dim_{n_dims})
        volume: Volume to transform, Tensor with shape
            (batch_size, n_channels, dim_1, ..., dim_{n_dims})
        deformation_voxel_size: Voxel size of the deformation. Voxel size of the volume
            is assumed to be 1.
        return_mask: Whether to return mask of values outside field of view
        coordinate_grid: If deformation voxel size or shape differs with the volume
                voxel size or shape, this option determines the location of origin of
                the deformation coordinates

    Returns:
        interpolated: Tensor with shape (batch_size, n_channels, dim_1, ..., dim_{n_dims})
        mask (optional): Tensor with shape (batch_size, 1, dim_1, ..., dim_{n_dims})
    """
    torch_device = deformation.device
    torch_dtype = deformation.dtype
    deformation_shape = tensor(deformation.shape[2:], device=torch_device, dtype=torch_dtype)
    volume_shape = tensor(volume.shape[2:], device=torch_device, dtype=torch_dtype)
    if coordinate_grid == 'centered':
        coordinates = generate_centered_coordinate_grid(
            grid_shape=deformation_shape,
            volume_shape=volume_shape,
            grid_voxel_size=deformation_voxel_size
        )
    elif coordinate_grid == 'zero_origin':
        coordinates = generate_coordinate_grid(
            grid_shape=deformation_shape,
            grid_voxel_size=deformation_voxel_size
        )
    return interpolate(
        volume=volume,
        points=deformation + coordinates,
        padding_mode='border',
        return_mask=return_mask)


def compose_deformations(
        deformation_1: Tensor,
        deformation_2: Tensor,
        return_mask: bool = False
    ) -> Tuple[Tensor, Optional[Tensor]]:
    """Compose two deformations

    Deformation here refers to deforming a volume using deform_volume function,
    deformation_1 is the one applied first.

    Args:
        deformation_1: Deformation represented as DDF, Tensor with
            shape (batch_size, n_dims, dim_1, ..., dim_{n_dims})
        deformation_2: Deformation represented as DDF, Tensor with
            shape (batch_size, n_dims, dim_1, ..., dim_{n_dims})

    Returns:
        Tensor with shape (batch_size, n_dims, dim_1, ..., dim_{n_dims})
        Optional: Tensor with shape (batch_size, 1, dim_1, ..., dim_{n_dims})
    """
    interpolated, mask = deform_volume(
        deformation=deformation_2,
        volume=deformation_1,
        return_mask=return_mask)
    composed = deformation_2 + interpolated
    return composed, mask


def integrate_svf(
        flow_field: Tensor,
        integration_steps: int = 7
    ) -> Tensor:
    """Integrate static velocity field (sfv) over unit time using scaling and squaring

    Args:
        flow_field: Tensor with shape (batch_size, n_dims, dim_1, ..., dim_{n_dims})
        integration_steps: How many scaling and squaring steps to use

    Returns:
        Tensor with shape (batch_size, n_dims, dim_1, ..., dim_{n_dims})
    """
    integrated_field = flow_field / 2**integration_steps
    for _ in range(integration_steps):
        integrated_field = compose_deformations(integrated_field, integrated_field)[0]
    return integrated_field


def transform_points(deformation: Tensor, points: Tensor) -> Tensor:
    """Transform points based on given deformation (in voxel coordinates)

    Args:
        deformation: Deformation represented as DDF, Tensor with
            shape (batch_size, n_dims, dim_1, ..., dim_{n_dims})
        points: Interpolation locations, Tensor with shape
            (batch_size, n_dims, *target_shape)

    Returns:
        Tensor with shape (batch_size, n_dims, target_shape)
    """
    sampled_deformation, _ = interpolate(
        volume=deformation,
        points=points,
        padding_mode='border')
    return points + sampled_deformation


def calculate_derivative(
        volume: Tensor,
        dim: int,
        scale: float,
        same_shape: bool = True,
        central = True) -> Tensor:
    """Calculate spatial derivative along a dimension

    Args:
        volume: Tensor with shape (batch_size, n_channels, dim_1, ..., dim_{n_dims})
        scale: Scaling of the dimension (of one pixel/voxel)
        same_shape: If True, output shape will not depend on the dimension
            over which the derivative is taken.
        central: Whether to use central difference [f(x + 1)  - f(x - 1)] / 2 or not
            f(x + 1) - f(x)

    Returns:
        if central and same_shape: Tensor with shape
            (batch_size, n_channels, dim_1 - 2, ..., dim_{n_dims} - 2)
        elif central and not same_shape: Tensor with shape
            (batch_size, n_channels, dim_1, ..., dim_{dim} - 2, ..., dim_{n_dims})
        elif not central and same_shape: Tensor with shape
            (batch_size, n_channels, dim_1 - 1, ..., dim_{n_dims} - 1)
        elif not central and not same_shape: Tensor with shape
            (batch_size, n_channels, dim_1, ..., dim_{dim} - 1, ..., dim_{n_dims})
    """
    n_dims = len(volume.shape) - 2
    other_crop = slice(1, -1) if central and same_shape else slice(None)
    if central:
        front_crop = slice(2, None)
        back_crop = slice(None, -2)
    else:
        front_crop = slice(1, None)
        back_crop = slice(None, -1)
    front_cropping_slice = (...,) + tuple(
        front_crop if i == dim else other_crop
        for i in range(n_dims))
    back_cropping_slice = (...,) + tuple(
        back_crop if i == dim else other_crop
        for i in range(n_dims))
    derivatives = (volume[front_cropping_slice] - volume[back_cropping_slice]) / scale
    if central:
        derivatives = derivatives / 2
    elif same_shape:
        front_cropping_slice_other_dims = (...,) + tuple(
            slice(None) if i == dim else slice(1, None)
            for i in range(n_dims))
        back_cropping_slice_other_dims = (...,) + tuple(
            slice(None) if i == dim else slice(None, -1)
            for i in range(n_dims))
        derivatives = (
            derivatives[front_cropping_slice_other_dims] +
            derivatives[back_cropping_slice_other_dims]) / 2
    return derivatives


def calculate_jacobian_matrices(
        vector_field: Tensor,
        scale: Tensor,
        central: bool = True) -> Tensor:
    """Calculate local Jacobian matrices of a vector field (over same space)

    Args:
        vector_field: Tensor with shape (batch_size, n_dims, dim_1, ..., dim_{n_dims}),
            vectors of the vector field are assumed to be in voxel coordinates for scaling
        scale: Tensor with shape (n_dims,), voxel sizes along each dimension
        central: Whether to calculate derivatives using central difference or not

    Returns:
        Tensor with shape (batch_size, n_dims, n_dims, dim_1 - 2, ..., dim_{n_dims} - 2)
    """
    n_dims = vector_field.size(1)
    scaled_vector_field = vector_field * scale.view(1, n_dims, *((1,) * n_dims))
    return stack(
        tensors=[
            calculate_derivative(
                volume=scaled_vector_field,
                dim=dim,
                scale=float(scale[dim]),
                central=central,
                same_shape=True) for dim in range(n_dims)
        ],
        dim=2)


def _calculate_determinant_2d(matrix: Tensor) -> Tensor:
    """Calculate determinant of a 2D matrix

    Args:
        matrix: Tensor with shape (batch_size, 2, 2, *any_shape)

    Returns:
        Tensor with shape (batch_size, *any_shape)
    """
    return matrix[:, 0, 0] * matrix[:, 1, 1] - matrix[:, 0, 1] * matrix[:, 1, 0]


def _calculate_determinant_3d(matrix: Tensor) -> Tensor:
    """Calculate determinant of a 3D matrix

    Args:
        matrix: Tensor with shape (batch_size, 3, 3, *any_shape)

    Returns:
        Tensor with shape (batch_size, *any_shape)
    """
    return (
        matrix[:, 0, 0] * (
            matrix[:, 1, 1] * matrix[:, 2, 2] -
            matrix[:, 1, 2] * matrix[:, 2, 1]) +
        matrix[:, 0, 1] * (
            matrix[:, 1, 2] * matrix[:, 2, 0] -
            matrix[:, 1, 0] * matrix[:, 2, 2]) +
        matrix[:, 0, 2] * (
            matrix[:, 1, 0] * matrix[:, 2, 1] -
            matrix[:, 1, 1] * matrix[:, 2, 0])
    )


def calculate_determinant(matrix: Tensor) -> Tensor:
    """Calculate determinant of a matrix

    Args:
        matrix: Tensor with shape (batch_size, n_dims, n_dims, *any_shape)

    Returns:
        Tensor with shape (batch_size, *any_shape)
    """
    n_dims = matrix.size(1)
    if n_dims == 2:
        return _calculate_determinant_2d(matrix)
    if n_dims == 3:
        return _calculate_determinant_3d(matrix)
    return det(matrix.transpose(2, -1).transpose(1, -2))


def calculate_jacobian_determinants(jacobian_matrices: Tensor) -> Tensor:
    """Calculate Jacobian determinants of local Jacobian matrices of deformation displacement field

    Args:
        jacobian_matrices: Local Jacobian matrices of a ddf representing the deformation,
            Tensor with shape (batch_size, n_dims, n_dims, *any_shape)

    Returns:
        Tensor with shape (batch_size, *any_shape)
    """
    n_dims = jacobian_matrices.size(1)
    n_other_dims = jacobian_matrices.dim() - 3
    return calculate_determinant(
        matrix=jacobian_matrices + eye(n_dims, device=jacobian_matrices.device).view(
            1,
            n_dims,
            n_dims,
            *((1,) * n_other_dims))
    )


def concatenate_transformations(transformations: Sequence[Tensor]) -> Tensor:
    """Concatenates multiple transformation matrices

    transformations: Sequence of arrays with shape (batch_size, n_dims, n_dims)
    """
    transformation = transformations[0]
    for next_transformation in transformations[1:]:
        transformation = matmul(next_transformation, transformation)
    return transformation


def _one_hot_vector(
        n_dims: int,
        dim: int,
        torch_dtype: dtype,
        torch_device: device
    ) -> Tensor:
    vector = zeros(n_dims, dtype=torch_dtype, device=torch_device)
    vector[dim] = 1
    return vector


def _insert_row_to(
        matrix: Tensor,
        row: Tensor,
        index: int
    ) -> Tensor:
    """Insert row before specified row

    matrix: Tensor with shape (batch_size, n_rows, n_columns)
    row: Tensor with shape (n_columns,) or (batch_size, n_columns)
    index: Insert row to this index
    """
    batch_size = matrix.size(0)
    if row.ndim == 1:
        expanded_row = row.expand(batch_size, -1)
    elif row.ndim == 2:
        expanded_row = row
    else:
        raise RuntimeError('Invalid input')
    return cat(
        [
            matrix[:, :index],
            expanded_row[:, None],
            matrix[:, index:]
        ],
        dim=1
    )


def _insert_column_to(
        matrix: Tensor,
        column: Tensor,
        index: int
    ) -> Tensor:
    """Insert column before specified row

    matrix: Tensor with shape (batch_size, n_rows, n_columns)
    column: Tensor with shape (n_rows,) or (batch_size, n_rows)
    index: Insert column to this index
    """
    batch_size = matrix.size(0)
    if column.ndim == 1:
        expanded_column = column.expand(batch_size, -1)
    elif column.ndim == 2:
        expanded_column = column
    else:
        raise RuntimeError('Invalid input')
    return cat(
        [
            matrix[..., :index],
            expanded_column[..., None],
            matrix[..., index:]
        ],
        dim=2
    )


def generate_homogenous_translation_matrix(translation: Tensor) -> Tensor:
    """Create homogenous translation matrix

    translation: Array with shape (batch_size, n_dims)
    """
    batch_size = translation.size(0)
    n_dims = translation.size(1)
    translation_matrix = eye(
        n_dims,
        n_dims,
        device=translation.device,
        dtype=translation.dtype).expand(batch_size, n_dims, n_dims)
    translation_matrix = _insert_row_to(
        matrix=_insert_column_to(
            matrix=translation_matrix,
            column=translation,
            index=n_dims
        ),
        row=_one_hot_vector(
            n_dims + 1,
            -1,
            torch_dtype=translation.dtype,
            torch_device=translation.device),
        index=n_dims
    )
    return translation_matrix


def _generate_rotation_matrix_2d(
        angle: Tensor
    ) -> Tensor:
    """Generate rotation matrix

    Args:
        angle: Tensor with shape (batch_size,)

    Returns: Tensor with shape (batch_size, 2 + 1, 2 + 1)
    """
    sin_angle = sin(angle)
    cos_angle = cos(angle)
    return stack(
        [
            stack(
                [
                    cos_angle,
                    -sin_angle
                ],
                dim=1),
            stack(
                [
                    sin_angle,
                    cos_angle
                ],
                dim=1)
        ],
        dim=1
    )


def _generate_rotation_matrix_3d(
        angles: Tensor
    ) -> Tensor:
    """Generate rotation matrix

    Args:
        angles: Tensor with shape (batch_size, 3)

    Returns: Tensor with shape (batch_size, 3 + 1, 3 + 1)
    """
    torch_device = angles.device
    torch_dtype = angles.dtype
    rotation_x = _insert_row_to(
        matrix=_insert_column_to(
            matrix=_generate_rotation_matrix_2d(angle=angles[:, 0]),
            column=zeros(2, dtype=torch_dtype, device=torch_device),
            index=0
        ),
        row=tensor([1, 0, 0], dtype=torch_dtype, device=torch_device),
        index=0
    )
    rotation_y = _insert_row_to(
        matrix=_insert_column_to(
            matrix=_generate_rotation_matrix_2d(angle=angles[:, 1]),
            column=zeros(2, dtype=torch_dtype, device=torch_device),
            index=1
        ),
        row=tensor([0, 1, 0], dtype=torch_dtype, device=torch_device),
        index=1
    )
    rotation_z = _insert_row_to(
        matrix=_insert_column_to(
            matrix=_generate_rotation_matrix_2d(angle=angles[:, 2]),
            column=zeros(2, dtype=torch_dtype, device=torch_device),
            index=2
        ),
        row=tensor([0, 0, 1], dtype=torch_dtype, device=torch_device),
        index=2
    )
    return concatenate_transformations([rotation_x, rotation_y, rotation_z])


def generate_rotation_matrix(angles: Tensor, n_dims: int) -> Tensor:
    """Generate rotation matrix

    angles: NDArray with shape (batch_size, n_rotation_axes)
    n_dims: Dimensionality of the rotation matrix
    """
    if n_dims == 2:
        return _generate_rotation_matrix_2d(angles[:, 0])
    if n_dims == 3:
        return _generate_rotation_matrix_3d(angles)
    raise ValueError('Not implemented for dimensions higher than 3')


def generate_homogenous_matrix_from_rigid_transformation(
        angles: Tensor,
        translations: Tensor,
        origin: Tensor,
        invert: bool = False
    ) -> Tensor:
    """Generate homogous transformation matrix representing rotation around given origin

    angles: Tensor with shape (batch_size, n_rotation_axes)
    translations: Tensor with shape (batch_size, n_dims)
    origin: Tensor with shape (batch_size, n_dims)
    """
    n_dims = origin.shape[1]
    torch_dtype = angles.dtype
    torch_device = angles.device
    if invert:
        angles = -angles
    rotation = generate_rotation_matrix(angles, n_dims)
    homogenous_rotation = _insert_row_to(
        matrix=_insert_column_to(
            matrix=rotation,
            column=zeros(n_dims, dtype=torch_dtype, device=torch_device),
            index=n_dims
        ),
        row=_one_hot_vector(n_dims + 1, -1, torch_dtype=torch_dtype, torch_device=torch_device),
        index=n_dims
    )
    backward_shift = -origin
    forward_shift = origin + translations
    if invert:
        backward_shift = -backward_shift
        forward_shift = -forward_shift
    backward_translation = generate_homogenous_translation_matrix(backward_shift)
    forward_translation = generate_homogenous_translation_matrix(forward_shift)
    transformations = [
        backward_translation,
        homogenous_rotation,
        forward_translation]
    if invert:
        transformations.reverse()
    return concatenate_transformations(transformations)


def generate_affine_ddf(
        affine_transformation: Tensor,
        deformation_shape: Tensor,
        deformation_voxel_size: Optional[Tensor] = None
    ) -> Tensor:
    """Generates ddf from affine transformation

    Origin is assumed to be in the middle of the volume

    Args:
        affine_transformation: Transformation represented as homogenous
            transformation matrix with shape (batch_size, n_dims + 1, n_dims + 1)
        deformation_shape: Tensor with shape (n_dims,)
        volume_shape: Optional Tensor with shape (n_dims,)
        deformation_voxel_size: Voxel size of the returned ddf, Optional Tensor with shape (n_dims,)

    Returns:
        Tensor with shape (batch_size, n_dims, deformation_shape[0], ..., deformation_shape[n_dims])
    """
    batch_size = affine_transformation.size(0)
    n_dims = deformation_shape.size(0)
    torch_device = affine_transformation.device
    torch_dtype = affine_transformation.dtype
    coordinates = generate_coordinate_grid(
        grid_shape=deformation_shape,
        grid_voxel_size=deformation_voxel_size
    )
    coordinates_flattened = coordinates.view(n_dims, -1)
    homogenous_coordinates = ones(
        (1, coordinates_flattened.size(1)),
        dtype=torch_dtype,
        device=torch_device)
    homogenous_coordinates_flattened = cat(
        (coordinates_flattened, homogenous_coordinates),
        dim=0
    )
    transformed_homogenous_coordinates_flattened = matmul(
        affine_transformation,
        homogenous_coordinates_flattened[None]
    )
    transformed_coordinates = (
        transformed_homogenous_coordinates_flattened[...,:-1, :].reshape(
            (batch_size,) + coordinates.shape
        )
    )
    affine_ddf = transformed_coordinates - coordinates[None]
    return affine_ddf


def apply_affine_transformation_to_deformation(
        affine_transformation: Tensor,
        deformation: Tensor,
        deformation_voxel_size: Optional[Tensor] = None
    ) -> Tensor:
    """Generates ddf from affine transformation

    Origin is assumed to be in the middle of the volume

    Args:
        affine_transformation: Transformation represented as homogenous
            transformation matrix with shape (batch_size, n_dims + 1, n_dims + 1)
        deformation: Deformation representeda as ddf,
            Tensor with shape (batch_size, n_dims, dim_1, ..., dim_{n_dims})
        volume_shape: Optional Tensor with shape (n_dims,)
        deformation_voxel_size: Optional Tensor with shape (n_dims,)

    Returns:
        Tensor with shape (batch_size, n_dims, deformation_shape[0], ..., deformation_shape[n_dims])
    """
    batch_size = affine_transformation.size(0)
    torch_device = deformation.device
    torch_dtype = deformation.dtype
    deformation_shape = tensor(deformation.shape[2:], device=torch_device)
    n_dims = deformation_shape.size(0)
    coordinates = generate_coordinate_grid(
        grid_shape=deformation_shape,
        grid_voxel_size=deformation_voxel_size
    )
    deformation_coordinates = coordinates[None] + deformation
    coordinates_flattened = deformation_coordinates.view(batch_size, n_dims, -1)
    homogenous_coordinates = ones(
        (batch_size, 1, coordinates_flattened.size(2)),
        dtype=torch_dtype,
        device=torch_device)
    homogenous_coordinates_flattened = cat(
        (coordinates_flattened, homogenous_coordinates),
        dim=1
    )
    transformed_homogenous_coordinates_flattened = matmul(
        affine_transformation,
        homogenous_coordinates_flattened
    )
    transformed_coordinates = (
        transformed_homogenous_coordinates_flattened[:, :-1, :].reshape(deformation.shape)
    )
    transformed_ddf = transformed_coordinates - coordinates[None]
    return transformed_ddf
