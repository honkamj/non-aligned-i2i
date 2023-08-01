"""Algorithms for generating random deformation fields"""

from typing import Optional, Sequence, Tuple, cast

from numpy import (
    arange,
    asarray,
    concatenate,
    cos,
    diag_indices,
    eye,
    matmul,
    meshgrid,
    ndarray,
    newaxis,
    ones,
    outer,
    pi,
    sin,
    stack,
    swapaxes,
    tile,
    tril_indices,
    triu_indices,
    zeros,
)
from numpy.random import RandomState
from numpy.typing import ArrayLike
from scipy.linalg import expm  # type: ignore
from scipy.special import binom  # type: ignore


def concatenate_transformations(transformations: Sequence[ndarray]) -> ndarray:
    """Concatenates multiple transformation matrices

    transformations: Sequence of arrays with shape (batch_size, n_dims, n_dims)
    """
    transformation = transformations[0]
    for next_transformation in transformations[1:]:
        transformation = matmul(next_transformation, transformation)
    return transformation


def generate_batch_eye(n_dims: int, batch_size: int) -> ndarray:
    """Generates batch of identity matrices"""
    return tile(eye(n_dims), reps=(batch_size, 1, 1))


def generate_homogenous_translation_matrix(translation: ndarray) -> ndarray:
    """Create homogenous translation matrix

    translation: Array with shape (batch_size, n_dims)
    """
    batch_size = translation.shape[0]
    n_dims = translation.shape[1]
    translation_matrix = generate_batch_eye(n_dims + 1, batch_size)
    translation_matrix[:, :-1, -1] = translation
    return translation_matrix


def _generate_rotation_matrix_2d(angle: ndarray) -> ndarray:
    """Generate homogous transformation matrix representing rotation around given origin

    angle: NDArray with shape (batch_size,)
    origin: NDArray with shape (batch_size, 2)
    """
    batch_size = angle.shape[0]
    n_dims = 2
    rotation = zeros((batch_size, n_dims, n_dims), dtype=angle.dtype)
    angle_cos, angle_sin = cos(angle), sin(angle)
    rotation[:, 0, 0] = angle_cos
    rotation[:, 0, 1] = -angle_sin
    rotation[:, 1, 0] = angle_sin
    rotation[:, 1, 1] = angle_cos
    return rotation


def _generate_rotation_matrix_3d(angles: ndarray) -> ndarray:
    """Generate homogous transformation matrix representing rotation around given origin

    angles: NDArray with shape (batch_size, 3)
    origin: NDArray with shape (batch_size, 3)
    """
    batch_size = angles.shape[0]
    n_dims = 3
    rotation_x = generate_batch_eye(n_dims, batch_size)
    rotation_y = generate_batch_eye(n_dims, batch_size)
    rotation_z = generate_batch_eye(n_dims, batch_size)
    mask_x = (False, True, True)
    mask_y = (True, False, True)
    mask_z = (True, True, False)
    rotation_x[:, outer(mask_x, mask_x)] = _generate_rotation_matrix_2d(angles[:, 2]).reshape(
        batch_size, -1
    )
    rotation_y[:, outer(mask_y, mask_y)] = _generate_rotation_matrix_2d(angles[:, 1]).reshape(
        batch_size, -1
    )
    rotation_z[:, outer(mask_z, mask_z)] = _generate_rotation_matrix_2d(angles[:, 0]).reshape(
        batch_size, -1
    )
    return concatenate_transformations((rotation_x, rotation_y, rotation_z))


def generate_rotation_matrix(angles: ndarray, n_dims: int) -> ndarray:
    """Generate rotation matrix

    angles: NDArray with shape (batch_size, n_rotation_axes)
    n_dims: Dimensionality of the rotation matrix
    """
    if n_dims == 2:
        return _generate_rotation_matrix_2d(angles[:, 0])
    if n_dims == 3:
        return _generate_rotation_matrix_3d(angles)
    raise ValueError("Not implemented for dimensions higner than 3")


def generate_scaling_and_shear_matrix(log_scales: ndarray, log_shears: ndarray) -> ndarray:
    """Generate scaling matrix

    log_scales: NDArray with shape (batch_size, n_dims)
    log_shears NDArray with shape (batch_size, n_shear_axes)
    """
    batch_size = log_scales.shape[0]
    n_dims = log_scales.shape[1]
    log_scale_and_shear_matrix = zeros((batch_size, n_dims, n_dims))
    log_scale_and_shear_matrix[(slice(None),) + diag_indices(n_dims)] = log_scales
    log_scale_and_shear_matrix[(slice(None),) + triu_indices(n_dims, k=1)] = log_shears
    log_scale_and_shear_matrix[(slice(None),) + tril_indices(n_dims, k=-1)] = log_shears
    scale_and_shear_matrix = expm(log_scale_and_shear_matrix)
    return scale_and_shear_matrix


def generate_homogenous_rotation_matrix(angles: ndarray, origin: ndarray) -> ndarray:
    """Generate homogous transformation matrix representing rotation around given origin

    angles: NDArray with shape (batch_size, n_rotation_axes)
    origin: NDArray with shape (batch_size, n_dims)
    """
    batch_size = angles.shape[0]
    n_dims = origin.shape[1]
    homogenous_rotation = zeros((batch_size, n_dims + 1, n_dims + 1), dtype=angles.dtype)
    homogenous_rotation[:, -1, -1] = 1
    homogenous_rotation[:, :-1, :-1] = generate_rotation_matrix(angles, n_dims)
    backward_translation = generate_homogenous_translation_matrix(-origin)
    forward_translation = generate_homogenous_translation_matrix(origin)
    return concatenate_transformations(
        (backward_translation, homogenous_rotation, forward_translation)
    )


def generate_homogenous_scaling_and_shear_matrix(
    log_scales: ndarray, log_shears: ndarray, origin: ndarray
) -> ndarray:
    """Generate homogous transformation matrix representing scaling around given origin

    log_scales: NDArray with shape (batch_size, n_dims)
    log_shears: NDArray with shape (batch_size, n_shear_axes)
    origin: NDArray with shape (batch_size, n_dims)
    """
    batch_size = log_scales.shape[0]
    n_dims = origin.shape[1]
    homogenous_scaling_and_shear = zeros(
        (batch_size, n_dims + 1, n_dims + 1), dtype=log_scales.dtype
    )
    homogenous_scaling_and_shear[:, -1, -1] = 1
    homogenous_scaling_and_shear[:, :-1, :-1] = generate_scaling_and_shear_matrix(
        log_scales, log_shears
    )
    backward_translation = generate_homogenous_translation_matrix(-origin)
    forward_translation = generate_homogenous_translation_matrix(origin)
    return concatenate_transformations(
        (backward_translation, homogenous_scaling_and_shear, forward_translation)
    )


def generate_dense_grid(
    shape: Sequence[int], batch_size: int, grid_voxel_size: Optional[Sequence[float]] = None
) -> ndarray:
    """Generate dense grid with given shape

    Returns:
        Array with shape (batch_size, n_dims, *shape)
    """
    if grid_voxel_size is None:
        grid_voxel_size_not_none: Sequence[float] = cast(
            Sequence[float], ones(len(shape), dtype="float")
        )
    else:
        grid_voxel_size_not_none = grid_voxel_size
    return stack(
        [
            stack(
                meshgrid(
                    *(
                        dim_voxel_size * arange(dim_size, dtype="float")
                        for dim_size, dim_voxel_size in zip(shape, grid_voxel_size_not_none)
                    ),
                    indexing="ij"
                ),
                axis=0,
            )
            for _ in range(batch_size)
        ],
        axis=0,
    )


def apply_transformation_to_grid(grid: ndarray, transformations: ndarray) -> ndarray:
    """Applies transformation to coordinate grid

    Args:
        grid: Array with shape (batch_size, n_dims, dim_1, ..., dim_{n_dims})
        transformations: Array with shape (batch_size, n_dims + 1, n_dims + 1)
    """
    batch_size = grid.shape[0]
    n_dims = grid.shape[1]
    shape = grid.shape[2:]
    coordinates = swapaxes(grid, 1, -1).reshape(batch_size, -1, n_dims, 1)
    n_points = coordinates.shape[1]
    homogenous_coordinates = concatenate([coordinates, ones((batch_size, n_points, 1, 1))], axis=2)
    transformed_coordinates = matmul(transformations[:, newaxis], homogenous_coordinates)[:, :, :-1]
    return swapaxes(transformed_coordinates.reshape(batch_size, *shape, n_dims), 1, -1)


def infer_origin(shape: ArrayLike) -> ndarray:
    """Calculate origin of array in voxel coordinates"""
    shape_array = asarray(shape)
    return (shape_array - 1) / 2


def generate_random_orthogonal_rotations(
    batch_size: int, origin: Sequence[float], random_state: RandomState
) -> ndarray:
    """Generates random rotations which are multiplies of 90 degrees with center origin"""
    n_dims = len(origin)
    n_rotation_axes = int(binom(n_dims, 2))
    multiplicity = random_state.randint(low=0, high=4, size=(batch_size, n_rotation_axes)).astype(
        "float32"
    )
    radians = multiplicity * pi / 2
    origin_np = asarray(origin)
    return generate_homogenous_rotation_matrix(radians, origin_np[newaxis])


def generate_random_rotations(
    batch_size: int,
    radian_range: Sequence[Tuple[float, float]],
    origin: Sequence[float],
    random_state: RandomState,
) -> ndarray:
    """Generates random rotations  with center origin"""
    radian_range_np = asarray(radian_range)
    n_rotation_axes = radian_range_np.shape[0]
    radians = (
        random_state.rand(batch_size, n_rotation_axes)
        * (radian_range_np[:, 1][newaxis] - radian_range_np[:, 0][newaxis])
        + radian_range_np[:, 0][newaxis]
    )
    origin_np = asarray(origin)
    return generate_homogenous_rotation_matrix(radians, origin_np[newaxis])


def generate_random_scalings_and_shears(
    batch_size: int,
    log_scale_scale: Sequence[float],
    log_shear_scale: Sequence[float],
    origin: Sequence[float],
    random_state: RandomState,
) -> ndarray:
    """Generates random scalings with center origin"""
    log_scale_scale_np = asarray(log_scale_scale)
    log_shear_scale_np = asarray(log_shear_scale)
    n_dims = log_scale_scale_np.shape[0]
    log_scales = random_state.randn(batch_size, n_dims) * log_scale_scale_np
    log_shears = random_state.randn(batch_size, log_shear_scale_np.shape[0]) * log_shear_scale_np
    origin_np = asarray(origin)
    return generate_homogenous_scaling_and_shear_matrix(
        log_scales=log_scales, log_shears=log_shears, origin=origin_np[newaxis]
    )


def generate_random_translations(
    batch_size: int, translation_range: Sequence[Tuple[float, float]], random_state: RandomState
) -> ndarray:
    """Create random homogenous translation matrix"""
    translation_range_np = asarray(translation_range)
    n_dims = translation_range_np.shape[0]
    translations = (
        random_state.rand(batch_size, n_dims)
        * (translation_range_np[:, 1][newaxis] - translation_range_np[:, 0][newaxis])
        + translation_range_np[:, 0][newaxis]
    )
    return generate_homogenous_translation_matrix(translation=translations)


def generate_random_gaussian_translations(
    batch_size: int, translation_std: float, n_dims: int, random_state: RandomState
) -> ndarray:
    """Create random homogenous translation matrix"""
    translations = random_state.normal(loc=0.0, scale=translation_std, size=(batch_size, n_dims))
    return generate_homogenous_translation_matrix(translation=translations)


def generate_random_flips(
    batch_size: int, n_dims: int, shape: Sequence[int], random_state: RandomState
) -> ndarray:
    """Generates random rotations  with center origin"""
    transformations = zeros((batch_size, n_dims + 1, n_dims + 1), dtype="float")
    diagonal_indices = diag_indices(n_dims)
    for index in range(batch_size):
        flips = 2 * random_state.randint(2, size=n_dims) - 1
        transformations[index, :-1, :-1][diagonal_indices] = flips
    transformations[:, -1, -1] = 1
    origin = infer_origin(shape)
    backward_translation = generate_homogenous_translation_matrix(-origin[newaxis])
    forward_translation = generate_homogenous_translation_matrix(origin[newaxis])
    reflection_matrix_around_origin = matmul(
        forward_translation, matmul(transformations, backward_translation)
    )
    return reflection_matrix_around_origin
