"""Data tools for rigid transformations"""

from typing import Optional, Sequence, Tuple

from numpy import asarray, ndarray, pi
from numpy.random import RandomState

from algorithm.rigid_transformation import (
    apply_transformation_to_grid,
    concatenate_transformations,
    generate_dense_grid,
    generate_random_flips,
    generate_random_orthogonal_rotations,
    generate_random_rotations,
    generate_random_scalings_and_shears,
    generate_random_translations,
    infer_origin,
)


def generate_simulated_affine_displacement_fields(
    batch_size: int,
    degree_ranges: Sequence[Tuple[float, float]],
    log_scale_scales: Sequence[float],
    log_shear_scales: Sequence[float],
    translation_ranges: Sequence[Tuple[float, float]],
    shape: Sequence[int],
    random_state: RandomState,
    generate_flips: bool = True,
    generate_orthogonal_rotations: bool = True,
    voxel_size: Optional[Sequence[float]] = None,
) -> ndarray:
    """Generate batch of affine transformation displacement fields"""
    n_dims = len(shape)
    if voxel_size is None:
        voxel_size = [1.0] * n_dims
    random_transformation = generate_simulated_affine_transformations(
        batch_size=batch_size,
        degree_ranges=degree_ranges,
        log_scale_scales=log_scale_scales,
        log_shear_scales=log_shear_scales,
        translation_ranges=translation_ranges,
        shape=shape,
        random_state=random_state,
        generate_flips=generate_flips,
        generate_orthogonal_rotations=generate_orthogonal_rotations,
        voxel_size=voxel_size,
    )
    grid = generate_dense_grid(shape, batch_size, voxel_size)
    voxel_size_np = asarray(voxel_size)[(None, ...) + (None,) * n_dims]
    augmented_rigid_displacement_fields = (
        apply_transformation_to_grid(grid=grid, transformations=random_transformation) - grid
    ) / voxel_size_np
    return augmented_rigid_displacement_fields


def generate_simulated_affine_transformations(
    batch_size: int,
    degree_ranges: Sequence[Tuple[float, float]],
    log_scale_scales: Sequence[float],
    log_shear_scales: Sequence[float],
    translation_ranges: Sequence[Tuple[float, float]],
    shape: Sequence[int],
    random_state: RandomState,
    generate_flips: bool = True,
    generate_orthogonal_rotations: bool = True,
    voxel_size: Optional[Sequence[float]] = None,
) -> ndarray:
    """Generate batch of random affine transformations"""
    n_dims = len(shape)
    if voxel_size is None:
        voxel_size = [1.0] * n_dims
    radian_ranges = [
        (degree_range[0] * pi / 180, degree_range[1] * pi / 180) for degree_range in degree_ranges
    ]
    origin = infer_origin(shape) * asarray(voxel_size)
    random_transformations = [
        generate_random_scalings_and_shears(
            batch_size, log_scale_scales, log_shear_scales, origin, random_state
        )
    ]
    if generate_flips:
        random_transformations += [generate_random_flips(batch_size, n_dims, shape, random_state)]
    if generate_orthogonal_rotations:
        random_transformations += [
            generate_random_orthogonal_rotations(batch_size, origin, random_state)
        ]
    random_transformations += [
        generate_random_rotations(batch_size, radian_ranges, origin, random_state),
        generate_random_translations(batch_size, translation_ranges, random_state),
    ]
    return concatenate_transformations(random_transformations)
