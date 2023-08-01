"""Largest interior rectangle algorithm"""


from itertools import product
from typing import Generator, Optional, Sequence, Tuple, Union

from numpy import abs as np_abs
from numpy import all as np_all
from numpy import any as np_any
from numpy import (arange, array, maximum, minimum, ndarray, ndindex, prod,
                   stack)
from numpy import sum as np_sum
from numpy import zeros, zeros_like


def _extended_summed_area_table(mask: ndarray) -> ndarray:
    """Calculate summed area table of the mask with additional zeros on one side

    mask: NDArray with shape (dim_1, ..., dim_{n_dims})
    """
    extended_mask = zeros(array(mask.shape) + 1, dtype=mask.dtype)
    extended_mask[(slice(1, None),) * mask.ndim] = mask
    summed_area_table = extended_mask
    for axis in range(summed_area_table.ndim):
        summed_area_table = summed_area_table.cumsum(axis)
    return summed_area_table


def _ndindex_between(
        start_index: Union[Sequence[int], ndarray],
        end_index: Union[Sequence[int], ndarray]
    ) -> Generator[Tuple[int, ...], None, None]:
    """Iterates multidimensional indiices starting from startindex"""
    for indices in ndindex(*(end_index[dim] - start_index[dim] for dim in range(len(end_index)))):
        yield tuple(indices[dim] + start_index[dim] for dim in range(len(end_index)))


def _rectangle_size(
        first_corner: Union[Sequence[int], ndarray],
        second_corner: Union[Sequence[int], ndarray]
    ) -> int:
    """Calculates size of a rectangele based on opposing corner points"""
    return prod(np_abs(array(second_corner) - array(first_corner)) + 1)


def _rectangle_mass_2d(
        extended_summed_area_table: ndarray,
        first_corner: Union[Sequence[int], ndarray],
        second_corner: Union[Sequence[int], ndarray]
    ) -> int:
    """Returns mass inside a rectangle based on summed area table

    Returns also the size of the rectangle
    """
    return (
        extended_summed_area_table[second_corner[0] + 1, second_corner[1] + 1]
        + extended_summed_area_table[first_corner[0], first_corner[1]]
        - extended_summed_area_table[first_corner[0], second_corner[1] + 1]
        - extended_summed_area_table[second_corner[0] + 1, first_corner[1]]
    )


def _rectangle_mass(
        extended_summed_area_table: ndarray,
        first_corner: Union[Sequence[int], ndarray],
        second_corner: Union[Sequence[int], ndarray]
    ) -> int:
    """Returns mass inside a rectangle based on summed area table

    Returns also the size of the rectangle
    """
    n_dims = len(first_corner)
    if n_dims == 2:
        return _rectangle_mass_2d(extended_summed_area_table, first_corner, second_corner)
    corner_coordinates = stack([first_corner, array(second_corner) + 1], axis=0)
    mass = 0
    for corners in product((0, 1), repeat=n_dims):
        corner_coordinate = corner_coordinates[corners, arange(corner_coordinates.shape[1])]
        sign = (-1)**(n_dims - np_sum(corners))
        mass += sign * extended_summed_area_table[tuple(corner_coordinate)]
    return mass


def _slice_between_corners(
        first_corner: Union[Sequence[int], ndarray],
        second_corner: Union[Sequence[int], ndarray]
    ) -> Tuple[slice, ...]:
    """Returns tuple of slices based on opposing corners"""
    return tuple(
        slice(first_coord, second_coord + 1)
        for first_coord, second_coord in zip(
            first_corner,
            second_corner
        )
    )


def _largest_interior_rectangle_corners(
        mask: ndarray,
        extended_summed_area_table: ndarray,
        first_corner_search_space: Tuple[ndarray, ndarray],
        second_corner_search_space: Tuple[ndarray, ndarray]
    ) -> Optional[Tuple[ndarray, ndarray]]:
    """Find mask of largest interior rectangle

    mask: NDArray with shape (dim_1, ..., dim_{n_dims})
    """
    largest_rectangle_size: int = 0
    largest_rectangle_corners: Optional[Tuple[ndarray, ndarray]] = None
    first_corner_start = maximum(0, first_corner_search_space[0])
    first_corner_end = minimum(mask.shape, first_corner_search_space[1] + 1)
    for first_corner in _ndindex_between(
            first_corner_start,
            first_corner_end):
        if mask[first_corner] == 0:
            continue
        second_corner_start = maximum(first_corner, second_corner_search_space[0])
        second_corner_end = maximum(
            minimum(mask.shape, second_corner_search_space[1] + 1),
            second_corner_start
        )
        for second_corner in _ndindex_between(
                second_corner_start,
                second_corner_end):
            if mask[second_corner] == 0:
                continue
            rectangle_size = _rectangle_size(first_corner, second_corner)
            if rectangle_size <= largest_rectangle_size:
                continue
            rectangle_mass = _rectangle_mass(
                extended_summed_area_table,
                first_corner,
                second_corner)
            if rectangle_mass == rectangle_size:
                largest_rectangle_size = rectangle_size
                largest_rectangle_corners = (
                    array(first_corner),
                    array(second_corner)
                )
    return largest_rectangle_corners


def largest_interior_rectangle(
        mask: ndarray,
        search_space_per_step: float
    ) -> ndarray:
    """Find mask of largest interior rectangle

    Heuristic algorithm, assumes that (within search_space_per_step) middle point is True

    mask: NDArray with shape (dim_1, ..., dim_{n_dims})
    """
    if np_all(mask > 0):
        return mask
    first_corner = array(mask.shape) // 2
    second_corner = array(mask.shape) // 2
    extended_summed_area_table = _extended_summed_area_table(mask)
    while (
            _rectangle_mass(extended_summed_area_table, first_corner, second_corner)
            == _rectangle_size(first_corner, second_corner)
        ):
        if np_any(first_corner == 0) or np_any(second_corner == array(mask.shape) - 1):
            break
        first_corner -= 1
        second_corner += 1
    for dim in range(mask.ndim):
        while (
            _rectangle_mass(extended_summed_area_table, first_corner, second_corner)
            == _rectangle_size(first_corner, second_corner)
        ):
            if first_corner[dim] == 0:
                break
            first_corner[dim] -= 1
        first_corner[dim] += 1
        while (
            _rectangle_mass(extended_summed_area_table, first_corner, second_corner)
            == _rectangle_size(first_corner, second_corner)
        ):
            if second_corner[dim] == mask.shape[dim] - 1:
                break
            second_corner[dim] += 1
        second_corner[dim] -= 1
    while True:
        corners = _largest_interior_rectangle_corners(
            mask,
            first_corner_search_space=(
                first_corner - search_space_per_step,
                first_corner + search_space_per_step
            ),
            second_corner_search_space=(
                second_corner - search_space_per_step,
                second_corner + search_space_per_step
            ),
            extended_summed_area_table=extended_summed_area_table
        )
        if corners is None:
            raise RuntimeError('Finding largest interior rectangle failed!')
        new_first_corner, new_second_corner = corners
        if np_all(first_corner == new_first_corner) and np_all(second_corner == new_second_corner):
            break
        first_corner = new_first_corner
        second_corner = new_second_corner
    mask = zeros_like(mask)
    mask[_slice_between_corners(new_first_corner, new_second_corner)] = 1
    return mask
