"""Simple 1d interpolation algorithm"""

from torch import Tensor, clamp, ge
from torch import sum as torch_sum


def interpolate_1d(interpolation_x: Tensor, data_x: Tensor, data_y: Tensor) -> Tensor:
    """One-dimensional linear interpolation for monotonically increasing sample
    points.

    Taken from https://github.com/pytorch/pytorch/issues/50334

    Returns the one-dimensional piecewise linear interpolant to a function with
    given discrete data points (xp, fp), evaluated at x.

    Args:
        x: x-coordinates at which to evaluate the interpolated
            values.
        xp: x-coordinates of the data points, must be increasing.
        data_y: y-coordinates of the data points, same length as `xp`.

    Returns:
        the interpolated values, same size as `x`.
    """
    slope = (data_y[1:] - data_y[:-1]) / (data_x[1:] - data_x[:-1])
    constant = data_y[:-1] - (slope * data_x[:-1])

    indicies = torch_sum(ge(interpolation_x[:, None], data_x[None, :]), 1) - 1
    indicies = clamp(indicies, 0, len(slope) - 1)

    return slope[indicies] * interpolation_x + constant[indicies]
