"""Deformation related utility functions"""

from torch import Tensor, meshgrid, linspace, stack
from matplotlib.pyplot import plot # type: ignore

from algorithm.torch.deformation.primitive  import transform_points


def visualize_deformation_2d(deformations: Tensor, n_rows: int, n_cols: int, **kwargs) -> None:
    """Visualizes combined deformation field as deformed grid

    Args:
        deformation: Tensor with shape (n_deformations, n_dims, dim_1, dim_2)
        grid_spacing: Spacing of lines in grid (in pixels)
    """
    device = deformations.device
    rows, cols = meshgrid(
        linspace(0, deformations.size(2) - 1, n_rows, device=device),
        linspace(0, deformations.size(3) - 1, n_cols, device=device)
    )
    transformed_grid = stack([rows, cols], dim=0)
    for deformation in deformations:
        transformed_grid = transform_points(
            deformation=deformation[None],
            points=transformed_grid[None])[0]
    transformed_grid = transformed_grid.cpu().detach()
    for row_index in range(transformed_grid.size(1)):
        plot(
            transformed_grid[1, row_index, :],
            transformed_grid[0, row_index, :],
            color='gray',
            **kwargs)
    for col_index in range(transformed_grid.size(2)):
        plot(
            transformed_grid[1, :, col_index],
            transformed_grid[0, :, col_index],
            color='gray',
            **kwargs)
