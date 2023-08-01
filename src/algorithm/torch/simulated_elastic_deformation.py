"""Generate random elastic deformations"""


from typing import Optional, Sequence, Tuple

from attr import define
from numpy import array as np_array
from numpy import asarray, ceil  # pylint: disable=no-name-in-module
from numpy.linalg import inv
from numpy.random import RandomState
from scipy.ndimage import gaussian_filter  # type: ignore
from torch import Tensor, device, exp, float32, from_numpy, linspace, meshgrid, stack
from torch import sum as torch_sum
from torch import tensor

from algorithm.rigid_transformation import (
    concatenate_transformations,
    generate_homogenous_rotation_matrix,
    generate_homogenous_translation_matrix,
    infer_origin,
)
from algorithm.torch.cubic_spline_upsampling import CubicSplineUpsampling
from algorithm.torch.deformation.primitive import (
    apply_affine_transformation_to_deformation,
    compose_deformations,
    generate_affine_ddf,
    integrate_svf,
)


@define
class GaussianElasticDeformationDefinition:
    """Defines elastic deformation consisting of rigid component
    and svf integrated gaussian bump

    width: Width of the elastic component along each dimension
        (as proportion to the whole width)
    magnitude: Signed magnitude of the elastic component along each
        dimension (in world coordinates)
    center: Center of the elastic component along each dimension (from 0 to 1)
    rotation: Rigid rotation in radians
    translation: Translation along each dimension (in world coordinates)
    """

    width: Sequence[float]
    magnitude: Sequence[float]
    center: Sequence[float]
    rotation: Sequence[float]
    translation: Sequence[float]

    @property
    def n_dims(self) -> int:
        """Amount of dimensions"""
        return len(self.width)


def _sample_tuple_between_sequences(
    lower_limit: Sequence[float], upper_limit: Sequence[float], random_state: RandomState
) -> Tuple[float, ...]:
    return tuple(
        random_state.rand(len(lower_limit)) * (np_array(upper_limit) - np_array(lower_limit))
        + np_array(lower_limit)
    )


def sample_random_gaussian_deformations(
    batch_size: int,
    lower_limit: GaussianElasticDeformationDefinition,
    upper_limit: GaussianElasticDeformationDefinition,
    random_state: RandomState,
) -> Sequence[GaussianElasticDeformationDefinition]:
    """Sample random deformation between the given random deformations"""
    return [
        GaussianElasticDeformationDefinition(
            width=_sample_tuple_between_sequences(
                lower_limit=lower_limit.width,
                upper_limit=upper_limit.width,
                random_state=random_state,
            ),
            magnitude=_sample_tuple_between_sequences(
                lower_limit=lower_limit.magnitude,
                upper_limit=upper_limit.magnitude,
                random_state=random_state,
            ),
            center=_sample_tuple_between_sequences(
                lower_limit=lower_limit.center,
                upper_limit=upper_limit.center,
                random_state=random_state,
            ),
            rotation=_sample_tuple_between_sequences(
                lower_limit=lower_limit.rotation,
                upper_limit=upper_limit.rotation,
                random_state=random_state,
            ),
            translation=_sample_tuple_between_sequences(
                lower_limit=lower_limit.translation,
                upper_limit=upper_limit.translation,
                random_state=random_state,
            ),
        )
        for _ in range(batch_size)
    ]


def _elastic_deformation_dim(
    shape: Tuple[int, ...],
    deformation_definition: GaussianElasticDeformationDefinition,
    dim: int,
    dim_voxel_size: float,
    torch_device: device,
) -> Tensor:
    grid = stack(
        meshgrid(
            *(linspace(0, 1, dim_size, device=torch_device, dtype=float32) for dim_size in shape),
            indexing="ij"
        ),
        dim=-1,
    )
    deformation = (
        (
            exp(
                -(1 / 2)
                * torch_sum(
                    (
                        (
                            grid
                            - tensor(
                                deformation_definition.center, device=torch_device, dtype=float32
                            )
                        )
                        / deformation_definition.width[dim]
                    )
                    ** 2,
                    dim=-1,
                )
            )
        )
        * deformation_definition.magnitude[dim]
        / dim_voxel_size
    )

    return deformation


def generate_gaussian_ddf(
    shape: Tuple[int, ...],
    deformation_definition: GaussianElasticDeformationDefinition,
    voxel_size: Optional[Sequence[float]] = None,
    invert=False,
) -> Tensor:
    """Generate deformation as ddf in voxel coordinates

    Args:
        shape: Shape of the generated deformation
        deformation_definition: Defines the deformation
        voxel_size: Voxel size along each dimension
        invert: Invert the deformation
    """
    if voxel_size is None:
        voxel_size = [1.0] * len(shape)
    torch_device = device("cpu")
    n_dims = len(shape)
    svf = stack(
        [
            _elastic_deformation_dim(
                shape=shape,
                deformation_definition=deformation_definition,
                dim=dim,
                dim_voxel_size=voxel_size[dim],
                torch_device=torch_device,
            )
            for dim in range(n_dims)
        ],
        dim=0,
    )
    if invert:
        svf = -svf
    elastic_component = integrate_svf(svf[None].contiguous(), integration_steps=7)
    origin = infer_origin(shape)
    random_rigid_transformation = concatenate_transformations(
        [
            generate_homogenous_rotation_matrix(
                angles=np_array([deformation_definition.rotation], dtype="float32"),
                origin=(origin * np_array(voxel_size))[None],
            ),
            generate_homogenous_translation_matrix(
                translation=(np_array([deformation_definition.translation], dtype="float32"))
            ),
        ]
    )
    voxel_size_torch = tensor(voxel_size, device=torch_device, dtype=float32)
    voxel_size_torch_broadcast = voxel_size_torch[(None, ...) + (None,) * n_dims]
    if invert:
        inverse_random_rigid_transformation = inv(random_rigid_transformation[0])[None]
        affine_deformation = (
            generate_affine_ddf(
                affine_transformation=from_numpy(
                    inverse_random_rigid_transformation.astype("float32")
                ),
                deformation_shape=tensor(shape, device=torch_device, dtype=float32),
                deformation_voxel_size=voxel_size_torch,
            )
            / voxel_size_torch_broadcast
        )
        composed_deformation = compose_deformations(
            deformation_1=elastic_component, deformation_2=affine_deformation
        )[0]
    else:
        composed_deformation = (
            apply_affine_transformation_to_deformation(
                affine_transformation=from_numpy(random_rigid_transformation.astype("float32")),
                deformation=elastic_component * voxel_size_torch_broadcast,
                deformation_voxel_size=voxel_size_torch,
            )
            / voxel_size_torch_broadcast
        )
    return composed_deformation[0].detach()


def sample_random_noise_ddf(
    shape: Tuple[int, ...],
    downsampling_factor: Sequence[int],
    noise_mean: Sequence[float],
    noise_std: Sequence[float],
    gaussian_filter_std: Sequence[float],
    rotation_bounds: tuple[Sequence[float], Sequence[float]],
    translation_bounds: tuple[Sequence[float], Sequence[float]],
    random_state: RandomState,
    voxel_size: Optional[Sequence[float]] = None,
) -> tuple[Tensor, Tensor]:
    """Sample elastic deformation consisting of a rigid component
    and a white noise based elastic component

    The elastic component is built as low resolution white noise which
    is first gaussian filtered, then upsampled using cubic spline interpolation and
    finally svf integrated

    Args:
        shape: Shape of the generated deformation
        downsampling_factor: Downsampling factor of the low resolution svf
            smoothed, upsampled and integrated to produce the final deformation
        gaussian_filter_std: Std of the gaussian filter applied to the low resolution random
            white noise before upsampling (in world coordinates)
        noise_mean: Noise mean of the elastic deformation (in world coordinates)
        noise_std: Noise standard deviation of the elastic deformation (in world coordinates)
        rotation_bounds: Lower and upper bound of rotation in radians, along each rotation axis
        translation: Lower and upper bound of translation (in world coordinates), along
            each dimension
        voxel_size: Voxel size along each dimension
        invert: Invert the deformation
    """
    if voxel_size is None:
        voxel_size = [1.0] * len(shape)
    torch_device = device("cpu")
    n_dims = len(shape)
    voxel_size_np = asarray(voxel_size, dtype="float32")
    voxel_noise_std = asarray(noise_std, dtype="float32") / voxel_size_np
    voxel_noise_mean = asarray(noise_mean, dtype="float32") / voxel_size_np
    voxel_gaussian_std = asarray(gaussian_filter_std, dtype="float32") / voxel_size_np
    downsampled_shape = ceil(asarray(shape) / asarray(downsampling_factor)).astype("int")
    downsampled_svf = from_numpy(
        random_state.randn(*(n_dims,) + tuple(downsampled_shape)).astype("float32")
        * voxel_noise_std[(...,) + (None,) * n_dims] + voxel_noise_mean[(...,) + (None,) * n_dims]
    )
    downsampled_svf = stack(
        [
            from_numpy(gaussian_filter(downsampled_svf[dim].numpy(), sigma=voxel_gaussian_std))
            for dim in range(n_dims)
        ],
        dim=0,
    )
    svf = CubicSplineUpsampling(upsampling_factor=downsampling_factor)(downsampled_svf[None])[
        (...,) + tuple(slice(dim_size) for dim_size in shape)
    ]
    elastic_component = integrate_svf(svf.contiguous(), integration_steps=7)
    inverse_elastic_component = integrate_svf(-svf.contiguous(), integration_steps=7)
    origin = infer_origin(shape)
    translation = _sample_tuple_between_sequences(
        lower_limit=translation_bounds[0],
        upper_limit=translation_bounds[1],
        random_state=random_state,
    )
    rotation = _sample_tuple_between_sequences(
        lower_limit=rotation_bounds[0],
        upper_limit=rotation_bounds[1],
        random_state=random_state,
    )
    random_rigid_transformation = concatenate_transformations(
        [
            generate_homogenous_rotation_matrix(
                angles=np_array([rotation], dtype="float32"),
                origin=(origin * np_array(voxel_size))[None],
            ),
            generate_homogenous_translation_matrix(
                translation=(np_array([translation], dtype="float32"))
            ),
        ]
    )
    voxel_size_torch = tensor(voxel_size, device=torch_device, dtype=float32)
    voxel_size_torch_broadcast = voxel_size_torch[(None, ...) + (None,) * n_dims]

    inverse_random_rigid_transformation = inv(random_rigid_transformation[0])[None]
    affine_deformation = (
        generate_affine_ddf(
            affine_transformation=from_numpy(inverse_random_rigid_transformation.astype("float32")),
            deformation_shape=tensor(shape, device=torch_device, dtype=float32),
            deformation_voxel_size=voxel_size_torch,
        )
        / voxel_size_torch_broadcast
    )
    inverse_composed_deformation = compose_deformations(
        deformation_1=inverse_elastic_component, deformation_2=affine_deformation
    )[0][0].detach()

    composed_deformation = (
        apply_affine_transformation_to_deformation(
            affine_transformation=from_numpy(random_rigid_transformation.astype("float32")),
            deformation=elastic_component * voxel_size_torch_broadcast,
            deformation_voxel_size=voxel_size_torch,
        )
        / voxel_size_torch_broadcast
    )[0].detach()

    return composed_deformation, inverse_composed_deformation
