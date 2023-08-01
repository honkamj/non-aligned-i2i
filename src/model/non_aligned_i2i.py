"""I2I with non-aligned data

Uses U-Nets as workhorses
"""

from math import pi
from typing import Any, Callable, Mapping, Optional, Sequence, Set, Tuple

from scipy.special import binom  # type: ignore
from torch import Tensor, cat, tensor
from torch.nn import LeakyReLU, Module, ReLU

from algorithm.torch.deformation.masked import AffineTransformation, MaskedDeformation, MaskedVolume
from algorithm.torch.deformation.primitive import (
    generate_coordinate_grid,
    generate_homogenous_matrix_from_rigid_transformation,
    integrate_svf,
)
from util.executor import ExecutionGraph, NodeDefinition, RequiredNodeExecutor

from .resnet_rigid_reg_encoder import ResNetEncoderNd
from .unet import UNetNd


class I2IRegNetNd(Module):
    """Module combining I2I translation and registration"""

    def __init__(
        self,
        n_input_channels: int,
        n_output_channels: int,
        n_features_per_block_i2i: Sequence[int],
        n_features_per_block_intra_modality_reg: Sequence[int],
        n_features_per_block_cross_modality_reg: Sequence[int],
        n_features_per_block_rigid_reg: Sequence[int],
        input_image_shape: Tuple[int, ...],
        voxel_size: Sequence[float],
        final_activation: Callable[[Tensor], Tensor],
        max_rotation_degrees: Optional[float] = None,
        max_translation: Optional[float] = None,
        n_normalization_groups: Optional[int] = None,
    ) -> None:
        super().__init__()
        n_dims: int = len(input_image_shape)
        self._voxel_size = voxel_size
        self.i2i_unet = UNetNd(
            n_dims=n_dims,
            n_input_channels=n_input_channels,
            n_output_channels=n_output_channels,
            activation=LeakyReLU(),
            final_activation=final_activation,
            n_features_per_block=n_features_per_block_i2i,
            n_normalization_groups=n_normalization_groups,
        )
        self.rigid_registration_encoder = ResNetEncoderNd(
            n_dims=n_dims,
            n_input_channels=n_input_channels + n_output_channels + 2,
            input_image_shape=input_image_shape,
            n_features_per_block=n_features_per_block_rigid_reg,
            activation=ReLU(),
            n_outputs=n_dims + self._get_num_rotation_axes(n_dims),
        )
        if max_rotation_degrees is not None:
            self._max_rotation: Optional[float] = max_rotation_degrees * pi / 180
        else:
            self._max_rotation = None
        self._max_translation = max_translation
        self.cross_modality_registration_unet = UNetNd(
            n_dims=n_dims,
            n_input_channels=n_input_channels + n_output_channels + 2,
            n_output_channels=n_dims,
            activation=ReLU(),
            final_activation=lambda x: x,
            n_features_per_block=n_features_per_block_cross_modality_reg,
            n_normalization_groups=None,
        )
        self.intra_modality_registration_unet = UNetNd(
            n_dims=n_dims,
            n_input_channels=n_dims + 2 * n_output_channels + 2,
            n_output_channels=n_dims,
            activation=ReLU(),
            final_activation=lambda x: x,
            n_features_per_block=n_features_per_block_intra_modality_reg,
            n_normalization_groups=None,
        )
        self._input_image_shape = input_image_shape
        self._node_executor = self._define_node_executor()

    @staticmethod
    def _get_num_rotation_axes(n_dims: int) -> int:
        return int(binom(n_dims, 2))

    def _define_node_executor(self) -> RequiredNodeExecutor:
        graph_definition = (
            NodeDefinition(
                input_names={"forward_random_transformation"},
                output_names={"inverse_random_transformation"},
                executable=self._invert_random_transformation,
            ),
            NodeDefinition(
                input_names={"input", "label"},
                output_names={"rigid_angles", "rigid_translations"},
                executable=self._generate_rigid_transformation_parameters,
            ),
            NodeDefinition(
                input_names={"input", "rigid_angles", "rigid_translations"},
                output_names={"forward_rigid_transformation"},
                executable=self._generate_forward_rigid_transformation,
            ),
            NodeDefinition(
                input_names={"input", "rigid_angles", "rigid_translations"},
                output_names={"inverse_rigid_transformation"},
                executable=self._generate_inverse_rigid_transformation,
            ),
            NodeDefinition(
                input_names={"label", "forward_rigid_transformation"},
                output_names={"rigidly_deformed_label"},
                executable=self._rigidly_deform_label,
            ),
            NodeDefinition(
                input_names={"rigidly_deformed_label", "input"},
                output_names={"cross_modality_deformation_svf"},
                executable=self._generate_cross_modality_deformation_svf,
            ),
            NodeDefinition(
                input_names={
                    "input",
                    "cross_modality_deformation_svf",
                    "forward_rigid_transformation",
                    "inverse_rigid_transformation",
                },
                output_names={
                    "forward_cross_modality_deformation",
                    "inverse_cross_modality_deformation",
                    "forward_elastic_cross_modality_deformation",
                    "inverse_elastic_cross_modality_deformation",
                },
                executable=self._integrate_cross_modality_deformation,
            ),
            NodeDefinition(
                input_names={"label", "forward_cross_modality_deformation"},
                output_names={"cross_modality_deformed_label"},
                executable=self._cross_modality_deform_label,
            ),
            NodeDefinition(
                input_names={
                    "inverse_elastic_cross_modality_deformation",
                    "cross_modality_deformed_label",
                    "output",
                },
                output_names={"intra_modality_deformation_svf"},
                executable=self._generate_intra_modality_deformation_svf,
            ),
            NodeDefinition(
                input_names={"input", "intra_modality_deformation_svf"},
                output_names={"forward_intra_modality_deformation"},
                executable=self._forward_integrate_intra_modality_deformation,
            ),
            NodeDefinition(
                input_names={"input", "intra_modality_deformation_svf"},
                output_names={"inverse_intra_modality_deformation"},
                executable=self._inverse_integrate_intra_modality_deformation,
            ),
            NodeDefinition(
                input_names={"inverse_intra_modality_deformation", "inverse_random_transformation"},
                output_names={"output_commutation_deformation"},
                executable=self._compose_output_commutation_deformation,
            ),
            NodeDefinition(
                input_names={"output_commutation_deformation", "left_random_deformed_output"},
                output_names={"commutation_deformed_output"},
                executable=self._deform_commutation_output,
            ),
            NodeDefinition(
                input_names={
                    "output_commutation_deformation",
                    "forward_random_transformation",
                    "left_random_deformed_output",
                },
                output_names={
                    "random_commutation_deformation",
                    "random_commutation_deformed_output",
                },
                executable=self._random_commutation_deform_output,
            ),
            NodeDefinition(
                input_names={
                    "forward_cross_modality_deformation",
                    "inverse_cross_modality_deformation",
                    "forward_elastic_cross_modality_deformation",
                    "inverse_elastic_cross_modality_deformation",
                    "forward_intra_modality_deformation",
                    "inverse_intra_modality_deformation",
                },
                output_names={
                    "forward_deformation",
                    "inverse_deformation",
                    "forward_elastic_deformation",
                    "inverse_elastic_deformation",
                },
                executable=self._compose_deformations,
            ),
            NodeDefinition(
                input_names={
                    "inverse_rigid_transformation",
                    "inverse_elastic_cross_modality_deformation",
                    "inverse_intra_modality_deformation",
                },
                output_names={"inverse_deformation"},
                executable=self._compose_inverse_deformation,
            ),
            NodeDefinition(
                input_names={"output", "inverse_intra_modality_deformation"},
                output_names={"deformed_output"},
                executable=self._intra_modality_deform_output,
            ),
            NodeDefinition(
                input_names={
                    "label",
                    "forward_random_transformation",
                    "forward_cross_modality_deformation",
                },
                output_names={"random_cross_modality_deformed_label"},
                executable=self._random_deform_cross_modality_deformed_label,
            ),
            NodeDefinition(
                input_names={"input", "forward_random_transformation"},
                output_names={"random_deformed_input"},
                executable=self._random_deform_input,
            ),
            NodeDefinition(
                input_names={"output", "forward_random_transformation"},
                output_names={"right_random_deformed_output"},
                executable=self._random_deform_output,
            ),
            NodeDefinition(
                input_names={"input", "forward_random_transformation"},
                output_names={"output", "left_random_deformed_output"},
                executable=self._apply_i2i,
            ),
            NodeDefinition(
                input_names={"input", "inverse_deformation"},
                output_names={"predicted_deformation_left_deformed_output"},
                executable=self._predicted_deformation_left_deform_output,
            ),
            NodeDefinition(
                input_names={"output", "inverse_deformation"},
                output_names={"predicted_deformation_right_deformed_output"},
                executable=self._predicted_deformation_right_deform_output,
            ),
        )
        return RequiredNodeExecutor(ExecutionGraph(graph_definition=graph_definition))

    @staticmethod
    def _invert_random_transformation(
        inputs: Mapping[str, Any], _args: Mapping[str, Any], _required_outputs: Set[str]
    ) -> Optional[Mapping[str, Any]]:
        return {"inverse_random_transformation": inputs["forward_random_transformation"].invert()}

    def _generate_rigid_transformation_parameters(
        self, inputs: Mapping[str, Any], _args: Mapping[str, Any], _required_outputs: Set[str]
    ) -> Optional[Mapping[str, Any]]:
        image_shape = tensor(
            self._input_image_shape,
            dtype=inputs["input"].volume.dtype,
            device=inputs["input"].volume.device,
        )
        scaling_factor = image_shape[(slice(None),) + (None,) * len(self._input_image_shape)]
        coordinate_grid = (
            generate_coordinate_grid(
                grid_shape=image_shape, grid_voxel_size=inputs["input"].voxel_size
            )
            / scaling_factor
        )
        stacked_coordinate_grid = coordinate_grid.expand(
            inputs["input"].batch_size, *coordinate_grid.shape
        )
        rigid_transformation_params = self.rigid_registration_encoder(
            volume=cat(
                [
                    inputs["input"].volume.detach(),
                    inputs["input"].generate_mask(),
                    inputs["label"].volume,
                    inputs["label"].generate_mask(),
                ],
                dim=1,
            ),
            coordinates=stacked_coordinate_grid,
        )
        n_dims = len(self._input_image_shape)
        n_rotation_axes = self._get_num_rotation_axes(n_dims)
        return {
            "rigid_angles": rigid_transformation_params[:, :n_rotation_axes],
            "rigid_translations": rigid_transformation_params[:, n_rotation_axes:],
        }

    def _generate_forward_rigid_transformation(
        self, inputs: Mapping[str, Any], _args: Mapping[str, Any], _required_outputs: Set[str]
    ) -> Optional[Mapping[str, Any]]:
        origin = (
            inputs["input"].voxel_size
            * (
                tensor(
                    self._input_image_shape,
                    dtype=inputs["rigid_angles"].dtype,
                    device=inputs["rigid_angles"].device,
                )[None]
                - 1
            )
            / 2
        )
        forward_rigid_transformation = generate_homogenous_matrix_from_rigid_transformation(
            angles=inputs["rigid_angles"], translations=inputs["rigid_translations"], origin=origin
        )
        return {"forward_rigid_transformation": AffineTransformation(forward_rigid_transformation)}

    def _generate_inverse_rigid_transformation(
        self, inputs: Mapping[str, Any], _args: Mapping[str, Any], _required_outputs: Set[str]
    ) -> Optional[Mapping[str, Any]]:
        origin = (
            inputs["input"].voxel_size
            * (
                tensor(
                    self._input_image_shape,
                    dtype=inputs["rigid_angles"].dtype,
                    device=inputs["rigid_angles"].device,
                )[None]
                - 1
            )
            / 2
        )
        inverse_rigid_transformation = generate_homogenous_matrix_from_rigid_transformation(
            angles=inputs["rigid_angles"],
            translations=inputs["rigid_translations"],
            origin=origin,
            invert=True,
        )
        return {"inverse_rigid_transformation": AffineTransformation(inverse_rigid_transformation)}

    def _rigidly_deform_label(
        self, inputs: Mapping[str, Any], _args: Mapping[str, Any], _required_outputs: Set[str]
    ) -> Mapping[str, Any]:
        return {
            "rigidly_deformed_label": inputs["label"].deform(inputs["forward_rigid_transformation"])
        }

    def _generate_cross_modality_deformation_svf(
        self, inputs: Mapping[str, Any], _args: Mapping[str, Any], _required_outputs: Set[str]
    ) -> Mapping[str, Any]:
        deformation_svf = self.cross_modality_registration_unet(
            cat(
                [
                    inputs["rigidly_deformed_label"].volume.detach(),
                    inputs["rigidly_deformed_label"].generate_mask(),
                    inputs["input"].volume,
                    inputs["input"].generate_mask(),
                ],
                dim=1,
            )
        )
        return {"cross_modality_deformation_svf": deformation_svf}

    def _integrate_cross_modality_deformation(
        self, inputs: Mapping[str, Any], _args: Mapping[str, Any], _required_outputs: Set[str]
    ) -> Mapping[str, Any]:
        forward_elastic_deformation_ddf = integrate_svf(
            inputs["cross_modality_deformation_svf"], integration_steps=7
        )
        inverse_elastic_deformation_ddf = integrate_svf(
            -inputs["cross_modality_deformation_svf"], integration_steps=7
        )
        forward_elastic_deformation = MaskedDeformation(
            deformation=forward_elastic_deformation_ddf,
            voxel_size=inputs["input"].voxel_size,
            mask=None,
        )
        inverse_elastic_deformation = MaskedDeformation(
            deformation=inverse_elastic_deformation_ddf,
            voxel_size=inputs["input"].voxel_size,
            mask=None,
        )
        forward_deformation = forward_elastic_deformation.compose(
            inputs["forward_rigid_transformation"].detach()
        )
        inverse_deformation = (
            inputs["inverse_rigid_transformation"].detach().compose(inverse_elastic_deformation)
        )
        return {
            "forward_cross_modality_deformation": forward_deformation,
            "inverse_cross_modality_deformation": inverse_deformation,
            "forward_elastic_cross_modality_deformation": forward_elastic_deformation,
            "inverse_elastic_cross_modality_deformation": inverse_elastic_deformation,
        }

    @staticmethod
    def _cross_modality_deform_label(
        inputs: Mapping[str, Any], _args: Mapping[str, Any], _required_outputs: Set[str]
    ) -> Mapping[str, Any]:
        return {
            "cross_modality_deformed_label": inputs["label"].deform(
                inputs["forward_cross_modality_deformation"]
            )
        }

    def _generate_intra_modality_deformation_svf(
        self, inputs: Mapping[str, Any], _args: Mapping[str, Any], _required_outputs: Set[str]
    ) -> Mapping[str, Any]:
        deformation_svf = self.intra_modality_registration_unet(
            cat(
                [
                    inputs["inverse_elastic_cross_modality_deformation"].get_deformation().detach(),
                    inputs["cross_modality_deformed_label"].volume.detach(),
                    inputs["cross_modality_deformed_label"].generate_mask(),
                    inputs["output"].volume,
                    inputs["output"].generate_mask(),
                ],
                dim=1,
            )
        )
        return {"intra_modality_deformation_svf": deformation_svf}

    def _forward_integrate_intra_modality_deformation(
        self, inputs: Mapping[str, Any], _args: Mapping[str, Any], _required_outputs: Set[str]
    ) -> Mapping[str, Any]:
        forward_elastic_deformation_ddf = integrate_svf(
            inputs["intra_modality_deformation_svf"], integration_steps=7
        )
        forward_elastic_deformation = MaskedDeformation(
            deformation=forward_elastic_deformation_ddf,
            voxel_size=inputs["input"].voxel_size,
            mask=None,
        )
        return {"forward_intra_modality_deformation": forward_elastic_deformation}

    def _inverse_integrate_intra_modality_deformation(
        self, inputs: Mapping[str, Any], _args: Mapping[str, Any], _required_outputs: Set[str]
    ) -> Mapping[str, Any]:
        inverse_elastic_deformation_ddf = integrate_svf(
            -inputs["intra_modality_deformation_svf"], integration_steps=7
        )
        inverse_elastic_deformation = MaskedDeformation(
            deformation=inverse_elastic_deformation_ddf,
            voxel_size=inputs["input"].voxel_size,
            mask=None,
        )
        return {"inverse_intra_modality_deformation": inverse_elastic_deformation}

    def _compose_output_commutation_deformation(
        self, inputs: Mapping[str, Any], _args: Mapping[str, Any], _required_outputs: Set[str]
    ) -> Mapping[str, Any]:
        return {
            "output_commutation_deformation": inputs["inverse_intra_modality_deformation"].compose(
                inputs["inverse_random_transformation"]
            )
        }

    def _deform_commutation_output(
        self, inputs: Mapping[str, Any], _args: Mapping[str, Any], _required_outputs: Set[str]
    ) -> Mapping[str, Any]:
        return {
            "commutation_deformed_output": inputs["left_random_deformed_output"].deform(
                inputs["output_commutation_deformation"]
            )
        }

    def _random_commutation_deform_output(
        self, inputs: Mapping[str, Any], _args: Mapping[str, Any], _required_outputs: Set[str]
    ) -> Mapping[str, Any]:
        deformation_composition = inputs["forward_random_transformation"].compose(
            inputs["output_commutation_deformation"]
        )
        return {
            "random_commutation_deformation": deformation_composition,
            "random_commutation_deformed_output": inputs["left_random_deformed_output"].deform(
                deformation_composition
            ),
        }

    @staticmethod
    def _compose_deformations(
        inputs: Mapping[str, Any], _args: Mapping[str, Any], _required_outputs: Set[str]
    ) -> Mapping[str, Any]:
        return {
            "forward_deformation": inputs["forward_intra_modality_deformation"].compose(
                inputs["forward_cross_modality_deformation"].detach()
            ),
            "inverse_deformation": inputs["inverse_cross_modality_deformation"].compose(
                inputs["inverse_intra_modality_deformation"].detach()
            ),
            "forward_elastic_deformation": inputs["forward_intra_modality_deformation"].compose(
                inputs["forward_elastic_cross_modality_deformation"].detach()
            ),
            "inverse_elastic_deformation": inputs["inverse_elastic_cross_modality_deformation"]
            .detach()
            .compose(inputs["inverse_intra_modality_deformation"]),
        }

    @staticmethod
    def _compose_inverse_deformation(
        inputs: Mapping[str, Any], _args: Mapping[str, Any], _required_outputs: Set[str]
    ) -> Mapping[str, Any]:
        return {
            "inverse_deformation": inputs["inverse_rigid_transformation"].compose(
                inputs["inverse_elastic_cross_modality_deformation"].compose(
                    inputs["inverse_intra_modality_deformation"].detach()
                )
            )
        }

    @staticmethod
    def _intra_modality_deform_output(
        inputs: Mapping[str, Any], _args: Mapping[str, Any], _required_outputs: Set[str]
    ) -> Mapping[str, Any]:
        return {
            "deformed_output": inputs["output"].deform(inputs["inverse_intra_modality_deformation"])
        }

    @staticmethod
    def _random_deform_cross_modality_deformed_label(
        inputs: Mapping[str, Any], _args: Mapping[str, Any], _required_outputs: Set[str]
    ) -> Mapping[str, Any]:
        composed_deformation = inputs["forward_random_transformation"].compose(
            inputs["forward_cross_modality_deformation"]
        )
        return {
            "random_cross_modality_deformed_label": inputs["label"].deform(composed_deformation)
        }

    @staticmethod
    def _random_deform_input(
        inputs: Mapping[str, Any], _args: Mapping[str, Any], _required_outputs: Set[str]
    ) -> Mapping[str, Any]:
        return {
            "random_deformed_input": inputs["input"].deform(inputs["forward_random_transformation"])
        }

    @staticmethod
    def _random_deform_output(
        inputs: Mapping[str, Any], _args: Mapping[str, Any], _required_outputs: Set[str]
    ) -> Mapping[str, Any]:
        return {
            "right_random_deformed_output": inputs["output"].deform(
                inputs["forward_random_transformation"]
            )
        }

    def _apply_i2i(
        self, inputs: Mapping[str, Any], _args: Mapping[str, Any], required_outputs: Set[str]
    ) -> Mapping[str, Any]:
        batch_size = inputs["input"].batch_size
        left_random_deformed_output_needed = (
            "left_random_deformed_output" in required_outputs
            or "random_commutation_deformed_output" in required_outputs
            or "commutation_deformed_output" in required_outputs
        )
        if left_random_deformed_output_needed:
            random_deformed_input = inputs["input"].deform(inputs["forward_random_transformation"])
            combined_input = cat([inputs["input"].volume, random_deformed_input.volume], dim=0)
        else:
            combined_input = inputs["input"].volume
        combined_output = self.i2i_unet(combined_input)
        output = MaskedVolume(
            volume=combined_output[:batch_size],
            voxel_size=inputs["input"].voxel_size,
            mask=inputs["input"].mask,
        )
        if left_random_deformed_output_needed:
            left_random_deformed_output = MaskedVolume(
                volume=combined_output[batch_size:],
                voxel_size=random_deformed_input.voxel_size,
                mask=random_deformed_input.mask,
            )
            return {"output": output, "left_random_deformed_output": left_random_deformed_output}
        return {"output": output}

    def _predicted_deformation_left_deform_output(
        self, inputs: Mapping[str, Any], _args: Mapping[str, Any], _required_outputs: Set[str]
    ) -> Mapping[str, Any]:
        deformed_input = inputs["input"].deform(inputs["inverse_deformation"].detach())
        output = self.i2i_unet(deformed_input.volume)
        output = MaskedVolume(
            volume=output, voxel_size=deformed_input.voxel_size, mask=deformed_input.mask
        )
        return {"predicted_deformation_left_deformed_output": output}

    def _predicted_deformation_right_deform_output(
        self, inputs: Mapping[str, Any], _args: Mapping[str, Any], _required_outputs: Set[str]
    ) -> Mapping[str, Any]:
        deformed_output = inputs["output"].deform(inputs["inverse_deformation"].detach())
        return {"predicted_deformation_right_deformed_output": deformed_output}

    def forward(
        self,
        input_volume: Tensor,
        label_volume: Optional[Tensor] = None,
        input_mask: Optional[Tensor] = None,
        label_mask: Optional[Tensor] = None,
        forward_random_transformation: Optional[Tensor] = None,
        desired_outputs: Tuple[str] = ("output",),
    ) -> Tuple[Tensor, ...]:
        """Forward training function

        Args:
            input_image: Tensor with shape (batch_size, n_input_channels, dim_1, ..., dim_{n_dims})
            label_image: Tensor with shape (batch_size, n_output_channels, dim_1, ..., dim_{n_dims})
            input_mask: Tensor with shape (batch_size, 1, dim_1, ..., dim_{n_dims})
            label_mask: Tensor with shape (batch_size, 1, dim_1, ..., dim_{n_dims})
            forward_random_transformation: Optional Tensor with shape
                (batch_size, n_dims + 1, n_dims + 1)
            desired_outputs: Which outputs to compute.
        """
        voxel_size = tensor(self._voxel_size, dtype=input_volume.dtype, device=input_volume.device)
        return self._node_executor.execute(
            inputs={
                "input": MaskedVolume(input_volume, voxel_size, input_mask),
                "label": MaskedVolume(label_volume, voxel_size, label_mask)
                if label_volume is not None
                else None,
                "forward_random_transformation": AffineTransformation(forward_random_transformation)
                if forward_random_transformation is not None
                else None,
            },
            output_names=desired_outputs,
            arguments={},
        )
