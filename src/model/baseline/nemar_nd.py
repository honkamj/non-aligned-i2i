"""I2I with non-aligned data

Uses U-Nets as workhorses
"""

from math import pi
from typing import Any, Callable, Mapping, Optional, Sequence, Set, Tuple

from scipy.special import binom  # type: ignore
from torch import Tensor, cat, tensor
from torch.nn import LeakyReLU, Module, ReLU

from algorithm.torch.deformation.masked import MaskedDeformation, MaskedVolume
from algorithm.torch.deformation.primitive import (
    integrate_svf,
)
from util.executor import ExecutionGraph, NodeDefinition, RequiredNodeExecutor

from ..unet import UNetNd


class NemarNd(Module):
    """Module corresponding to the method https://github.com/moabarar/nemar/
    but with network architectures similar to our method"""

    def __init__(
        self,
        n_input_channels: int,
        n_output_channels: int,
        n_features_per_block_i2i: Sequence[int],
        n_features_per_block_cross_modality_reg: Sequence[int],
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
        self._input_image_shape = input_image_shape
        self._node_executor = self._define_node_executor()

    @staticmethod
    def _get_num_rotation_axes(n_dims: int) -> int:
        return int(binom(n_dims, 2))

    def _define_node_executor(self) -> RequiredNodeExecutor:
        graph_definition = (
            NodeDefinition(
                input_names={"label", "input"},
                output_names={"cross_modality_deformation_svf"},
                executable=self._generate_cross_modality_deformation_svf,
            ),
            NodeDefinition(
                input_names={"cross_modality_deformation_svf", "input"},
                output_names={"forward_cross_modality_deformation"},
                executable=self._integrate_cross_modality_deformation,
            ),
            NodeDefinition(
                input_names={"output", "forward_cross_modality_deformation"},
                output_names={"cross_modality_deformed_output"},
                executable=self._cross_modality_deform_output,
            ),
            NodeDefinition(
                input_names={"input", "forward_cross_modality_deformation"},
                output_names={"cross_modality_deformed_input"},
                executable=self._cross_modality_deform_input,
            ),
            NodeDefinition(
                input_names={"input"},
                output_names={"output"},
                executable=self._apply_i2i,
            ),
            NodeDefinition(
                input_names={"cross_modality_deformed_input"},
                output_names={"output_cross_modality_deformed"},
                executable=self._apply_i2i_to_deformed,
            ),
        )
        return RequiredNodeExecutor(ExecutionGraph(graph_definition=graph_definition))

    def _generate_cross_modality_deformation_svf(
        self, inputs: Mapping[str, Any], _args: Mapping[str, Any], _required_outputs: Set[str]
    ) -> Mapping[str, Any]:
        deformation_svf = self.cross_modality_registration_unet(
            cat(
                [
                    inputs["label"].volume.detach(),
                    inputs["label"].generate_mask(),
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
        ddf = integrate_svf(inputs["cross_modality_deformation_svf"], integration_steps=7)
        deformation = MaskedDeformation(
            deformation=ddf,
            voxel_size=inputs["input"].voxel_size,
            mask=None,
        )
        return {"forward_cross_modality_deformation": deformation}

    @staticmethod
    def _cross_modality_deform_output(
        inputs: Mapping[str, Any], _args: Mapping[str, Any], _required_outputs: Set[str]
    ) -> Mapping[str, Any]:
        return {
            "cross_modality_deformed_output": inputs["output"].deform(
                inputs["forward_cross_modality_deformation"]
            )
        }

    @staticmethod
    def _cross_modality_deform_input(
        inputs: Mapping[str, Any], _args: Mapping[str, Any], _required_outputs: Set[str]
    ) -> Mapping[str, Any]:
        return {
            "cross_modality_deformed_input": inputs["input"].deform(
                inputs["forward_cross_modality_deformation"]
            )
        }

    def _apply_i2i(
        self, inputs: Mapping[str, Any], _args: Mapping[str, Any], _required_outputs: Set[str]
    ) -> Mapping[str, Any]:
        output = MaskedVolume(
            volume=self.i2i_unet(inputs["input"].volume),
            voxel_size=inputs["input"].voxel_size,
            mask=inputs["input"].mask,
        )
        return {"output": output}

    def _apply_i2i_to_deformed(
        self, inputs: Mapping[str, Any], _args: Mapping[str, Any], _required_outputs: Set[str]
    ) -> Mapping[str, Any]:
        output_cross_modality_deformed = MaskedVolume(
            volume=self.i2i_unet(inputs["cross_modality_deformed_input"].volume),
            voxel_size=inputs["cross_modality_deformed_input"].voxel_size,
            mask=inputs["cross_modality_deformed_input"].mask,
        )
        return {"output_cross_modality_deformed": output_cross_modality_deformed}

    def forward(
        self,
        input_volume: Tensor,
        label_volume: Optional[Tensor] = None,
        input_mask: Optional[Tensor] = None,
        label_mask: Optional[Tensor] = None,
        desired_outputs: Tuple[str] = ("output",),
    ) -> Tuple[Tensor, ...]:
        """Forward training function

        Args:
            input_image: Tensor with shape (batch_size, n_input_channels, dim_1, ..., dim_{n_dims})
            label_image: Tensor with shape (batch_size, n_output_channels, dim_1, ..., dim_{n_dims})
            input_mask: Tensor with shape (batch_size, 1, dim_1, ..., dim_{n_dims})
            label_mask: Tensor with shape (batch_size, 1, dim_1, ..., dim_{n_dims})
            desired_outputs: Which outputs to compute.
        """
        voxel_size = tensor(self._voxel_size, dtype=input_volume.dtype, device=input_volume.device)
        return self._node_executor.execute(
            inputs={
                "input": MaskedVolume(input_volume, voxel_size, input_mask),
                "label": MaskedVolume(label_volume, voxel_size, label_mask)
                if label_volume is not None
                else None,
            },
            output_names=desired_outputs,
            arguments={},
        )
