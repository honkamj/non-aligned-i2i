"""U-Net implementation for pytorch with basic parametrization"""

from typing import Callable, Optional, Sequence

from torch import Tensor, cat
from torch.nn import GroupNorm, Module, ModuleList

from .ndimensional_operators import conv_nd, conv_transpose_nd


class DoubleConvolutionNd(Module):
    """Double convolution"""
    def __init__(
            self,
            n_dims: int,
            n_input_channels: int,
            n_output_channels: int,
            activation: Callable[[Tensor], Tensor],
            n_normalization_groups: Optional[int] = None
        ) -> None:
        super().__init__()
        self.activation = activation
        self.conv1 = conv_nd(n_dims)(
            in_channels=n_input_channels,
            out_channels=n_output_channels,
            kernel_size=3,
            padding=1)
        self.conv2 = conv_nd(n_dims)(
            in_channels=n_output_channels,
            out_channels=n_output_channels,
            kernel_size=3,
            padding=1)
        self._n_normalization_groups = n_normalization_groups
        if n_normalization_groups is not None:
            self.group_norm_1 = GroupNorm(
                num_groups=n_normalization_groups,
                num_channels=n_output_channels)
            self.group_norm_2 = GroupNorm(
                num_groups=n_normalization_groups,
                num_channels=n_output_channels)

    def forward(self, input_tensor: Tensor) -> Tensor:
        """Double convolve 2D input

        Args:
            input_tensor: Tensor with shape (batch_size, n_input_channels, width, height)

        Returns:
            Tensor with shape (batch_size, n_output_channels, width, height)
        """
        output = self.conv1(input_tensor)
        if self._n_normalization_groups is not None:
            output = self.group_norm_1(output)
        output = self.activation(output)
        output = self.conv2(output)
        if self._n_normalization_groups is not None:
            output = self.group_norm_2(output)
        output = self.activation(output)
        return output


class DownsamplingBlockNd(Module):
    """2D downsampling block with double convolution"""
    def __init__(
            self,
            n_dims: int,
            n_input_channels: int,
            n_output_channels: int,
            activation: Callable[[Tensor], Tensor],
            n_normalization_groups: Optional[int] = None
        ) -> None:
        """Downsampling block with double convolution"""
        super().__init__()
        self.downsampling = conv_nd(n_dims)(
            in_channels=n_input_channels,
            out_channels=n_output_channels,
            kernel_size=2,
            stride=2)
        self.double_conv = DoubleConvolutionNd(
            n_dims=n_dims,
            n_input_channels=n_output_channels,
            n_output_channels=n_output_channels,
            activation=activation,
            n_normalization_groups=n_normalization_groups)

    def forward(self, input_tensor: Tensor) -> Tensor:
        """Double convolve 2D input

        Args:
            input_tensor: Tensor with shape (batch_size, n_input_channels, 2 * width, 2 * height)

        Returns:
            Tensor with shape (batch_size, n_output_channels, width, height)
        """
        output = self.downsampling(input_tensor)
        output = self.double_conv(output)
        return output


class UpsamplingBlockNd(Module):
    """Upsampling block with double convolution"""
    def __init__(
            self,
            n_dims: int,
            n_input_channels: int,
            n_skip_channels: int,
            n_output_channels: int,
            activation: Callable[[Tensor], Tensor],
            n_normalization_groups: Optional[int] = None
        ) -> None:
        super().__init__()
        self.upsampling = conv_transpose_nd(n_dims)(
            in_channels=n_input_channels,
            out_channels=n_input_channels,
            kernel_size=2,
            stride=2)
        self.double_conv = DoubleConvolutionNd(
            n_dims=n_dims,
            n_input_channels=n_input_channels + n_skip_channels,
            n_output_channels=n_output_channels,
            activation=activation,
            n_normalization_groups=n_normalization_groups)

    def forward(self, input_tensor: Tensor, skip_tensor: Tensor) -> Tensor:
        """Upsample 2D input

        Args:
            input_tensor: Tensor with shape (batch_size, n_input_channels, width, height)
            skip_tensor: Tensor with shape (batch_size, n_skip_channels, 2 * width, 2 * height)

        Returns:
            Tensor with shape (batch_size, n_output_channels, 2 * width, 2 * height)
        """
        upsampled = self.upsampling(input_tensor)
        concatenated = cat([upsampled, skip_tensor], dim=1)
        output = self.double_conv(concatenated)
        return output


class UNetNd(Module):
    """Unet"""
    def __init__(
            self,
            n_dims,
            n_input_channels: int,
            n_output_channels: Optional[int],
            activation: Callable[[Tensor], Tensor],
            final_activation: Optional[Callable[[Tensor], Tensor]],
            n_features_per_block: Sequence[int],
            n_normalization_groups: Optional[int] = None
        ) -> None:
        super().__init__()
        self.initial_conv = DoubleConvolutionNd(
            n_dims=n_dims,
            n_input_channels=n_input_channels,
            n_output_channels=n_features_per_block[0],
            activation=activation)

        self.downsampling_blocks = ModuleList()
        for i in range(len(n_features_per_block) - 1):
            self.downsampling_blocks.append(
                DownsamplingBlockNd(
                    n_dims=n_dims,
                    n_input_channels=n_features_per_block[i],
                    n_output_channels=n_features_per_block[i + 1],
                    activation=activation,
                    n_normalization_groups=n_normalization_groups))

        self.upsampling_blocks = ModuleList()
        for i in reversed(range(1, len(n_features_per_block))):
            n_upsampling_normalization_groups = (
                n_normalization_groups
                if i > 1
                else None
            )
            self.upsampling_blocks.append(
                UpsamplingBlockNd(
                    n_dims=n_dims,
                    n_input_channels=n_features_per_block[i],
                    n_skip_channels=n_features_per_block[i - 1],
                    n_output_channels=n_features_per_block[i - 1],
                    activation=activation,
                    n_normalization_groups=n_upsampling_normalization_groups))
        if n_output_channels is None:
            self.final_conv = None
        else:
            self.final_conv = conv_nd(n_dims)(
                in_channels=n_features_per_block[0],
                out_channels=n_output_channels,
                kernel_size=1)
        if final_activation is None:
            self.final_activation = None
        else:
            self.final_activation = final_activation

    def forward(self, input_tensor: Tensor) -> Tensor:
        """Upsample 2D input

        Args:
            input_tensor: Tensor with shape (batch_size, n_input_channels, width, height)

        Returns:
            Tensor with shape (batch_size, n_output_channels, width, height)
        """
        downsampling_outputs = [self.initial_conv(input_tensor)]
        for i, downsampling_block in enumerate(self.downsampling_blocks):
            downsampling_outputs.append(downsampling_block(downsampling_outputs[-1]))
        output = downsampling_outputs[-1]
        for i, upsampling_block in enumerate(self.upsampling_blocks):
            skip_index = len(self.downsampling_blocks) - i - 1
            output = upsampling_block(output, downsampling_outputs[skip_index])
        if self.final_conv is not None:
            output = self.final_conv(output)
        if self.final_activation is not None:
            output = self.final_activation(output)
        return output
