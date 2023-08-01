"""ResNet style encoder network"""

from typing import Callable, Sequence, Tuple

from torch import Tensor, cat
from torch.nn import Linear, Module, ModuleList, ReLU, Sequential

from .ndimensional_operators import avg_pool_nd, conv_nd


class ResBlockDownNd(Module):
    """Downsampling ResNet block"""
    def __init__(
            self,
            n_dims: int,
            n_input_channels: int,
            n_output_channels: int,
            activation: Callable[[Tensor], Tensor]
        ) -> None:
        super().__init__()
        self.activation = activation
        self.conv1 = conv_nd(n_dims)(
            in_channels=n_input_channels,
            out_channels=n_output_channels,
            kernel_size=4,
            stride=2,
            padding=1)
        self.conv2 = conv_nd(n_dims)(
            in_channels=n_output_channels,
            out_channels=n_output_channels,
            kernel_size=3,
            padding=1)
        self.average_pool = avg_pool_nd(n_dims)(
            kernel_size=2
        )
        self.projection = conv_nd(n_dims)(
            in_channels=n_input_channels,
            out_channels=n_output_channels,
            kernel_size=1,
            padding=0)

    def forward(self, volume: Tensor) -> Tensor:
        """Forward function of the block"""
        main_path = self.activation(self.conv1(volume))
        main_path = self.activation(self.conv2(main_path))
        skip = self.average_pool(volume)
        skip = self.projection(skip)
        output = skip + main_path
        return output


class ResBlockNd(Module):
    """ResNet block"""
    def __init__(
            self,
            n_dims: int,
            n_input_channels: int,
            n_output_channels: int,
            activation: Callable[[Tensor], Tensor]
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

    def forward(self, volume: Tensor) -> Tensor:
        """Forward function of the block"""
        main_path = self.activation(self.conv1(volume))
        main_path = self.activation(self.conv2(main_path))
        output = volume + main_path
        return output


class ResNetEncoderNd(Module):
    """Resnet encoder"""
    def __init__(
            self,
            n_dims: int,
            n_input_channels: int,
            input_image_shape: Tuple[int, ...],
            n_features_per_block: Sequence[int],
            activation: Callable[[Tensor], Tensor],
            n_outputs: int
        ) -> None:
        """Resnet encoder"""
        super().__init__()
        self.initial_conv = conv_nd(n_dims)(
            n_input_channels + n_dims,
            n_features_per_block[0],
            kernel_size=3,
            padding=1)
        self.initial_block = ResBlockNd(
            n_dims=n_dims,
            n_input_channels=n_features_per_block[0],
            n_output_channels=n_features_per_block[0],
            activation=activation
        )
        n_features_per_block_with_initial = (
            [n_features_per_block[0]] +
            list(n_features_per_block)
        )
        self.res_blocks = ModuleList(
            (
                ResBlockDownNd(
                    n_dims=n_dims,
                    n_input_channels=n_features_per_block_with_initial[i],
                    n_output_channels=n_features_per_block_with_initial[i + 1],
                    activation=activation)
                for i in range(len(n_features_per_block))
            )
        )
        self.final_block = ResBlockNd(
            n_dims=n_dims,
            n_input_channels=n_features_per_block[-1],
            n_output_channels=n_features_per_block[-1],
            activation=activation
        )
        self.global_avg_pooling = Sequential(
            ReLU(),
            avg_pool_nd(n_dims)(
                kernel_size=tuple(
                    input_image_dim_size // 2**len(n_features_per_block)
                    for input_image_dim_size in input_image_shape
                )
            )
        )
        self.linear = Linear(n_features_per_block[-1], n_outputs)

    def forward(
            self,
            volume: Tensor,
            coordinates: Tensor
        ) -> Tensor:
        """Forward function of ResNet discriminator

        volume: Tensor with shape (batch_size, n_channels, dim_1, ..., dim_{n_dims})
        coordinates: Coordinate volume Tensor with shape
            (batch_size, n_dims, dim_1, ..., dim_{n_dims})
        """
        batch_size = volume.size(0)
        combined_input = cat(
            [volume, coordinates],
            dim=1
        )
        output = self.initial_conv(combined_input)
        output = self.initial_block(output)
        for res_block in self.res_blocks:
            output = res_block(output)
        output = self.final_block(output)
        output = self.global_avg_pooling(output)
        output = self.linear(output.view(batch_size, -1))
        return output
