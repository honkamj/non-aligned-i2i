"""ResNet style discriminator with spectral normalization"""

from typing import Callable, Optional, Sequence, Tuple

from torch import Tensor, cat
from torch.nn import Linear, Module, ReLU, Sequential
from torch.nn.utils.parametrizations import spectral_norm

from .ndimensional_operators import avg_pool_nd, conv_nd, max_pool_nd


class SpectralNormResBlockDownWithMaxPoolingNd(Module):
    """Spectrally normalized downsampling ResNet block"""
    def __init__(
            self,
            n_dims: int,
            n_input_channels: int,
            n_output_channels: int,
            activation: Callable[[Tensor], Tensor]
        ) -> None:
        super().__init__()
        self.activation = activation
        self.max_pool = max_pool_nd(n_dims)(
            kernel_size=2
        )
        self.conv1 = spectral_norm(
            conv_nd(n_dims)(
                in_channels=n_input_channels,
                out_channels=n_output_channels,
                kernel_size=3,
                padding=1))
        self.conv2 = spectral_norm(
            conv_nd(n_dims)(
                in_channels=n_output_channels,
                out_channels=n_output_channels,
                kernel_size=3,
                padding=1))
        self.average_pool = avg_pool_nd(n_dims)(
            kernel_size=2
        )
        self.projection = spectral_norm(
            conv_nd(n_dims)(
                in_channels=n_input_channels,
                out_channels=n_output_channels,
                kernel_size=1,
                padding=0))

    def forward(self, input_tensor: Tensor) -> Tensor:
        """Forward function of the block"""
        main_path = self.max_pool(input_tensor)
        main_path = self.activation(self.conv1(main_path))
        main_path = self.activation(self.conv2(main_path))
        skip = self.average_pool(input_tensor)
        skip = self.projection(skip)
        output = skip + main_path
        return output


class SpectralNormResBlockDownNd(Module):
    """Spectrally normalized downsampling ResNet block"""
    def __init__(
            self,
            n_dims: int,
            n_input_channels: int,
            n_output_channels: int,
            activation: Callable[[Tensor], Tensor]
        ) -> None:
        super().__init__()
        self.activation = activation
        self.conv1 = spectral_norm(
            conv_nd(n_dims)(
                in_channels=n_input_channels,
                out_channels=n_output_channels,
                kernel_size=4,
                stride=2,
                padding=1))
        self.conv2 = spectral_norm(
            conv_nd(n_dims)(
                in_channels=n_output_channels,
                out_channels=n_output_channels,
                kernel_size=3,
                padding=1))
        self.average_pool = avg_pool_nd(n_dims)(
            kernel_size=2
        )
        self.projection = spectral_norm(
            conv_nd(n_dims)(
                in_channels=n_input_channels,
                out_channels=n_output_channels,
                kernel_size=1,
                padding=0))

    def forward(self, input_tensor: Tensor) -> Tensor:
        """Forward function of the block"""
        main_path = self.activation(self.conv1(input_tensor))
        main_path = self.activation(self.conv2(main_path))
        skip = self.average_pool(input_tensor)
        skip = self.projection(skip)
        output = skip + main_path
        return output


class SpectralNormResBlockNd(Module):
    """Spectrally normalized ResNet block"""
    def __init__(
            self,
            n_dims: int,
            n_input_channels: int,
            n_output_channels: int,
            activation: Callable[[Tensor], Tensor]
        ) -> None:
        super().__init__()
        self.activation = activation
        self.conv1 = spectral_norm(
            conv_nd(n_dims)(
                in_channels=n_input_channels,
                out_channels=n_output_channels,
                kernel_size=3,
                padding=1))
        self.conv2 = spectral_norm(
            conv_nd(n_dims)(
                in_channels=n_output_channels,
                out_channels=n_output_channels,
                kernel_size=3,
                padding=1))

    def forward(self, input_tensor: Tensor) -> Tensor:
        """Forward function of the block"""
        main_path = self.activation(self.conv1(input_tensor))
        main_path = self.activation(self.conv2(main_path))
        output = input_tensor + main_path
        return output


class SpectralNormResNetDiscriminator(Module):
    """Resnet discriminator"""
    def __init__(
            self,
            n_dims: int,
            n_input_channels: int,
            n_output_channels: int,
            input_image_shape: Tuple[int, ...],
            n_features_per_block: Sequence[int],
            activation: Callable[[Tensor], Tensor]
        ) -> None:
        """Resnet discriminator"""
        super().__init__()
        if n_output_channels + n_input_channels == n_features_per_block[0]:
            self.initial_conv = lambda x: x
        else:
            self.initial_conv = conv_nd(n_dims)(
                n_input_channels + n_output_channels,
                n_features_per_block[0],
                kernel_size=1,
                padding=1)
        self.initial_block_joint = SpectralNormResBlockNd(
            n_dims=n_dims,
            n_input_channels=n_features_per_block[0],
            n_output_channels=n_features_per_block[0],
            activation=activation
        )
        n_features_per_block_joint_with_initial = (
            [n_features_per_block[0]] +
            list(n_features_per_block)
        )
        self.joint_blocks = Sequential(
            *(
                SpectralNormResBlockDownNd(
                    n_dims=n_dims,
                    n_input_channels=n_features_per_block_joint_with_initial[i],
                    n_output_channels=n_features_per_block_joint_with_initial[i + 1],
                    activation=activation)
                for i in range(len(n_features_per_block))
            )
        )
        self.final_block = SpectralNormResBlockNd(
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
        self.linear = spectral_norm(Linear(n_features_per_block[-1], 1))

    def forward(
            self,
            input_volume: Optional[Tensor],
            output_volume: Tensor
        ) -> Tensor:
        """Forward function of ResNet discriminator"""
        batch_size = output_volume.size(0)
        if input_volume is None:
            joint_input = output_volume
        else:
            joint_input = cat(
                [input_volume, output_volume],
                dim=1
            )
        output = self.initial_conv(joint_input)
        output = self.initial_block_joint(output)
        output = self.joint_blocks(output)
        output = self.final_block(output)
        output = self.global_avg_pooling(output)
        output = self.linear(output.view(batch_size, -1))[..., -1]
        return output
