"""Interface to models"""

from typing import Any, Mapping, Sequence, Tuple, Union
from torch import device

from torch.nn import DataParallel, LeakyReLU
from model.baseline.nemar_nd import NemarNd

from model.non_aligned_i2i import I2IRegNetNd
from model.spectral_norm_resnet_discriminator import \
    SpectralNormResNetDiscriminator
from model.unet import UNetNd
from util.activation import str_to_activation


def init_non_aligned_i2i_model(
        model_config: Mapping[str, Any],
        data_loader_config: Mapping[str, Any],
        devices: Sequence[device]
    ) -> Tuple[
        Union[I2IRegNetNd, DataParallel],
        I2IRegNetNd
    ]:
    """Init non-aligned I2I model from config"""
    non_aligned_i2i_module = I2IRegNetNd(
        n_input_channels=model_config['n_input_channels'],
        n_output_channels=model_config['n_output_channels'],
        n_features_per_block_i2i=model_config['n_features_per_block_i2i'],
        n_features_per_block_intra_modality_reg=\
            model_config['n_features_per_block_intra_modality_reg'],
        n_features_per_block_cross_modality_reg=\
            model_config['n_features_per_block_cross_modality_reg'],
        n_features_per_block_rigid_reg=model_config['n_features_per_block_rigid_reg'],
        input_image_shape=data_loader_config['patch_size'],
        voxel_size=data_loader_config['voxel_size'],
        final_activation=str_to_activation(model_config['final_activation']),
        n_normalization_groups=model_config['n_normalization_groups']
    )
    if len(devices) > 1:
        non_aligned_i2i: Union[I2IRegNetNd, DataParallel] = DataParallel(
            non_aligned_i2i_module,
            devices)
    else:
        non_aligned_i2i = non_aligned_i2i_module
    return non_aligned_i2i, non_aligned_i2i_module


def init_i2i_unet_model(
        data_loader_config: Mapping[str, Any],
        model_config: Mapping[str, Any],
        devices: Sequence[device]
    ) -> Tuple[
        Union[UNetNd, DataParallel],
        UNetNd
    ]:
    """Init UNet model from config"""
    unet_module = UNetNd(
        n_dims=len(data_loader_config['patch_size']),
        n_input_channels=model_config['n_input_channels'],
        n_output_channels=model_config['n_output_channels'],
        n_features_per_block=model_config['n_features_per_block_i2i'],
        activation=LeakyReLU(),
        final_activation=str_to_activation(model_config['final_activation']),
        n_normalization_groups=model_config['n_normalization_groups']
    )
    if len(devices) > 1:
        unet: Union[UNetNd, DataParallel] = DataParallel(
            unet_module,
            devices)
    else:
        unet = unet_module
    return unet, unet_module


def init_spectral_norm_discriminator(
        model_config: Mapping[str, Any],
        data_loader_config: Mapping[str, Any],
        devices: Sequence[device]
    ) -> Tuple[
        Union[SpectralNormResNetDiscriminator, DataParallel],
        SpectralNormResNetDiscriminator
    ]:
    """Initialize spectrally normalized discriminator"""
    discriminator_module = SpectralNormResNetDiscriminator(
        n_dims=len(data_loader_config['patch_size']),
        n_input_channels=(
            model_config['n_input_channels']
            if model_config.get('conditional_discriminator', True)
            else 0
        ),
        n_output_channels=model_config['n_output_channels'],
        input_image_shape=data_loader_config['patch_size'],
        n_features_per_block=model_config['n_features_per_block_discriminator'],
        activation=LeakyReLU()
    )
    if len(devices) > 1:
        discriminator: Union[SpectralNormResNetDiscriminator, DataParallel] = (
            DataParallel(discriminator_module, devices)
        )
    else:
        discriminator = discriminator_module
    return discriminator, discriminator_module


def init_nemar_nd_model(
        model_config: Mapping[str, Any],
        data_loader_config: Mapping[str, Any],
        devices: Sequence[device]
    ) -> Tuple[
        Union[NemarNd, DataParallel],
        NemarNd
    ]:
    """Init non-aligned I2I model from config"""
    nemar_module = NemarNd(
        n_input_channels=model_config['n_input_channels'],
        n_output_channels=model_config['n_output_channels'],
        n_features_per_block_i2i=model_config['n_features_per_block_i2i'],
        n_features_per_block_cross_modality_reg=\
            model_config['n_features_per_block_cross_modality_reg'],
        input_image_shape=data_loader_config['patch_size'],
        voxel_size=data_loader_config['voxel_size'],
        final_activation=str_to_activation(model_config['final_activation']),
        n_normalization_groups=model_config['n_normalization_groups']
    )
    if len(devices) > 1:
        nemar: Union[NemarNd, DataParallel] = DataParallel(
            nemar_module,
            devices)
    else:
        nemar = nemar_module
    return nemar, nemar_module
