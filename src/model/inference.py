"""Inference related functions"""

from typing import Any, Callable, Mapping, Sequence

from torch import Tensor, device
from model.baseline.interface import load_reggan_model

from model.interface import init_nemar_nd_model, init_non_aligned_i2i_model, init_i2i_unet_model
from util.import_util import import_object
from util.training import load_model


I2IInferenceFunctionObtainer = Callable[
    [
        Mapping[str, Any], # model config
        Mapping[str, Any], # data loader config
        Sequence[device], # num gpu
        int, # epoch
        str, # target directory of the trainings
    ],
    Callable[[Tensor], Tensor]
]


def get_generic_i2i_inference_function(
        model_config: Mapping[str, Any],
        data_loader_config: Mapping[str, Any],
        devices: Sequence[device],
        epoch: int,
        target_dir: str
    ) -> Callable[[Tensor], Tensor]:
    """Init generic I2I inference function from config"""
    obtainer_func: I2IInferenceFunctionObtainer = import_object(
        model_config['inference_function_obtainer'])
    return obtainer_func(
        model_config,
        data_loader_config,
        devices,
        epoch,
        target_dir)


def get_i2i_unet_inference_function(
        model_config: Mapping[str, Any],
        data_loader_config: Mapping[str, Any],
        devices: Sequence[device],
        epoch: int,
        target_dir: str
    ) -> Callable[[Tensor], Tensor]:
    """Init non-aligned I2I inference function from config"""
    unet, unet_module = init_i2i_unet_model(
        model_config=model_config,
        data_loader_config=data_loader_config,
        devices=devices)
    load_model(target_dir, epoch, 'training', unet_module, torch_device=device('cpu'))
    unet.to(devices[0])
    unet.eval()
    return unet


def get_non_aligned_i2i_inference_function(
        model_config: Mapping[str, Any],
        data_loader_config: Mapping[str, Any],
        devices: Sequence[device],
        epoch: int,
        target_dir: str
    ) -> Callable[[Tensor], Tensor]:
    """Init non-aligned I2i inference function from config"""
    non_aligned_i2i, non_aligned_i2i_module = init_non_aligned_i2i_model(
        model_config=model_config,
        data_loader_config=data_loader_config,
        devices=devices)
    load_model(target_dir, epoch, 'training', non_aligned_i2i_module, torch_device=device('cpu'))
    non_aligned_i2i_module.i2i_unet.to(devices[0])
    non_aligned_i2i_module.i2i_unet.eval()
    def _infer(input_image: Tensor) -> Tensor:
        (
            predicted_label,
        ) = non_aligned_i2i(
            input_volume=input_image,
            label_volume=None,
            input_mask=None,
            label_mask=None,
            forward_random_transformation=None,
            desired_outputs=(
                'output',
            )
        )
        return predicted_label.volume
    return _infer


def get_nemar_nd_inference_function(
        model_config: Mapping[str, Any],
        data_loader_config: Mapping[str, Any],
        devices: Sequence[device],
        epoch: int,
        target_dir: str
    ) -> Callable[[Tensor], Tensor]:
    """Init non-aligned I2i inference function from config"""
    nemar, nemar_module = init_nemar_nd_model(
        model_config=model_config,
        data_loader_config=data_loader_config,
        devices=devices)
    load_reggan_model(
        target_dir=target_dir,
        epoch=epoch,
        prefix="training",
        torch_device=devices[0],
        generator_model=nemar_module.i2i_unet,
        discriminator_model=None,
        reg_model=None,
        generator_optimizer=None,
        discriminator_optimizer=None,
        reg_optimizer=None,
    )
    nemar_module.i2i_unet.to(devices[0])
    nemar_module.i2i_unet.eval()
    def _infer(input_image: Tensor) -> Tensor:
        (
            predicted_label,
        ) = nemar(
            input_volume=input_image,
            label_volume=None,
            input_mask=None,
            label_mask=None,
            desired_outputs=(
                'output',
            )
        )
        return predicted_label.volume
    return _infer


# The function follows a predefined interface
# pylint: disable=unused-argument
def get_input_inference_function(
        model_config: Mapping[str, Any],
        data_loader_config: Mapping[str, Any],
        devices: Sequence[device],
        epoch: int,
        target_dir: str
    ) -> Callable[[Tensor], Tensor]:
    """Inference function which directly outputs the input"""
    def _infer(input_image: Tensor) -> Tensor:
        return input_image
    return _infer
