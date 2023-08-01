"""Get inference functions"""

from typing import Any, Callable, Mapping, Sequence

from torch import Tensor, device

from model.baseline.cycle_gan_and_pix2pix import create_model
from util.training import load_model


class _Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)


def get_cycle_gan_inference_function(
        model_config: Mapping[str, Any],
        data_loader_config: Mapping[str, Any],  # pylint: disable=unused-argument
        devices: Sequence[device],
        epoch: int,
        target_dir: str,
    ) -> Callable[[Tensor], Tensor]:
    """Init non-aligned I2i inference function from config"""
    cycle_gan_and_pix2pix_config = dict(model_config)
    cycle_gan_and_pix2pix_config["isTrain"] = False
    if devices[0].type != "cpu":
        cycle_gan_and_pix2pix_config["gpu_ids"] = [torch_device.index for torch_device in devices]
    else:
        cycle_gan_and_pix2pix_config["gpu_ids"] = []
    opt = _Struct(**cycle_gan_and_pix2pix_config)
    squeeze_dim = cycle_gan_and_pix2pix_config.get("squeeze_dim")

    model = create_model(opt)
    model.setup(opt)
    load_model(target_dir, epoch, 'training', model, torch_device=devices[0])
    def _infer(input_image: Tensor) -> Tensor:
        if squeeze_dim is not None:
            input_image = input_image.squeeze(squeeze_dim + 2)
        output: Tensor = model.translate_A_to_B(input_image)
        if squeeze_dim is not None:
            output = output.unsqueeze(squeeze_dim + 2)
        return output
    return _infer


def get_pix2pix_inference_function(
        model_config: Mapping[str, Any],
        data_loader_config: Mapping[str, Any],  # pylint: disable=unused-argument
        devices: Sequence[device],
        epoch: int,
        target_dir: str,
    ) -> Callable[[Tensor], Tensor]:
    """Init non-aligned I2i inference function from config"""
    cycle_gan_and_pix2pix_config = dict(model_config)
    cycle_gan_and_pix2pix_config["isTrain"] = False
    if devices[0].type != "cpu":
        cycle_gan_and_pix2pix_config["gpu_ids"] = [torch_device.index for torch_device in devices]
    else:
        cycle_gan_and_pix2pix_config["gpu_ids"] = []
    opt = _Struct(**cycle_gan_and_pix2pix_config)
    squeeze_dim = cycle_gan_and_pix2pix_config.get("squeeze_dim")

    model = create_model(opt)
    model.setup(opt)
    load_model(target_dir, epoch, 'training', model, torch_device=devices[0])
    def _infer(input_image: Tensor) -> Tensor:
        if squeeze_dim is not None:
            input_image = input_image.squeeze(squeeze_dim + 2)
        output: Tensor = model.translate_A_to_B(input_image)
        if squeeze_dim is not None:
            output = output.unsqueeze(squeeze_dim + 2)
        return output
    return _infer


def get_nemar_inference_function(
        model_config: Mapping[str, Any],
        data_loader_config: Mapping[str, Any],  # pylint: disable=unused-argument
        devices: Sequence[device],
        epoch: int,
        target_dir: str,
    ) -> Callable[[Tensor], Tensor]:
    """Init non-aligned I2i inference function from config"""
    cycle_gan_and_pix2pix_config = dict(model_config)
    cycle_gan_and_pix2pix_config["isTrain"] = False
    if devices[0].type != "cpu":
        cycle_gan_and_pix2pix_config["gpu_ids"] = [torch_device.index for torch_device in devices]
    else:
        cycle_gan_and_pix2pix_config["gpu_ids"] = []
    opt = _Struct(**cycle_gan_and_pix2pix_config)
    squeeze_dim = cycle_gan_and_pix2pix_config.get("squeeze_dim")

    model = create_model(opt)
    model.setup(opt)
    load_model(target_dir, epoch, 'training', model, torch_device=devices[0])
    def _infer(input_image: Tensor) -> Tensor:
        if squeeze_dim is not None:
            input_image = input_image.squeeze(squeeze_dim + 2)
        output: Tensor = model.translate_A_to_B(input_image)
        if squeeze_dim is not None:
            output = output.unsqueeze(squeeze_dim + 2)
        return output
    return _infer
