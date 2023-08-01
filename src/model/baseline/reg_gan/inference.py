"""Get inference functions"""

from typing import Any, Callable, Mapping, Sequence

from torch import Tensor, device

from model.baseline.interface import load_reggan_model
from model.baseline.reg_gan.nice_gan import ResnetGenerator2


def get_reg_gan_inference_function(
    model_config: Mapping[str, Any],
    data_loader_config: Mapping[str, Any],  # pylint: disable=unused-argument
    devices: Sequence[device],
    epoch: int,
    target_dir: str,
) -> Callable[[Tensor], Tensor]:
    """Init non-aligned I2i inference function from config"""
    squeeze_dim = model_config.get("squeeze_dim")
    model = ResnetGenerator2(
        model_config["input_nc"],
        model_config["output_nc"],
        img_size=model_config["size"],
        n_blocks=model_config["n_blocks"],
        final_activation=model_config["final_activation"]
    )
    model.to(devices[0])
    load_reggan_model(target_dir, epoch, "training", generator_model=model, torch_device=devices[0])

    def _infer(input_image: Tensor) -> Tensor:
        if squeeze_dim is not None:
            input_image = input_image.squeeze(squeeze_dim + 2)
        output: Tensor = model(input_image)
        if squeeze_dim is not None:
            output = output.unsqueeze(squeeze_dim + 2)
        return output

    return _infer
