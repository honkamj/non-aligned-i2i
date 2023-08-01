"""Simple training script for training straightforward i2i network"""

from json import dumps as json_dumps
from json import loads as json_loads
from os.path import join
from typing import Any, Mapping, Optional, Sequence

from numpy.random import RandomState
from torch import Tensor, cat, device, float32, no_grad, ones_like, tensor
from tqdm import tqdm  # type: ignore

from algorithm.torch.deformation.masked import AffineTransformation, MaskedVolume
from data.interface import init_generic_training_data_loader
from loss.masked_loss import masked_mae_loss
from model.baseline.cycle_gan_and_pix2pix import create_model
from model.baseline.cycle_gan_and_pix2pix.nemar import NEMARModel
from util.training import (
    LossAverager,
    TrainingFunction,
    load_model,
    obtain_arguments_and_train,
    save_model,
)


class _Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)


def _augment(
    volumes: Sequence[Tensor],
    masks: Sequence[Tensor],
    background_values: Sequence[Optional[float]],
    random_deformations: Sequence[Tensor],
    voxel_size: Tensor,
) -> tuple[list[Tensor], list[Tensor]]:
    output_volumes = []
    output_masks = []
    for volume, mask, background_value, random_deformation in zip(
        volumes, masks, background_values, random_deformations
    ):
        random_transformation = AffineTransformation(random_deformation)
        masked_volume = MaskedVolume(volume=volume, voxel_size=voxel_size, mask=mask)
        deformed_masked_volume = masked_volume.deform(random_transformation)
        augmented_volume = deformed_masked_volume.volume.detach()
        augmented_mask = deformed_masked_volume.generate_mask()
        if background_value is not None:
            augmented_volume = augmented_volume * augmented_mask + background_value * (
                1 - augmented_mask
            )
        output_volumes.append(augmented_volume)
        output_masks.append(augmented_mask)
    return output_volumes, output_masks


def _squeeze_dim(
    volumes: Sequence[Tensor], random_deformations: Tensor, squeeze_dim: int
) -> tuple[list[Tensor], Tensor]:
    output_volumes = []
    for volume in volumes:
        output_volumes.append(volume.squeeze(squeeze_dim + 2))
    output_random_deformations = cat(
        (
            random_deformations[:, :, :squeeze_dim],
            random_deformations[:, :, squeeze_dim + 1 :],
        ),
        dim=2,
    )
    output_random_deformations = cat(
        (
            output_random_deformations[:, :, :, :squeeze_dim],
            output_random_deformations[:, :, :, squeeze_dim + 1 :],
        ),
        dim=3,
    )
    return output_volumes, output_random_deformations


def _train(
    config: Mapping[str, Any],
    target_dir: str,
    seed: int,
    continue_from_epoch: Optional[int],
    devices: Sequence[device],
) -> None:
    torch_device = devices[0]

    validation_seed = RandomState(seed=seed).randint(2**32, dtype="uint32")
    data_loader_config_validation_copy = json_loads(json_dumps(config["data_loader"]))
    data_loader_config_validation_copy["paired"] = True
    data_loader_config_validation_copy["n_random_deformations"] = 1
    (validation_data_loader, _generate_new_validation_variant,) = init_generic_training_data_loader(
        config["data"],
        data_loader_config_validation_copy,
        validation_seed,
        "validate",
        data_loader_config_validation_copy["n_validation_workers"],
    )
    data_loader, generate_new_variant = init_generic_training_data_loader(
        config["data"], config["data_loader"], seed, "train", config["data_loader"]["n_workers"]
    )

    cycle_gan_and_pix2pix_config = dict(config["model"])
    cycle_gan_and_pix2pix_config["isTrain"] = True
    if devices[0].type != "cpu":
        cycle_gan_and_pix2pix_config["gpu_ids"] = [torch_device.index for torch_device in devices]
    else:
        cycle_gan_and_pix2pix_config["gpu_ids"] = []
    opt = _Struct(**cycle_gan_and_pix2pix_config)
    squeeze_dim = cycle_gan_and_pix2pix_config.get("squeeze_dim")
    n_total_epochs = (
        cycle_gan_and_pix2pix_config["n_epochs"] + cycle_gan_and_pix2pix_config["n_epochs_decay"]
    )

    model = create_model(opt)
    model.setup(opt)

    voxel_size = tensor(config["data_loader"]["voxel_size"], dtype=float32, device=torch_device)
    if squeeze_dim is not None:
        voxel_size = cat((voxel_size[:squeeze_dim], voxel_size[squeeze_dim + 1 :]))

    if continue_from_epoch is None:
        initial_epoch = 0
    else:
        initial_epoch = load_model(
            target_dir=target_dir,
            epoch=continue_from_epoch,
            prefix="training",
            model=model,
            torch_device=torch_device,
            discriminator_model=None,
            optimizer=None,
            discriminator_optimizer=None,
        )
    for _ in range(initial_epoch):
        generate_new_variant()
        model.update_learning_rate()

    epoch_tqdm = tqdm(
        range(initial_epoch, n_total_epochs),
        unit="epoch",
        initial=initial_epoch,
        total=n_total_epochs,
    )
    augmentation_input_background_value = config["training"].get(
        "augmentation_input_background_value"
    )
    augmentation_label_background_value = config["training"].get(
        "augmentation_label_background_value"
    )
    for epoch in epoch_tqdm:
        loss_averager = LossAverager()
        data_tqdm = tqdm(data_loader, leave=False)
        try:
            for (
                input_image,
                label_image,
                input_mask,
                label_mask,
                random_deformations,
            ) in data_tqdm:
                input_image = input_image.to(torch_device)
                label_image = label_image.to(torch_device)
                input_mask = input_mask.to(torch_device)
                label_mask = label_mask.to(torch_device)
                random_deformations = random_deformations.to(torch_device)
                if squeeze_dim is not None:
                    (
                        input_image,
                        label_image,
                        input_mask,
                        label_mask,
                    ), random_deformations = _squeeze_dim(
                        volumes=[input_image, label_image, input_mask, label_mask],
                        random_deformations=random_deformations,
                        squeeze_dim=squeeze_dim,
                    )
                if config["training"].get("augment_using_the_random_transformations", False):
                    if random_deformations.size(1) > 1:
                        random_deformations_list = [
                            random_deformations[:, 0],
                            random_deformations[:, 1],
                        ]
                    else:
                        random_deformations_list = [
                            random_deformations[:, 0],
                            random_deformations[:, 0],
                        ]
                    (input_image, label_image), (input_mask, label_mask) = _augment(
                        volumes=[input_image, label_image],
                        masks=[input_mask, label_mask],
                        background_values=[
                            augmentation_input_background_value,
                            augmentation_label_background_value,
                        ],
                        random_deformations=random_deformations_list,
                        voxel_size=voxel_size,
                    )
                data = {
                    "A": input_image,
                    "B": label_image,
                    "A_mask": input_mask,
                    "B_mask": label_mask,
                }
                model.set_input(data)
                model.optimize_parameters()
                loss_averager.count(model.get_current_losses())
                data_tqdm.set_description(str(loss_averager))
        except KeyboardInterrupt:
            save_model(
                target_dir=target_dir,
                epoch=epoch + 1,
                prefix="training_dump",
                model=model,
                discriminator_model=None,
                optimizer=None,
                discriminator_optimizer=None,
            )
            return
        generate_new_variant()
        model.update_learning_rate()
        save_model(
            target_dir=target_dir,
            epoch=epoch + 1,
            prefix="training",
            model=model,
            discriminator_model=None,
            optimizer=None,
            discriminator_optimizer=None,
        )
        loss_averager.save_to_json(epoch=epoch + 1, filename=join(target_dir, "loss_history.json"))
        epoch_tqdm.write(f"End of epoch {epoch + 1}, {loss_averager}")
        epoch_tqdm.write("Calculating validation loss...")
        with no_grad():
            validation_loss_averager = LossAverager()
            validation_data_tqdm = tqdm(validation_data_loader)
            for (
                input_image,
                label_image,
                input_mask,
                label_mask,
                random_deformations,
            ) in validation_data_tqdm:
                input_image = input_image.to(torch_device)
                label_image = label_image.to(torch_device)
                input_mask = input_mask.to(torch_device)
                label_mask = label_mask.to(torch_device)
                random_deformations = random_deformations.to(torch_device)

                if squeeze_dim is not None:
                    (
                        input_image,
                        label_image,
                        input_mask,
                        label_mask,
                    ), random_deformations = _squeeze_dim(
                        volumes=[input_image, label_image, input_mask, label_mask],
                        random_deformations=random_deformations,
                        squeeze_dim=squeeze_dim,
                    )
                if config["training"].get("augment_using_the_random_transformations", False):
                    if random_deformations.size(1) > 1:
                        raise ValueError("Validation data must be paired!")
                    random_deformations_list = [
                        random_deformations[:, 0],
                        random_deformations[:, 0],
                    ]
                    (input_image, label_image), (input_mask, label_mask) = _augment(
                        volumes=[input_image, label_image],
                        masks=[input_mask, label_mask],
                        background_values=[
                            augmentation_input_background_value,
                            augmentation_label_background_value,
                        ],
                        random_deformations=random_deformations_list,
                        voxel_size=voxel_size,
                    )

                if isinstance(model, NEMARModel):
                    prediction = model.translate_and_deform_A_to_B(
                        input_image, label_image, augmentation_label_background_value
                    )
                else:
                    prediction = model.translate_A_to_B(input_image)
                if config["model"].get("use_masking", True):
                    combined_mask = input_mask * label_mask
                else:
                    combined_mask = ones_like(input_mask)
                similarity_validation_loss = masked_mae_loss(
                    input_1=prediction, input_2=label_image, mask=combined_mask
                )
                validation_loss_averager.count(
                    {
                        "sim": float(similarity_validation_loss),
                    }
                )
            validation_loss_averager.save_to_json(
                epoch=epoch + 1, filename=join(target_dir, "validation_loss_history.json")
            )


if __name__ == "__main__":
    training_func: TrainingFunction = _train
    obtain_arguments_and_train(training_func)
