"""Simple training script for training straightforward i2i network"""

from os.path import join
from typing import Any, Callable, Mapping, Optional, Sequence

from numpy.random import RandomState
from torch import Tensor, device, float32, no_grad, tensor
from torch.optim import Adam
from tqdm import tqdm  # type: ignore

from algorithm.torch.deformation.masked import AffineTransformation, MaskedVolume
from algorithm.torch.mask import mask_and
from data.interface import init_generic_training_data_loader
from loss.masked_loss import masked_mae_loss, masked_mse_loss
from model.interface import init_i2i_unet_model
from util.training import (
    LossAverager,
    TrainingFunction,
    load_model,
    obtain_arguments_and_train,
    save_model,
)


def _get_similarity_loss(
    loss_config: Mapping[str, Any]
) -> Callable[[Tensor, Tensor, Tensor], Tensor]:
    if loss_config["similarity_loss"] == "mse":
        output_loss = masked_mse_loss
    elif loss_config["similarity_loss"] == "mae":
        output_loss = masked_mae_loss
    else:
        raise ValueError("Specifiy output loss type!")

    return output_loss


def _augment(
    volumes: Sequence[Tensor],
    masks: Sequence[Tensor],
    random_deformation: Tensor,
    voxel_size: Tensor,
) -> tuple[list[Tensor], list[Tensor], Tensor]:
    output_volumes = []
    output_masks = []
    for volume, mask in zip(volumes, masks):
        random_transformation = AffineTransformation(random_deformation)
        masked_volume = MaskedVolume(volume=volume, voxel_size=voxel_size, mask=mask)
        deformed_masked_volume = masked_volume.deform(random_transformation)
        output_volumes.append(deformed_masked_volume.volume.detach())
        output_masks.append(deformed_masked_volume.generate_mask())
    return output_volumes, output_masks, mask_and(output_masks)


def _train(
    config: Mapping[str, Any],
    target_dir: str,
    seed: int,
    continue_from_epoch: Optional[int],
    devices: Sequence[device],
) -> None:
    training_config = config["training"]

    validation_seed = RandomState(seed=seed).randint(2**32, dtype="uint32")
    validation_data_loader, _generate_new_validation_variant = init_generic_training_data_loader(
        config["data"],
        config["data_loader"],
        validation_seed,
        "validate",
        config["data_loader"]["n_validation_workers"],
    )
    data_loader, generate_new_variant = init_generic_training_data_loader(
        config["data"], config["data_loader"], seed, "train", config["data_loader"]["n_workers"]
    )

    torch_device = devices[0]

    unet, unet_module = init_i2i_unet_model(config["data_loader"], config["model"], devices=devices)
    optimizer_generator = Adam(
        params=unet.parameters(), lr=training_config["learning_rate_generator"]
    )
    unet.to(torch_device)

    loss_config = training_config["loss"]
    voxel_size = tensor(config["data_loader"]["voxel_size"], dtype=float32, device=torch_device)
    calculate_output_similarity_loss = _get_similarity_loss(loss_config)

    if continue_from_epoch is None:
        initial_epoch = 0
    else:
        initial_epoch = load_model(
            target_dir=target_dir,
            epoch=continue_from_epoch,
            prefix="training",
            model=unet_module,
            torch_device=torch_device,
            discriminator_model=None,
            optimizer=optimizer_generator,
            discriminator_optimizer=None,
        )
    for _ in range(initial_epoch):
        generate_new_variant()
    unet.train()

    epoch_tqdm = tqdm(
        range(initial_epoch, training_config["n_epochs"]),
        unit="epoch",
        initial=initial_epoch,
        total=training_config["n_epochs"],
    )
    for epoch in epoch_tqdm:
        loss_averager = LossAverager()
        data_tqdm = tqdm(data_loader, leave=False)
        try:
            for (input_image, label_image, input_mask, label_mask, random_deformation) in data_tqdm:
                input_image = input_image.to(torch_device)
                label_image = label_image.to(torch_device)
                input_mask = input_mask.to(torch_device)
                label_mask = label_mask.to(torch_device)
                random_deformation = random_deformation.to(torch_device)

                if config["training"].get("augment_using_the_random_transformations", False):
                    (input_image, label_image), (input_mask, label_mask), combined_mask = _augment(
                        volumes=[input_image, label_image],
                        masks=[input_mask, label_mask],
                        random_deformation=random_deformation,
                        voxel_size=voxel_size
                    )
                else:
                    combined_mask = input_mask * label_mask

                unet.zero_grad()
                output = unet(input_image)
                similarity_loss = calculate_output_similarity_loss(
                    output, label_image, combined_mask
                )
                generator_loss = loss_config["similarity_weight"] * similarity_loss
                loss_averager.count({"loss": float(generator_loss), "sim": float(similarity_loss)})
                generator_loss.backward()
                optimizer_generator.step()
                data_tqdm.set_description(str(loss_averager))
        except KeyboardInterrupt:
            save_model(
                target_dir=target_dir,
                epoch=epoch + 1,
                prefix="training_dump",
                model=unet_module,
                discriminator_model=None,
                optimizer=optimizer_generator,
                discriminator_optimizer=None,
            )
            return
        generate_new_variant()
        save_model(
            target_dir=target_dir,
            epoch=epoch + 1,
            prefix="training",
            model=unet_module,
            discriminator_model=None,
            optimizer=optimizer_generator,
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
                random_deformation,
            ) in validation_data_tqdm:
                input_image = input_image.to(torch_device)
                label_image = label_image.to(torch_device)
                input_mask = input_mask.to(torch_device)
                label_mask = label_mask.to(torch_device)
                random_deformation = random_deformation.to(torch_device)

                if config["training"].get("augment_using_the_random_transformations", False):
                    (input_image, label_image), (input_mask, label_mask), combined_mask = _augment(
                        volumes=[input_image, label_image],
                        masks=[input_mask, label_mask],
                        random_deformation=random_deformation,
                        voxel_size=voxel_size
                    )

                output = unet(input_image)
                similarity_validation_loss = calculate_output_similarity_loss(
                    output, label_image, combined_mask
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
