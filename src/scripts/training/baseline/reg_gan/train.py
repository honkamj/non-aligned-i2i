"""Simple training script for training RegGAN

Modified from https://github.com/Kid-Liet/Reg-GAN
"""

from os.path import join
from typing import Any, Mapping, Optional, Sequence

from numpy.random import RandomState
from torch import Tensor, cat, device, float32, no_grad, ones_like, tensor, zeros_like
from torch.nn import L1Loss, MSELoss
from torch.optim import Adam
from tqdm import tqdm  # type: ignore

from algorithm.torch.deformation.masked import AffineTransformation, MaskedVolume
from algorithm.torch.mask import mask_and
from data.interface import init_generic_training_data_loader
from model.baseline.interface import load_reggan_model, save_reggan_model
from model.baseline.reg_gan.nice_gan import Discriminator2, ResnetGenerator2
from model.baseline.reg_gan.reg import Reg
from model.baseline.reg_gan.transformer import Transformer_2D
from model.baseline.reg_gan.utils import smooothing_loss
from util.training import LossAverager, TrainingFunction, obtain_arguments_and_train


def _augment(
    volumes: Sequence[Tensor],
    masks: Sequence[Tensor],
    background_values: Sequence[Optional[float]],
    random_deformation: Tensor,
    voxel_size: Tensor,
) -> tuple[list[Tensor], list[Tensor], Tensor]:
    output_volumes = []
    output_masks = []
    for volume, mask, background_value in zip(volumes, masks, background_values):
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
    return output_volumes, output_masks, mask_and(output_masks)


def _squeeze_dim(
    volumes: Sequence[Tensor], random_deformation: Tensor, squeeze_dim: int
) -> tuple[list[Tensor], Tensor]:
    output_volumes = []
    for volume in volumes:
        output_volumes.append(volume.squeeze(squeeze_dim + 2))
    output_random_deformation = cat(
        (
            random_deformation[:, :squeeze_dim],
            random_deformation[:, squeeze_dim + 1 :],
        ),
        dim=1,
    )
    output_random_deformation = cat(
        (
            output_random_deformation[:, :, :squeeze_dim],
            output_random_deformation[:, :, squeeze_dim + 1 :],
        ),
        dim=2,
    )
    return output_volumes, output_random_deformation


def _train(
    config: Mapping[str, Any],
    target_dir: str,
    seed: int,
    continue_from_epoch: Optional[int],
    devices: Sequence[device],
) -> None:
    torch_device = devices[0]

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

    model_config = config["model"]
    training_config = config["training"]

    reg = Reg(
        model_config["size"][0],
        model_config["size"][1],
        model_config["input_nc"],
        model_config["input_nc"],
        cfg=model_config["reg_architecture_config"],
    )
    discriminator = Discriminator2(model_config["input_nc"])
    generator = ResnetGenerator2(
        model_config["input_nc"],
        model_config["output_nc"],
        img_size=model_config["size"],
        n_blocks=model_config["n_blocks"],
        final_activation=model_config["final_activation"],
    )
    reg.to(torch_device)
    discriminator.to(torch_device)
    generator.to(torch_device)
    optimizer_reg = Adam(reg.parameters(), lr=training_config["learning_rate"], betas=(0.5, 0.999))
    optimizer_generator = Adam(
        generator.parameters(), lr=training_config["learning_rate"], betas=(0.5, 0.999)
    )
    optimizer_discriminator = Adam(
        discriminator.parameters(), lr=training_config["learning_rate"], betas=(0.5, 0.999)
    )

    spatial_transform = Transformer_2D()
    spatial_transform.to(torch_device)

    n_epochs = training_config["n_epochs"]
    voxel_size = tensor(config["data_loader"]["voxel_size"], dtype=float32, device=torch_device)
    squeeze_dim = model_config.get("squeeze_dim")
    use_masking = model_config.get("use_masking", True)
    augmentation_input_background_value = config["training"].get(
        "augmentation_input_background_value"
    )
    augmentation_label_background_value = config["training"].get(
        "augmentation_label_background_value"
    )

    mse_loss = MSELoss()
    l1_loss = L1Loss()

    if squeeze_dim is not None:
        voxel_size = cat((voxel_size[:squeeze_dim], voxel_size[squeeze_dim + 1 :]))

    if continue_from_epoch is None:
        initial_epoch = 0
    else:
        initial_epoch = load_reggan_model(
            target_dir=target_dir,
            epoch=continue_from_epoch,
            prefix="training",
            torch_device=torch_device,
            generator_model=generator,
            discriminator_model=discriminator,
            reg_model=reg,
            generator_optimizer=optimizer_generator,
            discriminator_optimizer=optimizer_discriminator,
            reg_optimizer=optimizer_reg,
        )
    for _ in range(initial_epoch):
        generate_new_variant()

    epoch_tqdm = tqdm(
        range(initial_epoch, n_epochs),
        unit="epoch",
        initial=initial_epoch,
        total=n_epochs,
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
                random_deformation,
            ) in data_tqdm:
                input_image = input_image.to(torch_device)
                label_image = label_image.to(torch_device)
                input_mask = input_mask.to(torch_device)
                label_mask = label_mask.to(torch_device)
                random_deformation = random_deformation.to(torch_device)
                if squeeze_dim is not None:
                    (
                        input_image,
                        label_image,
                        input_mask,
                        label_mask,
                    ), random_deformation = _squeeze_dim(
                        volumes=[input_image, label_image, input_mask, label_mask],
                        random_deformation=random_deformation,
                        squeeze_dim=squeeze_dim,
                    )
                if config["training"].get("augment_using_the_random_transformations", False):
                    (input_image, label_image), (input_mask, label_mask), combined_mask = _augment(
                        volumes=[input_image, label_image],
                        masks=[input_mask, label_mask],
                        background_values=[
                            augmentation_input_background_value,
                            augmentation_label_background_value,
                        ],
                        random_deformation=random_deformation,
                        voxel_size=voxel_size,
                    )
                else:
                    combined_mask = input_mask * label_mask

                optimizer_reg.zero_grad()
                optimizer_generator.zero_grad()
                prediction = generator(input_image)
                deformation = reg(prediction, label_image)
                deformed_prediction = spatial_transform(prediction, deformation)
                if use_masking:
                    deformed_input_mask = (
                        spatial_transform(input_mask, deformation, padding_mode="zeros")
                        >= 1.0 - 1e-6
                    ).to(input_mask.dtype)
                    combined_l1_mask = deformed_input_mask * label_mask
                else:
                    combined_mask = ones_like(input_mask)
                    combined_l1_mask = combined_mask
                similarity_loss = model_config["Corr_lamda"] * l1_loss(
                    deformed_prediction * combined_l1_mask, label_image * combined_l1_mask
                )
                fake_lb_logit, fake_gb_logit, fake_b_cam_logit = discriminator(
                    prediction * combined_mask
                )
                adv_loss = model_config["Adv_lamda"] * (
                    mse_loss(fake_gb_logit, ones_like(fake_gb_logit, device=torch_device))
                    + mse_loss(fake_lb_logit, ones_like(fake_lb_logit, device=torch_device))
                    + mse_loss(fake_b_cam_logit, ones_like(fake_b_cam_logit, device=torch_device))
                )
                smoothness_loss = model_config["Smooth_lamda"] * smooothing_loss(deformation)

                generator_loss = smoothness_loss + adv_loss + similarity_loss

                losses = {
                    "similarity": float(similarity_loss),
                    "smoothness": float(smoothness_loss),
                    "generator_gan": float(adv_loss),
                }

                generator_loss.backward()
                optimizer_reg.step()
                optimizer_generator.step()

                optimizer_discriminator.zero_grad()
                with no_grad():
                    prediction_discriminator = generator(input_image)

                fake_lb_logit, fake_gb_logit, fake_b_cam_logit = discriminator(
                    prediction_discriminator * combined_mask
                )
                real_lb_logit, real_gb_logit, real_b_cam_logit = discriminator(
                    label_image * combined_mask
                )

                discriminator_loss = model_config["Adv_lamda"] * (
                    mse_loss(fake_gb_logit, zeros_like(fake_gb_logit, device=torch_device))
                    + mse_loss(fake_lb_logit, zeros_like(fake_lb_logit, device=torch_device))
                    + mse_loss(fake_b_cam_logit, zeros_like(fake_b_cam_logit, device=torch_device))
                    + mse_loss(real_gb_logit, ones_like(real_gb_logit, device=torch_device))
                    + mse_loss(real_lb_logit, ones_like(real_lb_logit, device=torch_device))
                    + mse_loss(real_b_cam_logit, ones_like(real_b_cam_logit, device=torch_device))
                )

                discriminator_loss.backward()
                optimizer_discriminator.step()

                losses["discriminator_gan"] = float(discriminator_loss)

                loss_averager.count(losses)
                data_tqdm.set_description(str(loss_averager))
        except KeyboardInterrupt:
            save_reggan_model(
                target_dir=target_dir,
                epoch=epoch + 1,
                prefix="training_dump",
                generator_model=generator,
                discriminator_model=discriminator,
                reg_model=reg,
                generator_optimizer=optimizer_generator,
                discriminator_optimizer=optimizer_discriminator,
                reg_optimizer=optimizer_reg,
            )
            return
        generate_new_variant()
        save_reggan_model(
            target_dir=target_dir,
            epoch=epoch + 1,
            prefix="training",
            generator_model=generator,
            discriminator_model=discriminator,
            reg_model=reg,
            generator_optimizer=optimizer_generator,
            discriminator_optimizer=optimizer_discriminator,
            reg_optimizer=optimizer_reg,
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

                if squeeze_dim is not None:
                    (
                        input_image,
                        label_image,
                        input_mask,
                        label_mask,
                    ), random_deformation = _squeeze_dim(
                        volumes=[input_image, label_image, input_mask, label_mask],
                        random_deformation=random_deformation,
                        squeeze_dim=squeeze_dim,
                    )
                if config["training"].get("augment_using_the_random_transformations", False):
                    (input_image, label_image), (input_mask, label_mask), _combined_mask = _augment(
                        volumes=[input_image, label_image],
                        masks=[input_mask, label_mask],
                        background_values=[
                            augmentation_input_background_value,
                            augmentation_label_background_value,
                        ],
                        random_deformation=random_deformation,
                        voxel_size=voxel_size,
                    )

                prediction = generator(input_image)
                deformation = reg(prediction, label_image)
                deformed_prediction = spatial_transform(prediction, deformation)
                if use_masking:
                    deformed_input_mask = (
                        spatial_transform(input_mask, deformation, padding_mode="zeros")
                        >= 1.0 - 1e-6
                    ).to(input_mask.dtype)
                    combined_l1_mask = deformed_input_mask * label_mask
                else:
                    combined_l1_mask = ones_like(input_mask)
                similarity_validation_loss = l1_loss(
                    deformed_prediction * combined_l1_mask, label_image * combined_l1_mask
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
