"""Training script for N-dimensional NEMAR model"""

from os.path import join
from typing import Any, Callable, Mapping, Optional, Sequence

from numpy.random import RandomState
from torch import Tensor, device, float32, full, no_grad, tensor
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from tqdm import tqdm  # type: ignore

from algorithm.torch.deformation.masked import AffineTransformation, MaskedVolume
from data.interface import init_generic_training_data_loader
from loss.deformation import (
    EmptyRigidityCoefficientGenerator,
    IRigidityCoefficientGenerator,
    RigidityLoss,
)
from loss.masked_loss import masked_mae_loss, masked_mse_loss
from model.baseline.interface import load_reggan_model, save_reggan_model
from model.interface import init_nemar_nd_model, init_spectral_norm_discriminator
from util.import_util import import_object
from util.training import LossAverager, TrainingFunction, obtain_arguments_and_train


def _get_similarity_loss(
    loss_config: Mapping[str, Any]
) -> Callable[[Tensor, Tensor, Tensor], Tensor]:
    if loss_config["similarity_loss"] == "mse":
        return masked_mse_loss
    if loss_config["similarity_loss"] == "mae":
        return masked_mae_loss
    raise ValueError("Invalid loss type")


def _augment(
    volumes: Sequence[Tensor],
    masks: Sequence[Tensor],
    random_deformation: Tensor,
    voxel_size: Tensor,
) -> tuple[list[Tensor], list[Tensor]]:
    output_volumes = []
    output_masks = []
    for volume, mask in zip(volumes, masks):
        random_transformation = AffineTransformation(random_deformation)
        masked_volume = MaskedVolume(volume=volume, voxel_size=voxel_size, mask=mask)
        deformed_masked_volume = masked_volume.deform(random_transformation)
        output_volumes.append(deformed_masked_volume.volume.detach())
        output_masks.append(deformed_masked_volume.generate_mask())
    return output_volumes, output_masks


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

    nemar, nemar_module = init_nemar_nd_model(
        config["model"], config["data_loader"], devices=devices
    )
    discriminator, discriminator_module = init_spectral_norm_discriminator(
        config["model"], config["data_loader"], devices=devices
    )
    optimizer_registration = Adam(
        params=nemar_module.cross_modality_registration_unet.parameters(),
        lr=training_config["learning_rate_generator"],
        betas=(0.5, 0.999),
    )
    optimizer_generator = Adam(
        params=nemar_module.i2i_unet.parameters(),
        lr=training_config["learning_rate_generator"],
        betas=(0.5, 0.999),
    )
    optimizer_discriminator = Adam(
        params=discriminator.parameters(),
        lr=training_config["learning_rate_discriminator"],
        betas=(0.5, 0.999),
    )
    nemar.to(torch_device)
    discriminator.to(torch_device)

    loss_config = training_config["loss"]
    voxel_size = tensor(config["data_loader"]["voxel_size"], dtype=float32, device=torch_device)
    calculate_rigidity_loss = RigidityLoss(**loss_config["deformation"])
    calculate_rigidity_loss.to(torch_device)
    rigidity_coefficient_config = loss_config.get("rigidity_coefficient")
    rigidity_coefficient_config = (
        {} if rigidity_coefficient_config is None else rigidity_coefficient_config
    )
    rigidity_coeff_generator: IRigidityCoefficientGenerator = (
        import_object(rigidity_coefficient_config["type"])
        if rigidity_coefficient_config.get("type") is not None
        else EmptyRigidityCoefficientGenerator
    )(**rigidity_coefficient_config.get("args", {}))
    rigidity_coeff_generator.to(torch_device)
    calculate_output_similarity_loss = _get_similarity_loss(loss_config)
    calculate_discriminator_loss = BCEWithLogitsLoss()
    real_label_value = 1.0
    fake_label_value = 0.0

    if continue_from_epoch is None:
        initial_epoch = 0
    else:
        initial_epoch = load_reggan_model(
            target_dir=target_dir,
            epoch=continue_from_epoch,
            prefix="training",
            torch_device=torch_device,
            generator_model=nemar_module.i2i_unet,
            discriminator_model=discriminator_module,
            reg_model=nemar_module.cross_modality_registration_unet,
            generator_optimizer=optimizer_generator,
            discriminator_optimizer=optimizer_discriminator,
            reg_optimizer=optimizer_registration,
        )
    for _ in range(initial_epoch):
        generate_new_variant()
    nemar.train()

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

                if config["training"].get("augment_using_the_random_transformations", False):
                    (input_image, label_image), (input_mask, label_mask) = _augment(
                        volumes=[input_image, label_image],
                        masks=[input_mask, label_mask],
                        random_deformation=random_deformation,
                        voxel_size=voxel_size,
                    )

                half_batch_size = input_image.size(0) // 2

                discriminator.zero_grad()
                (
                    output_cross_modality_deformed,
                    cross_modality_deformed_output,
                    label,
                    input_volume,
                ) = nemar(
                    input_volume=input_image[:half_batch_size],
                    label_volume=label_image[:half_batch_size],
                    input_mask=input_mask[:half_batch_size],
                    label_mask=label_mask[:half_batch_size],
                    desired_outputs=(
                        "output_cross_modality_deformed",
                        "cross_modality_deformed_output",
                        "label",
                        "input",
                    ),
                )
                label_real = full(
                    (half_batch_size,),
                    real_label_value,
                    dtype=input_image.dtype,
                    device=torch_device,
                )
                label_fake = full(
                    (half_batch_size,),
                    fake_label_value,
                    dtype=input_image.dtype,
                    device=torch_device,
                )
                combined_mask_discriminator = MaskedVolume.get_comparison_mask(
                    label,
                    output_cross_modality_deformed,
                )
                discriminator_output_real = discriminator(
                    input_volume=input_volume.volume,
                    output_volume=(label.volume * combined_mask_discriminator),
                )
                discriminator_error_real = calculate_discriminator_loss(
                    discriminator_output_real, label_real
                )
                discriminator_output_fake_rt = discriminator(
                    input_volume=input_volume.volume,
                    output_volume=(
                        cross_modality_deformed_output.volume.detach() * combined_mask_discriminator
                    ),
                )
                discriminator_output_fake_tr = discriminator(
                    input_volume=input_volume.volume,
                    output_volume=(
                        output_cross_modality_deformed.volume.detach() * combined_mask_discriminator
                    ),
                )
                discriminator_error_fake = calculate_discriminator_loss(
                    discriminator_output_fake_rt, label_fake
                ) + calculate_discriminator_loss(discriminator_output_fake_tr, label_fake)
                (
                    loss_config["discriminator_weight"]
                    * (discriminator_error_fake + discriminator_error_real)
                ).backward()
                optimizer_discriminator.step()

                nemar.zero_grad()
                (
                    output_cross_modality_deformed,
                    cross_modality_deformed_output,
                    forward_cross_modality_deformation,
                    label,
                    input_volume,
                ) = nemar(
                    input_volume=input_image[:half_batch_size],
                    label_volume=label_image[:half_batch_size],
                    input_mask=input_mask[:half_batch_size],
                    label_mask=label_mask[:half_batch_size],
                    desired_outputs=(
                        "output_cross_modality_deformed",
                        "cross_modality_deformed_output",
                        "forward_cross_modality_deformation",
                        "label",
                        "input",
                    ),
                )
                combined_mask_discriminator = MaskedVolume.get_comparison_mask(
                    label,
                    output_cross_modality_deformed,
                )
                discriminator_output_generator_fake_rt = discriminator(
                    input_volume=input_volume.volume,
                    output_volume=(
                        cross_modality_deformed_output.volume * combined_mask_discriminator
                    ),
                )
                discriminator_output_generator_fake_tr = discriminator(
                    input_volume=input_volume.volume,
                    output_volume=(
                        output_cross_modality_deformed.volume * combined_mask_discriminator
                    ),
                )
                discriminator_error_generator = calculate_discriminator_loss(
                    discriminator_output_generator_fake_rt, label_real
                ) + calculate_discriminator_loss(discriminator_output_generator_fake_tr, label_real)
                cross_modality_similarity_loss = calculate_output_similarity_loss(
                    label.volume,
                    cross_modality_deformed_output.volume,
                    MaskedVolume.get_comparison_mask(label, cross_modality_deformed_output),
                ) + calculate_output_similarity_loss(
                    label.volume,
                    output_cross_modality_deformed.volume,
                    MaskedVolume.get_comparison_mask(label, output_cross_modality_deformed),
                )
                forward_rigidity_loss = calculate_rigidity_loss(
                    forward_cross_modality_deformation.get_deformation(),
                    rigidity_coefficient=(
                        rigidity_coeff_generator.generate_input_space_volume(
                            input_volume=input_volume
                        )
                    ),
                )
                generator_loss = (
                    0.5
                    * loss_config["cross_modality_similarity_weight"]
                    * cross_modality_similarity_loss
                    + 0.5
                    * loss_config["discriminator_weight_generator"]
                    * discriminator_error_generator
                    + loss_config["forward_rigidity_weight"] * forward_rigidity_loss
                )
                loss_averager.count(
                    {
                        "loss": float(generator_loss),
                        "cross_sim": float(cross_modality_similarity_loss),
                        "fwd_rig": float(forward_rigidity_loss),
                        "g_d_err": float(discriminator_error_generator),
                        "d_err": float(discriminator_error_real + discriminator_error_fake),
                    }
                )
                generator_loss.backward()
                optimizer_generator.step()
                optimizer_registration.step()
                data_tqdm.set_description(str(loss_averager))
        except KeyboardInterrupt:
            save_reggan_model(
                target_dir=target_dir,
                epoch=epoch + 1,
                prefix="training_dump",
                generator_model=nemar_module.i2i_unet,
                discriminator_model=discriminator_module,
                reg_model=nemar_module.cross_modality_registration_unet,
                generator_optimizer=optimizer_generator,
                discriminator_optimizer=optimizer_discriminator,
                reg_optimizer=optimizer_registration,
            )
            return
        generate_new_variant()
        save_reggan_model(
            target_dir=target_dir,
            epoch=epoch + 1,
            prefix="training",
            generator_model=nemar_module.i2i_unet,
            discriminator_model=discriminator_module,
            reg_model=nemar_module.cross_modality_registration_unet,
            generator_optimizer=optimizer_generator,
            discriminator_optimizer=optimizer_discriminator,
            reg_optimizer=optimizer_registration,
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
                    (input_image, label_image), (input_mask, label_mask) = _augment(
                        volumes=[input_image, label_image],
                        masks=[input_mask, label_mask],
                        random_deformation=random_deformation,
                        voxel_size=voxel_size,
                    )

                (output_cross_modality_deformed, cross_modality_deformed_output, label,) = nemar(
                    input_volume=input_image[:half_batch_size],
                    label_volume=label_image[:half_batch_size],
                    input_mask=input_mask[:half_batch_size],
                    label_mask=label_mask[:half_batch_size],
                    desired_outputs=(
                        "output_cross_modality_deformed",
                        "cross_modality_deformed_output",
                        "label",
                    ),
                )
                cross_modality_similarity_loss = calculate_output_similarity_loss(
                    label.volume,
                    cross_modality_deformed_output.volume,
                    MaskedVolume.get_comparison_mask(label, cross_modality_deformed_output),
                ) + calculate_output_similarity_loss(
                    label.volume,
                    output_cross_modality_deformed.volume,
                    MaskedVolume.get_comparison_mask(label, output_cross_modality_deformed),
                )
                validation_loss_averager.count(
                    {
                        "sim": float(cross_modality_similarity_loss),
                    }
                )
            validation_loss_averager.save_to_json(
                epoch=epoch + 1, filename=join(target_dir, "validation_loss_history.json")
            )


if __name__ == "__main__":
    training_func: TrainingFunction = _train
    obtain_arguments_and_train(training_func)
