"""Training script for training combined registration and I2I"""

from math import pi
from os.path import join
from typing import Any, Callable, Mapping, Optional, Sequence

from numpy.random import RandomState
from torch import Tensor
from torch import abs as torch_abs
from torch import device, float32
from torch import mean as torch_mean
from torch import no_grad, sqrt, square
from torch import sum as torch_sum
from torch import tensor
from torch.optim import Adam
from tqdm import tqdm  # type: ignore

from algorithm.torch.deformation.masked import AffineTransformation, MaskedVolume
from data.interface import init_generic_training_data_loader
from loss.constraining import squared_constraint
from loss.deformation import (
    EmptyRigidityCoefficientGenerator,
    IRigidityCoefficientGenerator,
    RigidityLoss,
)
from loss.masked_loss import masked_mae_loss, masked_mse_loss
from model.interface import init_non_aligned_i2i_model
from util.import_util import import_object
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

    non_aligned_i2i, non_aligned_i2i_module = init_non_aligned_i2i_model(
        config["model"], config["data_loader"], devices=devices
    )
    optimizer = Adam(
        params=non_aligned_i2i.parameters(), lr=training_config["learning_rate_generator"]
    )
    non_aligned_i2i.to(torch_device)

    loss_config = training_config["loss"]
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
    use_implicit_di = loss_config.get("use_implicit_di", True)
    voxel_size = tensor(config["data_loader"]["voxel_size"], dtype=float32, device=torch_device)

    if continue_from_epoch is None:
        initial_epoch = 0
    else:
        initial_epoch = load_model(
            target_dir=target_dir,
            epoch=continue_from_epoch,
            prefix="training",
            model=non_aligned_i2i_module,
            optimizer=optimizer,
            torch_device=torch_device,
        )
    for _ in range(initial_epoch):
        generate_new_variant()
    non_aligned_i2i.train()

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
                    (input_image, label_image), (input_mask, label_mask) = _augment(
                        volumes=[input_image, label_image],
                        masks=[input_mask, label_mask],
                        random_deformation=random_deformation,
                        voxel_size=voxel_size,
                    )

                non_aligned_i2i.zero_grad()
                (
                    input_volume,
                    output,
                    left_random_deformed_output,
                    right_random_deformed_output,
                    deformed_output,
                    rigidly_deformed_label,
                    cross_modality_deformed_label,
                    forward_elastic_cross_modality_deformation,
                    inverse_elastic_cross_modality_deformation,
                    forward_deformation,
                    forward_elastic_deformation,
                    inverse_elastic_deformation,
                    forward_intra_modality_deformation,
                    rigid_angles,
                    rigid_translations,
                ) = non_aligned_i2i(
                    input_volume=input_image,
                    label_volume=label_image,
                    input_mask=input_mask,
                    label_mask=label_mask,
                    forward_random_transformation=random_deformation,
                    desired_outputs=(
                        "input",
                        "output",
                        "left_random_deformed_output",
                        "right_random_deformed_output",
                        "commutation_deformed_output" if use_implicit_di else "deformed_output",
                        "rigidly_deformed_label",
                        "cross_modality_deformed_label",
                        "forward_elastic_cross_modality_deformation",
                        "inverse_elastic_cross_modality_deformation",
                        "forward_deformation",
                        "forward_elastic_deformation",
                        "inverse_elastic_deformation",
                        "forward_intra_modality_deformation",
                        "rigid_angles",
                        "rigid_translations",
                    ),
                )
                rigid_similarity_loss = calculate_output_similarity_loss(
                    rigidly_deformed_label.volume,
                    output.volume.detach(),
                    MaskedVolume.get_comparison_mask(rigidly_deformed_label, output),
                )
                cross_modality_similarity_loss = calculate_output_similarity_loss(
                    cross_modality_deformed_label.volume,
                    output.volume.detach(),
                    MaskedVolume.get_comparison_mask(cross_modality_deformed_label, output),
                )
                intra_modality_similarity_loss = calculate_output_similarity_loss(
                    cross_modality_deformed_label.volume.detach(),
                    deformed_output.volume,
                    MaskedVolume.get_comparison_mask(
                        cross_modality_deformed_label, deformed_output
                    ),
                )
                di_loss = calculate_output_similarity_loss(
                    left_random_deformed_output.volume,
                    right_random_deformed_output.volume,
                    MaskedVolume.get_comparison_mask(
                        left_random_deformed_output, right_random_deformed_output
                    ),
                )
                forward_rigidity_loss = (
                    (
                        calculate_rigidity_loss(
                            forward_elastic_cross_modality_deformation.get_deformation(),
                            rigidity_coefficient=(
                                rigidity_coeff_generator.generate_input_space_volume(
                                    input_volume=input_volume
                                )
                            ),
                        )
                        + calculate_rigidity_loss(
                            forward_elastic_deformation.get_deformation(),
                            rigidity_coefficient=(
                                rigidity_coeff_generator.generate_input_space_volume(
                                    input_volume=input_volume
                                )
                            ),
                        )
                    )
                    if loss_config.get("forward_rigidity_weight") is not None
                    else 0.0
                )
                inverse_rigidity_loss = (
                    (
                        calculate_rigidity_loss(
                            inverse_elastic_cross_modality_deformation.get_deformation(),
                            rigidity_coefficient=(
                                rigidity_coeff_generator.generate_target_space_volume(
                                    target_volume=rigidly_deformed_label.detach()
                                )
                            ),
                        )
                        + calculate_rigidity_loss(
                            inverse_elastic_deformation.get_deformation(),
                            rigidity_coefficient=(
                                rigidity_coeff_generator.generate_target_space_volume(
                                    target_volume=rigidly_deformed_label.detach()
                                )
                            ),
                        )
                    )
                    if loss_config.get("inverse_rigidity_weight") is not None
                    else 0.0
                )
                rotation_constraining_loss = squared_constraint(
                    torch_abs(rigid_angles), loss_config["max_rotation"] * pi / 180
                )
                translation_constraining_loss = squared_constraint(
                    torch_abs(rigid_translations), loss_config["max_translation"]
                )
                intra_modality_deformation_mean_displacement = torch_mean(
                    sqrt(
                        torch_sum(
                            square(forward_intra_modality_deformation.get_deformation()), dim=1
                        )
                    )
                )
                deformation_mean_displacement = torch_mean(
                    sqrt(torch_sum(square(forward_deformation.get_deformation()), dim=1))
                )
                deformation_mean_y = torch_mean(forward_deformation.get_deformation()[:, 0])
                deformation_mean_x = torch_mean(forward_deformation.get_deformation()[:, 1])
                generator_loss = (
                    loss_config["cross_modality_similarity_weight"] * cross_modality_similarity_loss
                    + loss_config["intra_modality_similarity_weight"]
                    * intra_modality_similarity_loss
                    + loss_config["rigid_similarity_weight"] * rigid_similarity_loss
                    + loss_config["di_weight"] * di_loss
                    + loss_config.get("forward_rigidity_weight", 0.0) * forward_rigidity_loss * 0.5
                    + loss_config.get("inverse_rigidity_weight", 0.0) * inverse_rigidity_loss * 0.5
                    + rotation_constraining_loss
                    + translation_constraining_loss
                )
                loss_averager.count(
                    {
                        "loss": float(generator_loss),
                        "intra_sim": float(intra_modality_similarity_loss),
                        "cross_sim": float(cross_modality_similarity_loss),
                        "di": float(di_loss),
                        "rig_sim": float(rigid_similarity_loss),
                        "fwd_rig": float(forward_rigidity_loss),
                        "inv_rig": float(inverse_rigidity_loss),
                        "rot_cnstr": float(rotation_constraining_loss),
                        "trans_cnstr": float(translation_constraining_loss),
                        "intra_def_length": float(intra_modality_deformation_mean_displacement),
                        "def_length": float(deformation_mean_displacement),
                        "def_y": float(deformation_mean_y),
                        "def_x": float(deformation_mean_x),
                    }
                )
                generator_loss.backward()
                optimizer.step()
                data_tqdm.set_description(str(loss_averager))
        except KeyboardInterrupt:
            save_model(
                target_dir=target_dir,
                epoch=epoch + 1,
                prefix="training_dump",
                model=non_aligned_i2i_module,
                optimizer=optimizer,
            )
            return
        generate_new_variant()
        save_model(
            target_dir=target_dir,
            epoch=epoch + 1,
            prefix="training",
            model=non_aligned_i2i_module,
            optimizer=optimizer,
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

                (deformed_output, cross_modality_deformed_label,) = non_aligned_i2i(
                    input_volume=input_image,
                    label_volume=label_image,
                    input_mask=input_mask,
                    label_mask=label_mask,
                    forward_random_transformation=random_deformation,
                    desired_outputs=(
                        "commutation_deformed_output" if use_implicit_di else "deformed_output",
                        "cross_modality_deformed_label",
                    ),
                )
                intra_modality_similarity_validation_loss = calculate_output_similarity_loss(
                    cross_modality_deformed_label.volume.detach(),
                    deformed_output.volume,
                    MaskedVolume.get_comparison_mask(
                        cross_modality_deformed_label, deformed_output
                    ),
                )
                validation_loss_averager.count(
                    {
                        "intra_sim": float(intra_modality_similarity_validation_loss),
                    }
                )
            validation_loss_averager.save_to_json(
                epoch=epoch + 1, filename=join(target_dir, "validation_loss_history.json")
            )


if __name__ == "__main__":
    training_func: TrainingFunction = _train
    obtain_arguments_and_train(training_func)
