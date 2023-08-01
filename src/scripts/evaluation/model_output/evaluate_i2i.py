"""Evaluation script for baseline I2I"""

from json import load as json_load
from os.path import isfile, join
from typing import Any, Iterable, Mapping, Optional, Sequence

from matplotlib.pyplot import gca, imshow, show, subplot, title  # type: ignore
from torch import abs as torch_abs, device
from torch import mean, no_grad, square
from torch import sum as torch_sum
from tqdm import tqdm  # type: ignore

from data.interface import init_generic_evaluation_data_loader
from metrics.normalized_mutual_information import normalized_mutual_information
from metrics.peak_signal_to_noise_ratio import PSNRMassFunction
from metrics.structural_similarity_index import structural_similarity_index
from model.inference import get_generic_i2i_inference_function
from util.data import denormalize_tensor as denormalize
from util.deformation import visualize_deformation_2d
from util.training import EvaluatingFunction, LossAverager, obtain_arguments_and_evaluate


def _evaluate(
    config: Mapping[str, Any],
    target_dir: str,
    seed: int,
    epochs: Iterable[int],
    plot_outputs: bool,
    sequence_item: Optional[int],
    shuffle: bool,
    data_set: str,
    full_patch: bool,
    skip_evaluated: bool,
    devices: Sequence[device],
) -> None:
    if full_patch:
        target_file_name = join(target_dir, f"{data_set}_full_patch_evaluation_metrics.json")
    else:
        target_file_name = join(target_dir, f"{data_set}_evaluation_metrics.json")
    if isfile(target_file_name):
        with open(target_file_name, mode="r", encoding="utf-8") as target_file:
            evaluations = json_load(target_file)
    else:
        evaluations = {}
    torch_device = devices[0]
    if full_patch:
        config["data_loader"]["inference_batch_size"] = 1
    data_loader = init_generic_evaluation_data_loader(
        data_config=config["data"],
        data_loader_config=config["data_loader"],
        shuffle=shuffle,
        data_set=data_set,
        seed=seed,
        sequence_item=sequence_item,
    )
    for epoch in epochs:
        if str(epoch) in evaluations and skip_evaluated and sequence_item is None:
            print(f"Epoch {epoch} already evaluated.")
            continue
        print(f"Evaluating epoch {epoch}")
        inference_function = get_generic_i2i_inference_function(
            model_config=config["model"],
            data_loader_config=config["data_loader"],
            devices=devices,
            epoch=epoch,
            target_dir=target_dir,
        )
        data_tqdm = tqdm(data_loader)
        metric_averager = LossAverager()
        metric_averager.set_custom_mass_func(
            "PSNR", PSNRMassFunction(config["data_loader"]["label_signal_range"])
        )
        metric_averager.set_custom_mass_func(
            "PSNR_training", PSNRMassFunction(config["data_loader"]["label_signal_range"])
        )
        for (
            input_image,
            aligned_label_image,
            training_label_image,
            input_mask,
            aligned_label_mask,
            training_label_mask,
            label_deformation,
            patch_mask,
            evaluation_mask,
        ) in data_tqdm:
            if (full_patch and (mean(input_mask) != 1.0)) or torch_sum(input_mask) + torch_sum(
                training_label_mask
            ) == 0:
                continue
            batch_size = input_image.size(0)
            input_image = input_image.to(torch_device)
            aligned_label_image = aligned_label_image.to(torch_device)
            training_label_image = training_label_image.to(torch_device)
            input_mask = input_mask.to(torch_device)
            aligned_label_mask = aligned_label_mask.to(torch_device)
            training_label_mask = training_label_mask.to(torch_device)
            label_deformation = label_deformation.to(torch_device)
            patch_mask = patch_mask.to(torch_device)
            evaluation_mask = evaluation_mask.to(torch_device)

            output = inference_function(input_image)

            output = denormalize(
                image=output,
                mean_and_std=config["data_loader"]["normalization"]["label_mean_and_std"],
                mask=input_mask,
            )
            input_volume = denormalize(
                image=input_image,
                mean_and_std=config["data_loader"]["normalization"]["input_mean_and_std"],
                mask=input_mask,
            )
            aligned_label = denormalize(
                image=aligned_label_image,
                mean_and_std=config["data_loader"]["normalization"]["label_mean_and_std"],
                mask=aligned_label_mask,
            )
            training_label = denormalize(
                image=training_label_image,
                mean_and_std=config["data_loader"]["normalization"]["label_mean_and_std"],
                mask=training_label_mask,
            )
            absolute_error = (
                input_mask
                * aligned_label_mask
                * mean(torch_abs(output - aligned_label), dim=1, keepdim=True)
            )
            squared_error = (
                input_mask
                * aligned_label_mask
                * mean(square(output - aligned_label), dim=1, keepdim=True)
            )
            absolute_error_training = (
                input_mask
                * training_label_mask
                * mean(torch_abs(output - training_label), dim=1, keepdim=True)
            )
            squared_error_training = (
                input_mask
                * training_label_mask
                * mean(square(output - training_label), dim=1, keepdim=True)
            )
            if plot_outputs:
                for batch_index in range(batch_size):
                    subplot(2, 3, 1)
                    title("Prediction")
                    imshow(output[batch_index].detach().cpu().swapaxes(0, -1).swapaxes(0, 1).int())
                    subplot(2, 3, 2)
                    title("Ground truth label")
                    imshow(
                        aligned_label[batch_index]
                        .detach()
                        .cpu()
                        .swapaxes(0, -1)
                        .swapaxes(0, 1)
                        .int()
                    )
                    subplot(2, 3, 3)
                    title("Input")
                    imshow(
                        input_volume[batch_index]
                        .detach()
                        .cpu()
                        .swapaxes(0, -1)
                        .swapaxes(0, 1)
                        .int()
                    )
                    subplot(2, 3, 4)
                    title("Training label (training)")
                    imshow(
                        training_label[batch_index]
                        .detach()
                        .cpu()
                        .swapaxes(0, -1)
                        .swapaxes(0, 1)
                        .int()
                    )
                    subplot(2, 3, 5)
                    title("Label deformation")
                    gca().set_aspect("equal", adjustable="box")
                    gca().invert_yaxis()
                    visualize_deformation_2d(label_deformation[batch_index][None], 50, 50)
                    show()

            training_middle_mask = input_mask * training_label_mask
            aligned_middle_mask = input_mask * aligned_label_mask
            ground_truth_comparison_mask = (
                input_mask * aligned_label_mask * patch_mask * evaluation_mask
            )
            n_training_middle_mask_voxels = float(torch_sum(training_middle_mask))
            n_ground_truth_comparison_mask_voxels = float(torch_sum(ground_truth_comparison_mask))
            squared_sum = float(torch_sum(squared_error * ground_truth_comparison_mask))
            absolute_sum = float(torch_sum(absolute_error * ground_truth_comparison_mask))
            training_squared_sum = float(torch_sum(squared_error_training))
            training_absolute_sum = float(torch_sum(absolute_error_training))
            output_np = output.cpu().numpy()
            aligned_label_np = aligned_label.cpu().numpy()
            training_label_np = training_label.cpu().numpy()
            input_np = input_volume.cpu().numpy()
            aligned_middle_mask_np = aligned_middle_mask.cpu().numpy()
            training_middle_mask_np = training_middle_mask.cpu().numpy()
            ground_truth_comparison_mask_np = ground_truth_comparison_mask.cpu().numpy()
            (
                structural_similarity_sum,
                structural_similarity_averaging_mass,
            ) = structural_similarity_index(
                aligned_label_np,
                output_np,
                content_mask=aligned_middle_mask_np,
                evaluation_mask=ground_truth_comparison_mask_np,
                data_range=config["data_loader"]["label_signal_range"],
            )
            (
                training_structural_similarity_sum,
                training_structural_similarity_averaging_mass,
            ) = structural_similarity_index(
                training_label_np,
                output_np,
                content_mask=training_middle_mask_np,
                evaluation_mask=training_middle_mask_np,
                data_range=config["data_loader"]["label_signal_range"],
            )
            (nmi_sum, nmi_averaging_mass) = normalized_mutual_information(
                label=input_np, output=output_np, mask=ground_truth_comparison_mask_np
            )
            metric_averager.count({"MSE": squared_sum}, mass=n_ground_truth_comparison_mask_voxels)
            metric_averager.count({"MAE": absolute_sum}, mass=n_ground_truth_comparison_mask_voxels)
            metric_averager.count({"NMI": nmi_sum}, mass=nmi_averaging_mass)
            metric_averager.count({"PSNR": squared_sum}, mass=n_ground_truth_comparison_mask_voxels)
            metric_averager.count(
                {"SSIM": structural_similarity_sum}, mass=structural_similarity_averaging_mass
            )
            metric_averager.count(
                {"MSE_training": training_squared_sum}, mass=n_training_middle_mask_voxels
            )
            metric_averager.count(
                {"MAE_training": training_absolute_sum}, mass=n_training_middle_mask_voxels
            )
            metric_averager.count(
                {"PSNR_training": training_squared_sum}, mass=n_training_middle_mask_voxels
            )
            metric_averager.count(
                {"SSIM_training": training_structural_similarity_sum},
                mass=training_structural_similarity_averaging_mass,
            )
            data_tqdm.set_description(str(metric_averager))
        print(f"Epoch {epoch} evaluated, {metric_averager}")
        if sequence_item is None:
            metric_averager.save_to_json(
                epoch=epoch, filename=target_file_name, postfix="_full_patch" if full_patch else ""
            )


if __name__ == "__main__":
    evaluating_function: EvaluatingFunction = _evaluate
    with no_grad():
        obtain_arguments_and_evaluate(evaluating_function)
