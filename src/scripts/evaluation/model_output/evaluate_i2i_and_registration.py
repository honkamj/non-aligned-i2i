"""Evaluation script for combined registration and I2I"""


from os.path import isfile, join
from typing import Any, Iterable, Mapping, Optional, Sequence

from json import load as json_load
from matplotlib.pyplot import (colorbar, gca, imshow, show,  # type: ignore
                               subplot, title)
from torch import abs as torch_abs, device, float32, no_grad, tensor
from torch import flatten
from torch import max as torch_max
from torch import mean, sqrt, square
from torch import sum as torch_sum
from tqdm import tqdm  # type: ignore

from algorithm.torch.deformation.masked import MaskedDeformation, MaskedVolume
from data.interface import init_generic_evaluation_data_loader
from metrics.normalized_mutual_information import normalized_mutual_information
from metrics.peak_signal_to_noise_ratio import PSNRMassFunction
from metrics.structural_similarity_index import structural_similarity_index
from model.interface import init_non_aligned_i2i_model
from util.data import denormalize_masked_volume as denormalize
from util.deformation import visualize_deformation_2d
from util.training import (EvaluatingFunction, LossAverager,
                           load_model, obtain_arguments_and_evaluate)


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
        devices: Sequence[device]
    ) -> None:
    if full_patch:
        target_file_name = join(target_dir, f'{data_set}_full_patch_evaluation_metrics.json')
    else:
        target_file_name = join(target_dir, f'{data_set}_evaluation_metrics.json')
    if isfile(target_file_name):
        with open(target_file_name, mode='r', encoding='utf-8') as target_file:
            evaluations = json_load(target_file)
    else:
        evaluations = {}
    torch_device = devices[0]
    if full_patch:
        config['data_loader']['inference_batch_size'] = 1
    data_loader = init_generic_evaluation_data_loader(
        data_config=config['data'],
        data_loader_config=config['data_loader'],
        shuffle=shuffle,
        data_set=data_set,
        seed=seed,
        sequence_item=sequence_item
    )
    voxel_size = tensor(config['data_loader']['voxel_size'], dtype=float32, device=torch_device)
    input_mean_and_std = config['data_loader']['normalization']['input_mean_and_std']
    label_mean_and_std = config['data_loader']['normalization']['label_mean_and_std']
    non_aligned_i2i, non_aligned_i2i_module = init_non_aligned_i2i_model(
        config['model'],
        config['data_loader'],
        devices=devices)
    non_aligned_i2i.to(torch_device)
    non_aligned_i2i.eval()
    for epoch in epochs:
        if str(epoch) in evaluations and skip_evaluated and sequence_item is None:
            print(f'Epoch {epoch} already evaluated.')
            continue
        print(f'Evaluating epoch {epoch}')
        load_model(target_dir, epoch, 'training', non_aligned_i2i_module, torch_device=torch_device)
        metric_averager = LossAverager()
        metric_averager.set_custom_mass_func(
            'PSNR',
            PSNRMassFunction(config['data_loader']['label_signal_range'])
        )
        metric_averager.set_custom_mass_func(
            'PSNR_label',
            PSNRMassFunction(config['data_loader']['label_signal_range'])
        )
        metric_averager.set_custom_mass_func(
            'PSNR_training',
            PSNRMassFunction(config['data_loader']['label_signal_range'])
        )
        data_tqdm = tqdm(data_loader)
        for (
                input_image,
                aligned_label_image,
                training_label_image,
                input_mask,
                aligned_label_mask,
                training_label_mask,
                label_deformation_ddf,
                patch_mask,
                evaluation_mask
            ) in data_tqdm:
            if (
                    (full_patch and (mean(input_mask) != 1.0)) or
                    torch_sum(input_mask) + torch_sum(training_label_mask) == 0
                ):
                continue
            batch_size = input_image.size(0)
            input_image = input_image.to(torch_device)
            aligned_label_image = aligned_label_image.to(torch_device)
            training_label_image = training_label_image.to(torch_device)
            input_mask = input_mask.to(torch_device)
            aligned_label_mask = aligned_label_mask.to(torch_device)
            training_label_mask = training_label_mask.to(torch_device)
            label_deformation_ddf = label_deformation_ddf.to(torch_device)
            patch_mask = patch_mask.to(torch_device)
            evaluation_mask = evaluation_mask.to(torch_device)
            (
                input_volume,
                label,
                output,
                forward_deformation,
                inverse_cross_modality_deformation,
                inverse_deformation,
                deformed_label,
                deformed_output,
                deformation_svf,
                rigidly_deformed_label
            ) = non_aligned_i2i(
                input_volume=input_image,
                label_volume=training_label_image,
                input_mask=input_mask,
                label_mask=training_label_mask,
                desired_outputs=(
                    'input',
                    'label',
                    'output',
                    'forward_deformation',
                    'inverse_cross_modality_deformation',
                    'inverse_deformation',
                    'cross_modality_deformed_label',
                    'deformed_output',
                    'cross_modality_deformation_svf',
                    'rigidly_deformed_label'
                )
            )
            aligned_label = MaskedVolume(
                volume=aligned_label_image,
                voxel_size=label.voxel_size,
                mask=aligned_label_mask)
            label_deformation = MaskedDeformation(
                deformation=label_deformation_ddf,
                voxel_size=voxel_size,
                mask=None)

            input_volume = denormalize(input_volume, input_mean_and_std)
            label = denormalize(label, label_mean_and_std)
            aligned_label = denormalize(aligned_label, label_mean_and_std)
            rigidly_deformed_label = denormalize(rigidly_deformed_label, label_mean_and_std)
            output = denormalize(output, label_mean_and_std)
            deformed_label = denormalize(deformed_label, label_mean_and_std)
            deformed_output = denormalize(deformed_output, label_mean_and_std)

            absolute_error = mean(
                torch_abs(
                    output.volume - aligned_label.volume
                ),
                dim=1,
                keepdim=True
            ) * MaskedVolume.get_comparison_mask(output, aligned_label)
            squared_error = mean(
                square(
                    output.volume - aligned_label.volume
                ),
                dim=1,
                keepdim=True
            ) * MaskedVolume.get_comparison_mask(output, aligned_label)
            training_absolute_error = mean(
                torch_abs(
                    deformed_label.volume - deformed_output.volume
                ),
                dim=1,
                keepdim=True
            ) * MaskedVolume.get_comparison_mask(deformed_label, deformed_output)
            training_squared_error = mean(
                square(
                    deformed_label.volume - deformed_output.volume
                ),
                dim=1,
                keepdim=True
            ) * MaskedVolume.get_comparison_mask(deformed_label, deformed_output)
            absolute_value_svf = sqrt(
                torch_sum(
                    square(deformation_svf),
                    dim=1,
                    keepdim=True
                )
            )
            inverse_deformation_comparison_mask = input_volume.generate_mask(
                label_deformation) * label.generate_mask()
            deformation_error_diff = sqrt(
                torch_sum(
                    square(
                        inverse_deformation.get_deformation() -
                        label_deformation.get_deformation()),
                    dim=1,
                    keepdim=True
                )
            ) * inverse_deformation_comparison_mask
            cross_deformation_error_diff = sqrt(
                torch_sum(
                    square(
                        inverse_cross_modality_deformation.get_deformation() -
                        label_deformation.get_deformation()),
                    dim=1,
                    keepdim=True
                )
            ) * inverse_deformation_comparison_mask

            n_dims = len(input_image.shape) - 2
            if plot_outputs:
                if n_dims != 2:
                    raise ValueError('Only dimensionality 2 is supported for plotting')
                for batch_index in range(batch_size):
                    subplot(3, 5, 1)
                    title('Prediction')
                    imshow(output.volume[batch_index].detach().cpu()\
                        .swapaxes(0, -1).swapaxes(0, 1).int())
                    subplot(3, 5, 2)
                    title('Ground truth label')
                    imshow(
                        aligned_label.volume[batch_index].detach().cpu()\
                            .swapaxes(0, -1).swapaxes(0, 1).int())
                    subplot(3, 5, 3)
                    title('Input')
                    imshow(input_volume.volume[batch_index].detach().cpu()\
                        .swapaxes(0, -1).swapaxes(0, 1).int())
                    subplot(3, 5, 4)
                    title('Training label')
                    imshow(label.volume[batch_index].detach().cpu()\
                        .swapaxes(0, -1).swapaxes(0, 1).int())
                    subplot(3, 5, 5)
                    title('Deformed label')
                    imshow(deformed_label.volume[batch_index].detach().cpu()\
                        .swapaxes(0, -1).swapaxes(0, 1).int())
                    subplot(3, 5, 6)
                    title('Deformed output')
                    imshow(deformed_output.volume[batch_index].detach().cpu()\
                        .swapaxes(0, -1).swapaxes(0, 1).int())
                    subplot(3, 5, 7)
                    title('Predicted cross-modality deformation')
                    gca().set_aspect('equal', adjustable='box')
                    gca().invert_yaxis()
                    visualize_deformation_2d(
                        inverse_cross_modality_deformation\
                            .get_deformation()[batch_index][None], 50, 50)
                    subplot(3, 5, 8)
                    title('Predicted deformation')
                    gca().set_aspect('equal', adjustable='box')
                    gca().invert_yaxis()
                    visualize_deformation_2d(
                        forward_deformation\
                            .get_deformation()[batch_index][None], 50, 50)
                    subplot(3, 5, 9)
                    title('Predicted inverse deformation')
                    gca().set_aspect('equal', adjustable='box')
                    gca().invert_yaxis()
                    visualize_deformation_2d(
                        inverse_deformation\
                            .get_deformation()[batch_index][None], 50, 50)
                    subplot(3, 5, 10)
                    title('Rigidly deformed label')
                    imshow(rigidly_deformed_label.volume[batch_index].detach().cpu()\
                        .swapaxes(0, -1).swapaxes(0, 1).int())
                    subplot(3, 5, 11)
                    title('Label deformation')
                    gca().set_aspect('equal', adjustable='box')
                    gca().invert_yaxis()
                    visualize_deformation_2d(
                        label_deformation\
                            .get_deformation()[batch_index][None], 50, 50)
                    subplot(3, 5, 12)
                    title('Image AE')
                    gca().set_aspect('equal', adjustable='box')
                    gca().invert_yaxis()
                    imshow(
                        absolute_error[batch_index].detach().cpu()\
                            .swapaxes(0, -1).swapaxes(0, 1),
                        cmap='gray')
                    colorbar()
                    subplot(3, 5, 13)
                    title('Training AE')
                    gca().set_aspect('equal', adjustable='box')
                    gca().invert_yaxis()
                    imshow(
                        training_absolute_error[batch_index].detach().cpu()\
                            .swapaxes(0, -1).swapaxes(0, 1),
                        cmap='gray')
                    colorbar()
                    subplot(3, 5, 14)
                    title('Deformation error diff')
                    gca().set_aspect('equal', adjustable='box')
                    gca().invert_yaxis()
                    imshow(
                        deformation_error_diff[batch_index].detach().cpu()\
                            .swapaxes(0, -1).swapaxes(0, 1),
                        cmap='gray')
                    colorbar()
                    subplot(3, 5, 15)
                    title('Deformation svf length')
                    gca().set_aspect('equal', adjustable='box')
                    gca().invert_yaxis()
                    imshow(
                        absolute_value_svf[batch_index].detach().cpu()\
                            .swapaxes(0, -1).swapaxes(0, 1),
                        cmap='gray')
                    colorbar()
                    show()

            ground_truth_comparison_mask = MaskedVolume.get_comparison_mask(
                output,
                aligned_label) * patch_mask * evaluation_mask
            n_ground_truth_comparison_mask_voxels = float(torch_sum(ground_truth_comparison_mask))
            aligned_middle_mask = MaskedVolume.get_comparison_mask(output, aligned_label)
            training_middle_mask = MaskedVolume.get_comparison_mask(deformed_label, deformed_output)
            n_training_middle_mask_voxels = float(torch_sum(training_middle_mask))
            n_deformation_comparison_mask_voxels = float(
                torch_sum(inverse_deformation_comparison_mask))
            squared_sum = float(torch_sum(squared_error * ground_truth_comparison_mask))
            absolute_sum = float(torch_sum(absolute_error * ground_truth_comparison_mask))
            training_squared_sum = float(torch_sum(training_squared_error))
            training_absolute_sum = float(torch_sum(training_absolute_error))

            output_np = output.volume.cpu().numpy()
            aligned_label_np = aligned_label.volume.cpu().numpy()
            input_np = input_volume.volume.cpu().numpy()
            deformed_output_np = deformed_output.volume.cpu().numpy()
            deformed_label_np = deformed_label.volume.cpu().numpy()
            ground_truth_comparison_mask_np = ground_truth_comparison_mask.cpu().numpy()
            training_middle_mask_np = training_middle_mask.cpu().numpy()
            aligned_middle_mask_np = aligned_middle_mask.cpu().numpy()

            (
                structural_similarity_sum,
                structural_similarity_averaging_mass
            ) = structural_similarity_index(
                aligned_label_np,
                output_np,
                content_mask=aligned_middle_mask_np,
                evaluation_mask=ground_truth_comparison_mask_np,
                data_range=config['data_loader']['label_signal_range']
            )
            (
                training_structural_similarity_sum,
                training_structural_similarity_averaging_mass
            ) = structural_similarity_index(
                deformed_output_np,
                deformed_label_np,
                content_mask=training_middle_mask_np,
                evaluation_mask=training_middle_mask_np,
                data_range=config['data_loader']['label_signal_range']
            )
            (
                nmi_sum,
                nmi_averaging_mass
            ) = normalized_mutual_information(
                label=input_np,
                output=output_np,
                mask=ground_truth_comparison_mask_np
            )
            metric_averager.count(
                {
                    'MSE': squared_sum
                },
                mass=n_ground_truth_comparison_mask_voxels
            )
            metric_averager.count(
                {
                    'MAE': absolute_sum
                },
                mass=n_ground_truth_comparison_mask_voxels
            )
            metric_averager.count(
                {
                    'NMI': nmi_sum
                },
                mass=nmi_averaging_mass
            )
            metric_averager.count(
                {
                    'PSNR': squared_sum
                },
                mass=n_ground_truth_comparison_mask_voxels
            )
            metric_averager.count(
                {
                    'SSIM': structural_similarity_sum
                },
                mass=structural_similarity_averaging_mass
            )
            metric_averager.count(
                {
                    'MSE_training': training_squared_sum
                },
                mass=n_training_middle_mask_voxels
            )
            metric_averager.count(
                {
                    'MAE_training': training_absolute_sum
                },
                mass=n_training_middle_mask_voxels
            )
            metric_averager.count(
                {
                    'PSNR_training': training_squared_sum
                },
                mass=n_training_middle_mask_voxels
            )
            metric_averager.count(
                {
                    'SSIM_training': training_structural_similarity_sum
                },
                mass=training_structural_similarity_averaging_mass
            )
            metric_averager.count(
                {
                    'MAE_def': float(
                        torch_sum(deformation_error_diff)
                    )
                },
                mass=n_deformation_comparison_mask_voxels
            )

            metric_averager.count(
                {
                    'cross_MAE_def': float(
                        torch_sum(cross_deformation_error_diff)
                    )
                },
                mass=n_deformation_comparison_mask_voxels
            )
            metric_averager.count(
                {
                    'ME_def_x': float(
                        torch_sum(
                            (
                                inverse_deformation_comparison_mask *
                                (
                                    inverse_deformation.get_deformation() -
                                    label_deformation.get_deformation())
                            )[:, 0]
                        )
                    )
                },
                mass=n_deformation_comparison_mask_voxels
            )
            metric_averager.count(
                {
                    'ME_def_y': float(
                        torch_sum(
                            (
                                inverse_deformation_comparison_mask *
                                (
                                    inverse_deformation.get_deformation() -
                                    label_deformation.get_deformation()
                                )
                            )[:, 1]
                        )
                    )
                },
                mass=n_deformation_comparison_mask_voxels
            )
            metric_averager.count(
                {
                    'mean_max_AE_def': float(
                        torch_sum(
                            torch_max(
                                flatten(
                                    deformation_error_diff,
                                    start_dim=1,
                                    end_dim=-1
                                ),
                                dim=1
                            )[0]
                        )
                    )
                },
                mass=float(batch_size)
            )
            data_tqdm.set_description(str(metric_averager))
        print(f'Epoch {epoch} evaluated, {metric_averager}')
        if sequence_item is None:
            metric_averager.save_to_json(
                epoch=epoch,
                filename=target_file_name,
                postfix='_full_patch' if full_patch else '')


if __name__ == "__main__":
    evaluating_function: EvaluatingFunction = _evaluate
    with no_grad():
        obtain_arguments_and_evaluate(evaluating_function)
