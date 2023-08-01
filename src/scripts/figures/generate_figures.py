"""Generate figures for paper"""

from typing import Any, Mapping, Optional, Sequence

from matplotlib.pyplot import axis, close, figure, gca, imshow, savefig # type: ignore
from torch import device  # type: ignore
from tqdm import tqdm  # type: ignore

from data.interface import init_generic_training_data_loader
from model.interface import init_non_aligned_i2i_model
from util.data import denormalize_masked_volume
from util.deformation import visualize_deformation_2d
from util.training import (TrainingFunction, load_model,
                           obtain_arguments_and_train)


def _generate_plots(
        config: Mapping[str, Any],
        target_dir: str,
        seed: int,
        continue_from_epoch: Optional[int],
        devices: Sequence[device]) -> None:
    torch_device = devices[0]

    non_aligned_i2i, non_aligned_i2i_module = init_non_aligned_i2i_model(
        config['model'],
        config['data_loader'],
        devices=devices)
    non_aligned_i2i.to(torch_device)

    config['data_loader']['batch_size'] = 1
    data_loader, generate_new_variant = init_generic_training_data_loader(
        config['data'],
        config['data_loader'],
        seed,
        'train',
        config['data_loader']['n_workers'])
    input_mean_and_std = config['data_loader']['normalization']['input_mean_and_std']
    label_mean_and_std = config['data_loader']['normalization']['label_mean_and_std']
    if continue_from_epoch is None:
        initial_epoch = 0
    else:
        initial_epoch = load_model(
            target_dir=target_dir,
            epoch=continue_from_epoch,
            prefix='training',
            model=non_aligned_i2i_module,
            torch_device=torch_device)
    for _ in range(initial_epoch):
        generate_new_variant()
    non_aligned_i2i.train()
    data_tqdm = tqdm(data_loader, leave=False)
    for (
        input_image,
        label_image,
        input_mask,
        label_mask,
        random_deformation
    ) in data_tqdm:
        input_image = input_image.to(torch_device)
        label_image = label_image.to(torch_device)
        input_mask = input_mask.to(torch_device)
        label_mask = label_mask.to(torch_device)
        random_deformation = random_deformation.to(torch_device)
        (
            input_volume,
            label_volume,
            output,
            left_random_deformed_output,
            random_deformed_input,
            cross_modality_deformed_label,
            random_cross_modality_deformed_label,
            random_commutation_deformed_output,
            inverse_elastic_cross_modality_deformation,
            inverse_rigid_transformation,
            forward_intra_modality_deformation
        ) = non_aligned_i2i(
            input_volume=input_image,
            label_volume=label_image,
            input_mask=input_mask,
            label_mask=label_mask,
            forward_random_transformation=random_deformation,
            desired_outputs=(
                'input',
                'label',
                'output',
                'left_random_deformed_output',
                'random_deformed_input',
                'cross_modality_deformed_label',
                'random_cross_modality_deformed_label',
                'random_commutation_deformed_output',
                'inverse_elastic_cross_modality_deformation',
                'inverse_rigid_transformation',
                'forward_intra_modality_deformation'
            )
        )
        input_volume = denormalize_masked_volume(
            input_volume,
            input_mean_and_std)
        label_volume = denormalize_masked_volume(
            label_volume,
            label_mean_and_std)
        output = denormalize_masked_volume(
            output,
            label_mean_and_std)
        left_random_deformed_output = denormalize_masked_volume(
            left_random_deformed_output,
            label_mean_and_std)
        random_deformed_input = denormalize_masked_volume(
            random_deformed_input,
            input_mean_and_std)
        cross_modality_deformed_label = denormalize_masked_volume(
            cross_modality_deformed_label,
            label_mean_and_std)
        random_cross_modality_deformed_label = denormalize_masked_volume(
            random_cross_modality_deformed_label,
            label_mean_and_std)
        random_commutation_deformed_output = denormalize_masked_volume(
            random_commutation_deformed_output,
            label_mean_and_std)

        inverse_cross_modality_deformation =\
            inverse_rigid_transformation.compose(inverse_elastic_cross_modality_deformation)
        inverse_cross_modality_deformation_ddf =\
            inverse_cross_modality_deformation.get_deformation()
        forward_intra_modality_deformation_ddf =\
            forward_intra_modality_deformation.get_deformation()
        inverse_elastic_cross_modality_deformation_ddf =\
            inverse_elastic_cross_modality_deformation.get_deformation()

        images = [
            (input_volume, 'example_input.png'),
            (label_volume, 'example_training_label.png'),
            (output, 'example_prediction.png'),
            (left_random_deformed_output, 'example_left_random_deformed_prediction.png'),
            (random_deformed_input, 'example_random_deformed_input.png'),
            (cross_modality_deformed_label, 'example_cross_modality_deformed_label.png'),
            (
                random_cross_modality_deformed_label,
                'example_random_cross_modality_deformed_label.png'),
            (random_commutation_deformed_output, 'example_random_commutation_deformed_output.png')
        ]
        deformations = [
            (inverse_cross_modality_deformation_ddf, 'example_cross_modality_deformation.svg'),
            (forward_intra_modality_deformation_ddf, 'example_intra_modality_deformation.svg'),
            (
                inverse_elastic_cross_modality_deformation_ddf,
                'example_elastic_cross_modality_deformation.svg')
        ]
        for volume, filename in images:
            figure(figsize=(10, 10))
            imshow(volume.volume[0].detach().cpu().swapaxes(0, -1).swapaxes(0, 1).int())
            axis('off')
            savefig(f"figures/{filename}", bbox_inches='tight', pad_inches=0)
            close()
        for deformation, filename in deformations:
            figure(figsize=(10, 10))
            gca().set_aspect('equal', adjustable='box')
            gca().invert_yaxis()
            visualize_deformation_2d(
                deformation[0][None], 20, 20)
            axis('off')
            savefig(f"figures/{filename}", bbox_inches='tight', pad_inches=0)
            close()
        input("Press Enter to continue...")


if __name__ == "__main__":
    plot_generation_func: TrainingFunction = _generate_plots
    obtain_arguments_and_train(plot_generation_func)
