"""Interfaces to synthetic data functionality"""

from multiprocessing import get_context
from typing import Any, Callable, Mapping, Tuple, cast

from numpy.random import RandomState
from torch.utils.data import DataLoader

from data.dataset import TrainingDataset, training_worker_init_fn

from .sequence import TrainingDummySequence


def init_dummy_training_loader(
    _data_config: Mapping[str, Any],
    data_loader_config: Mapping[str, Any],
    seed: int,
    _data_set: str,
    num_workers: int,
) -> Tuple[DataLoader, Callable[[], None]]:
    """Initializes synthetic data loader with generate new variant function"""
    random_state = RandomState(seed)
    multiprocessing_context = get_context('spawn')
    dataset = TrainingDataset(
        sequence=TrainingDummySequence(
            patch_size=cast(Tuple[int, int], tuple(data_loader_config["patch_size"])), length=100
        ),
        rotation_degree_range=data_loader_config.get("rotation_degree_range"),
        log_scale_scale=data_loader_config.get("log_scale_scale"),
        log_shear_scale=data_loader_config.get("log_shear_scale"),
        translation_range=data_loader_config.get("translation_range"),
        zero_random_deformation_prob=data_loader_config.get("zero_random_deformation_prob", 0.0),
        generate_flips=data_loader_config.get("generate_flips", False),
        generate_orthogonal_rotations=data_loader_config.get(
            "generate_orthogonal_rotations", False
        ),
        n_random_deformations=data_loader_config.get("n_random_deformations"),
        input_noise_amplitude_range=data_loader_config.get("input_noise_amplitude_range"),
        label_noise_amplitude_range=data_loader_config.get("label_noise_amplitude_range"),
        random_state=random_state,
        voxel_size=(1.0, 1.0),
        multiprocessing_context=multiprocessing_context,
    )

    data_loader = DataLoader(
        dataset,
        batch_size=data_loader_config["batch_size"],
        num_workers=num_workers,
        worker_init_fn=training_worker_init_fn,
        drop_last=True,
        persistent_workers=True,
        prefetch_factor=4,
        multiprocessing_context=multiprocessing_context,
    )
    return data_loader, dataset.generate_new_variant
