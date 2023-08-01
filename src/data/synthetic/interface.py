"""Interfaces to synthetic data functionality"""

from math import pi
from multiprocessing import get_context
from os import listdir
from os.path import isfile, join
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple, cast

from numpy import asarray as as_np_array
from numpy.random import RandomState
from torch.utils.data import DataLoader, Dataset

from algorithm.torch.simulated_elastic_deformation import (
    GaussianElasticDeformationDefinition,
    sample_random_gaussian_deformations,
)
from data.dataset import EvaluationDataset, TrainingDataset, SequenceDataset, training_worker_init_fn
from data.util import get_data_root

from .sequence import EvaluationSyntheticSequence, TrainingSyntheticSequence


def _ininitialize_label_deformation_parameters(
    label_simulated_deformation_config: Mapping[str, Any]
) -> Tuple[GaussianElasticDeformationDefinition, GaussianElasticDeformationDefinition]:
    lower_limit = GaussianElasticDeformationDefinition(
        width=label_simulated_deformation_config["width_range"]["lower"],
        magnitude=label_simulated_deformation_config["magnitude_range"]["lower"],
        center=label_simulated_deformation_config["center_range"]["lower"],
        rotation=[
            label_simulated_deformation_config["degree_range"]["lower"][dim] * pi / 180
            for dim in range(len(label_simulated_deformation_config["degree_range"]["lower"]))
        ],
        translation=label_simulated_deformation_config["translation_range"]["lower"],
    )
    upper_limit = GaussianElasticDeformationDefinition(
        width=label_simulated_deformation_config["width_range"]["upper"],
        magnitude=label_simulated_deformation_config["magnitude_range"]["upper"],
        center=label_simulated_deformation_config["center_range"]["upper"],
        rotation=[
            label_simulated_deformation_config["degree_range"]["upper"][dim] * pi / 180
            for dim in range(len(label_simulated_deformation_config["degree_range"]["upper"]))
        ],
        translation=label_simulated_deformation_config["translation_range"]["upper"],
    )
    return lower_limit, upper_limit


def _get_input_paths(data_root: str) -> Sequence[str]:
    return [
        join(data_root, file_name)
        for file_name in listdir(data_root)
        if isfile(join(data_root, file_name))
    ]


def init_synthetic_training_loader(
    data_config: Mapping[str, Any],
    data_loader_config: Mapping[str, Any],
    seed: int,
    data_set: str,
    num_workers: int,
) -> Tuple[DataLoader, Callable[[], None]]:
    """Initializes synthetic data loader with generate new variant function"""
    data_root = get_data_root(data_config["root"])
    lower_deformation_limit, upper_deformation_limit = _ininitialize_label_deformation_parameters(
        data_loader_config["label_simulated_deformation"]
    )
    input_paths = _get_input_paths(join(data_root, data_set))
    random_state = RandomState(seed)
    multiprocessing_context = get_context('spawn')
    dataset = TrainingDataset(
        sequence=TrainingSyntheticSequence(
            input_paths=input_paths,
            deformations=sample_random_gaussian_deformations(
                batch_size=len(input_paths),
                lower_limit=lower_deformation_limit,
                upper_limit=upper_deformation_limit,
                random_state=random_state,
            ),
            input_mean_and_std=as_np_array(
                data_loader_config["normalization"]["input_mean_and_std"]
            ),
            label_mean_and_std=as_np_array(
                data_loader_config["normalization"]["label_mean_and_std"]
            ),
            patch_size=cast(Tuple[int, int], tuple(data_loader_config["patch_size"])),
            paired=data_loader_config.get("paired", True),
        ),
        rotation_degree_range=data_loader_config.get("rotation_degree_range"),
        log_scale_scale=data_loader_config.get("log_scale_scale"),
        log_shear_scale=data_loader_config.get("log_shear_scale"),
        translation_range=data_loader_config.get("translation_range"),
        generate_flips=data_loader_config.get("generate_flips", False),
        generate_orthogonal_rotations=data_loader_config.get(
            "generate_orthogonal_rotations", False
        ),
        n_random_deformations=data_loader_config.get("n_random_deformations"),
        zero_random_deformation_prob=data_loader_config.get("zero_random_deformation_prob", 0.0),
        random_state=random_state,
        input_noise_amplitude_range=data_loader_config.get("input_noise_amplitude_range"),
        label_noise_amplitude_range=data_loader_config.get("label_noise_amplitude_range"),
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


def init_synthetic_evaluation_loader(
    data_config: Mapping[str, Any],
    data_loader_config: Mapping[str, Any],
    shuffle: bool,
    dataset: str,
    seed: int,
    sequence_item: Optional[int],
) -> DataLoader:
    """Initializes synthetic data loader for inference"""
    data_root = get_data_root(data_config["root"])
    lower_deformation_limit, upper_deformation_limit = _ininitialize_label_deformation_parameters(
        data_loader_config["label_simulated_deformation"]
    )
    input_paths = _get_input_paths(join(data_root, dataset))
    random_state = RandomState(seed)
    evaluation_dataset: Dataset = EvaluationDataset(
        sequence=EvaluationSyntheticSequence(
            input_paths=input_paths,
            deformations=sample_random_gaussian_deformations(
                batch_size=len(input_paths),
                lower_limit=lower_deformation_limit,
                upper_limit=upper_deformation_limit,
                random_state=random_state,
            ),
            input_mean_and_std=as_np_array(
                data_loader_config["normalization"]["input_mean_and_std"]
            ),
            label_mean_and_std=as_np_array(
                data_loader_config["normalization"]["label_mean_and_std"]
            ),
            patch_size=cast(Tuple[int, int], tuple(data_loader_config["patch_size"])),
        )
    )
    if sequence_item is not None:
        evaluation_dataset = SequenceDataset([evaluation_dataset[sequence_item]])
    data_loader = DataLoader(
        evaluation_dataset,
        batch_size=data_loader_config["inference_batch_size"],
        num_workers=data_loader_config["n_inference_workers"],
        drop_last=False,
        shuffle=shuffle,
        persistent_workers=True,
        prefetch_factor=4,
        multiprocessing_context=get_context('spawn'),
    )
    return data_loader
