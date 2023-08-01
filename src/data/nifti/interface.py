"""Interfaces to nifti data functionality"""

from multiprocessing import get_context
from os import listdir, makedirs
from os.path import basename, isdir, join
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple

from numpy import asarray as as_np_array
from numpy.random import RandomState
from torch.utils.data import DataLoader, Dataset

from data.dataset import (
    EvaluationDataset,
    InferenceDataset,
    SequenceDataset,
    TrainingDataset,
    training_worker_init_fn,
)
from data.util import get_data_root
from util.data import optionally_as_ndarray

from .sequence import (
    EvaluationNiftiPatchSequence,
    InferenceNiftiPatchSequence,
    TrainingNiftiPatchSequence,
)


def get_nifti_samples(root_directory: str, dataset: str) -> Sequence[str]:
    """Get paths to nifti files in folder structure"""
    train_input_folder = join(root_directory, dataset)
    return [
        folder_name
        for folder_name in listdir(train_input_folder)
        if isdir(join(train_input_folder, folder_name))
    ]


def get_nifti_paths(
    root_directory: str, postfixes: Mapping[str, Any], dataset: str
) -> Mapping[str, Sequence[str]]:
    """Get paths to nifti files in folder structure"""
    train_input_folder = join(root_directory, dataset)
    samples = get_nifti_samples(root_directory, dataset)
    input_paths = [
        join(train_input_folder, sample, f'{sample}_{postfixes["input"]}') for sample in samples
    ]
    aligned_label_paths = [
        join(train_input_folder, sample, f'{sample}_{postfixes["label_aligned"]}')
        for sample in samples
    ]
    non_aligned_label_paths = [
        join(train_input_folder, sample, f'{sample}_{postfixes["label_non_aligned"]}')
        for sample in samples
    ]
    training_label_paths = [
        join(train_input_folder, sample, f'{sample}_{postfixes["label_training"]}')
        for sample in samples
    ]
    input_mask_paths = [
        join(train_input_folder, sample, f'{sample}_{postfixes["input_mask"]}')
        for sample in samples
    ]
    aligned_label_mask_paths = [
        join(train_input_folder, sample, f'{sample}_{postfixes["label_aligned_mask"]}')
        for sample in samples
    ]
    non_aligned_label_mask_paths = [
        join(train_input_folder, sample, f'{sample}_{postfixes["label_non_aligned_mask"]}')
        for sample in samples
    ]
    training_label_mask_paths = [
        join(train_input_folder, sample, f'{sample}_{postfixes["label_training_mask"]}')
        for sample in samples
    ]
    evaluation_mask_paths = [
        join(train_input_folder, sample, f'{sample}_{postfixes["evaluation_mask"]}')
        for sample in samples
    ]
    paths = {
        "input": input_paths,
        "label_aligned": aligned_label_paths,
        "label_non_aligned": non_aligned_label_paths,
        "label_training": training_label_paths,
        "input_mask": input_mask_paths,
        "label_aligned_mask": aligned_label_mask_paths,
        "label_non_aligned_mask": non_aligned_label_mask_paths,
        "label_training_mask": training_label_mask_paths,
        "evaluation_mask": evaluation_mask_paths,
    }
    if "deformation" in postfixes:
        paths["ground_truth_deformation"] = [
            join(train_input_folder, sample, f'{sample}_{postfixes["deformation"]}')
            for sample in samples
        ]
    return paths


def init_nifti_training_loader(
    data_config: Mapping[str, Any],
    data_loader_config: Mapping[str, Any],
    seed: int,
    data_set: str,
    num_workers: int,
) -> Tuple[DataLoader, Callable[[], None]]:
    """Initializes nifti data loader with generate new variant function"""
    paths = get_nifti_paths(get_data_root(data_config["root"]), data_config["postfixes"], data_set)
    multiprocessing_context = get_context("spawn")
    dataset = TrainingDataset(
        sequence=TrainingNiftiPatchSequence(
            input_paths=paths["input"],
            label_paths=paths["label_training"],
            input_mask_paths=paths["input_mask"],
            label_mask_paths=paths["label_training_mask"],
            input_mean_and_std=as_np_array(
                data_loader_config["normalization"]["input_mean_and_std"]
            ),
            label_mean_and_std=as_np_array(
                data_loader_config["normalization"]["label_mean_and_std"]
            ),
            input_min_and_max=optionally_as_ndarray(
                data_loader_config["normalization"].get("input_min_and_max")
            ),
            label_min_and_max=optionally_as_ndarray(
                data_loader_config["normalization"].get("label_min_and_max")
            ),
            stride=data_loader_config["training_theoretical_stride"],
            patch_size=data_loader_config["patch_size"],
            min_input_mask_ratio=data_loader_config["min_input_mask_ratio"],
            min_label_mask_ratio=data_loader_config["min_label_mask_ratio"],
            shuffling_cluster_size=data_loader_config["shuffling_cluster_size"],
            paired=data_loader_config.get("paired", True),
            mask_threshold=data_loader_config.get("mask_smaller_or_equal_values_as_invalid"),
        ),
        rotation_degree_range=data_loader_config.get("rotation_degree_range"),
        log_scale_scale=data_loader_config.get("log_scale_scale"),
        log_shear_scale=data_loader_config.get("log_shear_scale"),
        translation_range=data_loader_config.get("translation_range"),
        zero_random_deformation_prob=data_loader_config.get("zero_random_deformation_prob", 0.0),
        input_noise_amplitude_range=data_loader_config.get("input_noise_amplitude_range"),
        label_noise_amplitude_range=data_loader_config.get("label_noise_amplitude_range"),
        n_random_deformations=data_loader_config.get("n_random_deformations"),
        generate_flips=data_loader_config.get("generate_flips", False),
        generate_orthogonal_rotations=data_loader_config.get(
            "generate_orthogonal_rotations", False
        ),
        random_state=RandomState(seed),
        voxel_size=data_loader_config["voxel_size"],
        multiprocessing_context=multiprocessing_context,
    )

    data_loader = DataLoader(
        dataset,
        batch_size=data_loader_config["batch_size"],
        num_workers=num_workers,
        worker_init_fn=training_worker_init_fn,
        drop_last=True,
        persistent_workers=True,
        multiprocessing_context=multiprocessing_context,
    )
    return data_loader, dataset.generate_new_variant


def init_nifti_inference_loader(
    sample: str, data_config: Mapping[str, Any], data_loader_config: Mapping[str, Any]
) -> DataLoader:
    """Initialize tiff data loader for inference"""
    data_root = get_data_root(data_config["root"])
    input_path = join(data_root, sample, f"{basename(sample)}_{data_config['postfixes']['input']}")
    mask_path = join(
        data_root, sample, f"{basename(sample)}_{data_config['postfixes']['input_mask']}"
    )
    dataset = InferenceDataset(
        sequence=InferenceNiftiPatchSequence(
            input_paths=[input_path],
            mask_paths=[mask_path],
            input_mean_and_std=as_np_array(
                data_loader_config["normalization"]["input_mean_and_std"]
            ),
            input_min_and_max=optionally_as_ndarray(
                data_loader_config["normalization"].get("input_min_and_max")
            ),
            stride=data_loader_config["inference_stride"],
            patch_size=data_loader_config["patch_size"],
            fusing_mask_smoothing=data_loader_config.get("fusing_mask_smoothing", 0.0),
            mask_threshold=data_loader_config.get("mask_smaller_or_equal_values_as_invalid"),
        )
    )
    data_loader = DataLoader(
        dataset,
        batch_size=data_loader_config["inference_batch_size"],
        num_workers=data_loader_config["n_inference_workers"],
        persistent_workers=True,
        multiprocessing_context=get_context("spawn"),
    )
    return data_loader


def init_nifti_evaluation_loader(
    data_config: Mapping[str, Any],
    data_loader_config: Mapping[str, Any],
    shuffle: bool,
    dataset: str,
    _seed: int,
    sequence_item: Optional[int],
) -> DataLoader:
    """Initializes nifti data loader for inference"""
    paths = get_nifti_paths(get_data_root(data_config["root"]), data_config["postfixes"], dataset)
    evaluation_config = data_loader_config.get("evaluation", {})
    if evaluation_config.get("affine_transformation_cache_path") is None:
        affine_transformation_cache_path = None
    else:
        affine_transformation_cache_path = join(
            get_data_root(evaluation_config["affine_transformation_cache_path"]), dataset
        )
        makedirs(affine_transformation_cache_path, exist_ok=True)
    evaluation_dataset: Dataset = EvaluationDataset(
        sequence=EvaluationNiftiPatchSequence(
            input_paths=paths["input"],
            aligned_label_paths=paths["label_aligned"],
            non_aligned_label_paths=paths["label_non_aligned"],
            training_label_paths=paths["label_training"],
            input_mask_paths=paths["input_mask"],
            aligned_label_mask_paths=paths["label_aligned_mask"],
            non_aligned_label_mask_paths=paths["label_non_aligned_mask"],
            training_label_mask_paths=paths["label_training_mask"],
            evaluation_mask_paths=paths["evaluation_mask"],
            ground_truth_deformation_paths=paths.get("ground_truth_deformation"),
            input_mean_and_std=as_np_array(
                data_loader_config["normalization"]["input_mean_and_std"]
            ),
            label_mean_and_std=as_np_array(
                data_loader_config["normalization"]["label_mean_and_std"]
            ),
            input_min_and_max=optionally_as_ndarray(
                data_loader_config["normalization"].get("input_min_and_max")
            ),
            label_min_and_max=optionally_as_ndarray(
                data_loader_config["normalization"].get("label_min_and_max")
            ),
            stride=data_loader_config["inference_stride"],
            patch_size=data_loader_config["patch_size"],
            mask_threshold=data_loader_config.get("mask_smaller_or_equal_values_as_invalid"),
            voxel_size=data_loader_config["voxel_size"],
            use_affinely_registered_non_aligned_label_as_ground_truth=evaluation_config.get(
                "use_affinely_registered_non_aligned_label_as_ground_truth", False
            ),
            affine_transformation_cache_path=affine_transformation_cache_path,
            affine_registration_seed=evaluation_config.get("affine_registration_seed"),
        )
    )
    if sequence_item is not None:
        evaluation_dataset = SequenceDataset([evaluation_dataset[sequence_item]])
    data_loader = DataLoader(
        evaluation_dataset,
        batch_size=data_loader_config["inference_batch_size"],
        num_workers=data_loader_config["n_inference_workers"],
        persistent_workers=True,
        shuffle=shuffle,
        multiprocessing_context=get_context("spawn"),
    )
    return data_loader
