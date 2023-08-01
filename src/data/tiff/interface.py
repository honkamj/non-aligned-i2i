"""Interfaces to tiff data functionality"""

from multiprocessing import get_context
from os import makedirs
from os.path import join
from typing import Any, Callable, Mapping, Optional, Tuple

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
from data.util import get_data_root, obtain_paths
from util.data import optionally_as_ndarray

from .sequence import (
    EvaluationTiffPatchSequence,
    InferenceTiffPatchSequence,
    TrainingTiffPatchSequence,
)


def init_tiff_training_loader(
    data_config: Mapping[str, Any],
    data_loader_config: Mapping[str, Any],
    seed: int,
    data_set: str,
    num_workers: int,
) -> Tuple[DataLoader, Callable[[], None]]:
    """Initializes tiff data loader with generate new variant function"""
    type_to_paths = obtain_paths(
        data_root=get_data_root(data_config["root"]),
        type_to_postfix=data_config["postfixes"],
        samples=data_config["datasets"][data_set],
    )
    multiprocessing_context = get_context("spawn")
    dataset = TrainingDataset(
        sequence=TrainingTiffPatchSequence(
            input_paths=type_to_paths["input"],
            label_paths=type_to_paths["label_training"],
            input_mask_paths=type_to_paths["input_mask"],
            label_mask_paths=type_to_paths["label_training_mask"],
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
            mask_threshold=data_loader_config.get("mask_smaller_or_equal_values_as_invalid"),
            paired=data_loader_config.get("paired", True),
        ),
        rotation_degree_range=data_loader_config.get("rotation_degree_range"),
        log_scale_scale=data_loader_config.get("log_scale_scale"),
        log_shear_scale=data_loader_config.get("log_shear_scale"),
        translation_range=data_loader_config.get("translation_range"),
        n_random_deformations=data_loader_config.get("n_random_deformations"),
        zero_random_deformation_prob=data_loader_config.get("zero_random_deformation_prob", 0.0),
        generate_flips=data_loader_config.get("generate_flips", False),
        generate_orthogonal_rotations=data_loader_config.get(
            "generate_orthogonal_rotations", False
        ),
        input_noise_amplitude_range=data_loader_config.get("input_noise_amplitude_range"),
        label_noise_amplitude_range=data_loader_config.get("label_noise_amplitude_range"),
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
        prefetch_factor=2,
        multiprocessing_context=multiprocessing_context,
    )
    return data_loader, dataset.generate_new_variant


def init_tiff_inference_loader(
    sample: str, data_config: Mapping[str, Any], data_loader_config: Mapping[str, Any]
) -> DataLoader:
    """Initialize tiff data loader for inference"""
    type_to_paths = obtain_paths(
        data_root=get_data_root(data_config["root"]),
        type_to_postfix=data_config["postfixes"],
        samples=[sample],
    )
    dataset = InferenceDataset(
        sequence=InferenceTiffPatchSequence(
            input_paths=type_to_paths["input"],
            mask_paths=type_to_paths["input_mask"],
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


def init_tiff_evaluation_loader(
    data_config: Mapping[str, Any],
    data_loader_config: Mapping[str, Any],
    shuffle: bool,
    dataset: str,
    _seed: int,
    sequence_item: Optional[int],
) -> DataLoader:
    """Initializes synthetic data loader for inference"""
    type_to_paths = obtain_paths(
        data_root=get_data_root(data_config["root"]),
        type_to_postfix=data_config["postfixes"],
        samples=data_config["datasets"][dataset],
    )
    evaluation_config = data_loader_config.get("evaluation", {})
    if evaluation_config.get("affine_transformation_cache_path") is None:
        affine_transformation_cache_path = None
    else:
        affine_transformation_cache_path = join(
            get_data_root(evaluation_config["affine_transformation_cache_path"]), dataset
        )
        makedirs(affine_transformation_cache_path, exist_ok=True)
    evaluation_dataset: Dataset = EvaluationDataset(
        sequence=EvaluationTiffPatchSequence(
            input_paths=type_to_paths["input"],
            aligned_label_paths=type_to_paths["label_aligned"],
            non_aligned_label_paths=type_to_paths["label_non_aligned"],
            training_label_paths=type_to_paths["label_training"],
            input_mask_paths=type_to_paths["input_mask"],
            aligned_label_mask_paths=type_to_paths["label_aligned_mask"],
            non_aligned_label_mask_paths=type_to_paths["label_non_aligned_mask"],
            training_label_mask_paths=type_to_paths["label_training_mask"],
            evaluation_mask_paths=type_to_paths["evaluation_mask"],
            ground_truth_deformation_paths=type_to_paths.get("deformation"),
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
                "use_affinely_registered_training_label_as_ground_truth", False
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
