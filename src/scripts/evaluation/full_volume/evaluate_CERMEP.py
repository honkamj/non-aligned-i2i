"""Evaluation script for full volume evaluation without ground truth"""


from argparse import ArgumentParser
from contextlib import contextmanager
import ctypes
from json import load as json_load
from multiprocessing import Queue, get_context
from os import environ, listdir, makedirs
from os.path import isfile, join
from re import sub
from shutil import copy, rmtree
from subprocess import run
from tempfile import TemporaryDirectory
from threading import Thread
from typing import Any, Iterable, Mapping, NamedTuple, Optional, Sequence

from ants import ANTsImage, image_read, resample_image_to_target  # type: ignore
from ants.utils import crop_indices  # type: ignore
from numpy import abs as np_abs
from numpy import (
    interp,
    asarray,
    exp,
    ndarray,
    newaxis,
    zeros_like,
    prod,
    frombuffer,
    save,
    load,
    float32,
    where,
    any as np_any,
)
from numpy.linalg import norm
from numpy.random import RandomState
from torch import from_numpy, tensor
from torch.multiprocessing import spawn
from tqdm import tqdm  # type: ignore

from algorithm.torch.deformation.primitive import generate_coordinate_grid
from data.util import get_data_root
from metrics.peak_signal_to_noise_ratio import PSNRMassFunction
from metrics.structural_similarity_index import structural_similarity_index
from util.training import LossAverager, find_largest_epoch


class _EvaluationArgs(NamedTuple):
    """Evaluation arguments"""

    config: Mapping[str, Any]
    target_dir: str
    epoch: int
    data_set: str
    registration_mask_width: float
    evaluation_mask_width: float
    n_evaluation_processes: int
    seed: int
    transformations_cache_path: str
    n_samples_per_volume: int
    n_instances: int
    instance_index: int
    elastix_path: str
    elastix_threads: int

    def get_full_transformations_cache_path(self) -> str:
        """Get path to the directory where the transformations will be stored"""
        return join(
            self.transformations_cache_path,
            (
                f"{self.registration_mask_width:.12g}_"
                f"{self.n_samples_per_volume}_"
                f"{self.seed}"
            ),
        )

    def get_identifier(self) -> str:
        """Get string identifier which defines uniquely evaluation results"""
        return (
            f"{self.registration_mask_width:.12g}_{self.evaluation_mask_width:.12g}_"
            f"{self.n_samples_per_volume}_"
            f"{self.seed}"
        )


def _get_evaluation_locations(
    shape: Sequence[int],
    n_samples: int,
    voxel_size: ndarray,
    sampling_mask: ndarray,
    random_state: RandomState,
) -> ndarray:
    coordinate_grid = _generate_coordinate_grid(
        shape=shape,
        voxel_size=voxel_size,
    )
    n_voxels = asarray(shape).prod()
    n_dims = len(shape)
    indices = random_state.choice(
        a=n_voxels,
        size=n_samples,
        replace=False,
        p=sampling_mask.flatten() / sampling_mask.sum(),
    )
    sampled_coordinates = coordinate_grid.reshape(n_dims, -1)[:, indices]
    return sampled_coordinates


def _generate_registration_mask(
    reference_ants_image: ANTsImage,
    coordinate_grid: ndarray,
    center_point: ndarray,
    mask_radius: float,
) -> ANTsImage:
    n_dims = len(coordinate_grid.shape) - 1
    return reference_ants_image.new_image_like(
        (
            norm(coordinate_grid - center_point[(...,) + (newaxis,) * n_dims], axis=0) < mask_radius
        ).astype("float")
    )


def _generate_evaluation_mask(
    reference_ants_image: ANTsImage,
    coordinate_grid: ndarray,
    center_point: ndarray,
    mask_width: float,
) -> ANTsImage:
    n_dims = len(coordinate_grid.shape) - 1
    return reference_ants_image.new_image_like(
        exp(
            -norm(coordinate_grid - center_point[(...,) + (newaxis,) * n_dims], axis=0) ** 2
            / (2 * mask_width**2)
        )
    )


def _generate_coordinate_grid(shape: Sequence[int], voxel_size: ndarray):
    return generate_coordinate_grid(tensor(shape), grid_voxel_size=from_numpy(voxel_size)).numpy()


def _load_mr_invalid_region_mask(case_name: str, template: ANTsImage) -> ANTsImage:
    invalid_region_mask_path = join(
        "../data/CERMEP/test_set_invalid_region_masks",
        f"{case_name}_space-pet_mr_invalid_region_mask.nii.gz",
    )
    if isfile(invalid_region_mask_path):
        return image_read(invalid_region_mask_path)
    else:
        return template.new_image_like(zeros_like(template.numpy()))


def _load_ct_invalid_region_mask(case_name: str, template: ANTsImage) -> ANTsImage:
    invalid_region_mask_path = join(
        "../data/CERMEP/test_set_invalid_region_masks",
        f"{case_name}_space-pet_ct_invalid_region_mask.nii.gz",
    )
    if isfile(invalid_region_mask_path):
        return image_read(invalid_region_mask_path)
    else:
        return template.new_image_like(zeros_like(template.numpy()))


def _load_images(
    args: _EvaluationArgs, case_name: str
) -> tuple[ANTsImage, ANTsImage, ANTsImage, ANTsImage, ANTsImage, ANTsImage, ANTsImage, ANTsImage]:
    data_folder_path = join(get_data_root(args.config["data"]["root"]), args.data_set)
    inference_folder_path = join(
        args.target_dir, "inference", f"epoch{int(args.epoch):03d}", args.data_set
    )
    input_body_mask_path = join(
        data_folder_path,
        case_name,
        f"{case_name}_{args.config['data']['postfixes']['input_body_mask']}",
    )
    prediction_path = join(inference_folder_path, case_name, f"{case_name}_predicted.nii.gz")
    input_path = join(
        data_folder_path, case_name, f"{case_name}_{args.config['data']['postfixes']['input']}"
    )
    label_non_aligned_non_interpolated_path = join(
        data_folder_path,
        case_name,
        f"{case_name}_{args.config['data']['postfixes']['label_non_aligned_non_interpolated']}",
    )
    label_non_aligned_non_interpolated_body_mask_path = join(
        data_folder_path,
        case_name,
        (
            f"{case_name}_"
            f"{args.config['data']['postfixes']['label_non_aligned_non_interpolated_body_mask']}"
        ),
    )
    evaluation_mask_path = join(
        data_folder_path,
        case_name,
        f"{case_name}_{args.config['data']['postfixes']['evaluation_mask']}",
    )

    input_ants = image_read(input_path)
    input_body_mask_ants = image_read(input_body_mask_path)
    label_non_aligned_non_interpolated_ants = image_read(label_non_aligned_non_interpolated_path)
    label_non_aligned_non_interpolated_body_mask_ants = image_read(
        label_non_aligned_non_interpolated_body_mask_path
    )
    prediction_ants = image_read(prediction_path)
    evaluation_mask_ants = image_read(evaluation_mask_path)

    return (
        input_ants,
        input_body_mask_ants,
        label_non_aligned_non_interpolated_ants,
        label_non_aligned_non_interpolated_body_mask_ants,
        prediction_ants,
        evaluation_mask_ants,
        _load_mr_invalid_region_mask(case_name, input_ants),
        _load_ct_invalid_region_mask(case_name, input_ants),
    )


def _elastix_register(
    elastix_path: str,
    fixed_image: ANTsImage,
    fixed_mask_image: ANTsImage,
    moving_ct_image: ANTsImage,
    moving_mask_image: ANTsImage,
    elastix_threads: int,
    affine_transform_parameters_target_path: str,
    elastic_transform_parameters_target_path: str,
) -> ANTsImage:
    rigidity_volume_ants = moving_ct_image.new_image_like(
        interp(moving_ct_image.numpy(), xp=[280, 320], fp=[0.0, 1.0])
    )
    with TemporaryDirectory() as temp_dir:
        fixed_path = join(temp_dir, "fixed.nii.gz")
        fixed_mask_path = join(temp_dir, "fixed_mask.nii.gz")
        moving_mask_path = join(temp_dir, "moving_mask.nii.gz")
        moving_path = join(temp_dir, "moving.nii.gz")
        rigidity_path = join(temp_dir, "moving_rigidity.nii.gz")
        with open(
            "../data/CERMEP/elastix_evaluation_registration_params.txt", mode="r", encoding="utf-8"
        ) as params_file:
            modified_params = params_file.read().replace("moving_rigidity_path", rigidity_path)
        with open(
            join(temp_dir, "elastix_evaluation_registration_params.txt"), mode="w", encoding="utf-8"
        ) as modified_params_file:
            modified_params_file.write(modified_params)
        fixed_image.to_filename(fixed_path)
        fixed_mask_image.to_filename(fixed_mask_path)
        moving_mask_image.to_filename(moving_mask_path)
        moving_ct_image.to_filename(moving_path)
        rigidity_volume_ants.to_filename(rigidity_path)
        run(
            [
                join(elastix_path, "elastix"),
                "-f",
                fixed_path,
                "-fMask",
                fixed_mask_path,
                "-m",
                moving_path,
                "-mMask",
                moving_mask_path,
                "-p",
                "../data/CERMEP/elastix_evaluation_rigid_registration_params.txt",
                "-p",
                join(temp_dir, "elastix_evaluation_registration_params.txt"),
                "-out",
                temp_dir,
                "-threads",
                str(elastix_threads),
            ],
            check=True,
            capture_output=True,
        )
        registered_temp_path = join(temp_dir, "result.1.nii.gz")
        if not isfile(affine_transform_parameters_target_path):
            copy(
                join(temp_dir, "TransformParameters.0.txt"), affine_transform_parameters_target_path
            )
        if not isfile(elastic_transform_parameters_target_path):
            with open(
                join(temp_dir, "TransformParameters.1.txt"), mode="r", encoding="utf-8"
            ) as transform_file:
                transform = transform_file.read()
                escaped_target_path = affine_transform_parameters_target_path.replace('"', r"\"")
                transform = sub(
                    r'\(InitialTransformParametersFileName "[^"]+"\)',
                    f'(InitialTransformParametersFileName "{escaped_target_path}")',
                    transform,
                )
            with open(
                elastic_transform_parameters_target_path, mode="w", encoding="utf-8"
            ) as target_transform_file:
                target_transform_file.write(transform)
        return image_read(registered_temp_path)


def _transformix_apply_transform(
    elastix_path: str,
    elastix_threads: int,
    moving_image: ANTsImage,
    transform_parameters_path: str,
    default_pixel_value: Optional[float] = None,
) -> ANTsImage:
    with TemporaryDirectory() as temp_dir:
        if default_pixel_value is not None:
            with open(
                transform_parameters_path, mode="r", encoding="utf-8"
            ) as transform_parameters_file:
                transform_parameters = transform_parameters_file.read()
            transform_parameters_path = join(temp_dir, "modified_transform_params.txt")
            with open(
                transform_parameters_path, mode="w", encoding="utf-8"
            ) as transform_parameters_file:
                transform_parameters_file.write(
                    transform_parameters.replace(
                        "(DefaultPixelValue -1024.000000)",
                        f"(DefaultPixelValue {default_pixel_value})",
                    )
                )
        moving_path = join(temp_dir, "moving.nii.gz")
        moving_image.to_filename(moving_path)
        run(
            [
                join(elastix_path, "transformix"),
                "-in",
                moving_path,
                "-tp",
                transform_parameters_path,
                "-out",
                temp_dir,
                "-threads",
                str(elastix_threads),
            ],
            check=True,
            capture_output=True,
        )
        registered_temp_path = join(temp_dir, "result.nii.gz")
        return image_read(registered_temp_path)


def _bbox(img):
    img = img > 0
    x_axis = np_any(img, axis=(1, 2))
    y_axis = np_any(img, axis=(0, 2))
    z_axis = np_any(img, axis=(0, 1))

    xmin, xmax = where(x_axis)[0][[0, -1]]
    ymin, ymax = where(y_axis)[0][[0, -1]]
    zmin, zmax = where(z_axis)[0][[0, -1]]

    return (xmin, ymin, zmin), (xmax, ymax, zmax)


def _evaluation_process(
    _inference_process_rank: int,
    args: _EvaluationArgs,
    allocation_queue: Queue,
    output_queue: Queue,
    shared_error_arrays,
) -> None:
    transformations_cache_path = args.get_full_transformations_cache_path()
    makedirs(transformations_cache_path, exist_ok=True)
    case_name: Optional[str] = None
    while True:
        task: Optional[tuple[str, ndarray]] = allocation_queue.get(block=True, timeout=None)
        if task is None:
            return
        new_case_name, coordinate = task
        if case_name != new_case_name:
            case_name = new_case_name
            (
                original_input_ants,
                original_input_body_mask_ants,
                original_label_non_aligned_non_interpolated_ants,
                original_label_non_aligned_non_interpolated_body_mask_ants,
                original_prediction_ants,
                original_evaluation_mask_ants,
                original_mr_invalid_region_mask_ants,
                original_ct_invalid_region_mask_ants,
            ) = _load_images(args, case_name)
            coordinate_grid = _generate_coordinate_grid(
                shape=original_input_body_mask_ants.shape,
                voxel_size=asarray(args.config["data_loader"]["voxel_size"]),
            )
        evaluation_weighting_mask_ants = _generate_evaluation_mask(
            reference_ants_image=original_input_ants,
            coordinate_grid=coordinate_grid,
            center_point=coordinate,
            mask_width=args.evaluation_mask_width,
        )
        input_registration_mask_ants = _generate_registration_mask(
            reference_ants_image=original_input_ants,
            coordinate_grid=coordinate_grid,
            center_point=coordinate,
            mask_radius=args.registration_mask_width,
        )
        input_registration_mask_ants = input_registration_mask_ants.new_image_like(
            input_registration_mask_ants.numpy()
            * (1 - original_mr_invalid_region_mask_ants.numpy())
        )
        registration_mask_extended_ants = _generate_registration_mask(
            reference_ants_image=original_input_ants,
            coordinate_grid=coordinate_grid,
            center_point=coordinate,
            mask_radius=args.registration_mask_width + 10,
        )
        registration_mask_extended_label_ants = resample_image_to_target(
            registration_mask_extended_ants,
            original_label_non_aligned_non_interpolated_ants,
            interp_type="nearestNeighbor",
        )
        lower_ind, upper_ind = _bbox(input_registration_mask_ants.numpy())
        lower_ind_label, upper_ind_label = _bbox(registration_mask_extended_label_ants.numpy())
        input_ants = crop_indices(original_input_ants, lower_ind, upper_ind)
        label_non_aligned_non_interpolated_ants = crop_indices(
            original_label_non_aligned_non_interpolated_ants, lower_ind_label, upper_ind_label
        )
        input_registration_mask_ants = resample_image_to_target(
            input_registration_mask_ants, input_ants, interp_type="nearestNeighbor"
        )
        input_body_mask_ants = resample_image_to_target(
            original_input_body_mask_ants, input_ants, interp_type="nearestNeighbor"
        )
        label_non_aligned_non_interpolated_body_mask_ants = resample_image_to_target(
            original_label_non_aligned_non_interpolated_body_mask_ants,
            label_non_aligned_non_interpolated_ants,
        )
        prediction_ants = resample_image_to_target(original_prediction_ants, input_ants)
        evaluation_mask_ants = resample_image_to_target(original_evaluation_mask_ants, input_ants)
        evaluation_weighting_mask_ants = resample_image_to_target(
            evaluation_weighting_mask_ants, input_ants
        )
        ct_invalid_region_mask_ants = resample_image_to_target(
            original_ct_invalid_region_mask_ants,
            label_non_aligned_non_interpolated_ants,
            interp_type="nearestNeighbor",
        )
        ct_valid_region_mask_ants = ct_invalid_region_mask_ants.new_image_like(
            1 - ct_invalid_region_mask_ants.numpy()
        )

        coordinate_as_string = "+".join(f"{dim_coordinate:.12g}" for dim_coordinate in coordinate)
        affine_transformation_file_path = join(
            transformations_cache_path,
            f"{case_name}_{coordinate_as_string}_{args.registration_mask_width:.12g}_affine_"
            f"seed_{args.seed}.txt",
        )
        elastic_transformation_file_path = join(
            transformations_cache_path,
            f"{case_name}_{coordinate_as_string}_{args.registration_mask_width:.12g}_elastic_"
            f"seed_{args.seed}.txt",
        )
        if not isfile(affine_transformation_file_path) or not isfile(
            elastic_transformation_file_path
        ):
            registered_label_ants = _elastix_register(
                elastix_path=args.elastix_path,
                fixed_image=input_ants,
                fixed_mask_image=input_registration_mask_ants,
                moving_ct_image=label_non_aligned_non_interpolated_ants,
                moving_mask_image=ct_valid_region_mask_ants,
                elastix_threads=args.elastix_threads,
                affine_transform_parameters_target_path=affine_transformation_file_path,
                elastic_transform_parameters_target_path=elastic_transformation_file_path,
            )
        else:
            registered_label_ants = _transformix_apply_transform(
                elastix_path=args.elastix_path,
                elastix_threads=args.elastix_threads,
                moving_image=label_non_aligned_non_interpolated_ants,
                transform_parameters_path=elastic_transformation_file_path,
            )
        warped_label_non_aligned_non_interpolated_body_mask_ants = _transformix_apply_transform(
            elastix_path=args.elastix_path,
            elastix_threads=args.elastix_threads,
            moving_image=label_non_aligned_non_interpolated_body_mask_ants,
            transform_parameters_path=elastic_transformation_file_path,
            default_pixel_value=0,
        )
        warped_ct_invalid_region_mask_ants = _transformix_apply_transform(
            elastix_path=args.elastix_path,
            elastix_threads=args.elastix_threads,
            moving_image=original_ct_invalid_region_mask_ants,
            transform_parameters_path=elastic_transformation_file_path,
            default_pixel_value=0,
        )
        registered_label = registered_label_ants.numpy()
        prediction = prediction_ants.numpy()
        body_mask_agreement_mask = (
            input_body_mask_ants.numpy()
            - warped_label_non_aligned_non_interpolated_body_mask_ants.numpy()
            < 1e-5
        )
        evaluation_mask = (
            evaluation_weighting_mask_ants.numpy()
            * body_mask_agreement_mask
            * evaluation_mask_ants.numpy()
            * input_registration_mask_ants.numpy()
            * (1 - warped_ct_invalid_region_mask_ants.numpy())
        )
        mask_mass = evaluation_mask.sum()
        squared_sum = ((registered_label - prediction) ** 2 * evaluation_mask).sum()
        absolute_error_volume = np_abs(registered_label - prediction) * evaluation_mask
        error_sum = ((registered_label - prediction) * evaluation_mask).sum()
        absolute_sum = absolute_error_volume.sum()

        (
            structural_similarity_sum,
            structural_similarity_averaging_mass,
        ) = structural_similarity_index(
            label=registered_label[None, None],
            output=prediction_ants.numpy()[None, None],
            content_mask=body_mask_agreement_mask[None, None],
            evaluation_mask=evaluation_mask[None, None],
            data_range=args.config["data_loader"]["label_signal_range"],
        )
        metrics = {}
        metrics["MSE"] = (squared_sum, mask_mass)
        metrics["MAE"] = (absolute_sum, mask_mass)
        metrics["ME"] = (error_sum, mask_mass)
        metrics["PSNR"] = (squared_sum, mask_mass)
        metrics["SSIM"] = (structural_similarity_sum, structural_similarity_averaging_mass)

        # registered_label_ants.to_filename(f"/hdd/honkamj2/temp/{case_name}_CT_registered.nii.gz")

        absolute_error_volume_ants = resample_image_to_target(
            input_ants.new_image_like(absolute_error_volume), original_input_ants
        )
        with _load_shared_memory_array(shared_error_arrays[case_name]) as array:
            array += absolute_error_volume_ants.numpy()

        output_queue.put((case_name, metrics))


def _get_target_dir(args: _EvaluationArgs) -> str:
    return join(
        args.target_dir,
        "evaluation_without_aligned_ground_truth",
        args.get_identifier(),
        args.data_set,
    )


def _get_agregation_dir_for_case(case_name: str, args: _EvaluationArgs) -> str:
    return join(_get_target_dir(args), f"{case_name}_epoch{int(args.epoch):03d}")


def _do_instance_outputs_exist(case_names: Iterable[str], args: _EvaluationArgs) -> bool:
    return all(
        isfile(join(_get_agregation_dir_for_case(case_name, args), f"{args.instance_index}.json"))
        for case_name in case_names
    )


def _evaluated(case_names: Iterable[str], args: _EvaluationArgs) -> bool:
    return all(isfile(join(_get_target_dir(args), f"{case_name}.json")) for case_name in case_names)


def _evaluate(args: _EvaluationArgs) -> None:
    data_folder_path = join(get_data_root(args.config["data"]["root"]), args.data_set)
    case_names = sorted(listdir(data_folder_path))
    if _do_instance_outputs_exist(case_names + ["combined"], args):
        return
    multiprocessing_context = get_context("spawn")
    allocation_queue = multiprocessing_context.Queue(-1)
    output_queue = multiprocessing_context.Queue(-1)
    voxel_size = asarray(args.config["data_loader"]["voxel_size"])
    makedirs(_get_target_dir(args), exist_ok=True)
    random_state = RandomState(seed=args.seed)
    evaluation_rois: list[tuple[str, ndarray]] = []
    shared_error_arrays = {}
    for case_name in case_names:
        input_body_mask_path = join(
            data_folder_path,
            case_name,
            f"{case_name}_{args.config['data']['postfixes']['input_body_mask']}",
        )
        input_path = join(
            data_folder_path,
            case_name,
            f"{case_name}_{args.config['data']['postfixes']['input']}",
        )
        input_body_mask_ants = image_read(input_body_mask_path)
        invalid_region_mask = (
            _load_mr_invalid_region_mask(case_name, input_body_mask_ants).numpy()
            * _load_ct_invalid_region_mask(case_name, input_body_mask_ants).numpy()
        )
        volume_shape = input_body_mask_ants.shape

        coordinates = _get_evaluation_locations(
            shape=input_body_mask_ants.shape,
            n_samples=args.n_samples_per_volume,
            voxel_size=voxel_size,
            sampling_mask=input_body_mask_ants.numpy() * (1 - invalid_region_mask),
            random_state=random_state,
        )
        for coordinate_index in range(coordinates.shape[1]):
            evaluation_rois.append((case_name, coordinates[:, coordinate_index]))
        shared_error_array = multiprocessing_context.Array(ctypes.c_float, int(prod(volume_shape)))
        shared_error_array_np = frombuffer(
            shared_error_array.get_obj(), dtype=float32
        )  # type: ignore
        shared_error_array_np[:] = 0.0
        shared_error_arrays[case_name] = (shared_error_array, volume_shape, input_path)

    evaluated_evaluation_rois = [
        evaluation_rois[index]
        for index in range(len(evaluation_rois))
        if index % args.n_instances == args.instance_index
    ]
    n_evaluated_evaluation_rois = len(evaluated_evaluation_rois)
    for _ in range(args.n_evaluation_processes):
        allocation_queue.put(evaluated_evaluation_rois.pop(0))
    evaluation_result_listener = Thread(
        target=_evaluation_results_listener,
        args=(
            output_queue,
            allocation_queue,
            case_names,
            evaluated_evaluation_rois,
            n_evaluated_evaluation_rois,
            args,
            shared_error_arrays,
        ),
    )
    evaluation_result_listener.start()
    try:
        spawn(
            _evaluation_process,
            args=(
                args,
                allocation_queue,
                output_queue,
                shared_error_arrays,
            ),
            nprocs=args.n_evaluation_processes,
            join=True,
        )
    except (KeyboardInterrupt, SystemExit, Exception) as exception:
        _exit_thread(evaluation_result_listener, output_queue)
        raise exception
    evaluation_result_listener.join()


def _exit_thread(thread: Thread, exit_queue: Queue) -> None:
    exit_queue.put_nowait(None)
    thread.join()


@contextmanager
def _load_shared_memory_array(array_shape_and_ants_template_path):
    array, shape, _ = array_shape_and_ants_template_path
    with array.get_lock():
        shared_error_array_np = frombuffer(array.get_obj(), dtype="float32")
        shared_error_array_np.shape = shape
        yield shared_error_array_np


def _evaluation_results_listener(
    output_queue: Queue,
    allocation_queue: Queue,
    case_names: list[str],
    evaluation_rois: list[tuple[str, ndarray]],
    n_evaluation_rois: int,
    args: _EvaluationArgs,
    shared_error_arrays,
):
    case_metric_averagers: dict[str, LossAverager] = {}
    label_signal_range = args.config["data_loader"]["label_signal_range"]
    for case_name in case_names + ["combined"]:
        case_metric_averagers[case_name] = LossAverager()
        case_metric_averagers[case_name].set_custom_mass_func(
            "PSNR", PSNRMassFunction(label_signal_range)
        )
    evaluation_tqdm = tqdm(range(n_evaluation_rois), smoothing=0)
    for _ in evaluation_tqdm:
        evaluation_output = output_queue.get(block=True, timeout=None)
        if evaluation_output is None:
            return
        if evaluation_rois:
            allocation_queue.put(evaluation_rois.pop(0))
        else:
            allocation_queue.put(None)
        case_name, metrics = evaluation_output
        for metric, (value, mass) in metrics.items():
            case_metric_averagers[case_name].count({metric: value}, mass=mass)
            case_metric_averagers["combined"].count({metric: value}, mass=mass)
        evaluation_tqdm.set_description(str(case_metric_averagers["combined"]))
    for case_name in case_names + ["combined"]:
        case_agregation_dir = _get_agregation_dir_for_case(case_name, args)
        makedirs(case_agregation_dir, exist_ok=True)
        case_metric_averagers[case_name].save_state(
            join(case_agregation_dir, f"{args.instance_index}.json")
        )
        if case_name != "combined":
            with _load_shared_memory_array(shared_error_arrays[case_name]) as array:
                save(join(case_agregation_dir, f"{args.instance_index}_absolute_error.npy"), array)

        if all(
            isfile(join(case_agregation_dir, f"{index}.json")) for index in range(args.n_instances)
        ):
            agregation_averager = LossAverager()
            error_agregation_array = zeros_like(array)
            agregation_averager.set_custom_mass_func("PSNR", PSNRMassFunction(label_signal_range))
            for index in range(args.n_instances):
                agregation_averager.load_state(join(case_agregation_dir, f"{index}.json"))
                if case_name != "combined":
                    error_agregation_array += load(
                        join(case_agregation_dir, f"{index}_absolute_error.npy")
                    )
            agregation_averager.save_to_json(
                epoch=args.epoch,
                filename=join(
                    _get_target_dir(args),
                    f"{case_name}.json",
                ),
                postfix="",
            )
            if case_name != "combined":
                image_read(shared_error_arrays[case_name][2]).new_image_like(
                    error_agregation_array
                ).to_filename(
                    join(
                        _get_target_dir(args),
                        f"{case_name}_absolute_error.nii.gz",
                    )
                )
            rmtree(case_agregation_dir)


def _main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--config", help="Path to config file", type=str, required=False)
    parser.add_argument("--target-root", help="Path to output root", type=str, required=True)
    parser.add_argument("--model-name", help="Model name to evaluate", type=str, required=True)
    parser.add_argument(
        "--data-set",
        help="Data set to use",
        choices=["train", "test", "validate"],
        type=str,
        required=True,
    )
    epoch_args = parser.add_mutually_exclusive_group()
    epoch_args.add_argument(
        "--epoch", help="Evaluate this epoch", type=int, required=False, default=None
    )
    epoch_args.add_argument(
        "--min-epoch",
        help="Evaluate epochs starting from this",
        type=int,
        required=False,
        default=None,
    )
    epoch_args.add_argument(
        "--best-epoch",
        help="Use best epoch read from best_epoch.txt",
        default=False,
        action="store_true",
        required=False,
    )
    parser.add_argument(
        "--n-evaluation-locations-per-volume",
        help="Number of evaluation locations to sample per volume",
        type=int,
        default=20,
        required=False,
    )
    parser.add_argument(
        "--registration-mask-width",
        help=("Registration mask radius in mm."),
        type=float,
        default=100.0,
        required=False,
    )
    parser.add_argument(
        "--evaluation-mask-width",
        help="Gaussian evaluation mask width in mm",
        type=float,
        default=30.0,
        required=False,
    )
    parser.add_argument(
        "--n-evaluation-processes",
        help="Number of worker processes to use",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--seed",
        help="Seed for registration",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--transformations-cache-path",
        help="Save transformations to this path or read them from there if they already exist",
        type=str,
        required=False,
        default=None,
    )
    parser.add_argument(
        "--elastix-binary-path",
        help=(
            "Path to elastix 5.0.1 bin folder, note that for linux and mac "
            "path to the elastix libraries must be included in $LD_LIBRARY_PATH."
        ),
        required=True,
        type=str,
    )
    parser.add_argument(
        "--elastix-threads",
        help=("Number of threads to use in elastix registration"),
        required=True,
        type=int,
    )
    parser.add_argument(
        "--n-instances",
        help="Set for parallelization over multiple separate instances",
        type=int,
        required=False,
        default=1,
    )
    parser.add_argument(
        "--instance-index",
        help=(
            "Index of this instance, use together with --n-instances for parallelization over "
            "multiple separate instances"
        ),
        type=int,
        required=False,
        default=0,
    )
    args = parser.parse_args()
    target_dir = join(args.target_root, args.model_name)
    if args.config is None:
        config_path = join(target_dir, "training_config.json")
    else:
        config_path = args.config
    with open(config_path, mode="r", encoding="UTF-8") as config_file:
        config = json_load(config_file)
    if args.epoch is None:
        if args.best_epoch:
            with open(
                join(target_dir, "best_epoch.txt"), mode="r", encoding="UTF-8"
            ) as best_epoch_file:
                epochs: Iterable[int] = [int(best_epoch_file.read())]
        else:
            if args.min_epoch is None:
                min_epoch = 1
            else:
                min_epoch = args.min_epoch
            largest_epoch = find_largest_epoch(target_dir, "training")
            if largest_epoch is None:
                raise ValueError("Largest epoch not found!")
            epochs = reversed(range(min_epoch, largest_epoch + 1))
    else:
        epochs = [args.epoch]
    environ[
        "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"
    ] = "1"  # ANTs registration is not deterministic with multiple threads
    environ["OMP_NUM_THREADS"] = "1"
    for epoch in epochs:
        _evaluate(
            _EvaluationArgs(
                config=config,
                target_dir=target_dir,
                epoch=epoch,
                data_set=args.data_set,
                registration_mask_width=args.registration_mask_width,
                evaluation_mask_width=args.evaluation_mask_width,
                n_evaluation_processes=args.n_evaluation_processes,
                seed=args.seed,
                transformations_cache_path=args.transformations_cache_path,
                n_samples_per_volume=args.n_evaluation_locations_per_volume,
                n_instances=args.n_instances,
                instance_index=args.instance_index,
                elastix_path=args.elastix_binary_path,
                elastix_threads=args.elastix_threads,
            )
        )


if __name__ == "__main__":
    _main()
