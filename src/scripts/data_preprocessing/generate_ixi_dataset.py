"""Generates IXI dataset to desired path"""

from argparse import ArgumentParser
from json import load as json_load
from os import listdir, makedirs, remove
from os.path import basename, dirname, isdir, isfile, join
from shutil import rmtree
from subprocess import run
from tarfile import open as open_tar
from tempfile import TemporaryDirectory
from typing import List, Optional, Sequence, Tuple, cast
from urllib.request import urlretrieve

from ants import ANTsImage  # type: ignore
from ants import image_read, image_write, resample_image_to_target  # type: ignore
from ants.utils import iMath  # type: ignore
from ants.utils import n4_bias_field_correction as ants_n4_bias_field_correction
from intensity_normalization.normalize.fcm import FCMNormalize
from intensity_normalization.typing import Modality, TissueType
from nibabel import Nifti1Image  # type: ignore
from nibabel import load as nib_load
from numpy import flip, floor, zeros_like  # pylint: disable=no-name-in-module
from numpy import asarray, diag, isclose, ndarray, ones_like, pi
from numpy.random import RandomState
from scipy.ndimage import median_filter  # type: ignore
from torch import Tensor, from_numpy, no_grad
from torch import zeros as torch_zeros
from tqdm import tqdm  # type: ignore

from algorithm.torch.deformation.primitive import deform_volume
from algorithm.torch.simulated_elastic_deformation import sample_random_noise_ddf


IXI_T1_URL = "http://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/IXI-T1.tar"
IXI_T2_URL = "http://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/IXI-T2.tar"
IXI_PD_URL = "http://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/IXI-PD.tar"


ELASTIC_DEFORMATION_NOISE_MEAN_IN_MM = [10.0, 10.0, -10.0]
ELASTIC_DEFORMATION_NOISE_STD_IN_MM = [200.0, 200.0, 200.0]
ELASTIC_DEFORMATION_SMOOTHING_FILTER_STD_IN_MM = [10.0, 10.0, 10.0]
ROTATION_RANGE_IN_DEGREES = ([2.0, 2.0, 2.0], [10.0, 10.0, 10.0])
TRANSLATION_RANGE_IN_MM = ([2.0, 2.0, 2.0], [10.0, 10.0, 10.0])


def _download(path: str, url: str) -> None:
    makedirs(dirname(path), exist_ok=True)
    progress_bar = None
    previous_recieved = 0

    def _show_progress(block_num, block_size, total_size):
        nonlocal progress_bar, previous_recieved
        if progress_bar is None:
            progress_bar = tqdm(unit="B", total=total_size)
        downloaded = block_num * block_size
        if downloaded < total_size:
            progress_bar.update(downloaded - previous_recieved)
            previous_recieved = downloaded
        else:
            progress_bar.close()

    if not isfile(path):
        urlretrieve(url, path, _show_progress)


def _untar(path: str) -> None:
    target_dir = join(dirname(path), basename(path)[:-4])
    makedirs(target_dir, exist_ok=True)
    tar = open_tar(path)
    tar.extractall(target_dir)
    remove(path)


def _generate_resampled_t1(t1_path: str, t2_path: str, target_path: str) -> None:
    makedirs(target_path, exist_ok=True)
    for t1_filename in tqdm(
        sorted(
            t1_filename for t1_filename in listdir(t1_path) if isfile(join(t1_path, t1_filename))
        )
    ):
        t2_filename = t1_filename.replace("T1", "T2")
        t1_filepath = join(t1_path, t1_filename)
        t1_resampled_filepath = join(target_path, t1_filename)
        if isfile(t1_resampled_filepath):
            continue
        t2_filepath = join(t2_path, t2_filename)
        if not isfile(t2_filepath):
            continue
        t1_ants = image_read(t1_filepath)
        t2_ants = image_read(t2_filepath)
        resampled_t1 = resample_image_to_target(t1_ants, t2_ants, interp_type="linear")
        image_write(resampled_t1, t1_resampled_filepath)


def _calculate_body_mask(t1_ants: ANTsImage) -> ANTsImage:
    t1_data_median_filtered = median_filter(t1_ants.numpy(), size=3)
    t1_body_mask_mask = (t1_data_median_filtered > 0.1).astype(t1_data_median_filtered.dtype)
    t1_body_mask_mask_ants = t1_ants.new_image_like(t1_body_mask_mask)
    t1_body_mask_mask_ants = iMath(t1_body_mask_mask_ants, "ME", 1)
    t1_body_mask_mask_ants = iMath(t1_body_mask_mask_ants, "MD", 1)
    t1_body_mask_mask_ants = iMath(t1_body_mask_mask_ants, "GetLargestComponent")
    t1_body_mask_mask_ants = iMath(t1_body_mask_mask_ants, "MC", 8)
    t1_body_mask_mask_ants = iMath(t1_body_mask_mask_ants, "FillHoles").threshold_image(1, 2)
    return t1_body_mask_mask_ants


def _remove_temp_files(temp_folder_path: str) -> None:
    rmtree(temp_folder_path)


def _generate_brain_masks(t1_path: str, t2_path: str, target_path: str, robex_path: str) -> None:
    makedirs(target_path, exist_ok=True)
    for t1_filename in tqdm(
        sorted(
            t1_filename for t1_filename in listdir(t1_path) if isfile(join(t1_path, t1_filename))
        )
    ):
        mask_filename = t1_filename.replace("T1", "brain_mask")
        t2_filename = t1_filename.replace("T1", "T2")
        t1_filepath = join(t1_path, t1_filename)
        t2_filepath = join(t2_path, t2_filename)
        mask_filepath = join(target_path, mask_filename)
        if not isfile(t2_filepath) or isfile(mask_filepath):
            continue
        stripped_t1_filepath = join(target_path, "temp.nii")
        run(
            [robex_path, t1_filepath, stripped_t1_filepath, mask_filepath],
            cwd=dirname(robex_path),
            check=True,
        )
        remove(stripped_t1_filepath)
        mask_ants = image_read(mask_filepath)
        t2_ants = image_read(t2_filepath)
        resampled_mask = resample_image_to_target(mask_ants, t2_ants, interp_type="linear")
        image_write(resampled_mask, mask_filepath)


def _get_cases(pd_path: str) -> List[str]:
    return [pd_filename[:-10] for pd_filename in listdir(pd_path)]


def _is_in_dataset(case: str, dataset: str) -> bool:
    with open(f"../data/IXI/{dataset}_cases.json", encoding="utf-8") as cases_file:
        cases = json_load(cases_file)
    return case in cases


def _get_dataset(case: str) -> Optional[str]:
    options = ["train", "validate", "test"]
    for option in options:
        if _is_in_dataset(case, option):
            return option
    return None


def _crop_to_shape(
    volume: ndarray, shape: Tuple[int, ...], deformation_voxel_size: ndarray
) -> ndarray:
    with no_grad():
        n_dims = len(shape)
        volume_torch = from_numpy(volume)
        crop = torch_zeros(1, dtype=volume_torch.dtype).expand((n_dims,) + shape)
        cropped, mask = deform_volume(
            deformation=crop[None],
            volume=volume_torch[None, None],
            deformation_voxel_size=from_numpy(deformation_voxel_size),
            return_mask=True,
        )
        return (cropped * mask)[0, 0].numpy()


def _deform(
    volume: ndarray, deformation: Tensor, deformation_voxel_size: ndarray
) -> Tuple[ndarray, ndarray]:
    with no_grad():
        deformed, mask = deform_volume(
            deformation=deformation[None],
            volume=from_numpy(volume)[None, None],
            deformation_voxel_size=from_numpy(deformation_voxel_size),
            return_mask=True,
        )
        mask = cast(Tensor, mask)
        return (deformed * mask)[0, 0].numpy(), mask[0, 0].numpy()


def _reregister_deformed_pd_with_elastix(
    elastix_path: str,
    transformix_path: str,
    pd_nib: Nifti1Image,
    pd_mask_nib: Nifti1Image,
    t2_nib: Nifti1Image,
    elastix_threads: str,
) -> tuple[ndarray, ndarray]:
    with TemporaryDirectory() as temp_dir:
        pd_path = join(temp_dir, "pd.nii")
        pd_mask_path = join(temp_dir, "pd_mask.nii")
        t2_path = join(temp_dir, "t2.nii")
        pd_nib.to_filename(pd_path)
        pd_mask_nib.to_filename(pd_mask_path)
        t2_nib.to_filename(t2_path)
        run(
            [
                elastix_path,
                "-f",
                t2_path,
                "-m",
                pd_path,
                "-mMask",
                pd_mask_path,
                "-p",
                "../data/IXI/elastix_rigid_reregistration_params.txt",
                "-p",
                "../data/IXI/elastix_elastic_reregistration_params.txt",
                "-out",
                temp_dir,
                "-threads",
                elastix_threads,
            ],
            check=True,
        )
        registered_temp_path = join(temp_dir, "result.1.nii.gz")
        reregistered_pd = image_read(registered_temp_path).numpy()
        linear_interpolation_transformation_file = join(
            temp_dir, "TransformParameters.1.linear.txt"
        )
        with open(
            join(temp_dir, "TransformParameters.1.txt"), mode="r", encoding="utf-8"
        ) as transfrom_parameters_file:
            modified_contents = transfrom_parameters_file.read().replace(
                "(FinalBSplineInterpolationOrder 3)", "(FinalBSplineInterpolationOrder 1)"
            )
        with open(
            linear_interpolation_transformation_file, mode="w", encoding="utf-8"
        ) as modified_transfrom_parameters_file:
            modified_transfrom_parameters_file.write(modified_contents)
        makedirs(join(temp_dir, "mask_resampling"))
        run(
            [
                transformix_path,
                "-tp",
                linear_interpolation_transformation_file,
                "-out",
                join(temp_dir, "mask_resampling"),
                "-in",
                pd_mask_path,
                "-threads",
                elastix_threads,
            ],
            check=True,
        )
        resampled_mask_temp_path = join(temp_dir, "mask_resampling", "result.nii.gz")
        resampled_pd_mask = image_read(resampled_mask_temp_path).numpy()
        return reregistered_pd, resampled_pd_mask


def _create_border_touching_masks(mask: ndarray) -> list[ndarray]:
    n_dims = len(mask.shape)
    masks = []
    for dim in range(n_dims):
        new_mask = zeros_like(mask)
        dim_first_mask = new_mask.swapaxes(dim, 0)
        dim_first_mask[:1] = 1
        dim_first_mask[-1:] = -1
        new_mask = dim_first_mask.swapaxes(0, dim)
        new_mask = mask * new_mask
        masks.append(new_mask)
    return masks


def _calculate_valid_region_mask_from_deformed_border_touching_masks(
    deformed_border_touching_masks: Sequence[ndarray],
) -> ndarray:
    n_dims = len(deformed_border_touching_masks)
    mask = ones_like(deformed_border_touching_masks[0])
    for dim in range(n_dims):
        dim_first_mask = deformed_border_touching_masks[dim].swapaxes(dim, 0)
        lower_end_mask = (dim_first_mask > 0) & (dim_first_mask <= 1)
        upper_end_mask = (dim_first_mask < 0) & (dim_first_mask >= -1)
        lower_end_mask_flipped = flip(lower_end_mask, axis=0)
        lower_end_flipped_marginals = lower_end_mask_flipped.sum(axis=(1, 2))
        upper_end_marginals = upper_end_mask.sum(axis=(1, 2))
        dim_first_output_mask = mask.swapaxes(dim, 0)
        if upper_end_marginals.sum() > 0:
            upper_end_first_index = (upper_end_marginals > 0).argmax()
            dim_first_output_mask[upper_end_first_index:] = 0
        if lower_end_flipped_marginals.sum() > 0:
            lower_end_flipped_first_index = (lower_end_flipped_marginals > 0).argmax()
            dim_first_output_mask_flipped = flip(dim_first_output_mask, axis=0)
            dim_first_output_mask_flipped[lower_end_flipped_first_index:] = 0
            dim_first_output_mask = flip(dim_first_output_mask_flipped, axis=0)
        mask = dim_first_output_mask.swapaxes(0, dim)
    return mask


def _save(data: ndarray, affine: ndarray, path: str) -> None:
    Nifti1Image(data.astype("float32"), affine).to_filename(path)


def _generate_case(
    case: str,
    dataset: str,
    random_state: RandomState,
    pd_dir: str,
    t1_dir: str,
    t2_dir: str,
    brain_mask_dir: str,
    target_dir: str,
    elastix_path: str,
    transformix_path: str,
    elastix_threads: str,
) -> None:
    print(f"Starting case {case}")
    case_directory = join(target_dir, dataset, case)
    cropped_t2_path = join(case_directory, f"{case}_T2.nii.gz")
    cropped_pd_path = join(case_directory, f"{case}_PD_ground_truth.nii.gz")
    deformed_pd_path = join(case_directory, f"{case}_PD_deformed.nii.gz")
    reregistered_pd_path = join(case_directory, f"{case}_PD_reregistered.nii.gz")
    input_mask_path = join(case_directory, f"{case}_T2_valid_region_mask.nii.gz")
    label_mask_path = join(case_directory, f"{case}_PD_deformed_valid_region_mask.nii.gz")
    label_reregistered_mask_path = join(
        case_directory, f"{case}_PD_reregistered_valid_region_mask.nii.gz"
    )
    evaluation_mask_path = join(case_directory, f"{case}_evaluation_mask.nii.gz")
    forward_deformation_path = join(case_directory, f"{case}_deformation_to_PD_deformed.nii.gz")
    inverse_deformation_path = join(case_directory, f"{case}_deformation_from_PD_deformed.nii.gz")

    pd_original_path = join(pd_dir, f"{case}-PD.nii.gz")
    t1_original_path = join(t1_dir, f"{case}-T1.nii.gz")
    t2_original_path = join(t2_dir, f"{case}-T2.nii.gz")
    brain_mask_original_path = join(brain_mask_dir, f"{case}-brain_mask.nii.gz")

    pd_nib = nib_load(pd_original_path)
    t2_nib = nib_load(t2_original_path)
    pd_voxel_size = pd_nib.header.get_zooms()[:3]
    t2_voxel_size = t2_nib.header.get_zooms()[:3]

    # Ensure that coordinate systems match
    assert pd_voxel_size == t2_voxel_size
    assert (pd_nib.affine == t2_nib.affine).all()
    assert pd_nib.shape == t2_nib.shape

    if isclose(pd_voxel_size[:2], (0.9375, 0.9375), atol=1e-03).all():
        source_axial_voxel_size = (0.9375, 0.9375)
    else:
        source_axial_voxel_size = pd_voxel_size[:2]
    if isclose(pd_voxel_size[-1], 1.25, atol=1e-02).all():
        original_slice_thickness = 1.25
    else:
        original_slice_thickness = pd_voxel_size[-1]

    # Calculate voxel size and target shape
    original_voxel_size = source_axial_voxel_size + (original_slice_thickness,)
    target_voxel_size = (0.9375, 0.9375, 1.25)
    target_shape = tuple(
        floor(
            asarray(original_voxel_size) * (asarray(t2_nib.shape) - 1) / asarray(target_voxel_size)
            + 1
        ).astype(int)
    )

    # Sample random deformation
    rotation_range_in_radians = asarray(ROTATION_RANGE_IN_DEGREES) * pi / 180
    deformation_ddf, inverse_deformation_ddf = sample_random_noise_ddf(
        shape=target_shape,
        downsampling_factor=(1, 1, 1),
        noise_mean=ELASTIC_DEFORMATION_NOISE_MEAN_IN_MM,
        noise_std=ELASTIC_DEFORMATION_NOISE_STD_IN_MM,
        gaussian_filter_std=ELASTIC_DEFORMATION_SMOOTHING_FILTER_STD_IN_MM,
        rotation_bounds=(tuple(rotation_range_in_radians[0]), tuple(rotation_range_in_radians[1])),
        translation_bounds=TRANSLATION_RANGE_IN_MM,
        random_state=random_state,
        voxel_size=target_voxel_size,
    )

    # Skip the case if output alrady exists. The skipping is done after sampling the deformation
    # to ensure identical data set is always generated with any given seed.
    result_paths = [
        cropped_t2_path,
        cropped_pd_path,
        deformed_pd_path,
        reregistered_pd_path,
        input_mask_path,
        label_mask_path,
        label_reregistered_mask_path,
        evaluation_mask_path,
        forward_deformation_path,
        inverse_deformation_path,
    ]
    if all(isfile(result_path) for result_path in result_paths):
        print(f"Skipping case {case}")
        return

    # Normalize data
    brain_mask_data = image_read(brain_mask_original_path).numpy().astype("float")
    t1_ants = image_read(t1_original_path)
    t2_ants = image_read(t2_original_path)
    pd_ants = image_read(pd_original_path)
    fcm_norm_pre_bias_correction = FCMNormalize(tissue_type=TissueType.WM)
    t1_body_mask_ants = _calculate_body_mask(
        t1_ants.new_image_like(
            fcm_norm_pre_bias_correction(
                t1_ants.numpy().astype(float), mask=brain_mask_data, modality=Modality.T1
            )
        )
    )
    t2_body_mask_ants = _calculate_body_mask(
        t2_ants.new_image_like(
            fcm_norm_pre_bias_correction(t2_ants.numpy().astype(float), modality=Modality.T2)
        )
    )
    pd_body_mask_ants = _calculate_body_mask(
        pd_ants.new_image_like(
            fcm_norm_pre_bias_correction(pd_ants.numpy().astype(float), modality=Modality.PD)
        )
    )
    evaluation_mask_ants = iMath(t2_body_mask_ants, "MD", 5)

    bias_field_corrected_t2_ants = ants_n4_bias_field_correction(t2_ants, mask=t2_body_mask_ants)
    bias_field_corrected_t2_ants = t2_ants
    bias_field_corrected_pd_ants = ants_n4_bias_field_correction(pd_ants, mask=pd_body_mask_ants)
    bias_field_corrected_pd_ants = pd_ants
    bias_field_corrected_t1_ants = ants_n4_bias_field_correction(t1_ants, mask=t1_body_mask_ants)
    bias_field_corrected_t1_ants = t1_ants
    t2_data = bias_field_corrected_t2_ants.numpy().astype(float)
    pd_data = bias_field_corrected_pd_ants.numpy().astype(float)

    fcm_norm = FCMNormalize(tissue_type=TissueType.WM)
    fcm_norm(
        bias_field_corrected_t1_ants.numpy().astype(float),
        mask=brain_mask_data,
        modality=Modality.T1,
    )
    t2_data = fcm_norm(t2_data, modality=Modality.T2)
    pd_data = fcm_norm(pd_data, modality=Modality.PD)
    evaluation_mask_data = evaluation_mask_ants.numpy()

    # Resample to same voxel size
    relative_voxel_size = asarray(target_voxel_size) / asarray(original_voxel_size)

    cropped_evaluation_mask = _crop_to_shape(
        evaluation_mask_data, target_shape, relative_voxel_size
    )
    cropped_t2 = _crop_to_shape(t2_data, target_shape, relative_voxel_size)
    cropped_pd = _crop_to_shape(pd_data, target_shape, relative_voxel_size)

    affine = diag(target_voxel_size + (1,))

    # Transform the volume with the sampled deformation
    deformed_pd, _mask = _deform(
        volume=pd_data, deformation=deformation_ddf, deformation_voxel_size=relative_voxel_size
    )
    # Calculate mask which straightens out borders of the deformed volume. Otherwise
    # learning based registration methods could use the border for inferring the
    # transformation
    pd_border_touching_masks = _create_border_touching_masks(pd_body_mask_ants.numpy())
    pd_deformed_border_touching_masks = [
        _deform(
            volume=pd_border_touching_mask,
            deformation=deformation_ddf,
            deformation_voxel_size=relative_voxel_size,
        )[0]
        for pd_border_touching_mask in pd_border_touching_masks
    ]
    deformed_pd_valid_region_mask = (
        _calculate_valid_region_mask_from_deformed_border_touching_masks(
            deformed_border_touching_masks=pd_deformed_border_touching_masks
        )
    )
    deformed_pd = deformed_pd * deformed_pd_valid_region_mask

    # Re-register the deformed volume to the original space for comparison
    # with pix2pix methods
    (
        reregistered_pd,
        reregistered_deformed_pd_valid_region_mask,
    ) = _reregister_deformed_pd_with_elastix(
        pd_nib=Nifti1Image(deformed_pd, affine),
        pd_mask_nib=Nifti1Image(deformed_pd_valid_region_mask.astype("float32"), affine),
        t2_nib=Nifti1Image(cropped_t2, affine),
        elastix_path=elastix_path,
        transformix_path=transformix_path,
        elastix_threads=elastix_threads,
    )
    reregistered_pd_mask = (reregistered_pd != -1) & (
        reregistered_deformed_pd_valid_region_mask > 1 - 1e-6
    )
    reregistered_pd = reregistered_pd * reregistered_pd_mask

    # Save results
    makedirs(case_directory, exist_ok=True)
    _save(data=(cropped_evaluation_mask > 0), affine=affine, path=evaluation_mask_path)
    _save(data=cropped_t2, affine=affine, path=cropped_t2_path)
    _save(data=cropped_pd, affine=affine, path=cropped_pd_path)
    _save(data=deformed_pd, affine=affine, path=deformed_pd_path)
    _save(data=ones_like(cropped_t2), affine=affine, path=input_mask_path)
    _save(data=deformed_pd_valid_region_mask, affine=affine, path=label_mask_path)
    _save(data=reregistered_pd, affine=affine, path=reregistered_pd_path)
    _save(data=reregistered_pd_mask, affine=affine, path=label_reregistered_mask_path)
    _save(
        data=deformation_ddf.numpy().transpose(1, 2, 3, 0),
        affine=affine,
        path=forward_deformation_path,
    )
    _save(
        data=inverse_deformation_ddf.numpy().transpose(1, 2, 3, 0),
        affine=affine,
        path=inverse_deformation_path,
    )


def _main() -> None:
    parser = ArgumentParser()
    parser.add_argument(
        "--target",
        help="Path to where data is generated",
        type=str,
        required=False,
        default="../data/IXI/volumes",
    )
    parser.add_argument(
        "--seed", help="Seed used for random deformation generation", type=int, required=True
    )
    parser.add_argument("--robex-binary-path", help="Path to ROBEX binary", required=True, type=str)
    parser.add_argument(
        "--elastix-binary-path",
        help=(
            "Path to elastix 5.0.1 binary, note that for linux and mac "
            "path to the elastix libraries must be included in $LD_LIBRARY_PATH."
        ),
        required=True,
        type=str,
    )
    parser.add_argument(
        "--transformix-binary-path",
        help=(
            "Path to elastix 5.0.1 transformix binary, note that for linux and mac "
            "path to the elastix libraries must be included in $LD_LIBRARY_PATH."
        ),
        required=True,
        type=str,
    )
    parser.add_argument(
        "--elastix-threads",
        help=("Number of threads to use in elastix registration"),
        required=True,
        type=str,
    )
    parser.add_argument(
        "--do-not-remove-temp-files",
        help="In the end, remove the temporary files used for generating the data set",
        default=False,
        action="store_true",
        required=False,
    )
    args = parser.parse_args()
    target_path = args.target
    robex_path = args.robex_binary_path
    elastix_path = args.elastix_binary_path
    transformix_path = args.transformix_binary_path
    elastix_threads = args.elastix_threads
    remove_temporary_files = not args.do_not_remove_temp_files
    seed = args.seed
    makedirs(target_path, exist_ok=True)
    seed_file_path = join(target_path, "seed.txt")
    if isfile(seed_file_path):
        with open(seed_file_path, mode="r", encoding="utf-8") as seed_file:
            if seed_file.read() != str(seed):
                raise ValueError("Seed does not match the seed of the target location")
    else:
        with open(seed_file_path, mode="w", encoding="utf-8") as seed_file:
            seed_file.write(str(seed))
    print(join(target_path, "T1"))
    if not isdir(join(target_path, "temp", "T1")):
        answer = input(
            "The script will download IXI dataset which is released under "
            "CC BY-SA 3.0 license. Before continuing make sure that you agree "
            "to the terms of use of the data set at https://brain-development.org/ixi-dataset/. "
            "Do you want to continue? (y/n)"
        )
        if answer != "y":
            raise ValueError("You must agree to the data terms of use.")
        print("Downloading T1 files...")
        _download(join(target_path, "temp/T1.tar"), IXI_T1_URL)
        print("Extracting T1 files...")
        _untar(join(target_path, "temp/T1.tar"))
    if not isdir(join(target_path, "temp", "T2")):
        print("Downloading T2 files...")
        _download(join(target_path, "temp/T2.tar"), IXI_T2_URL)
        print("Extracting T2 files...")
        _untar(join(target_path, "temp/T2.tar"))
    if not isdir(join(target_path, "temp", "PD")):
        print("Downloading PD files...")
        _download(join(target_path, "temp/PD.tar"), IXI_PD_URL)
        print("Extracting PD files...")
        _untar(join(target_path, "temp/PD.tar"))
    print("Resampling T1 files...")
    _generate_resampled_t1(
        t1_path=join(target_path, "temp/T1"),
        t2_path=join(target_path, "temp/T2"),
        target_path=join(target_path, "temp/T1_resampled"),
    )
    print("Generating brain masks...")
    _generate_brain_masks(
        t1_path=join(target_path, "temp/T1_resampled"),
        t2_path=join(target_path, "temp/T2"),
        target_path=join(target_path, "temp/brain_mask"),
        robex_path=robex_path,
    )
    cases = sorted(_get_cases(join(target_path, "temp/PD")))
    print("Generating cases...")
    random_state = RandomState(seed=seed)
    for case in tqdm(cases):
        dataset = _get_dataset(case)
        if dataset is not None:
            _generate_case(
                case=case,
                dataset=dataset,
                random_state=random_state,
                pd_dir=join(target_path, "temp/PD"),
                t1_dir=join(target_path, "temp/T1_resampled"),
                t2_dir=join(target_path, "temp/T2"),
                brain_mask_dir=join(target_path, "temp/brain_mask"),
                target_dir=target_path,
                elastix_path=elastix_path,
                transformix_path=transformix_path,
                elastix_threads=elastix_threads,
            )
    if remove_temporary_files:
        print("Removing temporary files...")
        _remove_temp_files(join(target_path, "temp"))
    print(f"Dataset generated to {target_path}.")


if __name__ == "__main__":
    _main()
