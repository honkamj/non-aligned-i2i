"""Preprocess CERMEP-iDB-MRXFDG database dataset to desired path"""

from argparse import ArgumentParser
from json import load as json_load
from os import listdir, makedirs, remove, rmdir
from os.path import abspath, dirname, isfile, join
from shutil import move, rmtree
from subprocess import run
from tarfile import open as open_tar
from tempfile import TemporaryDirectory

from ants import ANTsImage  # type: ignore
from ants import image_read, resample_image_to_target
from ants.utils import iMath  # type: ignore
from ants.utils import n4_bias_field_correction
from intensity_normalization.normalize.fcm import FCMNormalize
from intensity_normalization.typing import Modality, TissueType
from numpy import ones
from scipy.ndimage import median_filter  # type: ignore
from tqdm import tqdm  # type: ignore


def _untar(path: str, target_dir: str) -> None:
    makedirs(target_dir, exist_ok=True)
    tar = open_tar(path)
    tar.extractall(target_dir)
    tar.close()


def _remove_temp_files(temp_folder_path: str) -> None:
    rmtree(temp_folder_path)


def _generate_brain_masks(coregistration_path: str, target_path: str, robex_path: str) -> None:
    for case_name in tqdm(listdir(coregistration_path)):
        print(f"Starting case {case_name}")
        makedirs(join(target_path, case_name), exist_ok=True)
        t1_filepath = join(coregistration_path, case_name, f"{case_name}_space-pet_T1w.nii.gz")
        mask_filepath = join(target_path, case_name, f"{case_name}_space-pet_brain_mask.nii.gz")
        stripped_t1_filepath = join(target_path, case_name, "temp.nii")
        run(
            [robex_path, t1_filepath, stripped_t1_filepath, mask_filepath],
            cwd=dirname(robex_path),
            check=True,
        )
        remove(stripped_t1_filepath)


def _calculate_t1_body_mask(t1_ants: ANTsImage) -> ANTsImage:
    t1_data_median_filtered = median_filter(t1_ants.numpy(), size=3)
    t1_body_mask_mask = (t1_data_median_filtered > 45).astype(t1_data_median_filtered.dtype)
    t1_body_mask_mask_ants = t1_ants.new_image_like(t1_body_mask_mask)
    t1_body_mask_mask_ants = iMath(t1_body_mask_mask_ants, "ME", 1)
    t1_body_mask_mask_ants = iMath(t1_body_mask_mask_ants, "MD", 1)
    t1_body_mask_mask_ants = iMath(t1_body_mask_mask_ants, "GetLargestComponent")
    t1_body_mask_mask_ants = iMath(t1_body_mask_mask_ants, "MC", 8)
    t1_body_mask_mask_ants = iMath(t1_body_mask_mask_ants, "FillHoles").threshold_image(1, 2)
    return t1_body_mask_mask_ants


def _generate_t1(
    coregistration_path: str,
    target_path: str,
) -> None:
    for case_name in tqdm(listdir(coregistration_path)):
        print(f"Starting case {case_name}")
        makedirs(join(target_path, case_name), exist_ok=True)
        t1_filepath = join(coregistration_path, case_name, f"{case_name}_space-pet_T1w.nii.gz")
        brain_mask_filepath = join(
            target_path, case_name, f"{case_name}_space-pet_brain_mask.nii.gz"
        )
        t1_normalized_filepath = join(
            target_path, case_name, f"{case_name}_space-pet_T1w_normalized.nii.gz"
        )
        t1_body_mask_filepath = join(
            target_path, case_name, f"{case_name}_space-pet_T1w_body_mask.nii.gz"
        )
        t1_valid_region_mask_filepath = join(
            target_path, case_name, f"{case_name}_space-pet_T1w_valid_region_mask.nii.gz"
        )
        evaluation_mask_filepath = join(
            target_path, case_name, f"{case_name}_space-pet_evaluation_mask.nii.gz"
        )
        t1_ants = image_read(t1_filepath)
        t1_body_mask_mask_ants = _calculate_t1_body_mask(t1_ants)
        t1_body_mask_mask_ants.to_filename(t1_body_mask_filepath)
        evaluation_mask_ants = iMath(t1_body_mask_mask_ants, "MD", 5)
        evaluation_mask_ants.to_filename(evaluation_mask_filepath)
        t1_bias_field_corrected_ants = n4_bias_field_correction(
            image=t1_ants,
            mask=t1_body_mask_mask_ants,
            convergence={"iters": [400, 300, 200, 100], "tol": 1e-7},
            rescale_intensities=False,
        )
        fcm_norm = FCMNormalize(tissue_type=TissueType.WM)
        brain_mask_ants = image_read(brain_mask_filepath)
        t1_normalized_data = fcm_norm(
            t1_bias_field_corrected_ants.numpy(),
            mask=brain_mask_ants.numpy(),
            modality=Modality.T1,
        )
        t1_normalized_ants = t1_bias_field_corrected_ants.new_image_like(t1_normalized_data)
        t1_normalized_ants.to_filename(t1_normalized_filepath)
        t1_valid_region_mask = ones(t1_ants.numpy().shape, dtype="uint8")
        t1_valid_region_mask_ants = t1_ants.new_image_like(t1_valid_region_mask)
        t1_valid_region_mask_ants.to_filename(t1_valid_region_mask_filepath)
        remove(brain_mask_filepath)


def _calculate_ct_body_mask(ct_ants: ANTsImage) -> ANTsImage:
    ct_median_filtered = median_filter(ct_ants.numpy(), size=3)
    ct_body_mask = (ct_median_filtered > -350).astype(ct_median_filtered.dtype)
    ct_body_mask_ants = ct_ants.new_image_like(ct_body_mask)
    ct_body_mask_ants = iMath(ct_body_mask_ants, "GetLargestComponent")
    ct_body_mask_ants = iMath(ct_body_mask_ants, "MC", 3)
    ct_body_mask_ants = iMath(ct_body_mask_ants, "FillHoles").threshold_image(1, 2)
    ct_body_mask_ants = iMath(ct_body_mask_ants, "MD", 1)
    return ct_body_mask_ants


def _generate_ct(target_path: str, coregistration_path: str) -> None:
    for case_name in tqdm(listdir(target_path)):
        print(f"Starting case {case_name}")
        t1_normalized_filepath = join(
            target_path, case_name, f"{case_name}_space-pet_T1w_normalized.nii.gz"
        )
        ct_rigidly_registered_non_interpolated_filepath = join(
            coregistration_path, case_name, f"{case_name}_space-pet_ct.nii.gz"
        )
        ct_rigidly_registered_masked_filepath = join(
            target_path, case_name, f"{case_name}_space-pet_ct_rigidly_registered_masked.nii.gz"
        )
        ct_rigidly_registered_non_interpolated_masked_filepath = join(
            target_path,
            case_name,
            f"{case_name}_space-pet_ct_rigidly_registered_non_interpolated_masked.nii.gz",
        )
        ct_rigidly_registered_body_mask_filepath = join(
            target_path,
            case_name,
            f"{case_name}_space-pet_ct_rigidly_registered_body_mask.nii.gz",
        )
        ct_rigidly_registered_non_interpolated_body_mask_filepath = join(
            target_path,
            case_name,
            f"{case_name}_space-pet_ct_rigidly_registered_non_interpolated_body_mask.nii.gz",
        )
        ct_rigidly_registered_valid_region_mask_filepath = join(
            target_path,
            case_name,
            f"{case_name}_space-pet_ct_rigidly_registered_valid_region_mask.nii.gz",
        )
        t1_normalized_ants = image_read(t1_normalized_filepath)
        ct_rigidly_registered_non_interpolated_ants = image_read(
            ct_rigidly_registered_non_interpolated_filepath
        )
        ct_rigidly_registered_ants = resample_image_to_target(
            ct_rigidly_registered_non_interpolated_ants, t1_normalized_ants, interp_type="bSpline"
        )
        ct_rigidly_registered_valid_region_mask = ones(
            ct_rigidly_registered_ants.numpy().shape, dtype="uint8"
        )
        ct_rigidly_registered_valid_region_mask_ants = ct_rigidly_registered_ants.new_image_like(
            ct_rigidly_registered_valid_region_mask
        )
        ct_rigidly_registered_valid_region_mask_ants.to_filename(
            ct_rigidly_registered_valid_region_mask_filepath
        )
        ct_rigidly_registered_body_mask_ants = _calculate_ct_body_mask(ct_rigidly_registered_ants)
        ct_rigidly_registered_non_interpolated_body_mask_ants = _calculate_ct_body_mask(
            ct_rigidly_registered_non_interpolated_ants
        )
        ct_rigidly_registered_body_mask_ants.to_filename(ct_rigidly_registered_body_mask_filepath)
        ct_rigidly_registered_non_interpolated_body_mask_ants.to_filename(
            ct_rigidly_registered_non_interpolated_body_mask_filepath
        )
        ct_rigidly_registered_non_interpolated_ants.new_image_like(
            ct_rigidly_registered_non_interpolated_body_mask_ants.numpy()
            * ct_rigidly_registered_non_interpolated_ants.numpy()
            - (1 - ct_rigidly_registered_non_interpolated_body_mask_ants.numpy()) * 1024
        ).to_filename(ct_rigidly_registered_non_interpolated_masked_filepath)
        ct_rigidly_registered_ants.new_image_like(
            ct_rigidly_registered_body_mask_ants.numpy() * ct_rigidly_registered_ants.numpy()
            - (1 - ct_rigidly_registered_body_mask_ants.numpy()) * 1024
        ).to_filename(ct_rigidly_registered_masked_filepath)


def _register_ct(target_path: str, elastix_path: str, elastix_threads: str) -> None:
    for case_name in tqdm(listdir(target_path)):
        print(f"Starting case {case_name}")
        makedirs(join(target_path, case_name), exist_ok=True)
        t1_filepath = join(target_path, case_name, f"{case_name}_space-pet_T1w_normalized.nii.gz")
        ct_filepath = join(
            target_path,
            case_name,
            f"{case_name}_space-pet_ct_rigidly_registered_non_interpolated_masked.nii.gz",
        )
        ct_registered_filepath = join(
            target_path, case_name, f"{case_name}_space-pet_ct_deformably_registered_masked.nii.gz"
        )
        with TemporaryDirectory() as temp_dir:
            run(
                [
                    elastix_path,
                    "-f",
                    t1_filepath,
                    "-m",
                    ct_filepath,
                    "-p",
                    "../data/CERMEP/elastix_registration_params.txt",
                    "-out",
                    temp_dir,
                    "-threads",
                    elastix_threads,
                ],
                check=True,
            )
            registered_temp_path = join(temp_dir, "result.0.nii.gz")
            if isfile(ct_registered_filepath):
                remove(ct_registered_filepath)
            move(registered_temp_path, ct_registered_filepath)


def _is_in_dataset(case: str, dataset: str) -> bool:
    with open(f"../data/CERMEP/{dataset}_cases.json", encoding="utf-8") as cases_file:
        cases = json_load(cases_file)
    return case in cases


def _distribute_files(
    source_path: str,
    target_path: str,
) -> None:
    datasets = ["train", "test", "validate"]
    for dataset in datasets:
        makedirs(join(target_path, dataset), exist_ok=True)
    for case_name in tqdm(listdir(source_path)):
        for dataset in datasets:
            if _is_in_dataset(case_name, dataset):
                move(src=join(source_path, case_name), dst=join(target_path, dataset, case_name))
                break
    rmdir(source_path)


def _main() -> None:
    parser = ArgumentParser()
    parser.add_argument(
        "--target",
        help="Path to where data is generated",
        type=str,
        required=False,
        default="../data/CERMEP/volumes",
    )
    parser.add_argument(
        "--source",
        help=(
            "Path to the CERMEP-iDB-MRXFDG database arhive (iDB-CERMEP-MRXFDG_MRI_ct.tar.gz). "
            "The database can be requested from the authors of the database. "
            '(Mérida, Inés, et al. "CERMEP-IDB-MRXFDG: a database of 37 normal adult '
            "human brain [18F] FDG PET, T1 and FLAIR MRI, and CT images available for "
            'research." EJNMMI research 11.1 (2021))'
        ),
        type=str,
        required=True,
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
        "--elastix-threads",
        help=("Number of threads to use in elastix registration"),
        required=True,
        type=str,
    )
    args = parser.parse_args()
    target_path = args.target
    temporary_target_path = join(target_path, "temp")
    robex_path = args.robex_binary_path
    elastix_path = args.elastix_binary_path
    elastix_threads = args.elastix_threads
    coregistration_path = join(
        target_path,
        (
            "raw/home/pool/DM/TEP/CERMEP_MXFDG/BASE/DATABASE_SENT"
            "/MRI_CT/derivatives/coregistration"
        ),
    )
    if not isfile(args.source):
        raise FileNotFoundError(
            f'CERMEP-iDB-MRXFDG database archive not found in the path "{abspath(args.source)}". '
            "The database can be requested from the authors of the database. "
            '(Mérida, Inés, et al. "CERMEP-IDB-MRXFDG: a database of 37 normal adult '
            "human brain [18F] FDG PET, T1 and FLAIR MRI, and CT images available for "
            'research." EJNMMI research 11.1 (2021))'
        )
    print("Extracting...")
    _untar(args.source, join(target_path, "raw"))
    print("Generating brain masks for T1 normalization using ROBEX...")
    _generate_brain_masks(
        coregistration_path=coregistration_path,
        target_path=temporary_target_path,
        robex_path=robex_path,
    )
    print("Preprocessing T1 MRI images...")
    _generate_t1(coregistration_path=coregistration_path, target_path=temporary_target_path)
    print("Preprocessing CT images...")
    _generate_ct(target_path=temporary_target_path, coregistration_path=coregistration_path)
    print("Deformably registering CT images using elastix...")
    _register_ct(
        target_path=temporary_target_path,
        elastix_path=elastix_path,
        elastix_threads=elastix_threads,
    )
    print("Splitting dataset")
    _distribute_files(source_path=temporary_target_path, target_path=target_path)
    print("Removing temporary files...")
    _remove_temp_files(join(target_path, "raw"))
    print(f"Dataset generated to {target_path}.")


if __name__ == "__main__":
    _main()
