"""Generates IXI dataset to desired path"""

from argparse import ArgumentParser
from json import load as json_load
from os import makedirs
from os.path import abspath, isfile, join
from shutil import rmtree
from tarfile import open as open_tar

from tifffile import imread as tiff_imread  # type: ignore
from tifffile import imwrite as tiff_imwrite


def _untar(path: str, target_dir: str) -> None:
    makedirs(target_dir, exist_ok=True)
    with open_tar(path) as tar:
        tar.extractall(target_dir)


def _get_cases() -> list[str]:
    divisions = ["test", "train", "validate"]
    cases = []
    for division in divisions:
        with open(
            f"../data/virtual_staining/{division}_cases.json", encoding="utf-8"
        ) as cases_file:
            cases.extend(json_load(cases_file))
    return cases


def _main() -> None:
    parser = ArgumentParser()
    parser.add_argument(
        "--source",
        help=(
            'Path to "virtual-staining-data.tar" archive downloadable from '
            "https://doi.org/10.23729/9ddc2fc5-9bdb-404c-be07-c9c9540a32de"
        ),
        type=str,
        required=True,
    )
    parser.add_argument(
        "--target",
        help="Path to where data is generated",
        type=str,
        required=False,
        default="../data/virtual_staining/images",
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
    remove_temporary_files = not args.do_not_remove_temp_files
    makedirs(target_path, exist_ok=True)
    if not isfile(args.source):
        raise FileNotFoundError(
            f'Virtual staining data archive not found in the path "{abspath(args.source)}". '
            "The archieve can be downloaded from "
            "https://doi.org/10.23729/9ddc2fc5-9bdb-404c-be07-c9c9540a32de "
            "and is released under Creative Commons Attribution 4.0 International "
            "(CC BY 4.0) license. "
        )
    print("Extracting histopathology data set files...")
    _untar(args.source, join(target_path, "temp"))
    print("Processing files...")
    # It is important to write the final tiff-files with tiff_imwrite as it modifies
    # the data structure such that the images can be read as a memory map.
    for case in _get_cases():
        makedirs(join(target_path, case))
        for postfix in ["mask", "stained", "unstained"]:
            data = tiff_imread(
                join(
                    target_path,
                    "temp",
                    "data",
                    case,
                    f"{case}_{postfix}.tif",
                )
            )
            tiff_imwrite(join(target_path, case, f"{case}_{postfix}.tif"), data)
    print("Removing temporary files...")
    if remove_temporary_files:
        rmtree(join(target_path, "temp"))
    print(f"Dataset generated to {target_path}.")


if __name__ == "__main__":
    _main()
