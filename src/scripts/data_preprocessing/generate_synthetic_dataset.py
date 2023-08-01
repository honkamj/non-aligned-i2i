"""Generates IXI dataset to desired path"""

from argparse import ArgumentParser
from json import load as json_load
from os import listdir, makedirs, remove
from os.path import basename, dirname, isfile, join
from shutil import move, rmtree
from zipfile import ZipFile
from typing import List, Optional
from urllib.request import urlretrieve

from tqdm import tqdm  # type: ignore


COCO_UNLABELED_2017_URL = 'http://images.cocodataset.org/zips/unlabeled2017.zip'


def _download(
        path: str,
        url: str
    ) -> None:
    makedirs(dirname(path), exist_ok=True)
    progress_bar = None
    previous_recieved = 0
    def _show_progress(block_num, block_size, total_size):
        nonlocal progress_bar, previous_recieved
        if progress_bar is None:
            progress_bar = tqdm(
                unit='B',
                total=total_size)
        downloaded = block_num * block_size
        if downloaded < total_size:
            progress_bar.update(downloaded - previous_recieved)
            previous_recieved = downloaded
        else:
            progress_bar.close()
    if not isfile(path):
        urlretrieve(
            url,
            path,
            _show_progress
        )


def _unzip(
        path: str
    ) -> None:
    target_dir = join(
        dirname(path),
        basename(path)[:-4]
    )
    makedirs(
        target_dir,
        exist_ok=True)
    with ZipFile(path, 'r') as zip_file:
        zip_file.extractall(target_dir)
    remove(path)


def _get_cases(source_path: str) -> List[str]:
    return listdir(source_path)


def _is_in_dataset(case: str, dataset: str) -> bool:
    with open(
            f'../data/coco_unlabeled2017/{dataset}_cases.json',
            encoding='utf-8'
        ) as cases_file:
        cases = json_load(cases_file)
    return case in cases


def _get_dataset(case: str) -> Optional[str]:
    options = ['train', 'validate', 'test']
    for option in options:
        if _is_in_dataset(case, option):
            return option
    return None


def _move_case(
        case: str,
        dataset: str,
        source_dir: str,
        target_dir: str
    ) -> None:
    source_file = join(source_dir, case)
    target_file = join(target_dir, dataset, case)
    makedirs(dirname(target_file), exist_ok=True)
    move(source_file, target_file)


def _remove_temp_files(temp_folder_path: str) -> None:
    rmtree(temp_folder_path)


def _main() -> None:
    parser = ArgumentParser()
    parser.add_argument(
        '--target',
        help='Path to where data is generated',
        type=str,
        required=False,
        default='../data/coco_unlabeled2017/images')
    parser.add_argument(
        '--do-not-remove-temp-files',
        help='In the end, remove the temporary files used for generating the data set',
        default=False,
        action="store_true",
        required=False)
    args = parser.parse_args()
    target_path = args.target
    remove_temporary_files = not args.do_not_remove_temp_files
    print('Downloading files...')
    answer = input(
        "The script will download COCO 2017 unlabeled dataset which is released under "
        "Creative Commons Attribution 4.0 License license. Before continuing make sure "
        "that you agree to the terms of use of the data set at "
        "https://cocodataset.org/#termsofuse. "
        "Do you want to continue? (y/n)"
    )
    if answer.lower() != "y":
        raise ValueError("You must agree to the data terms of use.")
    _download(join(target_path, 'temp/coco_unlabeled2017.zip'), COCO_UNLABELED_2017_URL)
    print('Extracting...')
    _unzip(join(target_path, 'temp/coco_unlabeled_2017.zip'))
    cases = _get_cases(join(target_path, 'temp/coco_unlabeled2017/unlabeled2017'))
    print('Creating dataset splits...')
    for case in tqdm(cases):
        dataset = _get_dataset(case)
        if dataset is not None:
            _move_case(
                case=case,
                dataset=dataset,
                source_dir=join(target_path, 'temp/coco_unlabeled2017/unlabeled2017'),
                target_dir=target_path)
    if remove_temporary_files:
        print('Removing temporary files...')
        _remove_temp_files(join(target_path, 'temp'))
    print(f'Dataset generated to {target_path}.')


if __name__ == "__main__":
    _main()
