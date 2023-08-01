"""Data handling related utility functions"""

from os.path import isdir
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

from util.training import get_path


def get_data_root(root_candidates: Iterable[str]) -> str:
    """Get data root from config"""
    root_directory: Optional[str] = None
    for root_candidate in root_candidates:
        if isdir(root_candidate):
            root_directory = root_candidate
    if root_directory is None:
        raise ValueError('Could not find data root!')
    return root_directory


def obtain_paths(
        data_root: str,
        type_to_postfix: Mapping[str, str],
        samples: Sequence[str]
    ) -> Mapping[str, Sequence[str]]:
    """Obtain paths to all samples of each file type"""
    type_to_paths: Dict[str, List[str]] = {}
    for file_type, postfix in type_to_postfix.items():
        type_to_paths[file_type] = [get_path(data_root, postfix, sample) for sample in samples]
    return type_to_paths
