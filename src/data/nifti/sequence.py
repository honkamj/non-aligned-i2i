"""Sequence for iterating multiple large tiffs as patches"""

from nibabel import load as nib_load  # type: ignore
from numpy import memmap, ndarray

from data.base import (BaseEvaluationArrayPatchSequence,
                       BaseInferenceArrayPatchSequence,
                       BaseTrainingArrayPatchSequence)


def _nifti_to_memmap(path: str) -> memmap:
    return nib_load(path).dataobj


class TrainingNiftiPatchSequence(BaseTrainingArrayPatchSequence):
    """Samples patches from multiple niftis"""
    def _to_array(self, path: str) -> ndarray:
        return _nifti_to_memmap(path)


class InferenceNiftiPatchSequence(BaseInferenceArrayPatchSequence):
    """Obtains patches from niftis for inference"""
    def _to_array(self, path: str) -> ndarray:
        return _nifti_to_memmap(path)


class EvaluationNiftiPatchSequence(BaseEvaluationArrayPatchSequence):
    """Obtains patches from niftis for evaluation"""
    def _to_array(self, path: str) -> ndarray:
        return _nifti_to_memmap(path)

    def _deformation_to_array(self, path: str) -> ndarray:
        return _nifti_to_memmap(path)
