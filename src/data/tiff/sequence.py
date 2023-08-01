"""Sequence for iterating multiple large tiffs as patches"""

from numpy import memmap, ndarray
from tifffile import memmap as tiff_memmap  # type: ignore

from data.base import (BaseEvaluationArrayPatchSequence,
                       BaseInferenceArrayPatchSequence,
                       BaseTrainingArrayPatchSequence)


def _tiff_to_memmap(path: str) -> memmap:
    return tiff_memmap(path, mode='r')


class TrainingTiffPatchSequence(BaseTrainingArrayPatchSequence):
    """Samples patches from multiple tiffs"""
    def _to_array(self, path: str) -> ndarray:
        return _tiff_to_memmap(path)


class InferenceTiffPatchSequence(BaseInferenceArrayPatchSequence):
    """Obtains patches from tiffs for inference"""
    def _to_array(self, path: str) -> ndarray:
        return _tiff_to_memmap(path)


class EvaluationTiffPatchSequence(BaseEvaluationArrayPatchSequence):
    """Obtains patches from tiffs for evaluation"""
    def _to_array(self, path: str) -> ndarray:
        return _tiff_to_memmap(path)

    def _deformation_to_array(self, path: str) -> ndarray:
        return _tiff_to_memmap(path)
