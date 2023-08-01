"""Image utility functions"""

from numpy import ndarray
from skimage.color import gray2rgb  # type: ignore


def gray_to_color_if_needed(image: ndarray) -> ndarray:
    """Converts grayscale image to color image

    Color images will not be altered.
    """
    if image.ndim != 3:
        return gray2rgb(image)
    return image
