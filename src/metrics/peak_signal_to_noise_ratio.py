"""Peak signal to noise ratio"""

from math import log10


class PSNRMassFunction:
    """PSNR mass function for metric gathering"""
    def __init__(self, max_value: float) -> None:
        self._max_value = max_value

    def __call__(
            self,
            square_sum: float,
            pixel_mass: float
        ) -> float:
        """Calculate PSNR from square sum and amount of pixels"""
        if pixel_mass == 0:
            return float('nan')
        return 20 * log10(self._max_value) - 10 * log10(square_sum) + 10 * log10(pixel_mass)
