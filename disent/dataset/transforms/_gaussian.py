import cv2
import numpy as np


# ========================================================================= #
# Gaussian Blur                                                             #
# ========================================================================= #


class GaussianBlurTransform(object):
    """
    Gaussian blur as described in the SimCLR paper (without the random application chance).
    - had random chance to choose blur radius
    - https://github.com/sthalles/SimCLR
    """
    
    def __init__(self, kernel_size, sigma_min=0.1, sigma_max=2.0):
        self.kernel_size = kernel_size
        self._sigma_min = sigma_min
        self._sigma_max = sigma_max

    def __call__(self, sample):
        # TODO: fix casting. Is this actually needed?
        # sample = np.array(sample)
        assert sample.dtype == np.float32  # or sample.dtype == np.float16
        sigma = (self._sigma_max - self._sigma_min) * np.random.random_sample() + self._sigma_min
        return cv2.GaussianBlur(sample, (self.kernel_size, self.kernel_size), sigma)


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
