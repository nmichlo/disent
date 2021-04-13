#  ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~
#  MIT License
#
#  Copyright (c) 2021 Nathan Juraj Michlo
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in
#  all copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.
#  ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~
from numbers import Number
from typing import List
from typing import Tuple
from typing import Union

import numpy as np
import torch

from disent.util.math import torch_gaussian_kernel_2d
import disent.transform.functional as F_d


# ========================================================================= #
# Transforms                                                                #
# ========================================================================= #


TorLorN = Union[Number, Tuple[Number, Number], List[Number], np.ndarray]
MmTuple = Union[TorLorN, Tuple[TorLorN, TorLorN], List[TorLorN], np.ndarray]


def _expand_to_min_max_tuples(input: MmTuple) -> Tuple[Tuple[Number, Number], Tuple[Number, Number]]:
    (xm, xM), (ym, yM) = np.broadcast_to(input, (2, 2)).tolist()
    if not all(isinstance(n, (float, int)) for n in [xm, xM, ym, yM]):
        raise ValueError('only scalars, tuples with shape (2,): [m, M] or tuples with shape (2, 2): [[xm, xM], [ym, yM]] are supported')
    return (xm, xM), (ym, yM)


class FftGaussianBlur(object):
    """
    randomly gaussian blur the input images.
    - similar api to kornia
    """

    def __init__(self, sigma: MmTuple = 1.0, truncate: MmTuple = 3.0, p: float = 0.5, random_mode='batch', random_same_xy=True):
        assert 0 <= p <= 1, f'probability of applying transform p={repr(p)} must be in range [0, 1]'
        self.sigma = _expand_to_min_max_tuples(sigma)
        self.trunc = _expand_to_min_max_tuples(truncate)
        self.p = p
        # random modes
        self.ran_batch, self.ran_channels = {
            'same':     (False, False),
            'batch':    (True,  False),
            'all':      (True,  True),
            'channels': (False, True),
        }[random_mode]
        # same random value for x and y
        if random_same_xy:
            assert self.sigma[0] == self.sigma[1]
            assert self.trunc[0] == self.trunc[1]
        self.b_idx = 0 if random_same_xy else 1
        # original input values for self.__repr__()
        self._orig_sigma = sigma
        self._orig_truncate = truncate
        self._orig_p = p
        self._orig_random_mode = random_mode
        self._orig_random_same_xy = random_same_xy

    def __call__(self, obs):
        # randomly return original
        if np.random.random() < (1 - self.p):
            return obs
        # get batch shape
        B, C, H, W = obs.shape
        # sigma & truncate
        sigma_m, sigma_M = torch.as_tensor(self.sigma, device=obs.device).T
        trunc_m, trunc_M = torch.as_tensor(self.trunc, device=obs.device).T
        # generate random values
        sigma = sigma_m + torch.rand((B if self.ran_batch else 1), (C if self.ran_channels else 1), 2, dtype=torch.float32, device=obs.device) * (sigma_M - sigma_m)
        trunc = trunc_m + torch.rand((B if self.ran_batch else 1), (C if self.ran_channels else 1), 2, dtype=torch.float32, device=obs.device) * (trunc_M - trunc_m)
        # generate kernel
        kernel = torch_gaussian_kernel_2d(
            sigma=sigma[..., 0], truncate=trunc[..., 0],
            sigma_b=sigma[..., self.b_idx], truncate_b=trunc[..., self.b_idx],  # TODO: we do generate unneeded random values if random_same_xy == True
            dtype=torch.float32, device=obs.device,
        )
        # apply kernel
        return F_d.conv2d_channel_wise_fft(signal=obs, kernel=kernel)

    def __repr__(self):
        return f'{self.__class__.__name__}(sigma={repr(self._orig_sigma)}, {repr(self._orig_truncate)}, p={repr(self._orig_p)}, random_mode={repr(self._orig_random_mode)}, random_same_xy={repr(self._orig_random_same_xy)})'


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
