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

import os
import re
from numbers import Number
from typing import List
from typing import Tuple
from typing import Union

import numpy as np
import torch

import disent
from disent.nn.modules import DisentModule
from disent.nn.functional import torch_box_kernel_2d
from disent.nn.functional import torch_conv2d_channel_wise_fft
from disent.nn.functional import torch_gaussian_kernel_2d


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


class _BaseFftBlur(DisentModule):
    """
    randomly gaussian blur the input images.
    - similar api to kornia
    """

    def __init__(self, p: float = 0.5, random_mode='batch', random_same_xy=True):
        super().__init__()
        # check arguments
        assert 0 <= p <= 1, f'probability of applying transform p={repr(p)} must be in range [0, 1]'
        self.p = p
        # random modes
        self.ran_batch, self.ran_channels = {
            'same':     (False, False),
            'batch':    (True,  False),
            'all':      (True,  True),
            'channels': (False, True),
        }[random_mode]
        # same random value for x and y
        self.b_idx = 0 if random_same_xy else 1

    def forward(self, obs):
        # randomly return original
        if np.random.random() < (1 - self.p):
            return obs
        # add or remove batch dim
        add_batch_dim = (obs.ndim == 3)
        if add_batch_dim:
            obs = obs[None, ...]
        # apply kernel
        kernel = self._make_kernel(obs.shape, obs.device)
        result = torch_conv2d_channel_wise_fft(signal=obs, kernel=kernel)
        # remove batch dim
        if add_batch_dim:
            result = result[0]
        # done!
        return result

    def _make_kernel(self, shape, device):
        raise NotImplementedError


class FftGaussianBlur(_BaseFftBlur):
    """
    randomly gaussian blur the input images.
    - similar api to kornia
    """

    def __init__(
        self,
        sigma: MmTuple = 1.0,
        truncate: MmTuple = 3.0,
        p: float = 0.5,
        random_mode='batch',
        random_same_xy=True
    ):
        super().__init__(p=p, random_mode=random_mode, random_same_xy=random_same_xy)
        self.sigma = _expand_to_min_max_tuples(sigma)
        self.trunc = _expand_to_min_max_tuples(truncate)
        # same random value for x and y
        if random_same_xy:
            assert self.sigma[0] == self.sigma[1]
            assert self.trunc[0] == self.trunc[1]

    def _make_kernel(self, shape, device):
        B, C, H, W = shape
        # sigma & truncate
        sigma_m, sigma_M = torch.as_tensor(self.sigma, device=device).T
        trunc_m, trunc_M = torch.as_tensor(self.trunc, device=device).T
        # generate random values
        sigma = sigma_m + torch.rand((B if self.ran_batch else 1), (C if self.ran_channels else 1), 2, dtype=torch.float32, device=device) * (sigma_M - sigma_m)
        trunc = trunc_m + torch.rand((B if self.ran_batch else 1), (C if self.ran_channels else 1), 2, dtype=torch.float32, device=device) * (trunc_M - trunc_m)
        # generate kernel
        return torch_gaussian_kernel_2d(
            sigma=sigma[..., 0], truncate=trunc[..., 0],
            sigma_b=sigma[..., self.b_idx], truncate_b=trunc[..., self.b_idx],  # TODO: we do generate unneeded random values if random_same_xy == True
            dtype=torch.float32, device=device,
        )


class FftBoxBlur(_BaseFftBlur):
    """
    randomly box blur the input images.
    - similar api to kornia
    """

    def __init__(
        self,
        radius: MmTuple = 1,
        p: float = 0.5,
        random_mode='batch',
        random_same_xy=True
    ):
        super().__init__(p=p, random_mode=random_mode, random_same_xy=random_same_xy)
        self.radius: Tuple[Tuple[int, int], Tuple[int, int]] = _expand_to_min_max_tuples(radius)
        # same random value for x and y
        if random_same_xy:
            assert self.radius[0] == self.radius[1]
        # check values
        values = np.array(self.radius).flatten().tolist()
        assert all(isinstance(x, int) for x in values), 'radius values must be integers'
        assert all((0 <= x) for x in values), 'radius values must be >= 0, resulting in diameter: 2*r+1'

    def _make_kernel(self, shape, device):
        B, C, H, W = shape
        # sigma & truncate
        (rym, ryM), (rxm, rxM) = self.radius
        # generate random values
        radius_y = torch.randint(low=rym, high=ryM+1, size=((B if self.ran_batch else 1), (C if self.ran_channels else 1)), device=device)
        radius_x = torch.randint(low=rxm, high=rxM+1, size=((B if self.ran_batch else 1), (C if self.ran_channels else 1)), device=device)
        # done computing kernel
        return torch_box_kernel_2d(
            radius=radius_y,
            radius_b=radius_x if (self.b_idx == 1) else radius_y,
            dtype=torch.float32, device=device
        )


# ========================================================================= #
# FFT Kernel                                                                #
# ========================================================================= #


class FftKernel(DisentModule):
    """
    2D Convolve an image
    """

    def __init__(self, kernel: Union[torch.Tensor, str], normalize: bool = True):
        super().__init__()
        # load & save the kernel -- no gradients allowed
        self._kernel: torch.Tensor
        self.register_buffer('_kernel', get_kernel(kernel, normalize=normalize), persistent=True)
        self._kernel.requires_grad = False

    def forward(self, obs):
        # add or remove batch dim
        add_batch_dim = (obs.ndim == 3)
        if add_batch_dim:
            obs = obs[None, ...]
        # apply kernel
        result = torch_conv2d_channel_wise_fft(signal=obs, kernel=self._kernel)
        # remove batch dim
        if add_batch_dim:
            result = result[0]
        # done!
        return result


# ========================================================================= #
# Kernels                                                                   #
# ========================================================================= #


def _normalise_kernel(kernel: torch.Tensor, normalize: bool) -> torch.Tensor:
    if normalize:
        with torch.no_grad():
            return kernel / kernel.sum()
    return kernel


def _check_kernel(kernel: torch.Tensor) -> torch.Tensor:
    # check kernel
    assert isinstance(kernel, torch.Tensor)
    assert kernel.dtype == torch.float32
    assert kernel.ndim == 4, f'invalid number of kernel dims, required 4, given: {repr(kernel.ndim)}'  # B, C, H, W
    assert kernel.shape[0] == 1, f'invalid size of first kernel dim, required (1, ?, ?, ?), given: {repr(kernel.shape)}'  # B
    assert kernel.shape[0] in (1, 3), f'invalid size of second kernel dim, required (?, 1 or 3, ?, ?), given: {repr(kernel.shape)}'  # C
    # done checks
    return kernel


_KERNELS = {
    # kernels that do not require arguments, just general factory functions
    # name: class/fn -- with no required args
}


_ARG_KERNELS = [
    # (REGEX, EXAMPLE, FACTORY_FUNC)
    # - factory function takes at min one arg: fn(reduction) with one arg after that per regex capture group
    # - regex expressions are tested in order, expressions should be mutually exclusive or ordered such that more specialized versions occur first.
    (re.compile(r'^(box)_r(\d+)$'), 'box_r31', lambda kern, radius: torch_box_kernel_2d(radius=int(radius))[None, ...]),
    (re.compile(r'^(gau)_r(\d+)$'), 'gau_r31', lambda kern, radius: torch_gaussian_kernel_2d(sigma=int(radius) / 4.0, truncate=4.0)[None, None, ...]),
]


# NOTE: this function compliments make_reconstruction_loss in frameworks/helper/reconstructions.py
def _make_kernel(name: str) -> torch.Tensor:
    if name in _KERNELS:
        # search normal losses!
        return _KERNELS[name]()
    else:
        # regex search kernels, and call with args!
        for r, _, fn in _ARG_KERNELS:
            result = r.search(name)
            if result is not None:
                return fn(*result.groups())
    # we couldn't find anything
    raise KeyError(f'Invalid kernel name: {repr(name)} Examples of argument based kernels include: {[example for _, example, _ in _ARG_KERNELS]}')


def make_kernel(name: str, normalize: bool = False):
    kernel = _make_kernel(name)
    kernel = _normalise_kernel(kernel, normalize=normalize)
    kernel = _check_kernel(kernel)
    return kernel


def _get_kernel(name_or_path: str) -> torch.Tensor:
    if '/' not in name_or_path:
        try:
            return _make_kernel(name_or_path)
        except KeyError:
            pass
    if os.path.isfile(name_or_path):
        return torch.load(name_or_path)
    raise KeyError(f'Invalid kernel path or name: {repr(name_or_path)} Examples of argument based kernels include: {[example for _, example, _ in _ARG_KERNELS]}, otherwise specify a valid path to a kernel file save with torch.')


def get_kernel(kernel: Union[str, torch.Tensor], normalize: bool = False):
    kernel = _get_kernel(kernel) if isinstance(kernel, str) else torch.clone(kernel)
    kernel = _normalise_kernel(kernel, normalize=normalize)
    kernel = _check_kernel(kernel)
    return kernel


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
