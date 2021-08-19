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
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
from PIL.Image import Image
import torch
import torchvision.transforms.functional as F_tv


# ========================================================================= #
# Functional Transforms                                                     #
# ========================================================================= #


def noop(obs):
    """
    Transform that does absolutely nothing!
    """
    return obs


def check_tensor(obs, low: Optional[float] = 0., high: Optional[float] = 1., dtype=torch.float32):
    """
    Check that the input is a tensor, its datatype matches, and
    that it is in the required range.
    """
    # check is a tensor
    assert torch.is_tensor(obs), 'observation is not a tensor'
    # check type
    if dtype is not None:
        assert obs.dtype == dtype, f'tensor type {obs.dtype} is not required type {dtype}'
    # check range | TODO: are assertion strings inefficient?
    if low is not None:
        assert low <= obs.min(), f'minimum value of tensor {obs.min()} is less than allowed minimum value: {low}'
    if high is not None:
        assert obs.max() <= high, f'maximum value of tensor {obs.max()} is greater than allowed maximum value: {high}'
    # DONE!
    return obs


Obs = Union[np.ndarray, Image]
SizeType = Union[int, Tuple[int, int]]


def to_uint_tensor(
    obs: Obs,
    size: Optional[SizeType] = None,
    channel_to_front: bool = True
) -> torch.Tensor:
    # resize image
    if size is not None:
        if not isinstance(obs, Image):
            obs = F_tv.to_pil_image(obs)
        obs = F_tv.resize(obs, size=size)
    # to numpy
    if not isinstance(obs, np.ndarray):
        obs = np.array(obs)
    # to tensor
    obs = torch.from_numpy(obs)
    # move axis
    if channel_to_front:
        obs = torch.moveaxis(obs, -1, -3)
    # checks
    assert obs.dtype == torch.uint8
    # done!
    return obs


def to_standardised_tensor(
    obs: Obs,
    size: Optional[SizeType] = None,
    cast_f32: bool = False,
    check: bool = True,
    check_range: bool = True,
) -> torch.Tensor:
    """
    Basic transform that should be applied to
    any dataset before augmentation.

    1. resize if size is specified
    2. convert to tensor in range [0, 1]
    """
    # resize image
    if size is not None:
        if not isinstance(obs, Image):
            obs = F_tv.to_pil_image(obs)
        obs = F_tv.resize(obs, size=size)
    # transform to tensor
    obs = F_tv.to_tensor(obs)
    # cast if needed
    if cast_f32:
        obs = obs.to(torch.float32)
    # check that tensor is valid
    if check:
        if check_range:
            obs = check_tensor(obs, low=0, high=1, dtype=torch.float32)
        else:
            obs = check_tensor(obs, low=None, high=None, dtype=torch.float32)
    return obs


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
