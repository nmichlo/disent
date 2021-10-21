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
from typing import Sequence

import torch
import disent.dataset.transform.functional as F_d


# ========================================================================= #
# Transforms                                                                #
# ========================================================================= #
from disent.util.deprecate import deprecated


class Noop(object):
    """
    Transform that does absolutely nothing!

    See: disent.transform.functional.noop
    """

    def __call__(self, obs):
        return obs

    def __repr__(self):
        return f'{self.__class__.__name__}()'


class CheckTensor(object):
    """
    Check that the data is a tensor, the right dtype, and in the required range.

    See: disent.transform.functional.check_tensor
    """

    def __init__(
        self,
        low: Optional[float] = 0.,
        high: Optional[float] = 1.,
        dtype: Optional[torch.dtype] = torch.float32,
    ):
        self._low = low
        self._high = high
        self._dtype = dtype

    def __call__(self, obs):
        return F_d.check_tensor(obs, low=self._low, high=self._high, dtype=self._dtype)

    def __repr__(self):
        kwargs = dict(low=self._low, high=self._high, dtype=self._dtype)
        kwargs = ", ".join(f"{k}={repr(v)}" for k, v in kwargs.items() if (v is not None))
        return f'{self.__class__.__name__}({kwargs})'


class ToImgTensorF32(object):
    """
    Basic transform that should be applied to most datasets, making sure
    the image tensor is float32 and a specified size.

    Steps:
        1. resize image if size is specified
        2. if we have integer inputs, divide by 255
        3. add missing channel to greyscale image
        4. move channels to first dim (H, W, C) -> (C, H, W)
        5. normalize using mean and std, values might thus be outside of the range [0, 1]

    See: disent.transform.functional.to_img_tensor_f32
    """

    def __init__(
        self,
        size: Optional[F_d.SizeType] = None,
        mean: Optional[Sequence[float]] = None,
        std: Optional[Sequence[float]] = None,
    ):
        self._size = size
        self._mean = tuple(mean) if (mean is not None) else None
        self._std = tuple(std) if (std is not None) else None

    def __call__(self, obs) -> torch.Tensor:
        return F_d.to_img_tensor_f32(obs, size=self._size, mean=self._mean, std=self._std)

    def __repr__(self):
        kwargs = dict(size=self._size, mean=self._mean, std=self._std)
        kwargs = ", ".join(f"{k}={repr(v)}" for k, v in kwargs.items() if (v is not None))
        return f'{self.__class__.__name__}({kwargs})'


class ToImgTensorU8(object):
    """
    Basic transform that makes sure the image tensor is uint8 and a specified size.

    Steps:
    1. resize image if size is specified
    2. add missing channel to greyscale image
    3. move channels to first dim (H, W, C) -> (C, H, W)

    See: disent.transform.functional.to_img_tensor_u8
    """

    def __init__(
        self,
        size: Optional[F_d.SizeType] = None,
    ):
        self._size = size

    def __call__(self, obs) -> torch.Tensor:
        return F_d.to_img_tensor_u8(obs, size=self._size)

    def __repr__(self):
        kwargs = f'size={repr(self._size)}' if (self._size is not None) else ''
        return f'{self.__class__.__name__}({kwargs})'


# ========================================================================= #
# Deprecated                                                                #
# ========================================================================= #


@deprecated('ToStandardisedTensor renamed to ToImgTensorF32')
class ToStandardisedTensor(ToImgTensorF32):
    pass


@deprecated('ToUint8Tensor renamed to ToImgTensorU8')
class ToUint8Tensor(ToImgTensorU8):
    pass


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
