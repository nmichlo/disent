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

from typing import Any
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import TypeVar
from typing import Union

import numpy as np
from PIL.Image import Image
import torch
import torchvision.transforms.functional as F_tv


# ========================================================================= #
# Types                                                                     #
# ========================================================================= #


_T = TypeVar('_T')

Obs = Union[np.ndarray, Image]

SizeType = Union[int, Tuple[int, int]]


# ========================================================================= #
# Functional Transforms                                                     #
# ========================================================================= #


def noop(obs: _T) -> _T:
    """
    Transform that does absolutely nothing!
    """
    return obs


def check_tensor(
    obs: Any,
    low: Optional[float] = 0.,
    high: Optional[float] = 1.,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Check that the input is a tensor, its datatype matches, and
    that it is in the required range.
    """
    # check is a tensor
    assert torch.is_tensor(obs), 'observation is not a tensor'
    # check type
    if dtype is not None:
        assert obs.dtype == dtype, f'tensor type {obs.dtype} is not required type {dtype}'
    # check range
    if low is not None:
        assert low <= obs.min(), f'minimum value of tensor {obs.min()} is less than allowed minimum value: {low}'
    if high is not None:
        assert obs.max() <= high, f'maximum value of tensor {obs.max()} is greater than allowed maximum value: {high}'
    # DONE!
    return obs


# ========================================================================= #
# Normalized Image Tensors                                                  #
# ========================================================================= #


def to_img_tensor_u8(
    obs: Obs,
    size: Optional[SizeType] = None,
) -> torch.Tensor:
    """
    Basic transform that makes sure the image tensor is uint8 and a specified size.

    Steps:
    1. resize image if size is specified
    2. add missing channel to greyscale image
    3. move channels to first dim (H, W, C) -> (C, H, W)
    """
    # resize image
    if size is not None:
        if not isinstance(obs, Image):
            obs = F_tv.to_pil_image(obs)
        obs = F_tv.resize(obs, size=size)
    # to numpy
    if isinstance(obs, Image):
        obs = np.array(obs)
    # add missing axis
    if obs.ndim == 2:
        obs = obs[:, :, None]
    # to tensor & move axis
    obs = torch.from_numpy(obs)
    obs = torch.moveaxis(obs, -1, -3)
    # checks
    assert obs.ndim == 3
    assert obs.dtype == torch.uint8
    # done!
    return obs


def to_img_tensor_f32(
    obs: Obs,
    size: Optional[SizeType] = None,
    mean: Optional[Sequence[float]] = None,
    std: Optional[Sequence[float]] = None,
) -> torch.Tensor:
    """
    Basic transform that should be applied to most datasets, making sure
    the image tensor is float32 and a specified size.

    Steps:
        1. resize image if size is specified
        2. if we have integer inputs, divide by 255
        3. add missing channel to greyscale image
        4. move channels to first dim (H, W, C) -> (C, H, W)
        5. normalize using mean and std, values might thus be outside of the range [0, 1]
    """
    # resize image
    if size is not None:
        if not isinstance(obs, Image):
            obs = F_tv.to_pil_image(obs)
        obs = F_tv.resize(obs, size=size)
    # transform to tensor, add missing dims & move channel dim to front
    obs = F_tv.to_tensor(obs)
    # checks
    assert obs.ndim == 3, f'obs has does not have 3 dimensions, got: {obs.ndim} for shape: {obs.shape}'
    assert obs.dtype == torch.float32, f'obs is not dtype torch.float32, got: {obs.dtype}'
    # apply mean and std, we obs is of the shape (C, H, W)
    if (mean is not None) or (std is not None):
        obs = F_tv.normalize(obs, mean=0. if (mean is None) else mean, std=1. if (std is None) else std, inplace=True)
        assert obs.dtype == torch.float32, f'after normalization, tensor should remain as dtype torch.float32, got: {obs.dtype}'
    # done!
    return obs


# ========================================================================= #
# Custom Normalized Image - Faster Than Above                               #
# ========================================================================= #


# def to_img_tensor_f32(
#     x: Obs,
#     size: Optional[SizeType] = None,
#     channel_to_front: bool = None,
# ):
#     """
#     Basic transform that should be applied to
#     any dataset before augmentation.
#
#     1. resize if size is specified
#     2. if needed convert integers to float32 by dividing by 255
#     3. normalize using mean and std, values might thus be outside of the range [0, 1]
#
#     Convert PIL or uint8 inputs to float32
#     - input images should always have 2 (H, W) or 3 channels (H, W, C)
#     - output image always has size (C, H, W) with channels moved to the first dim
#     """
#     return _to_img_tensor(x, size=size, channel_to_front=channel_to_front, to_float32=True)
#
#
# def to_img_tensor_u8(
#     x: Obs,
#     size: Optional[SizeType] = None,
#     channel_to_front: bool = None,
# ):
#     """
#     Convert PIL or uint8 inputs to float32
#     - input images should always have 2 (H, W) or 3 channels (H, W, C)
#     - output image always has size (C, H, W) with channels moved to the first dim
#     """
#     return _to_img_tensor(x, size=size, channel_to_front=channel_to_front, to_float32=False)
#
#
# def _to_img_tensor(
#     x: Obs,
#     size: Optional[SizeType] = None,
#     channel_to_front: bool = None,
#     to_float32: bool = True,
# ) -> torch.Tensor:
#     assert isinstance(x, (np.ndarray, Image)), f'input is not an numpy.ndarray or PIL.Image.Image, got: {type(x)}'
#     # optionally resize the image, returns a numpy array or a PIL.Image.Image
#     x = _resize_if_needed(x, size=size)
#     # convert image to numpy
#     if isinstance(x, Image):
#         x = np.array(x)
#     # make sure 2D becomes 3D
#     if x.ndim == 2:
#         x = x[:, :, None]
#     assert x.ndim == 3, f'obs has invalid number of dimensions, required 2 or 3, got: {x.ndim} for shape: {x.shape}'
#     # convert to float32 if int or uint
#     if to_float32:
#         if x.dtype.kind in ('i', 'u'):
#             x = x.astype('float32') / 255  # faster than with torch
#     # convert to torch.Tensor and move channels (H, W, C) -> (C, H, W)
#     x = torch.from_numpy(x)
#     if channel_to_front or (channel_to_front is None):
#         x = torch.moveaxis(x, -1, 0)  # faster than the numpy version
#     # final check
#     if to_float32:
#         assert x.dtype == torch.float32, f'obs dtype invalid, required: {torch.float32}, got: {x.dtype}'
#     else:
#         assert x.dtype == torch.uint8, f'obs dtype invalid, required: {torch.uint8}, got: {x.dtype}'
#     # done
#     return x
#
#
# # ========================================================================= #
# # Resize Image Helper                                                       #
# # ========================================================================= #
#
#
# _PIL_INTERPOLATE_MODES = {
#     'nearest': 0,
#     'lanczos': 1,
#     'bilinear': 2,
#     'bicubic': 3,
#     'box': 4,
#     'hamming': 5,
# }
#
#
# def _resize_if_needed(img: Union[np.ndarray, Image], size: Optional[Union[Tuple[int, int], int]] = None) -> Union[np.ndarray, Image]:
#     # skip resizing
#     if size is None:
#         return img
#     # normalize size
#     if isinstance(size, int):
#         size = (size, size)
#     # get current image size
#     if isinstance(img, Image):
#         in_size = (img.height, img.width)
#     else:
#         assert img.ndim in (2, 3), f'image must have 2 or 3 dims, got shape: {img.shape}'
#         in_size = img.shape[:2]
#     # skip if the same size
#     if in_size == size:
#         return img
#     # normalize the image
#     if not isinstance(img, Image):
#         assert img.dtype == 'uint8'
#         # normalize image
#         if img.ndim == 3:
#             c = img.shape[-1]
#             assert c in (1, 3), f'image channel dim must be of size 1 or 3, got shape: {img.shape}'
#             img, mode = (img, 'RGB') if (c == 3) else (img[:, :, 0], 'L')
#         else:
#             mode = 'L'
#         # convert
#         img = PIL.Image.fromarray(img, mode=mode)
#     # resize
#     return img.resize(size, resample=_PIL_INTERPOLATE_MODES['bilinear'])


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
