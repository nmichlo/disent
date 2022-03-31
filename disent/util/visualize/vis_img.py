#  ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~
#  MIT License
#
#  Copyright (c) 2022 Nathan Juraj Michlo
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


import warnings
from functools import lru_cache
from numbers import Number
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import torch
from numpy.core.multiarray import normalize_axis_index
from PIL import Image


# ========================================================================= #
# Type Hints                                                                #
# ========================================================================= #


# from torch.testing._internal.common_utils import numpy_to_torch_dtype_dict
_NP_TO_TORCH_DTYPE = {
    np.dtype('bool'):       torch.bool,
    np.dtype('uint8'):      torch.uint8,
    np.dtype('int8'):       torch.int8,
    np.dtype('int16'):      torch.int16,
    np.dtype('int32'):      torch.int32,
    np.dtype('int64'):      torch.int64,
    np.dtype('float16'):    torch.float16,
    np.dtype('float32'):    torch.float32,
    np.dtype('float64'):    torch.float64,
    np.dtype('complex64'):  torch.complex64,
    np.dtype('complex128'): torch.complex128
}


MinMaxHint = Union[Number, Tuple[Number, ...], np.ndarray]


@lru_cache()
def _dtype_min_max(dtype: torch.dtype) -> Tuple[Union[float, int], Union[float, int]]:
    """Get the min and max values for a dtype"""
    dinfo = torch.finfo(dtype) if dtype.is_floating_point else torch.iinfo(dtype)
    return dinfo.min, dinfo.max


@lru_cache()
def _check_image_dtype(dtype: torch.dtype):
    """Check that a dtype can hold image values"""
    # check that the datatype is within the right range -- this is not actually necessary if the below is correct!
    dmin, dmax = _dtype_min_max(dtype)
    imin, imax = (0, 1) if dtype.is_floating_point else (0, 255)
    assert (dmin <= imin) and (imax <= dmax), f'The dtype: {repr(dtype)} with range [{dmin}, {dmax}] cannot store image values in the range [{imin}, {imax}]'
    # check the datatype is allowed
    if dtype not in _ALLOWED_DTYPES:
        raise TypeError(f'The dtype: {repr(dtype)} is not allowed, must be one of: {list(_ALLOWED_DTYPES)}')
    # return the min and max values
    return imin, imax


# ========================================================================= #
# Image Helper Functions                                                    #
# ========================================================================= #


def torch_image_has_valid_range(tensor: torch.Tensor, check_mode: Optional[str] = None) -> bool:
    """
    Check that the range of values in the image is correct!
    """
    if check_mode not in {'error', 'warn', 'bool', None}:
        raise KeyError(f'invalid check_mode: {repr(check_mode)}')
    # get the range for the dtype
    imin, imax = _check_image_dtype(tensor.dtype)
    # get the values
    m = tensor.amin().cpu().numpy()
    M = tensor.amax().cpu().numpy()
    if (m < imin) or (imax < M):
        if check_mode == 'error':
            raise ValueError(f'images value range: [{m}, {M}] is outside of the required range: [{imin}, {imax}], for dtype: {repr(tensor.dtype)}')
        elif check_mode == 'warn':
            warnings.warn(f'images value range: [{m}, {M}] is outside of the required range: [{imin}, {imax}], for dtype: {repr(tensor.dtype)}')
        return False
    return True


@torch.no_grad()
def torch_image_clamp(tensor: torch.Tensor, clamp_mode: str = 'warn') -> torch.Tensor:
    """
    Clamp the image based on the dtype
    Valid `clamp_mode`s are {'warn', 'error', 'clamp'}
    """
    # check range of values
    if clamp_mode in ('warn', 'error'):
        torch_image_has_valid_range(tensor, check_mode=clamp_mode)
    elif clamp_mode != 'clamp':
        raise KeyError(f'invalid clamp mode: {repr(clamp_mode)}')
    # get the range for the dtype
    imin, imax = _check_image_dtype(tensor.dtype)
    # clamp!
    return torch.clamp(tensor, imin, imax)


@torch.no_grad()
def torch_image_to_dtype(tensor: torch.Tensor, out_dtype: torch.dtype):
    """
    Convert an image to the specified dtype
    - Scaling is automatically performed based on the input and output dtype
      Floats should be in the range [0, 1], integers should be in the range [0, 255]
    - if precision will be lost (), then the values are clamped!
    """
    _check_image_dtype(tensor.dtype)
    _check_image_dtype(out_dtype)
    # check scale
    torch_image_has_valid_range(tensor, check_mode='error')
    # convert
    if tensor.dtype.is_floating_point and (not out_dtype.is_floating_point):
        # [float -> int] -- cast after scaling
        return torch.clamp(tensor * 255, 0, 255).to(out_dtype)
    elif (not tensor.dtype.is_floating_point) and out_dtype.is_floating_point:
        # [int -> float] -- cast before scaling
        return torch.clamp(tensor.to(out_dtype) / 255, 0, 1)
    else:
        # [int -> int] | [float -> float]
        return tensor.to(out_dtype)


@torch.no_grad()
def torch_image_normalize_channels(
    tensor: torch.Tensor,
    in_min: MinMaxHint,
    in_max: MinMaxHint,
    channel_dim: int = -1,
    out_dtype: Optional[torch.dtype] = None
):
    if out_dtype is None:
        out_dtype = tensor.dtype
    # check dtypes
    _check_image_dtype(out_dtype)
    assert out_dtype.is_floating_point, f'out_dtype must be a floating point, got: {repr(out_dtype)}'
    # get norm values padded to the dimension of the channel
    in_min, in_max = _torch_channel_broadcast_scale_values(in_min, in_max, in_dtype=tensor.dtype, dim=channel_dim, ndim=tensor.ndim)
    # convert
    tensor = tensor.to(out_dtype)
    in_min = torch.as_tensor(in_min, dtype=tensor.dtype, device=tensor.device)
    in_max = torch.as_tensor(in_max, dtype=tensor.dtype, device=tensor.device)
    # warn if the values are the same
    if torch.any(in_min == in_max):
        m = in_min.cpu().detach().numpy()
        M = in_min.cpu().detach().numpy()
        warnings.warn(f'minimum: {m} and maximum: {M} values are the same, scaling values to zero.')
    # handle equal values
    divisor = in_max - in_min
    divisor[divisor == 0] = 1
    # normalize
    return (tensor - in_min) / divisor


# ========================================================================= #
# Argument Helper                                                           #
# ========================================================================= #


# float16 doesnt always work, rather convert to float32 first
_ALLOWED_DTYPES = {
    torch.float32, torch.float64,
    torch.uint8,
    torch.int, torch.int16, torch.int32, torch.int64,
    torch.long,
}


@lru_cache()
def _torch_to_images_normalise_args(in_tensor_shape: Tuple[int, ...], in_tensor_dtype: torch.dtype, in_dims: str, out_dims: str, in_dtype: Optional[torch.dtype], out_dtype: Optional[torch.dtype]):
    # check types
    if not isinstance(in_dims, str): raise TypeError(f'in_dims must be of type: {str}, but got: {type(in_dims)}')
    if not isinstance(out_dims, str): raise TypeError(f'out_dims must be of type: {str}, but got: {type(out_dims)}')
    # normalise dim names
    in_dims = in_dims.upper()
    out_dims = out_dims.upper()
    # check dim values
    if sorted(in_dims) != sorted('CHW'): raise KeyError(f'in_dims contains the symbols: {repr(in_dims)}, must contain only permutations of: {repr("CHW")}')
    if sorted(out_dims) != sorted('CHW'): raise KeyError(f'out_dims contains the symbols: {repr(out_dims)}, must contain only permutations of: {repr("CHW")}')
    # get dimension indices
    in_c_dim = in_dims.index('C') - len(in_dims)
    out_c_dim = out_dims.index('C') - len(out_dims)
    transpose_indices = tuple(in_dims.index(c) - len(in_dims) for c in out_dims)
    # check image tensor
    if len(in_tensor_shape) < 3:
        raise ValueError(f'images must have 3 or more dimensions corresponding to: (..., {", ".join(in_dims)}), but got shape: {in_tensor_shape}')
    if in_tensor_shape[in_c_dim] not in (1, 3):
        raise ValueError(f'images do not have the correct number of channels for dim "C", required: 1 or 3. Input format is (..., {", ".join(in_dims)}), but got shape: {in_tensor_shape}')
    # get default values
    if in_dtype is None: in_dtype = in_tensor_dtype
    if out_dtype is None: out_dtype = in_dtype
    # check dtypes allowed
    if in_dtype not in _ALLOWED_DTYPES: raise TypeError(f'in_dtype is not allowed, got: {repr(in_dtype)} must be one of: {list(_ALLOWED_DTYPES)}')
    if out_dtype not in _ALLOWED_DTYPES: raise TypeError(f'out_dtype is not allowed, got: {repr(out_dtype)} must be one of: {list(_ALLOWED_DTYPES)}')
    # done!
    return transpose_indices, in_dtype, out_dtype, out_c_dim


def _torch_channel_broadcast_scale_values(
    in_min: MinMaxHint,
    in_max: MinMaxHint,
    in_dtype: torch.dtype,
    dim: int,
    ndim: int,
) -> Tuple[List[Number], List[Number]]:
    return __torch_channel_broadcast_scale_values(
        in_min=tuple(np.array(in_min).reshape(-1).tolist()),  # TODO: this is slow?
        in_max=tuple(np.array(in_max).reshape(-1).tolist()),  # TODO: this is slow?
        in_dtype=in_dtype,
        dim=dim,
        ndim=ndim,
    )

@lru_cache()
@torch.no_grad()
def __torch_channel_broadcast_scale_values(
    in_min: MinMaxHint,
    in_max: MinMaxHint,
    in_dtype: torch.dtype,
    dim: int,
    ndim: int,
) -> Tuple[List[Number], List[Number]]:
    # get the default values
    in_min: np.ndarray = np.array((0.0 if in_dtype.is_floating_point else 0.0)   if (in_min is None) else in_min)
    in_max: np.ndarray = np.array((1.0 if in_dtype.is_floating_point else 255.0) if (in_max is None) else in_max)
    # add missing axes
    if in_min.ndim == 0: in_min = in_min[None]
    if in_max.ndim == 0: in_max = in_max[None]
    # checks
    assert in_min.ndim == 1
    assert in_max.ndim == 1
    assert np.all(in_min <= in_max), f'min values are not <= the max values: {in_min} !<= {in_max}'
    # normalize dim
    dim = normalize_axis_index(dim, ndim=ndim)
    # pad dim
    r_pad = ndim - (dim + 1)
    if r_pad > 0:
        in_min = in_min[(...,) + ((None,)*r_pad)]
        in_max = in_max[(...,) + ((None,)*r_pad)]
    # done!
    return in_min.tolist(), in_max.tolist()


# ========================================================================= #
# Image Conversion                                                          #
# ========================================================================= #


@torch.no_grad()
def torch_to_images(
    tensor: torch.Tensor,
    in_dims: str = 'CHW',  # we always treat numpy by default as HWC, and torch.Tensor as CHW
    out_dims: str = 'HWC',
    in_dtype: Optional[torch.dtype] = None,
    out_dtype: Optional[torch.dtype] = torch.uint8,
    clamp_mode: str = 'warn',  # clamp, warn, error
    always_rgb: bool = False,
    in_min: Optional[MinMaxHint] = None,
    in_max: Optional[MinMaxHint] = None,
    to_numpy: bool = False,
) -> Union[torch.Tensor, np.ndarray]:
    """
    Convert a batch of image-like tensors to images.
    A batch in this case consists of an arbitrary number of dimensions of a tensor,
    with the last 3 dimensions making up the actual images.

    Process:
    1. check input dtype
    2. move axis
    3. normalize
    4. clamp values
    5. auto scale and convert
    6. convert to rgb
    7. check output dtype

    example:
        Convert a tensor of non-normalised images (..., C, H, W) to a
        tensor of normalised and clipped images (..., H, W, C).
        - integer dtypes are expected to be in the range [0, 255]
        - float dtypes are expected to be in the range [0, 1]

    # TODO: add support for uneven in/out dims, eg. in_dims="HW", out_dims="HWC"
    """
    # 0.a. check tensor
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f'images must be of type: {torch.Tensor}, got: {type(tensor)}')
    # 0.b. get arguments
    transpose_indices, in_dtype, out_dtype, out_c_dim = _torch_to_images_normalise_args(
        in_tensor_shape=tuple(tensor.shape),
        in_tensor_dtype=tensor.dtype,
        in_dims=in_dims,
        out_dims=out_dims,
        in_dtype=in_dtype,
        out_dtype=out_dtype,
    )
    # 1. check input dtype
    if in_dtype != tensor.dtype:
        raise TypeError(f'images dtype: {repr(tensor.dtype)} does not match in_dtype: {repr(in_dtype)}')
    # 2. move axes
    tensor = tensor.permute(*(i-tensor.ndim for i in range(tensor.ndim-3)), *transpose_indices)
    # 3. normalise
    if (in_min is not None) or (in_max is not None):
        norm_dtype = (out_dtype if out_dtype.is_floating_point else torch.float32)
        tensor = torch_image_normalize_channels(tensor, in_min=in_min, in_max=in_max, channel_dim=out_c_dim, out_dtype=norm_dtype)
    # 4. clamp
    tensor = torch_image_clamp(tensor, clamp_mode=clamp_mode)
    # 5. auto scale and convert
    tensor = torch_image_to_dtype(tensor, out_dtype=out_dtype)
    # 6. convert to rgb
    if always_rgb:
        if tensor.shape[out_c_dim] == 1:
            tensor = torch.repeat_interleave(tensor, 3, dim=out_c_dim)  # torch.repeat is like np.tile, torch.repeat_interleave is like np.repeat
    # 7. check output dtype
    if out_dtype != tensor.dtype:
        raise RuntimeError(f'[THIS IS A BUG!]: After conversion, images tensor dtype: {repr(tensor.dtype)} does not match out_dtype: {repr(in_dtype)}')
    if torch.any(torch.isnan(tensor)):
        raise RuntimeError('[THIS IS A BUG!]: After conversion, images contain NaN values!')
    # convert to numpy
    if to_numpy:
        return tensor.detach().cpu().numpy()
    return tensor


def numpy_to_images(
    ndarray: np.ndarray,
    in_dims: str = 'HWC',  # we always treat numpy by default as HWC, and torch.Tensor as CHW
    out_dims: str = 'HWC',
    in_dtype:  Optional[Union[str, np.dtype]] = None,
    out_dtype: Optional[Union[str, np.dtype]] = np.dtype('uint8'),
    clamp_mode: str = 'warn',  # clamp, warn, error
    always_rgb: bool = False,
    in_min: Optional[MinMaxHint] = None,
    in_max: Optional[MinMaxHint] = None,
) -> np.ndarray:
    """
    Convert a batch of image-like arrays to images.
    A batch in this case consists of an arbitrary number of dimensions of an array,
    with the last 3 dimensions making up the actual images.
    - See the docs for: torch_to_images(...)
    """
    # convert numpy dtypes to torch
    if in_dtype is not None: in_dtype = _NP_TO_TORCH_DTYPE[np.dtype(in_dtype)]
    if out_dtype is not None: out_dtype = _NP_TO_TORCH_DTYPE[np.dtype(out_dtype)]
    # convert back
    array = torch_to_images(
        tensor=torch.from_numpy(ndarray),
        in_dims=in_dims,
        out_dims=out_dims,
        in_dtype=in_dtype,
        out_dtype=out_dtype,
        clamp_mode=clamp_mode,
        always_rgb=always_rgb,
        in_min=in_min,
        in_max=in_max,
        to_numpy=True,
    )
    # done!
    return array


def numpy_to_pil_images(
    ndarray: np.ndarray,
    in_dims: str = 'HWC',  # we always treat numpy by default as HWC, and torch.Tensor as CHW
    clamp_mode: str = 'warn',
    always_rgb: bool = False,
    in_min: Optional[MinMaxHint] = None,
    in_max: Optional[MinMaxHint] = None,
) -> Union[np.ndarray]:
    """
    Convert a numpy array containing images (..., C, H, W) to an array of PIL images (...,)
    """
    imgs = numpy_to_images(
        ndarray=ndarray,
        in_dims=in_dims,
        out_dims='HWC',
        in_dtype=None,
        out_dtype='uint8',
        clamp_mode=clamp_mode,
        always_rgb=always_rgb,
        in_min=in_min,
        in_max=in_max,
    )
    # all the cases (even ndim == 3)... bravo numpy, bravo!
    images = [Image.fromarray(imgs[idx]) for idx in np.ndindex(imgs.shape[:-3])]
    images = np.array(images, dtype=object).reshape(imgs.shape[:-3])
    # done
    return images


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
