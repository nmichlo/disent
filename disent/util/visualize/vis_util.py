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

import logging
import warnings
from typing import List
from typing import Union

import numpy as np
import scipy.stats
import torch
from PIL import Image

from disent.util import to_numpy


log = logging.getLogger(__name__)


# ========================================================================= #
# operations on Images/Animations                                           #
# ========================================================================= #


# get bg color -- TODO: add support for more
_BG_COLOR_DTYPE_MAP = {
    'uint8':   127, np.uint8:   127, torch.uint8:   127, np.dtype('uint8'):   127,
    'float16': 0.5, np.float16: 0.5, torch.float16: 0.5, np.dtype('float16'): 0.5,
    'float32': 0.5, np.float32: 0.5, torch.float32: 0.5, np.dtype('float32'): 0.5,
    'float64': 0.5, np.float64: 0.5, torch.float64: 0.5, np.dtype('float64'): 0.5,
}


def make_image_grid(images, pad=8, border=True, bg_color=None, num_cols=None):
    """
    Convert a list of images into a single image that is a grid of those images.
    :param images: list of input images, all the same size: (I, H, W, C) or (I, H, W)
    :param pad: the number of pixels between images
    :param border: if there should be a border around the grid
    :param bg_color: the background color to use for padding (can be a float, int or RGB tuple)
    :param num_cols: the number of output columns in the grid. None for auto square, -1 for rows==1, or > 0 for that many cols.
    :return: single output image:  (H', W') or (H', W', C)
    """
    # first, second, third channels are the (H, W, C)
    # get image sizes
    img_shape, ndim = np.array(images[0].shape), images[0].ndim
    assert ndim == 2 or ndim == 3, 'images have wrong number of channels'
    assert np.all(img_shape == img.shape for img in images), 'Images are not the same shape!'
    # get image size and channels
    img_size = img_shape[:2]
    if ndim == 3:
        assert (img_shape[2] == 1) or (img_shape[2] == 3), 'Invalid number of channels for an image.'
    # get bg color
    if bg_color is None:
        bg_color = _BG_COLOR_DTYPE_MAP[images[0].dtype]
    # grid sizes
    num_rows, num_cols = _get_grid_size(len(images), num_cols=num_cols)
    grid_size = (img_size + pad) * [num_rows, num_cols] + (pad if border else -pad)
    # image sizes including padding on one side
    deltas = img_size + pad
    offset = pad if border else 0
    # make image
    grid = np.full_like(images, fill_value=bg_color, shape=(*grid_size, *img_shape[2:]))
    # fill image
    for i, img in enumerate(images):
        y0, x0 = offset + deltas * [i // num_cols, i % num_cols]
        y1, x1 = img_size + [y0, x0]
        grid[y0:y1, x0:x1, ...] = img
    return grid


def make_animated_image_grid(list_of_animated_images, pad=8, border=True, bg_color=None, num_cols=None):
    """
    :param list_of_animated_images: list of input images, with the second dimension the number of frames: : (I, F, H, W, C) or (I, F, H, W)
    :param pad: the number of pixels between images
    :param border: if there should be a border around the grid
    :param bg_color: the background color to use for padding (can be a float, int or RGB tuple)
    :param num_cols: the number of output columns in the grid. None for auto square, -1 for rows==1, or > 0 for that many cols.
    :return: animated output image: (F, H', W') or (F, H', W', C)
    """
    # first channel is the image (I)
    # second channel is the frame (F)
    # third, fourth, fifth channels are the (H, W, C)
    # -- (I, F, H, W, C)
    frames = []
    for list_of_images in zip(*list_of_animated_images):
        frames.append(make_image_grid(list_of_images, pad=pad, border=border, bg_color=bg_color, num_cols=num_cols))
    return to_numpy(frames)


# ========================================================================= #
# Calculations/Heuristics                                                   #
# ========================================================================= #


def _get_grid_size(n, num_cols=None):
    """
    Determine the number of rows and columns, given the total number of elements n.
    - if num_cols is None:     rows x cols is as square as possible
    - if num_cols is a number: minimum number of rows needed is returned.
    - if num_cols <= 0:        only 1 row is returned
    :return: (num_rows, num_cols)
    """
    if num_cols is None:
        num_cols = int(np.ceil(n ** 0.5))
    elif num_cols <= 0:
        num_cols = n
    num_rows = (n + num_cols - 1) // num_cols
    return num_rows, num_cols


# ========================================================================= #
# Factor Cycle Generators                                                   #
# ========================================================================= #


def _get_interval_factor_traversal(factor_size, num_frames, start_index=0):
    """
    Cycles through the state space in a single cycle.
    eg. num_indices=5, num_frames=7 returns: [0,1,1,2,3,3,4]
    eg. num_indices=4, num_frames=7 returns: [0,0,1,2,2,2,3]  # TODO: this result is weird
    """
    grid = np.linspace(0, factor_size - 1, num=num_frames, endpoint=True)
    grid = np.int64(np.around(grid))
    grid = (start_index + grid) % factor_size
    return grid


def _get_cycle_factor_traversal(factor_size, num_frames):
    """
    Cycles through the state space in a single cycle.
    eg. num_indices=5, num_frames=7 returns: [0,1,3,4,3,2,1]
    eg. num_indices=4, num_frames=7 returns: [0,1,2,3,2,2,0]
    """
    grid = _get_interval_factor_traversal(factor_size=factor_size, num_frames=num_frames)
    grid = np.concatenate([grid[0::2], grid[1::2][::-1]])
    return grid


_FACTOR_TRAVERSALS = {
    'interval': _get_interval_factor_traversal,
    'cycle': _get_cycle_factor_traversal,
}


def get_idx_traversal(factor_size, num_frames, mode='interval'):
    try:
        traversal_fn = _FACTOR_TRAVERSALS[mode]
    except KeyError:
        raise KeyError(f'Invalid factor traversal mode: {repr(mode)}')
    return traversal_fn(factor_size=factor_size, num_frames=num_frames)


# ========================================================================= #
# Cycle Generators | FROM: disentanglement_lib                              #
# ========================================================================= #


def cycle_gaussian(starting_value, num_frames, loc=0., scale=1.):
    """
    Cycles through the quantiles of a Gaussian in a single cycle.
    # ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~ #
    Copyright 2018 The DisentanglementLib Authors. All rights reserved.
    Licensed under the Apache License, Version 2.0
    https://github.com/google-research/disentanglement_lib
    # ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~ #
    """
    starting_prob = scipy.stats.norm.cdf(starting_value, loc=loc, scale=scale)
    grid = np.linspace(starting_prob, starting_prob + 2., num=num_frames, endpoint=False)
    grid -= np.maximum(0, 2 * grid - 2)
    grid += np.maximum(0, -2 * grid)
    grid = np.minimum(grid, 0.999)
    grid = np.maximum(grid, 0.001)
    return np.array([scipy.stats.norm.ppf(i, loc=loc, scale=scale) for i in grid])


def cycle_interval(starting_value, num_frames, min_val, max_val):
    """
    Cycles through the state space in a single cycle.
    # ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~ #
    Copyright 2018 The DisentanglementLib Authors. All rights reserved.
    Licensed under the Apache License, Version 2.0
    https://github.com/google-research/disentanglement_lib
    # ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~ #
    """
    starting_in_01 = (starting_value - min_val) / (max_val - min_val)
    starting_in_01 = np.nan_to_num(starting_in_01)  # handle division by zero, prints warning
    grid = np.linspace(starting_in_01, starting_in_01 + 2., num=num_frames, endpoint=False)
    grid -= np.maximum(0, 2 * grid - 2)
    grid += np.maximum(0, -2 * grid)
    return grid * (max_val - min_val) + min_val


# ========================================================================= #
# Conversion/Util                                                           #
# ========================================================================= #


# TODO: this functionality is duplicated elsewhere!
# TODO: similar functions exist: output_image, to_img, to_imgs, reconstructions_to_images
def reconstructions_to_images(
    recon,
    mode: str = 'float',
    moveaxis: bool = True,
    recon_min: Union[float, List[float]] = 0.0,
    recon_max: Union[float, List[float]] = 1.0,
    warn_if_clipped: bool = True,
) -> Union[np.ndarray, Image.Image]:
    """
    Convert a batch of reconstructions to images.
    A batch in this case consists of an arbitrary number of dimensions of an array,
    with the last 3 dimensions making up the actual image. For example: (..., channels, size, size)

    NOTE: This function might not be efficient for large amounts of
          data due to assertions and initial recursive conversions to a numpy array.

    NOTE: kornia has a similar function!
    """
    img = to_numpy(recon)
    # checks
    assert img.ndim >= 3
    assert img.dtype in (np.float32, np.float64)
    # move channels axis
    if moveaxis:
        img = np.moveaxis(img, -3, -1)
    # check min and max
    recon_min = np.array(recon_min)
    recon_max = np.array(recon_max)
    assert recon_min.shape == recon_max.shape
    assert recon_min.ndim in (0, 1)  # supports channels or glbal min . max
    # scale image
    img = (img - recon_min) / (recon_max - recon_min)
    # check image bounds
    if warn_if_clipped:
        m, M = np.min(img), np.max(img)
        if m < 0 or M > 1:
            log.warning(f'images with range [{m}, {M}] have been clipped to the range [0, 1]')
    # do clipping
    img = np.clip(img, 0, 1)
    # convert
    if mode == 'float':
        return img
    elif mode == 'int':
        return np.uint8(img * 255)
    elif mode == 'pil':
        img = np.uint8(img * 255)
        # all the cases (even ndim == 3)... bravo numpy, bravo!
        images = [Image.fromarray(img[idx]) for idx in np.ndindex(img.shape[:-3])]
        images = np.array(images, dtype=object).reshape(img.shape[:-3])
        return images
    else:
        raise KeyError(f'Invalid mode: {repr(mode)} not in { {"float", "int", "pil"} }')


# ========================================================================= #
# Image Util                                                                #
# ========================================================================= #


# TODO: something like this should replace reconstructions_to_images above!
# def torch_image_clamp(tensor: torch.Tensor, clamp_mode='warn') -> torch.Tensor:
#     # get dtype max value
#     dtype_M = 1 if tensor.dtype.is_floating_point else 255
#     # handle different modes
#     if clamp_mode in ('warn', 'error'):
#         m, M = tensor.min().cpu().numpy(), tensor.max().cpu().numpy()
#         if (0 < m) or (M > dtype_M):
#             if clamp_mode == 'warn':
#                 warnings.warn(f'image values are out of bounds, expected values in the range: [0, {dtype_M}], received values in the range: {[m, M]}')
#             else:
#                 raise ValueError(f'image values are out of bounds, expected values in the range: [0, {dtype_M}], received values in the range: {[m, M]}')
#     elif clamp_mode != 'clamp':
#         raise KeyError(f'invalid clamp mode: {repr(clamp_mode)}')
#     # clamp values
#     return torch.clamp(tensor, 0, dtype_M)
#
#
# # float16 doesnt always work, rather convert to float32 first
# _ALLOWED_DTYPES = {
#     torch.float32, torch.float64,
#     torch.uint8,
#     torch.int, torch.int16, torch.int32, torch.int64,
#     torch.long,
# }
#
#
# @lru_cache()
# def _torch_to_images_normalise_args(in_tensor_shape: Tuple[int, ...], in_tensor_dtype: torch.dtype, in_dims: str, out_dims: str, in_dtype: Optional[torch.dtype], out_dtype: Optional[torch.dtype]):
#     # check types
#     if not isinstance(in_dims, str): raise TypeError(f'in_dims must be of type: {str}, but got: {type(in_dims)}')
#     if not isinstance(out_dims, str): raise TypeError(f'out_dims must be of type: {str}, but got: {type(out_dims)}')
#     # normalise dim names
#     in_dims = in_dims.upper()
#     out_dims = out_dims.upper()
#     # check dim values
#     if sorted(in_dims) != sorted('CHW'): raise KeyError(f'in_dims contains the symbols: {repr(in_dims)}, must contain only permutations of: {repr("CHW")}')
#     if sorted(out_dims) != sorted('CHW'): raise KeyError(f'out_dims contains the symbols: {repr(out_dims)}, must contain only permutations of: {repr("CHW")}')
#     # get dimension indices
#     in_c_dim = in_dims.index('C') - len(in_dims)
#     transpose_indices = tuple(in_dims.index(c) - len(in_dims) for c in out_dims)
#     # check image tensor
#     if len(in_tensor_shape) < 3:
#         raise ValueError(f'images must have 3 or more dimensions corresponding to: (..., {", ".join(in_dims)}), but got shape: {in_tensor_shape}')
#     if in_tensor_shape[in_c_dim] not in (1, 3):
#         raise ValueError(f'images do not have the correct number of channels for dim "C", required: 1 or 3. Input format is (..., {", ".join(in_dims)}), but got shape: {in_tensor_shape}')
#     # get default values
#     if in_dtype is None: in_dtype = in_tensor_dtype
#     if out_dtype is None: out_dtype = in_dtype
#     # check dtypes allowed
#     if in_dtype not in _ALLOWED_DTYPES: raise TypeError(f'in_dtype is not allowed, got: {repr(in_dtype)} must be one of: {list(_ALLOWED_DTYPES)}')
#     if out_dtype not in _ALLOWED_DTYPES: raise TypeError(f'out_dtype is not allowed, got: {repr(out_dtype)} must be one of: {list(_ALLOWED_DTYPES)}')
#     # done!
#     return transpose_indices, in_dtype, out_dtype
#
#
# def torch_to_images(
#     tensor: torch.Tensor,
#     in_dims: str = 'CHW',
#     out_dims: str = 'HWC',
#     in_dtype: Optional[torch.dtype] = None,
#     out_dtype: Optional[torch.dtype] = torch.uint8,
#     clamp_mode: str = 'warn',  # clamp, warn, error
# ) -> torch.Tensor:
#     """
#     Convert a batch of image-like tensors to images.
#     A batch in this case consists of an arbitrary number of dimensions of a tensor,
#     with the last 3 dimensions making up the actual images.
#
#     example:
#         Convert a tensor of non-normalised images (..., C, H, W) to a
#         tensor of normalised and clipped images (..., H, W, C).
#         - integer dtypes are expected to be in the range [0, 255]
#         - float dtypes are expected to be in the range [0, 1]
#     """
#     if not isinstance(tensor, torch.Tensor):
#         raise TypeError(f'images must be of type: {torch.Tensor}, got: {type(tensor)}')
#     # check arguments
#     transpose_indices, in_dtype, out_dtype = _torch_to_images_normalise_args(
#         in_tensor_shape=tuple(tensor.shape), in_tensor_dtype=tensor.dtype,
#         in_dims=in_dims, out_dims=out_dims,
#         in_dtype=in_dtype, out_dtype=out_dtype,
#     )
#     # check inputs
#     if in_dtype != tensor.dtype:
#         raise TypeError(f'images dtype: {repr(tensor.dtype)} does not match in_dtype: {repr(in_dtype)}')
#     # convert images
#     with torch.no_grad():
#         # check that input values are in the correct range
#         # move axes
#         tensor = tensor.permute(*(i-tensor.ndim for i in range(tensor.ndim-3)), *transpose_indices)
#         # convert outputs
#         if in_dtype != out_dtype:
#             if in_dtype.is_floating_point and (not out_dtype.is_floating_point):
#                 tensor = (tensor * 255).to(out_dtype)
#             elif (not in_dtype.is_floating_point) and out_dtype.is_floating_point:
#                 tensor = tensor.to(out_dtype) / 255
#             else:
#                 tensor = tensor.to(out_dtype)
#         # clamp
#         tensor = torch_image_clamp(tensor, clamp_mode=clamp_mode)
#     # check outputs
#     if out_dtype != tensor.dtype:  # pragma: no cover
#         raise RuntimeError(f'[THIS IS A BUG! PLEASE REPORT THIS!]: After conversion, images tensor dtype: {repr(tensor.dtype)} does not match out_dtype: {repr(in_dtype)}')
#     # done
#     return tensor
#
#
# def numpy_to_images(
#     ndarray: np.ndarray,
#     in_dims: str = 'CHW',
#     out_dims: str = 'HWC',
#     in_dtype: Optional[str, np.dtype] = None,
#     out_dtype: Optional[str, np.dtype] = np.dtype('uint8'),
#     clamp_mode: str = 'warn',  # clamp, warn, error
# ) -> np.ndarray:
#     """
#     Convert a batch of image-like arrays to images.
#     A batch in this case consists of an arbitrary number of dimensions of an array,
#     with the last 3 dimensions making up the actual images.
#     - See the docs for: torch_to_imgs(...)
#     """
#     # convert numpy dtypes to torch
#     if in_dtype is not None: in_dtype = getattr(torch, np.dtype(in_dtype).name)
#     if out_dtype is not None: out_dtype = getattr(torch, np.dtype(out_dtype).name)
#     # convert back
#     tensor = torch_to_images(tensor=torch.from_numpy(ndarray), in_dims=in_dims, out_dims=out_dims, in_dtype=in_dtype, out_dtype=out_dtype, clamp_mode=clamp_mode)
#     # done!
#     return tensor.numpy()
#
#
# def numpy_to_pil_images(ndarray: np.ndarray, in_dims: str = 'CHW', clamp_mode: str = 'warn'):
#     """
#     Convert a numpy array containing images (..., C, H, W) to an array of PIL images (...,)
#     """
#     imgs = numpy_to_images(ndarray=ndarray, in_dims=in_dims, out_dims='HWC', in_dtype=None, out_dtype='uint8', clamp_mode=clamp_mode)
#     # all the cases (even ndim == 3)... bravo numpy, bravo!
#     images = [Image.fromarray(imgs[idx]) for idx in np.ndindex(imgs.shape[:-3])]
#     images = np.array(images, dtype=object).reshape(imgs.shape[:-3])
#     # done
#     return images
#
#
# def test_torch_to_imgs():
#     inp_float = torch.rand(8, 3, 64, 64, dtype=torch.float32)
#     inp_uint8 = (inp_float * 127 + 63).to(torch.uint8)
#     # check runs
#     out = torch_to_imgs(inp_float)
#     assert out.dtype == torch.uint8
#     out = torch_to_imgs(inp_uint8)
#     assert out.dtype == torch.uint8
#     out = torch_to_imgs(inp_float, in_dtype=None, out_dtype=None)
#     assert out.dtype == inp_float.dtype
#     out = torch_to_imgs(inp_uint8, in_dtype=None, out_dtype=None)
#     assert out.dtype == inp_uint8.dtype
#
#
# def test_torch_to_imgs_permutations():
#     inp_float = torch.rand(8, 3, 64, 64, dtype=torch.float32)
#     inp_uint8 = (inp_float * 127 + 63).to(torch.uint8)
#
#     # general checks
#     def check_all(inputs, in_dtype=None):
#         float_results, int_results = [], []
#         for out_dtype in _ALLOWED_DTYPES:
#             out = torch_to_imgs(inputs, in_dtype=in_dtype, out_dtype=out_dtype)
#             (float_results if out_dtype.is_floating_point else int_results).append(torch.stack([
#                 out.min().to(torch.float64), out.max().to(torch.float64), out.mean(dtype=torch.float64)
#             ]))
#         for a, b in zip(float_results[:-1], float_results[1:]): assert torch.allclose(a, b)
#         for a, b in zip(int_results[:-1], int_results[1:]): assert torch.allclose(a, b)
#
#     # check type permutations
#     check_all(inp_float, torch.float32)
#     check_all(inp_uint8, torch.uint8)
#
#
# def test_torch_to_imgs_preserve_type():
#     for dtype in _ALLOWED_DTYPES:
#         tensor = (torch.rand(8, 3, 64, 64) * (1 if dtype.is_floating_point else 255)).to(dtype)
#         out = torch_to_imgs(tensor, in_dtype=dtype, out_dtype=dtype, clamp=True)
#         assert out.dtype == dtype
#
#
# def test_torch_to_imgs_args():
#     inp_float = torch.rand(8, 3, 64, 64, dtype=torch.float32)
#
#     # check tensor
#     with pytest.raises(TypeError, match="images tensor must be of type"):
#         torch_to_imgs(tensor=None)
#     with pytest.raises(ValueError, match='dim "C", required: 1 or 3'):
#         torch_to_imgs(tensor=torch.rand(8, 2, 16, 16, dtype=torch.float32))
#     with pytest.raises(ValueError, match='dim "C", required: 1 or 3'):
#         torch_to_imgs(tensor=torch.rand(8, 16, 16, 3, dtype=torch.float32))
#     with pytest.raises(ValueError, match='images tensor must have 3 or more dimensions corresponding to'):
#         torch_to_imgs(tensor=torch.rand(16, 16, dtype=torch.float32))
#
#     # check dims
#     with pytest.raises(TypeError, match="in_dims must be of type"):
#         torch_to_imgs(inp_float, in_dims=None)
#     with pytest.raises(TypeError, match="out_dims must be of type"):
#         torch_to_imgs(inp_float, out_dims=None)
#     with pytest.raises(KeyError, match="in_dims contains the symbols: 'INVALID', must contain only permutations of: 'CHW'"):
#         torch_to_imgs(inp_float, in_dims='INVALID')
#     with pytest.raises(KeyError, match="out_dims contains the symbols: 'INVALID', must contain only permutations of: 'CHW'"):
#         torch_to_imgs(inp_float, out_dims='INVALID')
#     with pytest.raises(KeyError, match="in_dims contains the symbols: 'CHWW', must contain only permutations of: 'CHW'"):
#         torch_to_imgs(inp_float, in_dims='CHWW')
#     with pytest.raises(KeyError, match="out_dims contains the symbols: 'CHWW', must contain only permutations of: 'CHW'"):
#         torch_to_imgs(inp_float, out_dims='CHWW')
#
#     # check dtypes
#     with pytest.raises(TypeError, match="images tensor dtype: torch.float32 does not match in_dtype: torch.uint8"):
#         torch_to_imgs(inp_float, in_dtype=torch.uint8)
#     with pytest.raises(TypeError, match='in_dtype is not allowed'):
#         torch_to_imgs(inp_float, in_dtype=torch.complex64)
#     with pytest.raises(TypeError, match='out_dtype is not allowed'):
#         torch_to_imgs(inp_float, out_dtype=torch.complex64)
#     with pytest.raises(TypeError, match='in_dtype is not allowed'):
#         torch_to_imgs(inp_float, in_dtype=torch.float16)
#     with pytest.raises(TypeError, match='out_dtype is not allowed'):
#         torch_to_imgs(inp_float, out_dtype=torch.float16)


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
