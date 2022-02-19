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


import numpy as np
import pytest
import torch
from PIL import Image

from disent.util.visualize.vis_img import _torch_to_images_normalise_args
from disent.util.visualize.vis_img import numpy_to_images
from disent.util.visualize.vis_img import numpy_to_pil_images
from disent.util.visualize.vis_img import torch_to_images
from disent.util.visualize.vis_img import _ALLOWED_DTYPES


# ========================================================================= #
# Tests                                                                     #
# ========================================================================= #


def test_torch_to_images_basic():
    inp_float = torch.rand(8, 3, 64, 64, dtype=torch.float32)
    inp_uint8 = (inp_float * 127 + 63).to(torch.uint8)
    # check runs
    out = torch_to_images(inp_float)
    assert out.dtype == torch.uint8
    out = torch_to_images(inp_uint8)
    assert out.dtype == torch.uint8
    out = torch_to_images(inp_float, in_dtype=None, out_dtype=None)
    assert out.dtype == inp_float.dtype
    out = torch_to_images(inp_uint8, in_dtype=None, out_dtype=None)
    assert out.dtype == inp_uint8.dtype


def test_torch_to_images_permutations():
    inp_float = torch.rand(8, 3, 64, 64, dtype=torch.float32)
    inp_uint8 = (inp_float * 127 + 63).to(torch.uint8)

    # general checks
    def check_all(inputs, in_dtype=None):
        float_results, int_results = [], []
        for out_dtype in _ALLOWED_DTYPES:
            out = torch_to_images(inputs, in_dtype=in_dtype, out_dtype=out_dtype)
            stats = torch.stack([out.min().to(torch.float64), out.max().to(torch.float64), out.to(dtype=torch.float64).mean()])
            (float_results if out_dtype.is_floating_point else int_results).append(stats)
        for a, b in zip(float_results[:-1], float_results[1:]): assert torch.allclose(a, b)
        for a, b in zip(int_results[:-1], int_results[1:]):     assert torch.allclose(a, b)

    # check type permutations
    check_all(inp_float, torch.float32)
    check_all(inp_uint8, torch.uint8)


def test_torch_to_images_preserve_type():
    for dtype in _ALLOWED_DTYPES:
        tensor = (torch.rand(8, 3, 64, 64) * (1 if dtype.is_floating_point else 255)).to(dtype)
        out = torch_to_images(tensor, in_dtype=dtype, out_dtype=dtype, clamp_mode='warn')
        assert out.dtype == dtype


def test_torch_to_images_arg_helper():
    assert _torch_to_images_normalise_args((64, 128, 3), torch.uint8, 'HWC', 'CHW', None, None) == ((-1, -3, -2), torch.uint8, torch.uint8, -3)
    assert _torch_to_images_normalise_args((64, 128, 3), torch.uint8, 'HWC', 'HWC', None, None) == ((-3, -2, -1), torch.uint8, torch.uint8, -1)


def test_torch_to_images_invalid_args():
    inp_float = torch.rand(8, 3, 64, 64, dtype=torch.float32)

    # check tensor
    with pytest.raises(TypeError, match="images must be of type"):
        torch_to_images(tensor=None)
    with pytest.raises(ValueError, match='dim "C", required: 1 or 3'):
        torch_to_images(tensor=torch.rand(8, 2, 16, 16, dtype=torch.float32))
    with pytest.raises(ValueError, match='dim "C", required: 1 or 3'):
        torch_to_images(tensor=torch.rand(8, 16, 16, 3, dtype=torch.float32))
    with pytest.raises(ValueError, match='images must have 3 or more dimensions corresponding to'):
        torch_to_images(tensor=torch.rand(16, 16, dtype=torch.float32))

    # check dims
    with pytest.raises(TypeError, match="in_dims must be of type"):
        torch_to_images(inp_float, in_dims=None)
    with pytest.raises(TypeError, match="out_dims must be of type"):
        torch_to_images(inp_float, out_dims=None)
    with pytest.raises(KeyError, match="in_dims contains the symbols: 'INVALID', must contain only permutations of: 'CHW'"):
        torch_to_images(inp_float, in_dims='INVALID')
    with pytest.raises(KeyError, match="out_dims contains the symbols: 'INVALID', must contain only permutations of: 'CHW'"):
        torch_to_images(inp_float, out_dims='INVALID')
    with pytest.raises(KeyError, match="in_dims contains the symbols: 'CHWW', must contain only permutations of: 'CHW'"):
        torch_to_images(inp_float, in_dims='CHWW')
    with pytest.raises(KeyError, match="out_dims contains the symbols: 'CHWW', must contain only permutations of: 'CHW'"):
        torch_to_images(inp_float, out_dims='CHWW')

    # check dtypes
    with pytest.raises(TypeError, match="images dtype: torch.float32 does not match in_dtype: torch.uint8"):
        torch_to_images(inp_float, in_dtype=torch.uint8)
    with pytest.raises(TypeError, match='in_dtype is not allowed'):
        torch_to_images(inp_float, in_dtype=torch.complex64)
    with pytest.raises(TypeError, match='out_dtype is not allowed'):
        torch_to_images(inp_float, out_dtype=torch.complex64)
    with pytest.raises(TypeError, match='in_dtype is not allowed'):
        torch_to_images(inp_float, in_dtype=torch.float16)
    with pytest.raises(TypeError, match='out_dtype is not allowed'):
        torch_to_images(inp_float, out_dtype=torch.float16)


def _check(target_shape, target_dtype, img, m=None, M=None):
    assert img.dtype == target_dtype
    assert img.shape == target_shape
    if isinstance(img, torch.Tensor): img = img.numpy()
    if m is not None: assert np.isclose(img.min(), m), f'min mismatch: {img.min()} (actual) != {m} (expected)'
    if M is not None: assert np.isclose(img.max(), M), f'max mismatch: {img.max()} (actual) != {M} (expected)'


def test_torch_to_images_adv():
    # CHW
    nchw_float = torch.rand(8, 3, 64, 32, dtype=torch.float32)
    nchw_uint8 = torch.randint(0, 255, (8, 3, 64, 32), dtype=torch.uint8)
    # HWC
    nhwc_float = torch.rand(8, 64, 32, 3, dtype=torch.float32)
    nhwc_uint8 = torch.randint(0, 255, (8, 64, 32, 3), dtype=torch.uint8)

    _check((8, 64, 32, 3), torch.uint8, torch_to_images(nchw_float))  # make sure default for numpy is CHW
    _check((8, 64, 32, 3), torch.uint8, torch_to_images(nchw_uint8))  # make sure default for numpy is CHW

    _check((8, 64, 32, 3), torch.uint8, torch_to_images(nchw_float, 'CHW'))
    _check((8, 64, 32, 3), torch.uint8, torch_to_images(nchw_uint8, 'CHW'))
    _check((8, 64, 32, 3), torch.uint8, torch_to_images(nhwc_float, 'HWC'))
    _check((8, 64, 32, 3), torch.uint8, torch_to_images(nhwc_uint8, 'HWC'))

    _check((8, 64, 32, 3), torch.uint8, torch_to_images(nchw_float, 'CHW', 'HWC'))
    _check((8, 64, 32, 3), torch.uint8, torch_to_images(nchw_uint8, 'CHW', 'HWC'))
    _check((8, 64, 32, 3), torch.uint8, torch_to_images(nhwc_float, 'HWC', 'HWC'))
    _check((8, 64, 32, 3), torch.uint8, torch_to_images(nhwc_uint8, 'HWC', 'HWC'))

    _check((8, 3, 64, 32), torch.uint8, torch_to_images(nchw_float, 'CHW', 'CHW'))
    _check((8, 3, 64, 32), torch.uint8, torch_to_images(nchw_uint8, 'CHW', 'CHW'))
    _check((8, 3, 64, 32), torch.uint8, torch_to_images(nhwc_float, 'HWC', 'CHW'))
    _check((8, 3, 64, 32), torch.uint8, torch_to_images(nhwc_uint8, 'HWC', 'CHW'))

    # random permute
    _check((8, 3, 64, 32), torch.uint8, torch_to_images(nchw_uint8, 'CHW', 'CHW'))
    _check((8, 3, 64, 32), torch.uint8, torch_to_images(nhwc_uint8, 'HWC', 'CHW'))
    _check((8, 3, 32, 64), torch.uint8, torch_to_images(nchw_uint8, 'CHW', 'CWH'))
    _check((8, 3, 32, 64), torch.uint8, torch_to_images(nhwc_uint8, 'HWC', 'CWH'))

    _check((8, 64, 3, 32), torch.uint8, torch_to_images(nchw_uint8, 'CHW', 'HCW'))
    _check((8, 64, 3, 32), torch.uint8, torch_to_images(nhwc_uint8, 'HWC', 'HCW'))
    _check((8, 32, 3, 64), torch.uint8, torch_to_images(nchw_uint8, 'CHW', 'WCH'))
    _check((8, 32, 3, 64), torch.uint8, torch_to_images(nhwc_uint8, 'HWC', 'WCH'))

    _check((8, 64, 32, 3), torch.uint8, torch_to_images(nchw_uint8, 'CHW', 'HWC'))
    _check((8, 64, 32, 3), torch.uint8, torch_to_images(nhwc_uint8, 'HWC', 'HWC'))
    _check((8, 32, 64, 3), torch.uint8, torch_to_images(nchw_uint8, 'CHW', 'WHC'))
    _check((8, 32, 64, 3), torch.uint8, torch_to_images(nhwc_uint8, 'HWC', 'WHC'))

    _check((8, 64, 32, 3), torch.float32, torch_to_images(nchw_float, 'CHW', 'HWC', out_dtype=torch.float32))
    _check((8, 64, 32, 3), torch.float32, torch_to_images(nchw_uint8, 'CHW', 'HWC', out_dtype=torch.float32))
    _check((8, 64, 32, 3), torch.float32, torch_to_images(nhwc_float, 'HWC', 'HWC', out_dtype=torch.float32))
    _check((8, 64, 32, 3), torch.float32, torch_to_images(nhwc_uint8, 'HWC', 'HWC', out_dtype=torch.float32))

    _check((8, 64, 32, 3), torch.float64, torch_to_images(nchw_float, 'CHW', 'HWC', out_dtype=torch.float64))
    _check((8, 64, 32, 3), torch.float64, torch_to_images(nchw_uint8, 'CHW', 'HWC', out_dtype=torch.float64))
    _check((8, 64, 32, 3), torch.float64, torch_to_images(nhwc_float, 'HWC', 'HWC', out_dtype=torch.float64))
    _check((8, 64, 32, 3), torch.float64, torch_to_images(nhwc_uint8, 'HWC', 'HWC', out_dtype=torch.float64))

    # random, but chance of this failing is almost impossible
    _check((8, 64, 32, 3), torch.uint8, torch_to_images(nchw_float, 'CHW', 'HWC', out_dtype=torch.uint8, in_min=0.25, in_max=0.75), m=0, M=255)
    _check((8, 64, 32, 3), torch.uint8, torch_to_images(nchw_uint8, 'CHW', 'HWC', out_dtype=torch.uint8, in_min=64,   in_max=192),  m=0, M=255)
    _check((8, 64, 32, 3), torch.uint8, torch_to_images(nhwc_float, 'HWC', 'HWC', out_dtype=torch.uint8, in_min=0.25, in_max=0.75), m=0, M=255)
    _check((8, 64, 32, 3), torch.uint8, torch_to_images(nhwc_uint8, 'HWC', 'HWC', out_dtype=torch.uint8, in_min=64,   in_max=192),  m=0, M=255)

    # random, but chance of this failing is almost impossible
    _check((8, 64, 32, 3), torch.float32, torch_to_images(nchw_float, 'CHW', 'HWC', out_dtype=torch.float32, in_min=0.25, in_max=0.75), m=0, M=1)
    _check((8, 64, 32, 3), torch.float32, torch_to_images(nchw_uint8, 'CHW', 'HWC', out_dtype=torch.float32, in_min=64,   in_max=192),  m=0, M=1)
    _check((8, 64, 32, 3), torch.float32, torch_to_images(nhwc_float, 'HWC', 'HWC', out_dtype=torch.float32, in_min=0.25, in_max=0.75), m=0, M=1)
    _check((8, 64, 32, 3), torch.float32, torch_to_images(nhwc_uint8, 'HWC', 'HWC', out_dtype=torch.float32, in_min=64,   in_max=192),  m=0, M=1)

    # random, but chance of this failing is almost impossible
    _check((8, 3, 64, 32), torch.uint8, torch_to_images(nchw_float, 'CHW', 'CHW', out_dtype=torch.uint8, in_min=0.25, in_max=0.75), m=0, M=255)
    _check((8, 3, 64, 32), torch.uint8, torch_to_images(nchw_uint8, 'CHW', 'CHW', out_dtype=torch.uint8, in_min=64,   in_max=192),  m=0, M=255)
    _check((8, 3, 64, 32), torch.uint8, torch_to_images(nhwc_float, 'HWC', 'CHW', out_dtype=torch.uint8, in_min=0.25, in_max=0.75), m=0, M=255)
    _check((8, 3, 64, 32), torch.uint8, torch_to_images(nhwc_uint8, 'HWC', 'CHW', out_dtype=torch.uint8, in_min=64,   in_max=192),  m=0, M=255)

    # random, but chance of this failing is almost impossible
    _check((8, 3, 64, 32), torch.float32, torch_to_images(nchw_float, 'CHW', 'CHW', out_dtype=torch.float32, in_min=0.25, in_max=0.75), m=0, M=1)
    _check((8, 3, 64, 32), torch.float32, torch_to_images(nchw_uint8, 'CHW', 'CHW', out_dtype=torch.float32, in_min=64,   in_max=192),  m=0, M=1)
    _check((8, 3, 64, 32), torch.float32, torch_to_images(nhwc_float, 'HWC', 'CHW', out_dtype=torch.float32, in_min=0.25, in_max=0.75), m=0, M=1)
    _check((8, 3, 64, 32), torch.float32, torch_to_images(nhwc_uint8, 'HWC', 'CHW', out_dtype=torch.float32, in_min=64,   in_max=192),  m=0, M=1)

    # check clamping
    with pytest.raises(ValueError, match='is outside of the required range'):
        torch_to_images(nchw_float, 'CHW', out_dtype=torch.float32, clamp_mode='error', in_min=0.25, in_max=0.75)
    with pytest.raises(ValueError, match='is outside of the required range'):
        torch_to_images(nchw_uint8, 'CHW', out_dtype=torch.float32, clamp_mode='error', in_min=64, in_max=192)
    with pytest.raises(ValueError, match='is outside of the required range'):
        torch_to_images(nhwc_float, 'HWC', out_dtype=torch.float32, clamp_mode='error', in_min=0.25, in_max=0.75)
    with pytest.raises(ValueError, match='is outside of the required range'):
        torch_to_images(nhwc_uint8, 'HWC', out_dtype=torch.float32, clamp_mode='error', in_min=64, in_max=192)
    with pytest.raises(KeyError, match="invalid clamp mode: 'asdf'"):
        torch_to_images(nhwc_uint8, 'HWC', out_dtype=torch.float32, clamp_mode='asdf', in_min=64, in_max=192)


def test_numpy_to_pil_image():
    # CHW
    nchw_float = np.random.rand(8, 3, 64, 32)
    nchw_uint8 = np.random.randint(0, 255, (8, 3, 64, 32), dtype='uint8')

    # HWC
    nhwc_float = np.random.rand(8, 64, 32, 3)
    nhwc_uint8 = np.random.randint(0, 255, (8, 64, 32, 3), dtype='uint8')

    with pytest.raises(ValueError, match='images do not have the correct number of channels for dim "C"'):
        numpy_to_images(nchw_float)  # make sure default for numpy is HWC
    with pytest.raises(ValueError, match='images do not have the correct number of channels for dim "C"'):
        numpy_to_images(nchw_uint8)  # make sure default for numpy is HWC

    _check((8, 64, 32, 3), 'uint8', numpy_to_images(nhwc_float))  # make sure default for numpy is HWC
    _check((8, 64, 32, 3), 'uint8', numpy_to_images(nhwc_uint8))  # make sure default for numpy is HWC
    _check((8, 64, 32, 3), 'uint8', numpy_to_images(nchw_float, 'CHW'))
    _check((8, 64, 32, 3), 'uint8', numpy_to_images(nchw_uint8, 'CHW'))

    for pil_images in [
        numpy_to_pil_images(nchw_float, 'CHW'),
        numpy_to_pil_images(nhwc_float),
    ]:
        assert isinstance(pil_images, np.ndarray)
        assert pil_images.shape == (8,)
        for pil_image in pil_images:
            pil_image: Image.Image
            assert isinstance(pil_image, Image.Image)
            assert pil_image.width == 32
            assert pil_image.height == 64

    # single image should be returned as an array of shape ()
    pil_image: np.ndarray = numpy_to_pil_images(np.random.rand(64, 32, 3))
    assert pil_image.shape == ()
    assert isinstance(pil_image, np.ndarray)
    assert isinstance(pil_image.tolist(), Image.Image)

    # check arb size
    pil_images: np.ndarray = numpy_to_pil_images(np.random.rand(4, 5, 2, 16, 32, 3))
    assert pil_images.shape == (4, 5, 2)


def test_numpy_image_min_max():
    # CHW
    nchw_float = np.random.rand(8, 3, 64, 32)
    nchw_uint8 = np.random.randint(0, 255, (8, 3, 64, 32), dtype='uint8')

    # HWC
    nhwc_float = np.random.rand(8, 64, 32, 3)
    nhwc_uint8 = np.random.randint(0, 255, (8, 64, 32, 3), dtype='uint8')

    _check((8, 64, 32, 3), 'uint8', numpy_to_images(nhwc_float, 'HWC', in_min=0, in_max=1))
    _check((8, 64, 32, 3), 'uint8', numpy_to_images(nhwc_uint8, 'HWC', in_min=0, in_max=255))
    _check((8, 64, 32, 3), 'uint8', numpy_to_images(nchw_float, 'CHW', in_min=0, in_max=1))
    _check((8, 64, 32, 3), 'uint8', numpy_to_images(nchw_uint8, 'CHW', in_min=0, in_max=255))

    # OUT: HWC

    _check((8, 64, 32, 3), 'uint8', numpy_to_images(nhwc_float, 'HWC', in_min=(0, 0, 0), in_max=(1, 1, 1)))
    _check((8, 64, 32, 3), 'uint8', numpy_to_images(nhwc_uint8, 'HWC', in_min=(0, 0, 0), in_max=(255, 255, 255)))
    _check((8, 64, 32, 3), 'uint8', numpy_to_images(nchw_float, 'CHW', in_min=(0, 0, 0), in_max=(1, 1, 1)))
    _check((8, 64, 32, 3), 'uint8', numpy_to_images(nchw_uint8, 'CHW', in_min=(0, 0, 0), in_max=(255, 255, 255)))

    _check((8, 64, 32, 3), 'uint8', numpy_to_images(nhwc_float, 'HWC', in_min=(0,), in_max=(1,)))    # should maybe disable this from working?
    _check((8, 64, 32, 3), 'uint8', numpy_to_images(nhwc_uint8, 'HWC', in_min=(0,), in_max=(255,)))  # should maybe disable this from working?
    _check((8, 64, 32, 3), 'uint8', numpy_to_images(nchw_float, 'CHW', in_min=(0,), in_max=(1,)))    # should maybe disable this from working?
    _check((8, 64, 32, 3), 'uint8', numpy_to_images(nchw_uint8, 'CHW', in_min=(0,), in_max=(255,)))  # should maybe disable this from working?

    _check((8, 64, 32, 1), 'uint8', numpy_to_images(nhwc_float[:, :, :, 0:1], 'HWC', in_min=(0,), in_max=(1,)))
    _check((8, 64, 32, 1), 'uint8', numpy_to_images(nhwc_uint8[:, :, :, 0:1], 'HWC', in_min=(0,), in_max=(255,)))
    _check((8, 64, 32, 1), 'uint8', numpy_to_images(nchw_float[:, 0:1, :, :], 'CHW', in_min=(0,), in_max=(1,)))
    _check((8, 64, 32, 1), 'uint8', numpy_to_images(nchw_uint8[:, 0:1, :, :], 'CHW', in_min=(0,), in_max=(255,)))

    _check((8, 64, 32, 1), 'uint8', numpy_to_images(nhwc_float[:, :, :, 0:1], 'HWC', in_min=0, in_max=1))
    _check((8, 64, 32, 1), 'uint8', numpy_to_images(nhwc_uint8[:, :, :, 0:1], 'HWC', in_min=0, in_max=255))
    _check((8, 64, 32, 1), 'uint8', numpy_to_images(nchw_float[:, 0:1, :, :], 'CHW', in_min=0, in_max=1))
    _check((8, 64, 32, 1), 'uint8', numpy_to_images(nchw_uint8[:, 0:1, :, :], 'CHW', in_min=0, in_max=255))

    # OUT: CHW

    _check((8, 3, 64, 32), 'uint8', numpy_to_images(nhwc_float, 'HWC', 'CHW', in_min=(0, 0, 0), in_max=(1, 1, 1)))
    _check((8, 3, 64, 32), 'uint8', numpy_to_images(nhwc_uint8, 'HWC', 'CHW', in_min=(0, 0, 0), in_max=(255, 255, 255)))
    _check((8, 3, 64, 32), 'uint8', numpy_to_images(nchw_float, 'CHW', 'CHW', in_min=(0, 0, 0), in_max=(1, 1, 1)))
    _check((8, 3, 64, 32), 'uint8', numpy_to_images(nchw_uint8, 'CHW', 'CHW', in_min=(0, 0, 0), in_max=(255, 255, 255)))

    _check((8, 3, 64, 32), 'uint8', numpy_to_images(nhwc_float, 'HWC', 'CHW', in_min=(0,), in_max=(1,)))    # should maybe disable this from working?
    _check((8, 3, 64, 32), 'uint8', numpy_to_images(nhwc_uint8, 'HWC', 'CHW', in_min=(0,), in_max=(255,)))  # should maybe disable this from working?
    _check((8, 3, 64, 32), 'uint8', numpy_to_images(nchw_float, 'CHW', 'CHW', in_min=(0,), in_max=(1,)))    # should maybe disable this from working?
    _check((8, 3, 64, 32), 'uint8', numpy_to_images(nchw_uint8, 'CHW', 'CHW', in_min=(0,), in_max=(255,)))  # should maybe disable this from working?

    _check((8, 1, 64, 32), 'uint8', numpy_to_images(nhwc_float[:, :, :, 0:1], 'HWC', 'CHW', in_min=(0,), in_max=(1,)))
    _check((8, 1, 64, 32), 'uint8', numpy_to_images(nhwc_uint8[:, :, :, 0:1], 'HWC', 'CHW', in_min=(0,), in_max=(255,)))
    _check((8, 1, 64, 32), 'uint8', numpy_to_images(nchw_float[:, 0:1, :, :], 'CHW', 'CHW', in_min=(0,), in_max=(1,)))
    _check((8, 1, 64, 32), 'uint8', numpy_to_images(nchw_uint8[:, 0:1, :, :], 'CHW', 'CHW', in_min=(0,), in_max=(255,)))

    _check((8, 1, 64, 32), 'uint8', numpy_to_images(nhwc_float[:, :, :, 0:1], 'HWC', 'CHW', in_min=0, in_max=1))
    _check((8, 1, 64, 32), 'uint8', numpy_to_images(nhwc_uint8[:, :, :, 0:1], 'HWC', 'CHW', in_min=0, in_max=255))
    _check((8, 1, 64, 32), 'uint8', numpy_to_images(nchw_float[:, 0:1, :, :], 'CHW', 'CHW', in_min=0, in_max=1))
    _check((8, 1, 64, 32), 'uint8', numpy_to_images(nchw_uint8[:, 0:1, :, :], 'CHW', 'CHW', in_min=0, in_max=255))


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
