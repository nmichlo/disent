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
import numpy as np
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


def check_tensor(obs, low=0., high=1., dtype=torch.float32):
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
    assert low <= obs.min(), f'minimum value of tensor {obs.min()} is less than allowed minimum value: {low}'
    assert obs.max() <= high, f'maximum value of tensor {obs.max()} is greater than allowed maximum value: {high}'
    # DONE!
    return obs


def to_standardised_tensor(obs, size=None, check=True):
    """
    Basic transform that should be applied to
    any dataset before augmentation.

    1. resize if size is specified
    2. convert to tensor in range [0, 1]
    """
    # resize image
    if size is not None:
        obs = F_tv.to_pil_image(obs)
        obs = F_tv.resize(obs, size=size)
    # transform to tensor
    obs = F_tv.to_tensor(obs)
    # check that tensor is valid
    if check:
        obs = check_tensor(obs, low=0, high=1, dtype=torch.float32)
    return obs


# ========================================================================= #
# convolve                                                                  #
# ========================================================================= #


def _check_conv2d_inputs(signal, kernel):
    assert signal.ndim == 4, 'signal must have 4 dimensions: BxCxHxW'
    assert kernel.ndim == 2, 'kernel must have 2 dimensions: HxW'
    kh, kw = kernel.shape
    assert kh % 2 != 0 and kw % 2 != 0, f'kernel dimension sizes must be odd: ({kh}, {kw})'


def conv2d_channel_wise(signal, kernel):
    """
    Apply the kernel to each channel separately!
    """
    _check_conv2d_inputs(signal, kernel)
    # split channels into singel images
    fsignal = signal.reshape(-1, 1, *signal.shape[2:])
    # convolve each channel image
    out = torch.nn.functional.conv2d(fsignal, kernel[None, None, ...], padding=(kernel.size(-2) // 2, kernel.size(-1) // 2))
    # reshape into original
    return out.reshape(-1, signal.shape[1], *out.shape[2:])


def conv2d_channel_wise_fft(signal, kernel):
    """
    The same as conv2d_channel_wise, but apply the kernel using fft.
    This is much more efficient for large filter sizes.

    Reference implementation is from: https://github.com/pyro-ppl/pyro/blob/ae55140acfdc6d4eade08b434195234e5ae8c261/pyro/ops/tensor_utils.py#L187
    """
    _check_conv2d_inputs(signal, kernel)
    # get last dimension sizes
    m = np.array(signal.shape[-2:])
    n = np.array(kernel.shape[-2:])
    # compute padding
    truncate = np.maximum(m, n)
    padded_size = m + n - 1
    # Compute convolution using fft.
    f_signal = torch.fft.rfft2(signal, s=tuple(padded_size))
    f_kernel = torch.fft.rfft2(kernel, s=tuple(padded_size))
    result = torch.fft.irfft2(f_signal * f_kernel, s=tuple(padded_size))
    # crop final result
    s = (padded_size - truncate) // 2
    f = s + truncate
    result = result[..., s[0]:f[0], s[1]:f[1]]
    # done...
    return result


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
