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


# ========================================================================= #
# convolve2d                                                                #
# ========================================================================= #


def _check_conv2d_inputs(signal, kernel):
    assert signal.ndim == 4, f'signal has {repr(signal.ndim)} dimensions, must have 4 dimensions instead: BxCxHxW'
    assert kernel.ndim == 2 or kernel.ndim == 4, f'kernel has {repr(kernel.ndim)} dimensions, must have 2 or 4 dimensions instead: HxW or BxCxHxW'
    # increase kernel size
    if kernel.ndim == 2:
        kernel = kernel[None, None, ...]
    # check kernel is an odd size
    kh, kw = kernel.shape[-2:]
    assert kh % 2 != 0 and kw % 2 != 0, f'kernel dimension sizes must be odd: ({kh}, {kw})'
    # check that broadcasting does not adjust the signal shape... TODO: relax this limitation?
    assert torch.broadcast_shapes(signal.shape[:2], kernel.shape[:2]) == signal.shape[:2]
    # done!
    return signal, kernel


def torch_conv2d_channel_wise(signal, kernel):
    """
    Apply the kernel to each channel separately!
    """
    signal, kernel = _check_conv2d_inputs(signal, kernel)
    # split channels into singel images
    fsignal = signal.reshape(-1, 1, *signal.shape[2:])
    # convolve each channel image
    out = torch.nn.functional.conv2d(fsignal, kernel, padding=(kernel.size(-2) // 2, kernel.size(-1) // 2))
    # reshape into original
    return out.reshape(-1, signal.shape[1], *out.shape[2:])


def torch_conv2d_channel_wise_fft(signal, kernel):
    """
    The same as torch_conv2d_channel_wise, but apply the kernel using fft.
    This is much more efficient for large filter sizes.

    Reference implementation is from: https://github.com/pyro-ppl/pyro/blob/ae55140acfdc6d4eade08b434195234e5ae8c261/pyro/ops/tensor_utils.py#L187
    """
    signal, kernel = _check_conv2d_inputs(signal, kernel)
    # get last dimension sizes
    sig_shape = np.array(signal.shape[-2:])
    ker_shape = np.array(kernel.shape[-2:])
    # compute padding
    padded_shape = sig_shape + ker_shape - 1
    # Compute convolution using fft.
    f_signal = torch.fft.rfft2(signal, s=tuple(padded_shape))
    f_kernel = torch.fft.rfft2(kernel, s=tuple(padded_shape))
    result = torch.fft.irfft2(f_signal * f_kernel, s=tuple(padded_shape))
    # crop final result
    s = (padded_shape - sig_shape) // 2
    f = s + sig_shape
    crop = result[..., s[0]:f[0], s[1]:f[1]]
    # done...
    return crop


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
