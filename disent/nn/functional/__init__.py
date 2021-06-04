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
from typing import Optional
from typing import Union

import numpy as np
import torch

from disent.nn.functional._generic_tensors import generic_as_int32
from disent.nn.functional._generic_tensors import generic_max
from disent.nn.functional._generic_tensors import TypeGenericTensor
from disent.nn.functional._generic_tensors import TypeGenericTorch


log = logging.getLogger(__name__)


# ========================================================================= #
# pytorch math correlation functions                                        #
# ========================================================================= #


def torch_cov_matrix(xs: torch.Tensor):
    """
    Calculate the covariance matrix of multiple samples (N) of random vectors of size (X)
    https://en.wikipedia.org/wiki/Covariance_matrix
    - The input shape is: (N, X)
    - The output shape is: (X, X)

    This should be the same as:
        np.cov(xs, rowvar=False, ddof=0)
    """
    # NOTE:
    #   torch.mm is strict matrix multiplication
    #   however if we multiply arrays with broadcasting:
    #   size(3, 1) * size(1, 2) -> size(3, 2)  # broadcast, not matmul
    #   size(1, 3) * size(2, 1) -> size(2, 3)  # broadcast, not matmul
    # CHECK:
    assert xs.ndim == 2  # (N, X)
    Rxx = torch.mean(xs[:, :, None] * xs[:, None, :], dim=0)  # (X, X)
    ux = torch.mean(xs, dim=0)  # (X,)
    Kxx = Rxx - (ux[:, None] * ux[None, :])  # (X, X)
    return Kxx


def torch_corr_matrix(xs: torch.Tensor):
    """
    Calculate the pearson's correlation matrix of multiple samples (N) of random vectors of size (X)
    https://en.wikipedia.org/wiki/Pearson_correlation_coefficient
    https://en.wikipedia.org/wiki/Covariance_matrix
    - The input shape is: (N, X)
    - The output shape is: (X, X)

    This should be the same as:
        np.corrcoef(xs, rowvar=False, ddof=0)
    """
    Kxx = torch_cov_matrix(xs)
    diag_Kxx = torch.rsqrt(torch.diagonal(Kxx))
    corr = Kxx * (diag_Kxx[:, None] * diag_Kxx[None, :])
    return corr


def torch_rank_corr_matrix(xs: torch.Tensor):
    """
    Calculate the spearman's rank correlation matrix of multiple samples (N) of random vectors of size (X)
    https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient
    - The input shape is: (N, X)
    - The output shape is: (X, X)

    Pearson's correlation measures linear relationships
    Spearman's correlation measures monotonic relationships (whether linear or not)
    - defined in terms of the pearson's correlation matrix of the rank variables

    TODO: check, be careful of repeated values, this might not give the correct input?
    """
    rs = torch.argsort(xs, dim=0, descending=False)
    return torch_corr_matrix(rs.to(xs.dtype))


# aliases
torch_pearsons_corr_matrix = torch_corr_matrix
torch_spearmans_corr_matrix = torch_rank_corr_matrix


# ========================================================================= #
# pytorch math helper functions                                             #
# ========================================================================= #


def torch_tril_mean(mat: torch.Tensor, diagonal=-1):
    """
    compute the mean of the lower triangular matrix.
    """
    # checks
    N, M = mat.shape
    assert N == M
    assert diagonal == -1
    # compute
    n = (N*(N-1))/2
    mean = torch.tril(mat, diagonal=diagonal).sum() / n
    # done
    return mean


# ========================================================================= #
# pytorch mean functions                                                    #
# ========================================================================= #


_DimTypeHint = Optional[Union[int, List[int]]]

_POS_INF = float('inf')
_NEG_INF = float('-inf')

_GENERALIZED_MEAN_MAP = {
    'maximum':   _POS_INF,
    'quadratic':  2,
    'arithmetic': 1,
    'geometric':  0,
    'harmonic':  -1,
    'minimum':   _NEG_INF,
}


def torch_mean_generalized(xs: torch.Tensor, dim: _DimTypeHint = None, p: Union[int, str] = 1, keepdim: bool = False):
    """
    Compute the generalised mean.
    - p is the power

    harmonic mean ≤ geometric mean ≤ arithmetic mean
    - If values have the same units: Use the arithmetic mean.
    - If values have differing units: Use the geometric mean.
    - If values are rates: Use the harmonic mean.
    """
    if isinstance(p, str):
        p = _GENERALIZED_MEAN_MAP[p]
    # compute the specific extreme cases
    if p == _POS_INF:
        return torch.max(xs, dim=dim, keepdim=keepdim).values if (dim is not None) else torch.max(xs, keepdim=keepdim)
    elif p == _NEG_INF:
        return torch.min(xs, dim=dim, keepdim=keepdim).values if (dim is not None) else torch.min(xs, keepdim=keepdim)
    # compute the number of elements being averaged
    if dim is None:
        dim = list(range(xs.ndim))
    n = torch.prod(torch.as_tensor(xs.shape)[dim])
    # warn if the type is wrong
    if p != 1:
        if xs.dtype != torch.float64:
            warnings.warn(f'Input tensor to generalised mean might not have the required precision, type is {xs.dtype} not {torch.float64}.')
    # compute the specific cases
    if p == 0:
        # geometric mean
        # orig numerically unstable: torch.prod(xs, dim=dim) ** (1 / n)
        return torch.exp((1 / n) * torch.sum(torch.log(xs), dim=dim, keepdim=keepdim))
    elif p == 1:
        # arithmetic mean
        return torch.mean(xs, dim=dim, keepdim=keepdim)
    else:
        # generalised mean
        return ((1/n) * torch.sum(xs ** p, dim=dim, keepdim=keepdim)) ** (1/p)


def torch_mean_quadratic(xs, dim: _DimTypeHint = None, keepdim: bool = False):
    return torch_mean_generalized(xs, dim=dim, p='quadratic', keepdim=keepdim)


def torch_mean_geometric(xs, dim: _DimTypeHint = None, keepdim: bool = False):
    return torch_mean_generalized(xs, dim=dim, p='geometric', keepdim=keepdim)


def torch_mean_harmonic(xs, dim: _DimTypeHint = None, keepdim: bool = False):
    return torch_mean_generalized(xs, dim=dim, p='harmonic', keepdim=keepdim)


# ========================================================================= #
# helper                                                                    #
# ========================================================================= #


def torch_normalize(tensor: torch.Tensor, dims=None, dtype=None):
    # get min & max values
    if dims is not None:
        m, M = tensor, tensor
        for dim in dims:
            m, M = m.min(dim=dim, keepdim=True).values, M.max(dim=dim, keepdim=True).values
    else:
        m, M = tensor.min(), tensor.max()
    # scale tensor
    return (tensor.to(dtype=dtype) - m) / (M - m)  # automatically converts to float32 if needed


# ========================================================================= #
# polyfill - in later versions of pytorch                                   #
# ========================================================================= #


def torch_nan_to_num(input, nan=0.0, posinf=None, neginf=None):
    output = input.clone()
    if nan is not None:
        output[torch.isnan(input)] = nan
    if posinf is not None:
        output[input == np.inf] = posinf
    if neginf is not None:
        output[input == -np.inf] = neginf
    return output


# ========================================================================= #
# PCA                                                                       #
# ========================================================================= #


def torch_pca_eig(X, center=True, scale=False):
    """
    perform PCA over X
    - X is of size (num_points, vec_size)

    NOTE: unlike PCA_svd, the number of vectors/values returned is always: vec_size
    """
    n, _ = X.shape
    # center points along axes
    if center:
        X = X - X.mean(dim=0)
    # compute covariance -- TODO: optimise this line
    covariance = (1 / (n-1)) * torch.mm(X.T, X)
    if scale:
        scaling = torch.sqrt(1 / torch.diagonal(covariance))
        covariance = torch.mm(torch.diagflat(scaling), covariance)
    # compute eigen values and eigen vectors
    eigenvalues, eigenvectors = torch.eig(covariance, True)
    # sort components by decreasing variance
    components = eigenvectors.T
    explained_variance = eigenvalues[:, 0]
    idxs = torch.argsort(explained_variance, descending=True)
    return components[idxs], explained_variance[idxs]


def torch_pca_svd(X, center=True):
    """
    perform PCA over X
    - X is of size (num_points, vec_size)

    NOTE: unlike PCA_eig, the number of vectors/values returned is: min(num_points, vec_size)
    """
    n, _ = X.shape
    # center points along axes
    if center:
        X = X - X.mean(dim=0)
    # perform singular value decomposition
    u, s, v = torch.svd(X)
    # sort components by decreasing variance
    # these are already sorted?
    components = v.T
    explained_variance = torch.mul(s, s) / (n-1)
    return components, explained_variance


def torch_pca(X, center=True, mode='svd'):
    if mode == 'svd':
        return torch_pca_svd(X, center=center)
    elif mode == 'eig':
        return torch_pca_eig(X, center=center, scale=False)
    else:
        raise KeyError(f'invalid torch_pca mode: {repr(mode)}')


# ========================================================================= #
# DCT                                                                       #
# ========================================================================= #


def _flatten_dim_to_end(input, dim):
    # get shape
    s = input.shape
    n = s[dim]
    # do operation
    x = torch.moveaxis(input, dim, -1)
    x = x.reshape(-1, n)
    return x, s, n


def _unflatten_dim_to_end(input, dim, shape):
    # get intermediate shape
    s = list(shape)
    s.append(s.pop(dim))
    # undo operation
    x = input.reshape(*s)
    x = torch.moveaxis(x, -1, dim)
    return x


def torch_dct(x, dim=-1):
    """
    Discrete Cosine Transform (DCT) Type II
    """
    x, x_shape, n = _flatten_dim_to_end(x, dim=dim)
    if n % 2 != 0:
        raise ValueError(f'dct does not support odd sized dimension! trying to compute dct over dimension: {dim} of tensor with shape: {x_shape}')

    # concatenate even and odd offsets
    v_evn = x[:, 0::2]
    v_odd = x[:, 1::2].flip([1])
    v = torch.cat([v_evn, v_odd], dim=-1)

    # fast fourier transform
    fft = torch.fft.fft(v)

    # compute real & imaginary forward weights
    k = torch.arange(n, dtype=x.dtype, device=x.device) * (-np.pi / (2 * n))
    k = k[None, :]
    wr = torch.cos(k) * 2
    wi = torch.sin(k) * 2

    # compute dct
    dct = torch.real(fft) * wr - torch.imag(fft) * wi

    # restore shape
    return _unflatten_dim_to_end(dct, dim, x_shape)


def torch_idct(dct, dim=-1):
    """
    Inverse Discrete Cosine Transform (Inverse DCT) Type III
    """
    dct, dct_shape, n = _flatten_dim_to_end(dct, dim=dim)
    if n % 2 != 0:
        raise ValueError(f'idct does not support odd sized dimension! trying to compute idct over dimension: {dim} of tensor with shape: {dct_shape}')

    # compute real & imaginary backward weights
    k = torch.arange(n, dtype=dct.dtype, device=dct.device) * (np.pi / (2 * n))
    k = k[None, :]
    wr = torch.cos(k) / 2
    wi = torch.sin(k) / 2

    dct_real = dct
    dct_imag = torch.cat([0*dct_real[:, :1], -dct_real[:, 1:].flip([1])], dim=-1)

    fft_r = dct_real * wr - dct_imag * wi
    fft_i = dct_real * wi + dct_imag * wr
    # to complex number
    fft = torch.view_as_complex(torch.stack([fft_r, fft_i], dim=-1))

    # inverse fast fourier transform
    v = torch.fft.ifft(fft)
    v = torch.real(v)

    # undo even and odd offsets
    x = torch.zeros_like(dct)
    x[:, 0::2]  = v[:, :(n+1)//2]  # (N+1)//2 == N-(N//2)
    x[:, 1::2] += v[:, (n+0)//2:].flip([1])

    # restore shape
    return _unflatten_dim_to_end(x, dim, dct_shape)


def torch_dct2(x, dim1=-1, dim2=-2):
    d = torch_dct(x, dim=dim1)
    d = torch_dct(d, dim=dim2)
    return d


def torch_idct2(d, dim1=-1, dim2=-2):
    x = torch_idct(d, dim=dim2)
    x = torch_idct(x, dim=dim1)
    return x


# ========================================================================= #
# Torch Dim Helper                                                          #
# ========================================================================= #


def torch_unsqueeze_l(input: torch.Tensor, n: int):
    """
    Add n new axis to the left.

    eg. a tensor with shape (2, 3) passed to this function
        with n=2 will input in an output shape of (1, 1, 2, 3)
    """
    assert n >= 0, f'number of new axis cannot be less than zero, given: {repr(n)}'
    return input[((None,)*n) + (...,)]


def torch_unsqueeze_r(input: torch.Tensor, n: int):
    """
    Add n new axis to the right.

    eg. a tensor with shape (2, 3) passed to this function
        with n=2 will input in an output shape of (2, 3, 1, 1)
    """
    assert n >= 0, f'number of new axis cannot be less than zero, given: {repr(n)}'
    return input[(...,) + ((None,)*n)]


# ========================================================================= #
# Kernels                                                                   #
# ========================================================================= #


# TODO: replace with meshgrid based functions from experiment/exp/06_metric
#       these are more intuitive and flexible


def get_kernel_size(sigma: TypeGenericTensor = 1.0, truncate: TypeGenericTensor = 4.0):
    """
    This is how sklearn chooses kernel sizes.
    - sigma is the standard deviation, and truncate is the number of deviations away to truncate
    - our version broadcasts sigma and truncate together, returning the max kernel size needed over all values
    """
    # compute radius
    radius = generic_as_int32(truncate * sigma + 0.5)
    # get maximum value
    radius = int(generic_max(radius))
    # compute diameter
    return 2 * radius + 1


def torch_gaussian_kernel(
    sigma: TypeGenericTorch = 1.0, truncate: TypeGenericTorch = 4.0, size: int = None,
    dtype=torch.float32, device=None,
):
    # broadcast tensors together -- data may reference single memory locations
    sigma = torch.as_tensor(sigma, dtype=dtype, device=device)
    truncate = torch.as_tensor(truncate, dtype=dtype, device=device)
    sigma, truncate = torch.broadcast_tensors(sigma, truncate)
    # compute default size
    if size is None:
        size: int = get_kernel_size(sigma=sigma, truncate=truncate)
    # compute kernel
    x = torch.arange(size, dtype=sigma.dtype, device=sigma.device) - (size - 1) / 2
    # pad tensors correctly
    x = torch_unsqueeze_l(x, n=sigma.ndim)
    s = torch_unsqueeze_r(sigma, n=1)
    # compute
    return torch.exp(-(x ** 2) / (2 * s ** 2)) / (np.sqrt(2 * np.pi) * s)


def torch_gaussian_kernel_2d(
    sigma: TypeGenericTorch = 1.0, truncate: TypeGenericTorch = 4.0, size: int = None,
    sigma_b: TypeGenericTorch = None, truncate_b: TypeGenericTorch = None, size_b: int = None,
    dtype=torch.float32, device=None,
):
    # set default values
    if sigma_b is None: sigma_b = sigma
    if truncate_b is None: truncate_b = truncate
    if size_b is None: size_b = size
    # compute kernel
    kh = torch_gaussian_kernel(sigma=sigma,   truncate=truncate,   size=size,   dtype=dtype, device=device)
    kw = torch_gaussian_kernel(sigma=sigma_b, truncate=truncate_b, size=size_b, dtype=dtype, device=device)
    return kh[..., :, None] * kw[..., None, :]


def torch_box_kernel(radius: TypeGenericTorch = 1, dtype=torch.float32, device=None):
    radius = torch.abs(torch.as_tensor(radius, device=device))
    assert radius.dtype in {torch.int32, torch.int64}, f'box kernel radius must be of integer type: {radius.dtype}'
    # box kernel values
    radius_max = radius.max()
    crange = torch.abs(torch.arange(radius_max * 2 + 1, dtype=dtype, device=device) - radius_max)
    # pad everything
    radius = radius[..., None]
    crange = crange[None, ...]
    # compute box kernel
    kernel = (crange <= radius).to(dtype) / (radius * 2 + 1)
    # done!
    return kernel


def torch_box_kernel_2d(
    radius: TypeGenericTorch = 1,
    radius_b: TypeGenericTorch = None,
    dtype=torch.float32, device=None
):
    # set default values
    if radius_b is None: radius_b = radius
    # compute kernel
    kh = torch_box_kernel(radius=radius,   dtype=dtype, device=device)
    kw = torch_box_kernel(radius=radius_b, dtype=dtype, device=device)
    return kh[..., :, None] * kw[..., None, :]


# ========================================================================= #
# convolve                                                                  #
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
# DEBUG                                                                     #
# ========================================================================= #


def debug_transform_tensors(obj):
    """
    recursively convert all tensors to their shapes for debugging
    """
    if isinstance(obj, (torch.Tensor, np.ndarray)):
        return obj.shape
    elif isinstance(obj, dict):
        return {debug_transform_tensors(k): debug_transform_tensors(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return list(debug_transform_tensors(v) for v in obj)
    elif isinstance(obj, tuple):
        return tuple(debug_transform_tensors(v) for v in obj)
    elif isinstance(obj, set):
        return {debug_transform_tensors(k) for k in obj}
    else:
        return obj


# ========================================================================= #
# END                                                                       #
# ========================================================================= #

