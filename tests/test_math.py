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
import pytest
import torch
from scipy.stats import gmean
from scipy.stats import hmean

from disent.dataset import DisentDataset
from disent.dataset.data import XYObjectData
from disent.dataset.sampling import RandomSampler
from disent.dataset.transform import ToImgTensorF32
from disent.nn.functional import torch_conv2d_channel_wise
from disent.nn.functional import torch_conv2d_channel_wise_fft
from disent.nn.functional import torch_corr_matrix
from disent.nn.functional import torch_cov_matrix
from disent.nn.functional import torch_dct
from disent.nn.functional import torch_dct2
from disent.nn.functional import torch_dist
from disent.nn.functional import torch_dist_hamming
from disent.nn.functional import torch_gaussian_kernel_2d
from disent.nn.functional import torch_idct
from disent.nn.functional import torch_idct2
from disent.nn.functional import torch_mean_generalized
from disent.nn.functional import torch_norm
from disent.nn.functional import torch_norm_euclidean
from disent.nn.functional import torch_norm_manhattan
from disent.util import to_numpy

# ========================================================================= #
# TESTS                                                                     #
# ========================================================================= #


def test_cov_corr():
    for i in range(5, 1000, 250):
        for j in range(2, 100, 25):
            # these match when torch.float64 is used, not when torch float32 is used...
            xs = torch.randn(i, j, dtype=torch.float64)

            np_cov = torch.from_numpy(np.cov(to_numpy(xs), rowvar=False, ddof=0)).to(xs.dtype)
            np_cor = torch.from_numpy(np.corrcoef(to_numpy(xs), rowvar=False, ddof=0)).to(xs.dtype)

            cov = torch_cov_matrix(xs)
            cor = torch_corr_matrix(xs)

            assert torch.allclose(np_cov, cov)
            assert torch.allclose(np_cor, cor)


def test_generalised_mean():
    xs = torch.abs(torch.randn(2, 1000, 3, dtype=torch.float64))

    # normal
    assert torch.allclose(torch_mean_generalized(xs, p="arithmetic", dim=1), torch.mean(xs, dim=1))
    assert torch.allclose(torch_mean_generalized(xs, p=1, dim=1), torch.mean(xs, dim=1))

    # scipy equivalents
    assert torch.allclose(torch_mean_generalized(xs, p="geometric", dim=1), torch.as_tensor(gmean(xs, axis=1)))
    assert torch.allclose(torch_mean_generalized(xs, p="harmonic", dim=1), torch.as_tensor(hmean(xs, axis=1)))
    assert torch.allclose(torch_mean_generalized(xs, p=0, dim=1), torch.as_tensor(gmean(xs, axis=1)))
    assert torch.allclose(torch_mean_generalized(xs, p=-1, dim=1), torch.as_tensor(hmean(xs, axis=1)))
    assert torch.allclose(
        torch_mean_generalized(xs, p=0), torch.as_tensor(gmean(xs, axis=None))
    )  # scipy default axis is 0
    assert torch.allclose(
        torch_mean_generalized(xs, p=-1), torch.as_tensor(hmean(xs, axis=None))
    )  # scipy default axis is 0

    # min max
    assert torch.allclose(torch_mean_generalized(xs, p="maximum", dim=1), torch.amax(xs, dim=1))
    assert torch.allclose(torch_mean_generalized(xs, p="minimum", dim=1), torch.amin(xs, dim=1))
    assert torch.allclose(torch_mean_generalized(xs, p=np.inf, dim=1), torch.amax(xs, dim=1))
    assert torch.allclose(torch_mean_generalized(xs, p=-np.inf, dim=1), torch.amin(xs, dim=1))


def test_p_norm():
    xs = torch.abs(torch.randn(2, 1000, 3, dtype=torch.float64))

    inf = float("inf")

    # torch equivalents
    assert torch.allclose(torch_norm(xs, p=1, dim=-1), torch.linalg.norm(xs, ord=1, dim=-1))
    assert torch.allclose(torch_norm(xs, p=2, dim=-1), torch.linalg.norm(xs, ord=2, dim=-1))
    assert torch.allclose(torch_norm(xs, p="maximum", dim=-1), torch.linalg.norm(xs, ord=inf, dim=-1))

    # torch equivalents -- less than zero [FAIL]
    with pytest.raises(ValueError, match="p-norm cannot have a p value less than 1"):
        assert torch.allclose(torch_norm(xs, p=0, dim=-1), torch.linalg.norm(xs, ord=0, dim=-1))
    with pytest.raises(ValueError, match="p-norm cannot have a p value less than 1"):
        assert torch.allclose(torch_norm(xs, p="minimum", dim=-1), torch.linalg.norm(xs, ord=-inf, dim=-1))

    # torch equivalents -- less than zero
    assert torch.allclose(torch_dist(xs, p=0, dim=-1), torch.linalg.norm(xs, ord=0, dim=-1))
    assert torch.allclose(torch_dist(xs, p="minimum", dim=-1), torch.linalg.norm(xs, ord=-inf, dim=-1))
    assert torch.allclose(torch_dist(xs, p=0, dim=-1), torch.linalg.norm(xs, ord=0, dim=-1))
    assert torch.allclose(torch_dist(xs, p="minimum", dim=-1), torch.linalg.norm(xs, ord=-inf, dim=-1))

    # test other axes
    ys = torch.flatten(torch.moveaxis(xs, 0, -1), start_dim=1, end_dim=-1)
    assert torch.allclose(torch_norm(xs, p=2, dim=[0, -1]), torch.linalg.norm(ys, ord=2, dim=-1))
    ys = torch.flatten(xs, start_dim=1, end_dim=-1)
    assert torch.allclose(torch_norm(xs, p=1, dim=[-2, -1]), torch.linalg.norm(ys, ord=1, dim=-1))
    ys = torch.flatten(torch.moveaxis(xs, -1, 0), start_dim=1, end_dim=-1)
    assert torch.allclose(torch_dist(xs, p=0, dim=[0, 1]), torch.linalg.norm(ys, ord=0, dim=-1))

    # check equal names
    assert torch.allclose(torch_dist(xs, dim=1, p="euclidean"), torch_norm_euclidean(xs, dim=1))
    assert torch.allclose(torch_dist(xs, dim=1, p=2), torch_norm_euclidean(xs, dim=1))
    assert torch.allclose(torch_dist(xs, dim=1, p="manhattan"), torch_norm_manhattan(xs, dim=1))
    assert torch.allclose(torch_dist(xs, dim=1, p=1), torch_norm_manhattan(xs, dim=1))
    assert torch.allclose(torch_dist(xs, dim=1, p="hamming"), torch_dist_hamming(xs, dim=1))
    assert torch.allclose(torch_dist(xs, dim=1, p=0), torch_dist_hamming(xs, dim=1))

    # check axes
    assert torch_dist(xs, dim=1, p=2, keepdim=False).shape == (2, 3)
    assert torch_dist(xs, dim=1, p=2, keepdim=True).shape == (2, 1, 3)
    assert torch_dist(xs, dim=-1, p=1, keepdim=False).shape == (2, 1000)
    assert torch_dist(xs, dim=-1, p=1, keepdim=True).shape == (2, 1000, 1)
    assert torch_dist(xs, dim=0, p=0, keepdim=False).shape == (1000, 3)
    assert torch_dist(xs, dim=0, p=0, keepdim=True).shape == (1, 1000, 3)
    assert torch_dist(xs, dim=[0, -1], p=-inf, keepdim=False).shape == (1000,)
    assert torch_dist(xs, dim=[0, -1], p=-inf, keepdim=True).shape == (1, 1000, 1)
    assert torch_dist(xs, dim=[0, 1], p=inf, keepdim=False).shape == (3,)
    assert torch_dist(xs, dim=[0, 1], p=inf, keepdim=True).shape == (1, 1, 3)

    # check norm over all
    assert torch_dist(xs, dim=None, p=0, keepdim=False).shape == ()
    assert torch_dist(xs, dim=None, p=0, keepdim=True).shape == (1, 1, 1)
    assert torch_dist(xs, dim=None, p=1, keepdim=False).shape == ()
    assert torch_dist(xs, dim=None, p=1, keepdim=True).shape == (1, 1, 1)
    assert torch_dist(xs, dim=None, p=2, keepdim=False).shape == ()
    assert torch_dist(xs, dim=None, p=2, keepdim=True).shape == (1, 1, 1)
    assert torch_dist(xs, dim=None, p=inf, keepdim=False).shape == ()
    assert torch_dist(xs, dim=None, p=inf, keepdim=True).shape == (1, 1, 1)


def test_dct():
    x = torch.randn(128, 3, 64, 32, dtype=torch.float64)

    # chceck +ve dims
    assert torch.allclose(x, torch_idct(torch_dct(x, dim=0), dim=0))
    with pytest.raises(ValueError, match="does not support odd sized dimension"):
        torch.allclose(x, torch_idct(torch_dct(x, dim=1), dim=1))
    assert torch.allclose(x, torch_idct(torch_dct(x, dim=2), dim=2))
    assert torch.allclose(x, torch_idct(torch_dct(x, dim=3), dim=3))

    # chceck -ve dims
    assert torch.allclose(x, torch_idct(torch_dct(x, dim=-4), dim=-4))
    with pytest.raises(ValueError, match="does not support odd sized dimension"):
        torch.allclose(x, torch_idct(torch_dct(x, dim=-3), dim=-3))
    assert torch.allclose(x, torch_idct(torch_dct(x, dim=-2), dim=-2))
    assert torch.allclose(x, torch_idct(torch_dct(x, dim=-1), dim=-1))

    # check defaults
    assert torch.allclose(torch_dct(x), torch_dct(x, dim=-1))
    assert torch.allclose(torch_idct(x), torch_idct(x, dim=-1))

    # check dct2
    assert torch.allclose(x, torch_idct2(torch_dct2(x)))
    assert torch.allclose(x, torch_idct2(torch_dct2(x)))

    # check defaults dct2
    assert torch.allclose(torch_dct2(x), torch_dct2(x, dim1=-1, dim2=-2))
    assert torch.allclose(torch_dct2(x), torch_dct2(x, dim1=-2, dim2=-1))
    assert torch.allclose(torch_idct2(x), torch_idct2(x, dim1=-1, dim2=-2))
    assert torch.allclose(torch_idct2(x), torch_idct2(x, dim1=-2, dim2=-1))
    # check order dct2
    assert torch.allclose(torch_dct2(x, dim1=-1, dim2=-2), torch_dct2(x, dim1=-2, dim2=-1))
    assert torch.allclose(torch_dct2(x, dim1=-1, dim2=-4), torch_dct2(x, dim1=-4, dim2=-1))
    assert torch.allclose(torch_dct2(x, dim1=-4, dim2=-1), torch_dct2(x, dim1=-1, dim2=-4))
    assert torch.allclose(torch_idct2(x, dim1=-1, dim2=-2), torch_idct2(x, dim1=-2, dim2=-1))
    assert torch.allclose(torch_idct2(x, dim1=-1, dim2=-4), torch_idct2(x, dim1=-4, dim2=-1))
    assert torch.allclose(torch_idct2(x, dim1=-4, dim2=-1), torch_idct2(x, dim1=-1, dim2=-4))


def test_fft_conv2d():
    data = XYObjectData()
    dataset = DisentDataset(data, RandomSampler(), transform=ToImgTensorF32(), augment=None)
    # sample data
    factors = dataset.gt_data.sample_random_factor_traversal(f_idx=2)
    batch = dataset.dataset_batch_from_factors(factors=factors, mode="input")
    # test torch_conv2d_channel_wise variants
    for i in range(1, 5):
        kernel = torch_gaussian_kernel_2d(sigma=i)
        out_cnv = torch_conv2d_channel_wise(signal=batch, kernel=kernel)[0]
        out_fft = torch_conv2d_channel_wise_fft(signal=batch, kernel=kernel)[0]
        assert torch.amax(torch.abs(out_cnv - out_fft)) < 1e-6


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
