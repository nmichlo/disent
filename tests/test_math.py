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
from scipy.stats import gmean
from scipy.stats import hmean

from disent.util import to_numpy
from disent.util.math import torch_corr_matrix
from disent.util.math import torch_cov_matrix
from disent.util.math import torch_mean_generalized


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
    assert torch.allclose(torch_mean_generalized(xs, p='arithmetic', dim=1), torch.mean(xs, dim=1))
    assert torch.allclose(torch_mean_generalized(xs, p=1, dim=1), torch.mean(xs, dim=1))

    # scipy equivalents
    assert torch.allclose(torch_mean_generalized(xs, p='geometric', dim=1), torch.as_tensor(gmean(xs, axis=1)))
    assert torch.allclose(torch_mean_generalized(xs, p='harmonic', dim=1), torch.as_tensor(hmean(xs, axis=1)))
    assert torch.allclose(torch_mean_generalized(xs, p=0, dim=1), torch.as_tensor(gmean(xs, axis=1)))
    assert torch.allclose(torch_mean_generalized(xs, p=-1, dim=1), torch.as_tensor(hmean(xs, axis=1)))
    assert torch.allclose(torch_mean_generalized(xs, p=0), torch.as_tensor(gmean(xs, axis=None)))  # scipy default axis is 0
    assert torch.allclose(torch_mean_generalized(xs, p=-1), torch.as_tensor(hmean(xs, axis=None)))  # scipy default axis is 0

    # min max
    assert torch.allclose(torch_mean_generalized(xs, p='maximum', dim=1), torch.max(xs, dim=1).values)
    assert torch.allclose(torch_mean_generalized(xs, p='minimum', dim=1), torch.min(xs, dim=1).values)
    assert torch.allclose(torch_mean_generalized(xs, p=np.inf, dim=1), torch.max(xs, dim=1).values)
    assert torch.allclose(torch_mean_generalized(xs, p=-np.inf, dim=1), torch.min(xs, dim=1).values)

