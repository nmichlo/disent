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

import pytest
import torch

from disent.dataset.data import XYObjectData
from disent.dataset import DisentDataset
from disent.metrics import *
from disent.dataset.transform import ToImgTensorF32
from disent.util.function import wrapped_partial


# ========================================================================= #
# TESTS                                                                     #
# ========================================================================= #


@pytest.mark.parametrize('metric_fn', [
    wrapped_partial(metric_mig,          num_train=7),
    wrapped_partial(metric_unsupervised, num_train=7),
    wrapped_partial(metric_dci,          num_train=7, num_test=7),
    wrapped_partial(metric_sap,          num_train=7, num_test=7),
    wrapped_partial(metric_factor_vae,   num_train=7, num_eval=7, num_variance_estimate=7),
])
def test_metrics(metric_fn):
    z_size = 8
    # ground truth data
    # TODO: DisentDataset should not be needed to compute metrics!
    dataset = DisentDataset(XYObjectData(), transform=ToImgTensorF32())
    # randomly sampled representation
    get_repr = lambda x: torch.randn(len(x), z_size)
    # evaluate
    metric_fn(dataset, get_repr)


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
