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
from typing import List
from typing import Optional

import psutil
import torch
import pytorch_lightning as pl
from torch.nn import Parameter
from tqdm import tqdm

import experiment.exp.helper as H
from disent.transform.functional import conv2d_channel_wise_fft
from disent.util import DisentLightningModule
from disent.util import DisentModule
from disent.util.math_loss import spearman_rank_loss


# ========================================================================= #
# EXP                                                                       #
# ========================================================================= #


def disentangle_loss(
    batch: torch.Tensor,
    factors: torch.Tensor,
    num_pairs: int,
    f_idxs: Optional[List[int]] = None,
    loss_fn: str = 'mse',
    mean_dtype=None,
) -> torch.Tensor:
    assert len(batch) == len(factors)
    assert batch.ndim == 4
    assert factors.ndim == 2
    # random pairs
    ia, ib = torch.randint(0, len(batch), size=(2, num_pairs), device=batch.device)
    # get pairwise distances
    b_dists = H.pairwise_loss(batch[ia], batch[ib], mode=loss_fn, mean_dtype=mean_dtype)  # avoid precision errors
    # compute factor distances
    if f_idxs is not None:
        f_dists = torch.abs(factors[ia, f_idxs] - factors[ib, f_idxs])
    else:
        f_dists = torch.abs(factors[ia] - factors[ib]).sum(dim=-1)
    # optimise metric
    loss = spearman_rank_loss(b_dists, -f_dists)  # decreasing overlap should mean increasing factor dist
    return loss



def train_module_to_disentangle(
    model,
    dataset='xysquares_1x1',
    batch_size=128,
    batch_samples_ratio=4.0,
    factor_idxs: List[int] = None,
    train_steps=10000,
    train_optimizer='radam',
    train_lr=1e-3,
    loss_fn='mse',
    step_callback=None,
):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # make dataset
    dataset = H.make_dataset(dataset)
    model = model.to(device=device)
    # make optimizer
    optimizer = H.make_optimizer(model, name=train_optimizer, lr=train_lr)
    # factors to optimise
    factor_idxs = None if (factor_idxs is None) else H.normalise_factor_idxs(dataset, factor_idxs)
    # train
    pbar = tqdm(range(train_steps+1), postfix={'loss': 0.0})
    for i in pbar:
        batch, factors = H.sample_batch_and_factors(dataset, num_samples=batch_size, factor_mode='sample_random', device=device)
        # feed forward batch
        aug_batch = model(batch)
        # compute pairwise distances of factors and batch, and optimize to correspond
        loss = disentangle_loss(batch=aug_batch, factors=factors, num_pairs=int(batch_size * batch_samples_ratio), f_idxs=factor_idxs, loss_fn=loss_fn)
        # ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~ #
        # update variables
        H.step_optimizer(optimizer, loss)
        pbar.set_postfix({'loss': float(loss)})
        if step_callback:
            step_callback(i)


# ========================================================================= #
# MAIN                                                                      #
# ========================================================================= #


class Kernel(DisentModule):
    def __init__(self, radius: int = 33, channels: int = 1):
        super().__init__()
        assert channels in (1, 3)
        kernel = torch.abs(torch.randn(1, channels, 2*radius+1, 2*radius+1, dtype=torch.float32))
        kernel = kernel / kernel.sum(dim=(0, 2, 3), keepdim=True)
        self._kernel = Parameter(kernel)

    def forward(self, xs):
        return conv2d_channel_wise_fft(xs, self._kernel)

    def show_img(self, i=None):
        H.show_img(self._kernel[0], i=i, step=250, scale=True)


if __name__ == '__main__':

    model = Kernel(radius=33, channels=3)
    train_module_to_disentangle(model=model, step_callback=model.show_img)


