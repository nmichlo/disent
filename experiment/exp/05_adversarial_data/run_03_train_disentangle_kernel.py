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
from torch.nn import Parameter
from torch.utils.data import DataLoader
from tqdm import tqdm

import experiment.exp.util.helper as H
from disent.transform.functional import conv2d_channel_wise_fft
from disent.util import DisentModule
from disent.util import seed
from disent.util.math_loss import spearman_rank_loss
from experiment.exp.util.io_util import GithubWriter
from experiment.exp.util.io_util import torch_save_bytes


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


def train_model_to_disentangle(
    model,
    dataset='xysquares_1x1',
    batch_size=128,
    batch_samples_ratio=16.0,
    factor_idxs: List[int] = None,
    train_steps=10000,
    train_optimizer='radam',
    lr=1e-3,
    loss_fn='mse',
    step_callback=None,
    weight_decay: float = 0
):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # make dataset, model & optimizer
    dataset = H.make_dataset(dataset, factors=True)
    dataloader = DataLoader(dataset, batch_sampler=H.StochasticBatchSampler(dataset, batch_size), num_workers=psutil.cpu_count(), pin_memory=True)
    model = model.to(device=device)
    optimizer = H.make_optimizer(model, name=train_optimizer, lr=lr, weight_decay=weight_decay)
    # factors to optimise
    factor_idxs = None if (factor_idxs is None) else H.normalise_factor_idxs(dataset, factor_idxs)
    # train
    pbar = tqdm(postfix={'loss': 0.0}, total=train_steps, position=0, leave=True)
    for i, batch in enumerate(H.yield_dataloader(dataloader, steps=train_steps)):
        (batch,), (factors,) = batch['x_targ'], batch['factors']
        batch, factors = batch.to(device), factors.to(device)
        # feed forward batch
        aug_batch = model(batch)
        # compute pairwise distances of factors and batch, and optimize to correspond
        loss = disentangle_loss(batch=aug_batch, factors=factors, num_pairs=int(batch_size * batch_samples_ratio), f_idxs=factor_idxs, loss_fn=loss_fn, mean_dtype=torch.float64)
        # ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~ #
        if hasattr(model, 'augment_loss'):
            loss = model.augment_loss(loss)
        # ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~ #
        # update variables
        H.step_optimizer(optimizer, loss)
        pbar.update()
        pbar.set_postfix({'loss': float(loss)})
        if step_callback:
            step_callback(i+1)


# ========================================================================= #
# MAIN                                                                      #
# ========================================================================= #


class Kernel(DisentModule):
    def __init__(self, radius: int = 33, channels: int = 1, offset: float = 0.0, scale: float = 0.001, abs_val: bool = False, rescale: bool = False):
        super().__init__()
        assert channels in (1, 3)
        kernel = torch.randn(1, channels, 2*radius+1, 2*radius+1, dtype=torch.float32)
        if abs_val:
            kernel = torch.abs(kernel)
        kernel = offset + kernel * scale
        if rescale:
            kernel = kernel / torch.abs(kernel).sum(dim=(0, 2, 3), keepdim=True)
        self._kernel = Parameter(kernel)
        self._i = 1

    def forward(self, xs):
        return conv2d_channel_wise_fft(xs, self._kernel)

    def show_img(self, i=None):
        H.show_img(self._kernel[0], i=i, step=2500, scale=True)
        self._i = i

    def augment_loss(self, loss):
        # symmetric loss
        k, kt = self._kernel[0], torch.transpose(self._kernel[0], -1, -2)
        symmetric_loss = 0
        symmetric_loss += H.unreduced_loss(torch.flip(k, dims=[-1]), k, mode='mae').mean()
        symmetric_loss += H.unreduced_loss(torch.flip(k, dims=[-2]), k, mode='mae').mean()
        symmetric_loss += H.unreduced_loss(torch.flip(k, dims=[-1]), kt, mode='mae').mean()
        symmetric_loss += H.unreduced_loss(torch.flip(k, dims=[-2]), kt, mode='mae').mean()
        # final loss
        return loss + symmetric_loss * 10


# ========================================================================= #
# Models                                                                    #
# ========================================================================= #


def main(
    radius=63,
    seed_=777
):
    seed(seed_)

    model = Kernel(radius=radius, channels=1, offset=0.002, scale=0.01, abs_val=False, rescale=False)

    kwargs = dict(
        train_optimizer='radam',
        model=model,
        step_callback=model.show_img,
        train_steps=10_000,
        lr=1e-3,
        weight_decay=1e-1,
    )

    ghw = GithubWriter('nmichlo/uploads')
    ghw.write_file(f'disent/adversarial_kernel/r{radius}_random.pt', content=torch_save_bytes(model._kernel))
    train_model_to_disentangle(dataset='xysquares_8x8', **kwargs)
    ghw.write_file(f'disent/adversarial_kernel/r{radius}_xy8x8.pt', content=torch_save_bytes(model._kernel))
    train_model_to_disentangle(dataset='xysquares_4x4', **kwargs)
    ghw.write_file(f'disent/adversarial_kernel/r{radius}_xy4x4.pt', content=torch_save_bytes(model._kernel))
    train_model_to_disentangle(dataset='xysquares_2x2', **kwargs)
    ghw.write_file(f'disent/adversarial_kernel/r{radius}_xy2x2.pt', content=torch_save_bytes(model._kernel))
    train_model_to_disentangle(dataset='xysquares_1x1', **kwargs)
    ghw.write_file(f'disent/adversarial_kernel/r{radius}_xy1x1.pt', content=torch_save_bytes(model._kernel))


if __name__ == '__main__':
    main(radius=63)
    main(radius=47)
    main(radius=31)
    main(radius=15)

