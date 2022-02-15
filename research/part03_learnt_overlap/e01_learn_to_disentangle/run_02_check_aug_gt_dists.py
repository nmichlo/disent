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
import torch.nn.functional as F
from matplotlib import pyplot as plt
from tqdm import tqdm

import research.code.util as H
from disent.nn.functional import torch_box_kernel_2d
from disent.nn.functional import torch_conv2d_channel_wise_fft
from disent.nn.functional import torch_gaussian_kernel_2d


# ========================================================================= #
# distance function                                                         #
# ========================================================================= #


def spearman_rank_dist(
    pred: torch.Tensor,
    targ: torch.Tensor,
    reduction='mean',
    nan_to_num=False,
):
    # add missing dim
    if pred.ndim == 1:
        pred, targ = pred.reshape(1, -1), targ.reshape(1, -1)
    assert pred.shape == targ.shape
    assert pred.ndim == 2
    # sort the last dimension of the 2D tensors
    pred = torch.argsort(pred).to(torch.float32)
    targ = torch.argsort(targ).to(torch.float32)
    # compute individual losses
    # TODO: this can result in nan values, what to do then?
    pred = pred - pred.mean(dim=-1, keepdim=True)
    pred = pred / pred.norm(dim=-1, keepdim=True)
    targ = targ - targ.mean(dim=-1, keepdim=True)
    targ = targ / targ.norm(dim=-1, keepdim=True)
    # replace nan values
    if nan_to_num:
        pred = torch.nan_to_num(pred, nan=0.0)
        targ = torch.nan_to_num(targ, nan=0.0)
    # compute the final loss
    loss = (pred * targ).sum(dim=-1)
    # reduce the loss
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'none':
        return loss
    else:
        raise KeyError(f'Invalid reduction mode: {repr(reduction)}')


def check_xy_squares_dists(kernel='box', repeats=100, samples=256, pairwise_samples=256, kernel_radius=32, show_prog=True):
    if kernel == 'box':
        kernel = torch_box_kernel_2d(radius=kernel_radius)[None, ...]
    elif kernel == 'max_box':
        crange = torch.abs(torch.arange(kernel_radius * 2 + 1) - kernel_radius)
        y, x = torch.meshgrid(crange, crange)
        d = torch.maximum(x, y) + 1
        d = d.max() - d
        kernel = (d.to(torch.float32) / d.sum())[None, None, ...]
    elif kernel == 'min_box':
        crange = torch.abs(torch.arange(kernel_radius * 2 + 1) - kernel_radius)
        y, x = torch.meshgrid(crange, crange)
        d = torch.minimum(x, y) + 1
        d = d.max() - d
        kernel = (d.to(torch.float32) / d.sum())[None, None, ...]
    elif kernel == 'manhat_box':
        crange = torch.abs(torch.arange(kernel_radius * 2 + 1) - kernel_radius)
        y, x = torch.meshgrid(crange, crange)
        d = (y + x) + 1
        d = d.max() - d
        kernel = (d.to(torch.float32) / d.sum())[None, None, ...]
    elif kernel == 'gaussian':
        kernel = torch_gaussian_kernel_2d(sigma=kernel_radius / 4.0, truncate=4.0)[None, None, ...]
    else:
        raise KeyError(f'invalid kernel mode: {repr(kernel)}')

    # make dataset
    dataset = H.make_dataset('xysquares')

    losses = []
    prog = tqdm(range(repeats), postfix={'loss': 0.0}) if show_prog else range(repeats)

    for i in prog:
        # get random samples
        factors = dataset.sample_factors(samples)
        batch = dataset.dataset_batch_from_factors(factors, mode='target')
        if torch.cuda.is_available():
            batch = batch.cuda()
            kernel = kernel.cuda()
        factors = torch.from_numpy(factors).to(dtype=torch.float32, device=batch.device)

        # random pairs
        ia, ib = torch.randint(0, len(batch), size=(2, pairwise_samples), device=batch.device)

        # compute factor distances
        f_dists = torch.abs(factors[ia] - factors[ib]).sum(dim=-1)

        # compute loss distances
        aug_batch = torch_conv2d_channel_wise_fft(batch, kernel)
        # TODO: aug - batch or aug - aug
        # b_dists = torch.abs(aug_batch[ia] - aug_batch[ib]).sum(dim=(-3, -2, -1))
        b_dists = F.mse_loss(aug_batch[ia], aug_batch[ib], reduction='none').sum(dim=(-3, -2, -1))

        # compute ranks
        # losses.append(float(torch.clamp(torch_mse_rank_loss(b_dists, f_dists), 0, 100)))
        # losses.append(float(torch.abs(torch.argsort(f_dists, descending=True) - torch.argsort(b_dists, descending=False)).to(torch.float32).mean()))
        losses.append(float(spearman_rank_dist(b_dists, f_dists)))

        if show_prog:
            prog.set_postfix({'loss': np.mean(losses)})

    return np.mean(losses), aug_batch[0]


def run_check_all_xy_squares_dists(show=False):
    for kernel in [
        'box',
        'max_box',
        'min_box',
        'manhat_box',
        'gaussian',
    ]:
        rs = list(range(1, 33, 4))
        ys = []
        for r in rs:
            ave_spearman, last_img = check_xy_squares_dists(kernel=kernel, repeats=32, samples=128, pairwise_samples=1024, kernel_radius=r, show_prog=False)
            H.plt_imshow(H.to_img(last_img, scale=True), show=show)
            ys.append(abs(ave_spearman))
            print(kernel, r, ':', r*2+1, abs(ave_spearman))
        plt.plot(rs, ys, label=kernel)
    plt.legend()
    plt.show()


# ========================================================================= #
# MAIN                                                                      #
# ========================================================================= #


if __name__ == '__main__':
    run_check_all_xy_squares_dists()
