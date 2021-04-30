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

import torch
import torch.nn.functional as F
from tqdm import tqdm

import experiment.exp.helper as H
from disent.transform.functional import conv2d_channel_wise_fft
from disent.util.math_loss import spearman_rank_loss


# ========================================================================= #
# EXP                                                                       #
# ========================================================================= #


def train_kernel_to_disentangle_xy(
    dataset='xysquares_1x1',
    kernel_radius=33,
    kernel_channels=False,
    batch_size=128,
    batch_samples_ratio=4.0,
    batch_factor_mode='sample_random',  # sample_random, sample_traversals
    batch_factor=None,
    batch_aug_both=True,
    train_steps=10000,
    train_optimizer='radam',
    train_lr=1e-3,
    loss_dist_mse=True,
    progress=True,
):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # make dataset
    dataset = H.make_dataset(dataset)
    # make trainable kernel
    kernel = torch.abs(torch.randn(1, 3 if kernel_channels else 1, 2*kernel_radius+1, 2*kernel_radius+1, dtype=torch.float32, device=device))
    kernel = kernel / kernel.sum(dim=(0, 2, 3), keepdim=True)
    kernel = torch.tensor(kernel, device=device, requires_grad=True)
    # make optimizer
    optimizer = H.make_optimizer(kernel, name=train_optimizer, lr=train_lr)

    # factor to optimise
    f_idx = H.normalise_factor_idx(dataset, batch_factor) if (batch_factor is not None) else None

    # train
    pbar = tqdm(range(train_steps+1), postfix={'loss': 0.0}, disable=not progress)
    for i in pbar:
        batch, factors = H.sample_batch_and_factors(dataset, num_samples=batch_size, factor_mode=batch_factor_mode, factor=batch_factor, device=device)
        # random pairs
        ia, ib = torch.randint(0, len(batch), size=(2, int(batch_size * batch_samples_ratio)), device=batch.device)
        # compute loss distances
        aug_batch = conv2d_channel_wise_fft(batch, kernel)
        (targ_a, targ_b) = (aug_batch[ia], aug_batch[ib]) if batch_aug_both else (aug_batch[ia], batch[ib])
        if loss_dist_mse:
            b_dists = F.mse_loss(targ_a, targ_b, reduction='none').sum(dim=(-3, -2, -1))
        else:
            b_dists = torch.abs(targ_a - targ_b).sum(dim=(-3, -2, -1))
        # compute factor distances
        if f_idx:
            f_dists = torch.abs(factors[ia, f_idx] - factors[ib, f_idx])
        else:
            f_dists = torch.abs(factors[ia] - factors[ib]).sum(dim=-1)
        # optimise metric
        loss = spearman_rank_loss(b_dists, -f_dists)  # decreasing overlap should mean increasing factor dist
        # ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~ #
        # update variables
        H.step_optimizer(optimizer, loss)
        H.show_img(kernel[0], i=i, step=100, scale=True)
        pbar.set_postfix({'loss': float(loss)})


# ========================================================================= #
# MAIN                                                                      #
# ========================================================================= #


if __name__ == '__main__':
    train_kernel_to_disentangle_xy()


