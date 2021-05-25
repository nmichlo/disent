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
from torch.utils.data import DataLoader

import experiment.exp.util as H
from disent.util.math_loss import multi_spearman_rank_loss
from disent.util.math_loss import torch_soft_rank


# ========================================================================= #
# tests                                                                     #
# ========================================================================= #


def run_differentiable_sorting_loss(dataset='xysquares', loss_mode='spearman', optimizer='adam', lr=1e-2):
    """
    test that the differentiable sorting works over a batch of images.
    """

    dataset = H.make_dataset(dataset)
    dataloader = DataLoader(dataset=dataset, batch_size=256, pin_memory=True, shuffle=True)

    y = H.get_single_batch(dataloader)
    # y += torch.randn_like(y) * 0.001  # prevent nan errors
    x = torch.randn_like(y, requires_grad=True)

    optimizer = H.make_optimizer(x, name=optimizer, lr=lr)

    for i in range(1001):
        if loss_mode == 'spearman':
            loss = multi_spearman_rank_loss(x, y, dims=(2, 3), nan_to_num=True)
        elif loss_mode == 'mse_rank':
            loss = 0.
            loss += F.mse_loss(torch_soft_rank(x, dims=(-3, -1)), torch_soft_rank(y, dims=(-3, -1)), reduction='mean')
            loss += F.mse_loss(torch_soft_rank(x, dims=(-3, -2)), torch_soft_rank(y, dims=(-3, -2)), reduction='mean')
        elif loss_mode == 'mse':
            loss += F.mse_loss(x, y, reduction='mean')
        else:
            raise KeyError(f'invalid loss mode: {repr(loss_mode)}')

        # update variables
        H.step_optimizer(optimizer, loss)
        H.show_img(x[0], i=i, step=250)

        # compute loss
        print(i, float(loss))


# ========================================================================= #
# MAIN                                                                      #
# ========================================================================= #


if __name__ == '__main__':
    run_differentiable_sorting_loss()

