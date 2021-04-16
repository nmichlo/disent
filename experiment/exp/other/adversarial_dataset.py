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

import torch_optimizer
import torchsort
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from disent.data.groundtruth import Shapes3dData
from disent.data.groundtruth import XYSquaresData
import torch
import torch.nn.functional as F

from disent.dataset.groundtruth import GroundTruthDataset
from disent.transform import ToStandardisedTensor
from disent.util.math_loss import multi_spearman_rank_loss
from disent.util.math_loss import spearman_rank_loss


# ========================================================================= #
# helper                                                                    #
# ========================================================================= #
from disent.util.math_loss import torch_soft_rank
from disent.util.math_loss import torch_soft_sort


def make_optimizer(model: torch.nn.Module, name: str = 'sgd', lr=1e-3):
    if isinstance(model, torch.nn.Module):
        params = model.parameters()
    elif isinstance(model, torch.Tensor):
        assert model.requires_grad
        params = [model]
    else:
        raise TypeError(f'cannot optimize type: {type(model)}')
    # make optimizer
    if name == 'sgd': return torch.optim.SGD(params, lr=lr)
    elif name == 'adam': return torch.optim.Adam(params, lr=lr)
    elif name == 'radam': return torch_optimizer.RAdam(params, lr=lr)
    else: raise KeyError(f'invalid optimizer name: {repr(name)}')


def make_data(name: str = 'xysquares'):
    if name == 'xysquares': data = XYSquaresData()
    elif name == 'shapes3d': data = Shapes3dData()
    else: raise KeyError(f'invalid data name: {repr(name)}')
    return data


def step_optimizer(optimizer, loss):
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def get_single_batch(dataloader, cuda=True):
    for batch in dataloader:
        (x_targ,) = batch['x_targ']
        break
    if cuda:
        x_targ = x_targ.cuda()
    return x_targ


def to_img(x, scale=False):
    assert x.dtype in {torch.float16, torch.float32, torch.float64, torch.complex32, torch.complex64}, f'unsupported dtype: {x.dtype}'
    x = x.detach().cpu()
    x = torch.abs(x)
    if scale:
        m, M = torch.min(x), torch.max(x)
        x = (x - m) / (M - m)
    x = torch.moveaxis(x, 0, -1)
    x = torch.clamp(x, 0, 1)
    x = (x * 255).to(torch.uint8)
    return x


def show_img(x, scale=False, i=None, step=None):
    if (i is None) or (step is None) or (i % step == 0):
        plt.imshow(to_img(x, scale=scale))
        plt.show()


# ========================================================================= #
# tests                                                                     #
# ========================================================================= #


def main():
    """
    test that the differentiable sorting works over a batch of images.
    """

    dataset = GroundTruthDataset(make_data('xysquares'), transform=ToStandardisedTensor())
    dataloader = DataLoader(dataset=dataset, batch_size=256, pin_memory=True, shuffle=True)

    y = get_single_batch(dataloader)
    # y += torch.randn_like(y) * 0.001  # prevent nan errors
    x = torch.randn_like(y, requires_grad=True)

    optimizer = make_optimizer(x, name='adam', lr=1e-2)

    for i in range(1001):
        # loss = multi_spearman_rank_loss(x, y, dims=(2, 3), nan_to_num=True)
        loss = 0.

        loss += F.mse_loss(
            torch_soft_rank(x, dims=(-3, -1)),
            torch_soft_rank(y, dims=(-3, -1)),
            reduction='mean',
        )
        loss += F.mse_loss(
            torch_soft_rank(x, dims=(-3, -2)),
            torch_soft_rank(y, dims=(-3, -2)),
            reduction='mean',
        )

        # loss += F.mse_loss(x, y, reduction='mean')
        # update variables
        step_optimizer(optimizer, loss)
        show_img(x[0], i=i, step=250)
        # compute loss
        print(i, float(loss))


# ========================================================================= #
# entrypoint                                                                #
# ========================================================================= #


if __name__ == '__main__':
    main()


