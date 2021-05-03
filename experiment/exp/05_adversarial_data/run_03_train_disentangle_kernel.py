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
import pytorch_lightning as pl
import torch
from torch.nn import Parameter
from tqdm import tqdm

import experiment.exp.util.helper as H
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


def train_model_to_disentangle(
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


class Disentangler(DisentLightningModule):

    def __init__(
        self,
        model,
        loss: str = 'mse',
        optimizer: str = 'radam',
        lr: float = 1e-3,
        train_factors: Optional[List[int]] = None,
        train_pair_ratio: float = 4.0,
    ):
        super().__init__()
        self._model = model
        self.save_hyperparameters()
        self._i = 0

    def configure_optimizers(self):
        return H.make_optimizer(self, name=self.hparams.optimizer, lr=self.hparams.lr)

    def training_step(self, batch, batch_idx):
        (xs,), (factors,) = batch['x_targ'], batch['factors']
        # feed forward batch
        aug_xs = self._model(xs)
        assert aug_xs.shape == xs.shape
        # compute pairwise distances of factors and batch, and optimize to correspond
        loss = disentangle_loss(
            batch=aug_xs, factors=factors,
            num_pairs=int(len(xs) * self.hparams.train_pair_ratio),
            f_idxs=self.hparams.train_factors,
            loss_fn=self.hparams.loss,
            mean_dtype=torch.float64,
        )
        # show
        self._i += 1
        H.show_imgs(aug_xs[:9], i=self._i, step=500)
        # log
        self.log('loss', loss)
        return loss

    def forward(self, xs) -> torch.Tensor:
        return self._model(xs)


def train_pl_model_to_disentangle(
    model: torch.nn.Module,
    dataset='xysquares_4x4',
    train_batch_size: int = 128,
    train_epochs: int = 10,
    # disentangler settings
    factor_idxs: Optional[List[int]] = None,
    loss_fn: str = 'mse',
    train_optimizer: str = 'radam',
    train_lr: float = 1e-3,
    train_pair_ratio: float = 4.0,
):
    # make data
    dataset = H.make_dataset(dataset, factors=True)
    shuffle = len(dataset) <= 16777216
    if not shuffle:
        print('WARNING: not shuffling, dataset too big!')
    # make dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=train_batch_size, shuffle=shuffle,
        num_workers=psutil.cpu_count(), pin_memory=torch.cuda.is_available()
    )
    # make module
    module = Disentangler(
        model,
        train_factors=factor_idxs, train_pair_ratio=train_pair_ratio,
        loss=loss_fn, optimizer=train_optimizer, lr=train_lr,
    )
    # train
    trainer = pl.Trainer(
        checkpoint_callback=False, terminate_on_nan=False,
        max_epochs=train_epochs, gpus=int(torch.cuda.is_available()),
    )
    trainer.fit(module, dataloader)


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
        H.show_img(self._kernel[0], i=i, step=1000, scale=True)


class NN(DisentModule):

    def __init__(self):
        super().__init__()
        self._layers = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=6, kernel_size=7, padding=1+1+1),
            torch.nn.InstanceNorm2d(num_features=3),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=6, out_channels=9, kernel_size=7, padding=1+1+1),
            torch.nn.InstanceNorm2d(num_features=3),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=9, out_channels=9, kernel_size=7, padding=1+1+1),
            torch.nn.InstanceNorm2d(num_features=3),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=9, out_channels=6, kernel_size=7, padding=1+1+1),
            torch.nn.InstanceNorm2d(num_features=3),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=6, out_channels=3, kernel_size=7, padding=1+1+1),
        )

    def forward(self, xs):
        return self._layers(xs)


if __name__ == '__main__':
    model = Kernel(radius=55, channels=1)
    train_pl_model_to_disentangle(model=model)


