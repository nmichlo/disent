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

import dataclasses
from typing import List
from typing import Optional

import hydra
import os

import logging
import psutil
import torch
import wandb
from omegaconf import OmegaConf
from torch.nn import Parameter
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from tqdm import tqdm

import experiment.exp.util.helper as H
from disent.transform.functional import conv2d_channel_wise_fft
from disent.util import DisentLightningModule
from disent.util import DisentModule
from disent.util import make_box_str
from disent.util import seed
from disent.util.math_loss import spearman_rank_loss
from experiment.exp.util.io_util import torch_write
from experiment.run import hydra_append_progress_callback
from experiment.run import hydra_check_cuda
from experiment.run import hydra_make_logger
from experiment.util.callbacks.callbacks_base import _PeriodicCallback
from experiment.util.hydra_utils import make_non_strict
from experiment.util.logger_util import wb_log_metrics


log = logging.getLogger(__name__)


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


class DisentangleModule(DisentLightningModule):

    def __init__(
        self,
        model,
        hparams,
    ):
        super().__init__()
        self.model = model
        self.hparams = hparams

    def configure_optimizers(self):
        return H.make_optimizer(self, name=self.hparams.optimizer.name, lr=self.hparams.optimizer.lr, weight_decay=self.hparams.optimizer.weight_decay)

    def training_step(self, batch, batch_idx):
        (batch,), (factors,) = batch['x_targ'], batch['factors']
        # feed forward batch
        aug_batch = self.model(batch)
        # compute pairwise distances of factors and batch, and optimize to correspond
        loss = disentangle_loss(
            batch=aug_batch,
            factors=factors,
            num_pairs=int(len(batch) * self.hparams.train.pairs_ratio),
            f_idxs=None,
            loss_fn=self.hparams.train.loss,
            mean_dtype=torch.float64,
        )
        # ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~ #
        if hasattr(self.model, 'augment_loss'):
            loss_aug = self.model.augment_loss(self)
        else:
            loss_aug = 0
        loss += loss_aug
        # ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~ #
        self.log('loss', loss)
        # ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~ #
        return loss

    def forward(self, batch):
        return self.model(batch)


# ========================================================================= #
# MAIN                                                                      #
# ========================================================================= #


class Kernel(DisentModule):
    def __init__(self, radius: int = 33, channels: int = 1, offset: float = 0.0, scale: float = 0.001):
        super().__init__()
        assert channels in (1, 3)
        kernel = torch.randn(1, channels, 2*radius+1, 2*radius+1, dtype=torch.float32)
        kernel = offset + kernel * scale
        self._kernel = Parameter(kernel)

    def forward(self, xs):
        return conv2d_channel_wise_fft(xs, self._kernel)

    def make_train_periodic_callback(self, cfg) -> _PeriodicCallback:
        class ImShowCallback(_PeriodicCallback):
            def do_step(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
                wb_log_metrics(
                    trainer.logger, {
                        'kernel': wandb.Image(H.to_img(pl_module.model._kernel[0], scale=True).numpy()),
                    }
                )
        return ImShowCallback(every_n_steps=cfg.exp.show_every_n_steps)

    def augment_loss(self, framework: DisentLightningModule):
        # symmetric loss
        k, kt = self._kernel[0], torch.transpose(self._kernel[0], -1, -2)
        loss_symmetric = 0
        loss_symmetric += H.unreduced_loss(torch.flip(k, dims=[-1]), k, mode='mae').mean()
        loss_symmetric += H.unreduced_loss(torch.flip(k, dims=[-2]), k, mode='mae').mean()
        loss_symmetric += H.unreduced_loss(torch.flip(k, dims=[-1]), kt, mode='mae').mean()
        loss_symmetric += H.unreduced_loss(torch.flip(k, dims=[-2]), kt, mode='mae').mean()
        # log loss
        framework.log('loss_symmetric', loss_symmetric)
        # final loss
        return loss_symmetric


# ========================================================================= #
# Run Hydra                                                                 #
# ========================================================================= #


ROOT_DIR = os.path.abspath(__file__ + '/../../../..')


@hydra.main(config_path=os.path.join(ROOT_DIR, 'experiment/config'), config_name="config_05_adversarial_03_gen")
def run_hydra(cfg):
    cfg = make_non_strict(cfg)
    # ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~ #
    # check CUDA setting
    cfg.trainer.setdefault('cuda', 'try_cuda')
    hydra_check_cuda(cfg)
    # CREATE LOGGER
    logger = hydra_make_logger(cfg)
    # TRAINER CALLBACKS
    callbacks = []
    hydra_append_progress_callback(callbacks, cfg)
    # print everything
    log.info('Final Config' + make_box_str(OmegaConf.to_yaml(cfg)))
    # ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~ #
    seed(cfg.exp.seed)
    assert cfg.dataset.spacing in {1, 2, 4, 8}
    # ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~ #
    dataset = H.make_dataset(f'xysquares_{cfg.dataset.spacing}x{cfg.dataset.spacing}', factors=True)
    dataloader = DataLoader(
        dataset,
        batch_sampler=H.StochasticBatchSampler(dataset, batch_size=128),
        num_workers=psutil.cpu_count(),
        pin_memory=True
    )
    model = Kernel(radius=cfg.kernel.radius, channels=cfg.kernel.channels, offset=0.002, scale=0.01)
    callbacks.append(model.make_train_periodic_callback(cfg))
    framework = DisentangleModule(model, cfg)
    # ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~ #
    if framework.logger:
        framework.logger.log_hyperparams(framework.hparams)
    # train
    trainer = pl.Trainer(
        log_every_n_steps=cfg.logging.setdefault('log_every_n_steps', 50),
        flush_logs_every_n_steps=cfg.logging.setdefault('flush_logs_every_n_steps', 100),
        logger=logger,
        callbacks=callbacks,
        gpus=1 if cfg.trainer.cuda else 0,
        max_epochs=cfg.trainer.setdefault('epochs', None),
        max_steps=cfg.trainer.setdefault('steps', 10000),
        progress_bar_refresh_rate=0,  # ptl 0.9
        terminate_on_nan=True,  # we do this here so we don't run the final metrics
        checkpoint_callback=False,
    )
    trainer.fit(framework, dataloader)
    # ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~ #
    # save kernel
    if cfg.exp.rel_save_dir is not None:
        assert not os.path.isabs(cfg.exp.rel_save_dir), f'rel_save_dir must be relative: {repr(cfg.exp.rel_save_dir)}'
        save_dir = os.path.join(ROOT_DIR, cfg.exp.rel_save_dir)
        assert os.path.isabs(save_dir), f'save_dir must be absolute: {repr(save_dir)}'
        # save kernel
        torch_write(os.path.join(save_dir, cfg.exp.save_name), framework.model._kernel)


# ========================================================================= #
# Entry Point                                                               #
# ========================================================================= #


if __name__ == '__main__':
    # NORMAL:
    # run()

    # HYDRA:
    # run experiment (12min * 4*8*2) / 60 ~= 12 hours
    # but speeds up as kernel size decreases, so might be shorter
    # EXP ARGS:
    # $ ... -m +weight_decay=1e-4,0.0 +radius=63,55,47,39,31,23,15,7 +spacing=8,4,2,1
    run_hydra()
