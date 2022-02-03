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

import logging
import os
from typing import List
from typing import Optional
from typing import Sequence

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
import wandb
from omegaconf import OmegaConf
from torch.nn import Parameter
from torch.utils.data import DataLoader

import disent.util.seeds
import research.code.util as H
from disent.nn.functional import torch_conv2d_channel_wise_fft
from disent.nn.loss.softsort import spearman_rank_loss
from disent.nn.modules import DisentLightningModule
from disent.nn.modules import DisentModule
from disent.util.lightning.callbacks import BaseCallbackPeriodic
from disent.util.lightning.logger_util import wb_log_metrics
from disent.util.seeds import seed
from disent.util.strings.fmt import make_box_str
from experiment.run import hydra_append_progress_callback
from experiment.run import hydra_get_gpus
from experiment.run import hydra_make_logger
from experiment.util.hydra_utils import make_non_strict


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
        f_dists = torch.abs(factors[ia][:, f_idxs] - factors[ib][:, f_idxs]).sum(dim=-1)
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
        disentangle_factor_idxs: Sequence[int] = None
    ):
        super().__init__()
        self.model = model
        self.hparams = hparams
        self._disentangle_factors = None if (disentangle_factor_idxs is None) else np.array(disentangle_factor_idxs)

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
            f_idxs=self._disentangle_factors,
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
    def __init__(self, radius: int = 33, channels: int = 1, offset: float = 0.0, scale: float = 0.001, train_symmetric_regularise: bool = True, train_norm_regularise: bool = True, train_nonneg_regularise: bool = True):
        super().__init__()
        assert channels in (1, 3)
        kernel = torch.randn(1, channels, 2*radius+1, 2*radius+1, dtype=torch.float32)
        kernel = offset + kernel * scale
        # normalise
        if train_nonneg_regularise:
            kernel = torch.abs(kernel)
        if train_norm_regularise:
            kernel = kernel / kernel.sum(dim=[-1, -2], keepdim=True)
        # store
        self._kernel = Parameter(kernel)
        # regularise options
        self._train_symmetric_regularise = train_symmetric_regularise
        self._train_norm_regularise = train_norm_regularise
        self._train_nonneg_regularise = train_nonneg_regularise

    def forward(self, xs):
        return torch_conv2d_channel_wise_fft(xs, self._kernel)

    def make_train_periodic_callback(self, cfg, dataset) -> BaseCallbackPeriodic:
        class ImShowCallback(BaseCallbackPeriodic):
            def do_step(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
                # get kernel image
                kernel = H.to_img(pl_module.model._kernel[0], scale=True).numpy()
                # augment function
                def augment_fn(batch):
                    return H.to_imgs(pl_module.forward(batch.to(pl_module.device)), scale=True)
                # get augmented traversals
                with torch.no_grad():
                    orig_wandb_image, orig_wandb_animation = H.visualize_dataset_traversal(dataset)
                    augm_wandb_image, augm_wandb_animation = H.visualize_dataset_traversal(dataset, augment_fn=augment_fn, data_mode='input')
                # log images to WANDB
                wb_log_metrics(trainer.logger, {
                    'kernel': wandb.Image(kernel),
                    'traversal_img_orig': orig_wandb_image, 'traversal_animation_orig': orig_wandb_animation,
                    'traversal_img_augm': augm_wandb_image, 'traversal_animation_augm': augm_wandb_animation,
                })
        return ImShowCallback(every_n_steps=cfg.exp.show_every_n_steps, begin_first_step=True)

    def augment_loss(self, framework: DisentLightningModule):
        augment_loss = 0
        # symmetric loss
        if self._train_symmetric_regularise:
            k, kt = self._kernel[0], torch.transpose(self._kernel[0], -1, -2)
            loss_symmetric = 0
            loss_symmetric += H.unreduced_loss(torch.flip(k, dims=[-1]), k,  mode='mae').mean()
            loss_symmetric += H.unreduced_loss(torch.flip(k, dims=[-2]), k,  mode='mae').mean()
            loss_symmetric += H.unreduced_loss(torch.flip(k, dims=[-1]), kt, mode='mae').mean()
            loss_symmetric += H.unreduced_loss(torch.flip(k, dims=[-2]), kt, mode='mae').mean()
            # log loss
            framework.log('loss_symmetric', loss_symmetric)
            # final loss
            augment_loss += loss_symmetric
        # sum of 1 loss, per channel
        if self._train_norm_regularise:
            k = self._kernel[0]
            # sum over W & H resulting in: (C, W, H) -> (C,)
            channel_sums = k.sum(dim=[-1, -2])
            channel_loss = H.unreduced_loss(channel_sums, torch.ones_like(channel_sums), mode='mae')
            norm_loss = channel_loss.mean()
            # log loss
            framework.log('loss_norm', norm_loss)
            # final loss
            augment_loss += norm_loss
        # no negatives regulariser
        if self._train_nonneg_regularise:
            k = self._kernel[0]
            nonneg_loss = torch.abs(k[k < 0].sum())
            # log loss
            framework.log('loss_non_negative', nonneg_loss)
            # regularise negatives
            augment_loss += nonneg_loss
        # return!
        return augment_loss


# ========================================================================= #
# Run Hydra                                                                 #
# ========================================================================= #


ROOT_DIR = os.path.abspath(__file__ + '/../../../..')


@hydra.main(config_path=os.path.join(ROOT_DIR, 'experiment/config'), config_name="config_adversarial_kernel")
def run_disentangle_dataset_kernel(cfg):
    cfg = make_non_strict(cfg)
    # ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~ #
    # TODO: some of this code is duplicated between this and the main experiment run.py
    # check CUDA setting
    cfg.trainer.setdefault('cuda', 'try_cuda')
    gpus = hydra_get_gpus(cfg)
    # CREATE LOGGER
    logger = hydra_make_logger(cfg)
    # TRAINER CALLBACKS
    callbacks = []
    hydra_append_progress_callback(callbacks, cfg)
    # ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~ #
    seed(disent.util.seeds.seed)
    # ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~ #
    # initialise dataset and get factor names to disentangle
    dataset = H.make_dataset(cfg.data.name, factors=True, data_root=cfg.default_settings.storage.data_root)
    disentangle_factor_idxs = dataset.gt_data.normalise_factor_idxs(cfg.kernel.disentangle_factors)
    cfg.kernel.disentangle_factors = tuple(dataset.gt_data.factor_names[i] for i in disentangle_factor_idxs)
    log.info(f'Dataset has ground-truth factors: {dataset.gt_data.factor_names}')
    log.info(f'Chosen ground-truth factors are: {tuple(cfg.kernel.disentangle_factors)}')
    # ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~ #
    # print everything
    log.info('Final Config' + make_box_str(OmegaConf.to_yaml(cfg)))
    # ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~ #
    dataloader = DataLoader(
        dataset,
        batch_sampler=H.StochasticBatchSampler(dataset, batch_size=cfg.dataset.batch_size),
        num_workers=cfg.dataset.num_workers,
        pin_memory=cfg.dataset.pin_memory,
    )
    model = Kernel(radius=cfg.kernel.radius, channels=cfg.kernel.channels, offset=0.002, scale=0.01, train_symmetric_regularise=cfg.kernel.regularize_symmetric, train_norm_regularise=cfg.kernel.regularize_norm, train_nonneg_regularise=cfg.kernel.regularize_nonneg)
    callbacks.append(model.make_train_periodic_callback(cfg, dataset=dataset))
    framework = DisentangleModule(model, cfg, disentangle_factor_idxs=disentangle_factor_idxs)
    # ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~ #
    if framework.logger:
        framework.logger.log_hyperparams(framework.hparams)
    # train
    trainer = pl.Trainer(
        log_every_n_steps=cfg.log.setdefault('log_every_n_steps', 50),
        flush_logs_every_n_steps=cfg.log.setdefault('flush_logs_every_n_steps', 100),
        logger=logger,
        callbacks=callbacks,
        gpus=1 if gpus else 0,
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
        H.torch_write(os.path.join(save_dir, cfg.exp.save_name), framework.model._kernel)


# ========================================================================= #
# Entry Point                                                               #
# ========================================================================= #


if __name__ == '__main__':
    # HYDRA:
    # run experiment (12min * 4*8*2) / 60 ~= 12 hours
    # but speeds up as kernel size decreases, so might be shorter
    # EXP ARGS:
    # $ ... -m optimizer.weight_decay=1e-4,0.0 kernel.radius=63,55,47,39,31,23,15,7 dataset.spacing=8,4,2,1
    run_disentangle_dataset_kernel()
