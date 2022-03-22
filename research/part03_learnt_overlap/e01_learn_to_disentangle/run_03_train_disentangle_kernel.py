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

import numpy as np
import pytorch_lightning as pl
import torch
import wandb
from omegaconf import OmegaConf
from pytorch_lightning.loggers import WandbLogger
from torch.nn import Parameter
from torch.utils.data import DataLoader

import research.code.util as H
from disent.nn.functional import torch_conv2d_channel_wise_fft
from disent.nn.loss.softsort import spearman_rank_loss
from disent.nn.modules import DisentLightningModule
from disent.nn.modules import DisentModule
from disent.util.lightning.callbacks import BaseCallbackPeriodic
from disent.util.lightning.logger_util import wb_log_metrics
from disent.util.seeds import seed
from disent.util.strings.fmt import make_box_str
from experiment.run import hydra_get_callbacks
from experiment.run import hydra_get_gpus
from experiment.run import hydra_make_logger
from experiment.util.hydra_main import hydra_main
from experiment.util.hydra_utils import make_non_strict


log = logging.getLogger(__name__)


# ========================================================================= #
# EXP                                                                       #
# ========================================================================= #


def disentangle_loss(
    batch: torch.Tensor,
    aug_batch: Optional[torch.Tensor],
    factors: torch.Tensor,
    num_pairs: int,
    f_idxs: Optional[List[int]] = None,
    loss_fn: str = 'mse',
    mean_dtype=None,
    corr_mode: str = 'improve',
    regularization_strength: float = 1.0,
    factor_sizes: Optional[torch.Tensor] = None,  # scale the distances | Must be the same approach as `GroundTruthDistSampler`
) -> torch.Tensor:
    assert len(batch) == len(factors)
    assert batch.ndim == 4
    assert factors.ndim == 2
    # random pairs
    ia, ib = torch.randint(0, len(batch), size=(2, num_pairs), device=batch.device)
    # get pairwise distances
    b_dists = H.pairwise_loss(batch[ia], batch[ib], mode=loss_fn, mean_dtype=mean_dtype)  # avoid precision errors
    if aug_batch is not None:
        assert aug_batch.shape == batch.shape
        b_dists += H.pairwise_loss(aug_batch[ia], aug_batch[ib], mode=loss_fn, mean_dtype=mean_dtype)
    # compute factor differences
    if f_idxs is not None:
        f_diffs = factors[ia][:, f_idxs] - factors[ib][:, f_idxs]
    else:
        f_diffs = factors[ia] - factors[ib]
    # scale the factor distances
    if factor_sizes is not None:
        assert factor_sizes.ndim == 1
        assert factor_sizes.shape == factors.shape[1:]
        scale = torch.maximum(torch.ones_like(factor_sizes), factor_sizes - 1)
        f_diffs = f_diffs / scale.detach()
    # compute factor distances
    f_dists = torch.abs(f_diffs).sum(dim=-1)
    # optimise metric
    if corr_mode == 'improve':  loss = spearman_rank_loss(b_dists, -f_dists, regularization_strength=regularization_strength)  # default one to use!
    elif corr_mode == 'invert': loss = spearman_rank_loss(b_dists, +f_dists, regularization_strength=regularization_strength)
    elif corr_mode == 'none':   loss = +torch.abs(spearman_rank_loss(b_dists, -f_dists, regularization_strength=regularization_strength))
    elif corr_mode == 'any':    loss = -torch.abs(spearman_rank_loss(b_dists, -f_dists, regularization_strength=regularization_strength))
    else: raise KeyError(f'invalid correlation mode: {repr(corr_mode)}')
    # done!
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
        self.hparams.update(hparams)
        self._disentangle_factors = None if (disentangle_factor_idxs is None) else np.array(disentangle_factor_idxs)

    def configure_optimizers(self):
        return H.make_optimizer(self, name=self.hparams.exp.optimizer.name, lr=self.hparams.exp.optimizer.lr, weight_decay=self.hparams.exp.optimizer.weight_decay)

    def training_step(self, batch, batch_idx):
        (x,), (f,) = batch['x_targ'], batch['factors']
        # feed forward batch
        y = self.model(x)
        # compute pairwise distances of factors and batch, and optimize to correspond
        loss_rank = disentangle_loss(
            batch     = x if self.hparams.exp.train.combined_loss else y,
            aug_batch = y if self.hparams.exp.train.combined_loss else None,
            factors=f,
            num_pairs=int(len(x) * self.hparams.exp.train.pairs_ratio),
            f_idxs=self._disentangle_factors,
            loss_fn=self.hparams.exp.train.loss,
            mean_dtype=torch.float64,
            regularization_strength=self.hparams.exp.train.reg_strength,
            factor_sizes=None,
        )
        # ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~ #
        if hasattr(self.model, 'augment_loss'):
            loss_aug = self.model.augment_loss(self)
        else:
            loss_aug = 0
        loss = loss_rank + loss_aug
        # ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~ #
        self.log('loss_rank', float(loss_rank), prog_bar=True)
        self.log('loss',      float(loss),      prog_bar=True)
        # ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~ #
        return loss

    def forward(self, x):
        return self.model(x)


# ========================================================================= #
# MAIN                                                                      #
# ========================================================================= #


_REPR_FN_INIT = {
    'none':   lambda x: x,
    'square': lambda x: torch.sqrt(torch.abs(x)),
    'abs':    lambda x: torch.abs(x),
    'exp':    lambda x: torch.log(torch.abs(x)),
}

_REPR_FN = {
    'none':   lambda x: x,
    'square': lambda x: torch.square(x),
    'abs':    lambda x: torch.abs(x),
    'exp':    lambda x: torch.exp(x),
}



class Kernel(DisentModule):

    def __init__(
        self,
        radius: int = 33,
        channels: int = 1,
        # loss settings
        train_symmetric_regularise: bool = True,
        train_norm_regularise: bool = True,
        train_nonneg_regularise: bool = True,
        train_regularize_l2_weight: Optional[float] = None,
        # kernel settings
        represent_mode: str = 'abs',
        init_offset: float = 0.0,
        init_scale: float = 0.001,
        init_sums_to_one: bool = True,
    ):
        super().__init__()
        assert channels in (1, 3)
        assert set(_REPR_FN_INIT.keys()) == set(_REPR_FN.keys())
        assert represent_mode in _REPR_FN, f'invalid represent_mode: {repr(represent_mode)}'
        # initialize
        with torch.no_grad():
            # randomly sample value
            if represent_mode == 'none':
                kernel = torch.rand(1, channels, 2*radius+1, 2*radius+1, dtype=torch.float32)
            else:
                kernel = torch.randn(1, channels, 2*radius+1, 2*radius+1, dtype=torch.float32)
            # scale values
            kernel = init_offset + kernel * init_scale
            if init_sums_to_one:
                kernel = kernel / kernel.sum(dim=[-1, -2], keepdim=True)
            # log params
            kernel = _REPR_FN_INIT[represent_mode](kernel)
            assert not torch.any(torch.isnan(kernel))
        # store
        self.__kernel = Parameter(kernel)
        self._represent_mode = represent_mode
        # regularise options
        self._train_symmetric_regularise = train_symmetric_regularise
        self._train_norm_regularise = train_norm_regularise
        self._train_nonneg_regularise = train_nonneg_regularise
        self._train_regularize_l2_weight = train_regularize_l2_weight

    @property
    def kernel(self) -> torch.Tensor:
        return _REPR_FN[self._represent_mode](self.__kernel)

    def forward(self, xs):
        return torch_conv2d_channel_wise_fft(xs, self.kernel)

    def make_train_periodic_callback(self, cfg, dataset) -> BaseCallbackPeriodic:
        class ImShowCallback(BaseCallbackPeriodic):
            def do_step(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
                # get kernel image
                img_kernel = H.to_img(pl_module.model.kernel[0], scale=True).numpy()
                img_kernel_log = H.to_img(torch.log(pl_module.model.kernel[0]), scale=True).numpy()
                # augment function
                def augment_fn(batch):
                    return H.to_imgs(pl_module.forward(batch.to(pl_module.device)), scale=True)
                # get augmented traversals
                with torch.no_grad():
                    orig_wandb_image, orig_wandb_animation = H.visualize_dataset_traversal(dataset, augment_fn=None,       data_mode='raw',   output_wandb=True)  # dataset returns (numpy?) HWC batches
                    augm_wandb_image, augm_wandb_animation = H.visualize_dataset_traversal(dataset, augment_fn=augment_fn, data_mode='input', output_wandb=True)  # dataset returns (tensor) CHW batches
                # log images to WANDB
                wb_log_metrics(trainer.logger, {
                    'kernel': wandb.Image(img_kernel),
                    'kernel_ln': wandb.Image(img_kernel_log),
                    'traversal_img_orig': orig_wandb_image, 'traversal_animation_orig': orig_wandb_animation,
                    'traversal_img_augm': augm_wandb_image, 'traversal_animation_augm': augm_wandb_animation,
                })
        return ImShowCallback(every_n_steps=cfg.exp.out.show_every_n_steps, begin_first_step=True)

    def augment_loss(self, framework: DisentLightningModule):
        augment_loss = 0
        # symmetric loss
        if self._train_symmetric_regularise:
            k, kt = self.kernel[0], torch.transpose(self.kernel[0], -1, -2)
            loss_symmetric = 0
            loss_symmetric += H.unreduced_loss(torch.flip(k, dims=[-1]), k,  mode='mae').mean()
            loss_symmetric += H.unreduced_loss(torch.flip(k, dims=[-2]), k,  mode='mae').mean()
            loss_symmetric += H.unreduced_loss(torch.flip(k, dims=[-1]), kt, mode='mae').mean()
            loss_symmetric += H.unreduced_loss(torch.flip(k, dims=[-2]), kt, mode='mae').mean()
            # log loss
            framework.log('loss_sym', float(loss_symmetric), prog_bar=True)
            # final loss
            augment_loss += loss_symmetric
        # regularize, try make kernel as small as possible
        if (self._train_regularize_l2_weight is not None) and (self._train_regularize_l2_weight > 0):
            k = self.kernel[0]
            loss_l2 = self._train_regularize_l2_weight * (k ** 2).mean()
            framework.log('loss_l2', float(loss_l2), prog_bar=True)
            augment_loss += loss_l2
        # sum of 1 loss, per channel
        if self._train_norm_regularise:
            k = self.kernel[0]
            # sum over W & H resulting in: (C, W, H) -> (C,)
            channel_sums = k.sum(dim=[-1, -2])
            channel_loss = H.unreduced_loss(channel_sums, torch.ones_like(channel_sums), mode='mse')
            norm_loss = channel_loss.mean()
            # log loss
            framework.log('loss_norm', float(norm_loss), prog_bar=True)
            # final loss
            augment_loss += norm_loss
        # no negatives regulariser
        if self._train_nonneg_regularise:
            k = self.kernel[0]
            nonneg_loss = torch.abs(k[k < 0].sum())
            # log loss
            framework.log('loss_nonneg', float(nonneg_loss), prog_bar=True)
            # regularise negatives
            augment_loss += nonneg_loss
        # stats
        framework.log('kernel_mean', float(self.kernel.mean()), prog_bar=False)
        framework.log('kernel_std', float(self.kernel.std()), prog_bar=False)
        # return!
        return augment_loss


# ========================================================================= #
# Run Hydra                                                                 #
# ========================================================================= #


def run_disentangle_dataset_kernel(cfg):
    cfg = make_non_strict(cfg)
    # ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~ #
    # TODO: some of this code is duplicated between this and the main experiment run.py
    # check CUDA setting
    cfg.trainer.setdefault('cuda', 'try_cuda')
    gpus = hydra_get_gpus(cfg)
    # CREATE LOGGER
    logger = hydra_make_logger(cfg)
    if isinstance(logger.experiment, WandbLogger):
        _ = logger.experiment  # initialize
    # TRAINER CALLBACKS
    callbacks = hydra_get_callbacks(cfg)
    # ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~ #
    seed(cfg.settings.job.seed)
    # ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~ #
    # initialise dataset and get factor names to disentangle
    dataset = H.make_dataset(cfg.exp.data.name, factors=True, data_root=cfg.dsettings.storage.data_root)
    disentangle_factor_idxs = dataset.gt_data.normalise_factor_idxs(cfg.exp.kernel.disentangle_factors)
    cfg.exp.kernel.disentangle_factors = tuple(dataset.gt_data.factor_names[i] for i in disentangle_factor_idxs)
    log.info(f'Dataset has ground-truth factors: {dataset.gt_data.factor_names}')
    log.info(f'Chosen ground-truth factors are: {tuple(cfg.exp.kernel.disentangle_factors)}')
    # ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~ #
    # print everything
    log.info('Final Config' + make_box_str(OmegaConf.to_yaml(cfg)))
    # ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~ #
    dataloader = DataLoader(
        dataset,
        batch_sampler=H.StochasticBatchSampler(dataset, batch_size=cfg.datamodule.dataloader.batch_size),
        num_workers=cfg.datamodule.dataloader.num_workers,
        pin_memory=cfg.datamodule.dataloader.pin_memory,
    )
    model = Kernel(radius=cfg.exp.kernel.radius, channels=cfg.exp.kernel.channels, init_offset=cfg.exp.kernel.init_offset, init_scale=cfg.exp.kernel.init_scale, train_symmetric_regularise=cfg.exp.kernel.regularize_symmetric, train_norm_regularise=cfg.exp.kernel.regularize_norm, train_nonneg_regularise=cfg.exp.kernel.regularize_nonneg, represent_mode=cfg.exp.kernel.represent_mode, init_sums_to_one=cfg.exp.kernel.init_sums_to_one, train_regularize_l2_weight=cfg.exp.kernel.regularize_l2_weight)
    callbacks.append(model.make_train_periodic_callback(cfg, dataset=dataset))
    framework = DisentangleModule(model, cfg, disentangle_factor_idxs=disentangle_factor_idxs)
    # ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~ #
    if logger:
        logger.log_hyperparams(cfg)
    # train
    trainer = pl.Trainer(
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        logger=logger,
        callbacks=callbacks,
        gpus=1 if gpus else 0,
        max_epochs=cfg.trainer.max_epochs,
        max_steps=cfg.trainer.max_steps,
        enable_progress_bar=False,
        # we do this here so we don't run the final metrics
        detect_anomaly=False,  # this should only be enabled for debugging torch and finding NaN values, slows down execution, not by much though?
        enable_checkpointing=False,
    )
    trainer.fit(framework, dataloader)
    # ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~ #
    # save kernel
    if cfg.exp.out.rel_save_dir is not None:
        assert not os.path.isabs(cfg.exp.out.rel_save_dir), f'rel_save_dir must be relative: {repr(cfg.exp.out.rel_save_dir)}'
        save_dir = os.path.join(ROOT_DIR, cfg.exp.out.rel_save_dir)
        assert os.path.isabs(save_dir), f'save_dir must be absolute: {repr(save_dir)}'
        # save kernel
        H.torch_write(os.path.join(save_dir, cfg.exp.out.save_name), framework.model.kernel.cpu().detach())


# ========================================================================= #
# Entry Point                                                               #
# ========================================================================= #


if __name__ == '__main__':
    # HYDRA:
    # run experiment (12min * 4*8*2) / 60 ~= 12 hours
    # but speeds up as kernel size decreases, so might be shorter
    # EXP ARGS:
    # $ ... -m optimizer.weight_decay=1e-4,0.0 kernel.radius=63,55,47,39,31,23,15,7 dataset.spacing=8,4,2,1

    ROOT_DIR = os.path.abspath(__file__ + '/../../../..')
    CONFIGS_THIS_EXP = os.path.abspath(os.path.join(__file__, '..', 'config'))
    CONFIGS_RESEARCH = os.path.abspath(os.path.join(__file__, '../../..', 'config'))

    # launch the action
    hydra_main(
        callback=run_disentangle_dataset_kernel,
        config_name='config_adversarial_kernel',
        search_dirs_prepend=[CONFIGS_THIS_EXP, CONFIGS_RESEARCH],
        log_level=logging.INFO,
    )
