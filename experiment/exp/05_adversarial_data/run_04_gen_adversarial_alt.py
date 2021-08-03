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

import itertools
import logging
import os
import time
import warnings
from argparse import Namespace
from typing import Iterator
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple

import dataset as dataset
import hydra
import numpy as np
import pytorch_lightning as pl
import torch
import wandb
from omegaconf import DictConfig
from omegaconf import OmegaConf
from pytorch_lightning.loggers import WandbLogger
from torch.nn import Parameter
from torch.utils.data import DataLoader
from torch.utils.data import IterableDataset
from torch.utils.data.dataset import T_co

import disent.util.seeds
import experiment.exp.util as H
from disent.dataset import DisentDataset
from disent.dataset.data import ArrayGroundTruthData
from disent.dataset.data import GroundTruthData
from disent.dataset.sampling import BaseDisentSampler
from disent.dataset.sampling import GroundTruthPairSampler
from disent.frameworks import DisentConfigurable
from disent.nn.modules import DisentLightningModule
from disent.nn.modules import DisentModule
from disent.util.strings.fmt import make_box_str
from disent.util.seeds import seed
from disent.nn.functional import torch_conv2d_channel_wise_fft
from disent.nn.loss.softsort import spearman_rank_loss
from experiment.run import hydra_append_progress_callback
from experiment.run import hydra_check_cuda
from experiment.run import hydra_make_logger
from disent.util.lightning.callbacks import BaseCallbackPeriodic
from experiment.util.hydra_utils import make_non_strict
from disent.util.lightning.logger_util import wb_log_metrics
from disent.util.math.random import randint2, sample_radius


log = logging.getLogger(__name__)


# ========================================================================= #
# Samplers                                                                  #
# ========================================================================= #


class AdversarialSampler_CloseFar(BaseDisentSampler):

    def __init__(
        self,
        close_p_k_range=(1, 1),
        close_p_radius_range=(1, 1),
        far_p_k_range=(1, -1),
        far_p_radius_range=(1, -1),
    ):
        super().__init__(3)
        self.sampler_close = GroundTruthPairSampler(p_k_range=close_p_k_range, p_radius_range=close_p_radius_range)
        self.sampler_far = GroundTruthPairSampler(p_k_range=far_p_k_range, p_radius_range=far_p_radius_range)

    def _init(self, gt_data: GroundTruthData):
        self.sampler_close.init(gt_data)
        self.sampler_far.init(gt_data)

    def _sample_idx(self, idx: int) -> Tuple[int, ...]:
        # sample indices
        anchor, pos = self.sampler_close(idx)
        _anchor, neg = self.sampler_far(idx)
        assert anchor == _anchor
        # return triple
        return anchor, pos, neg


# ========================================================================= #
# Adversarial Loss                                                          #
# ========================================================================= #


def _adversarial_const_loss(a_x: torch.Tensor, p_x: torch.Tensor, n_x: torch.Tensor, loss: str = 'mse', target: float = 0.1):
    # compute deltas
    p_deltas = H.pairwise_loss(a_x, p_x, mode=loss, mean_dtype=torch.float32)
    n_deltas = H.pairwise_loss(a_x, n_x, mode=loss, mean_dtype=torch.float32)
    # compute loss
    p_loss = torch.abs(target - p_deltas).mean()  # should this be l2 dist instead?
    n_loss = torch.abs(target - n_deltas).mean()  # should this be l2 dist instead?
    loss = p_loss + n_loss
    # done!
    return loss


def _adversarial_self_loss(a_x: torch.Tensor, p_x: torch.Tensor, n_x: torch.Tensor, loss: str = 'mse', target: None = None):
    if target is not None:
        warnings.warn(f'adversarial_self_loss does not support a value for target, this is kept for compatibility reasons!')
    # compute deltas
    p_deltas = H.pairwise_loss(a_x, p_x, mode=loss, mean_dtype=torch.float64)
    n_deltas = H.pairwise_loss(a_x, n_x, mode=loss, mean_dtype=torch.float64)
    # compute loss
    loss = torch.abs(n_deltas - p_deltas).mean()  # should this be l2 dist instead?
    # done!
    return loss


def _adversarial_inverse_loss(a_x: torch.Tensor, p_x: torch.Tensor, n_x: torch.Tensor, loss: str = 'mse', target: None = None):
    if target is not None:
        warnings.warn(f'adversarial_inverse_loss does not support a value for target, this is kept for compatibility reasons!')
    # compute deltas
    p_deltas = H.pairwise_loss(a_x, p_x, mode=loss, mean_dtype=torch.float32)
    n_deltas = H.pairwise_loss(a_x, n_x, mode=loss, mean_dtype=torch.float32)
    # compute loss (unbounded)
    loss = (n_deltas - p_deltas).mean()
    # done!
    return loss


_ADVERSARIAL_LOSS_FNS = {
    'const': _adversarial_const_loss,
    'self': _adversarial_self_loss,
    'inverse': _adversarial_inverse_loss,
}


def adversarial_loss(a_x: torch.Tensor, p_x: torch.Tensor, n_x: torch.Tensor, loss: str = 'mse', target: float = 0.1, adversarial_mode: str = 'self'):
    try:
        loss_fn = _ADVERSARIAL_LOSS_FNS[adversarial_mode]
    except KeyError:
        raise KeyError(f'invalid adversarial_mode={repr(adversarial_mode)}')
    return loss_fn(a_x=a_x, p_x=p_x, n_x=n_x, loss=loss, target=target)


# ========================================================================= #
# adversarial dataset generator                                             #
# ========================================================================= #


def make_adversarial_class(batch_optimizer: bool, gpu: bool, fp16: bool = True):
    # check values
    if fp16 and (not gpu):
        warnings.warn('`fp16=True` is not supported on CPU, overriding setting to `False`')
        fp16 = False

    # get dtypes
    SRC_DTYPE = torch.float16 if fp16 else torch.float32
    DST_DTYPE = torch.float32

    class AdversarialModel(pl.LightningModule):

        def __init__(
            self,
            # optimizer options
                optimizer_name: str = 'adam',
                optimizer_lr: float = 1e-2,
                optimizer_kwargs: Optional[dict] = None,
            # dataset config options
                dataset_name: str = 'cars3d',
                dataset_num_workers: int = 0,
                dataset_batch_size: int = 256,  # approx
                # batch_sample_mode: str = 'shuffle',  # range, shuffle, random
            # initialize params from dataset
                # params_masked: bool = True,
                # params_initial_noise: Optional[float] = None,
            # loss config options
                loss_fn: str = 'mse',
                loss_mode: str = 'self',
                loss_const_targ: Optional[float] = 0.1, # replace stochastic pairwise constant loss with deterministic loss target
                # loss_num_pairs: int = 1024 * 4,
                # loss_num_samples: int = 1024 * 4 * 2,  # only applies if loss_const_targ=None
                # loss_reg_out_of_bounds: bool = False,
                # loss_top_k: Optional[int] = None,
            # sampling config
                sampler_name: str = 'close_far',
                sampler_kwargs: Optional[dict] = None,
        ):
            super().__init__()
            # modify hparams
            if optimizer_kwargs is None: optimizer_kwargs = {}
            if sampler_kwargs is None: sampler_kwargs = {}
            # save hparams
            self.save_hyperparameters()
            # variables
            self.dataset: DisentDataset = None
            self.array: torch.Tensor = None
            self.sampler: BaseDisentSampler = None

        # ================================== #
        # setup                              #
        # ================================== #

        def prepare_data(self) -> None:
            # create dataset
            self.dataset = H.make_dataset(self.hparams.dataset_name, load_into_memory=True, load_memory_dtype=SRC_DTYPE)
            # load dataset into memory as fp16
            if batch_optimizer:
                self.array = self.dataset.gt_data.array
            else:
                self.array = torch.nn.Parameter(self.dataset.gt_data.array, requires_grad=True)  # move with model to correct device
            # create sampler
            assert self.hparams.sampler_name == 'close_far', '`close_far` is the only mode currently supported!'
            self.sampler = AdversarialSampler_CloseFar(**self.hparams.sampler_kwargs).init(self.dataset.gt_data)

        def _make_optimizer(self, params):
            return H.make_optimizer(
                params,
                name=self.hparams.optimizer_name,
                lr=self.hparams.optimizer_lr,
                **self.hparams.optimizer_kwargs,
            )

        def configure_optimizers(self):
            if batch_optimizer:
                return None
            else:
                return self._make_optimizer(self.array)

        # ================================== #
        # train step                         #
        # ================================== #

        def training_step(self, batch, batch_idx):
            # get indices
            (a_idx, p_idx, n_idx) = batch['idx']
            # generate batches & transfer to correct device
            if batch_optimizer:
                (a_x, p_x, n_x), (params, param_idxs, optimizer) = self._load_batch(a_idx, p_idx, n_idx)
            else:
                a_x = self.array[a_idx]
                p_x = self.array[p_idx]
                n_x = self.array[n_idx]
            # compute loss
            loss = adversarial_loss(
                a_x=a_x,
                p_x=p_x,
                n_x=n_x,
                loss=self.hparams.loss_fn,
                target=self.hparams.loss_const_targ,
                adversarial_mode=self.hparams.loss_mode,
            )
            # log results
            self.log_dict({
                'loss': loss.float(),
                'adv_loss': loss.float(),
            }, prog_bar=True)
            # done!
            if batch_optimizer:
                self._update_with_batch(loss, params, param_idxs, optimizer)
                return None
            else:
                return loss

        # ================================== #
        # dataset                            #
        # ================================== #

        def train_dataloader(self):
            # sampling in dataloader
            sampler = self.sampler
            data_len = len(self.dataset.gt_data)
            # generate the indices in a multi-threaded environment -- this is not deterministic if num_workers > 0
            class SamplerIndicesDataset(IterableDataset):
                def __getitem__(self, index) -> T_co:
                    raise RuntimeError('this should never be called on an iterable dataset')
                def __iter__(self) -> Iterator[T_co]:
                    while True:
                        yield {'idx': sampler(np.random.randint(0, data_len))}
            # create data loader!
            return DataLoader(
                SamplerIndicesDataset(),
                batch_size=self.hparams.dataset_batch_size,
                num_workers=self.hparams.dataset_num_workers,
                shuffle=False,
            )

        # ================================== #
        # optimizer for each batch mode      #
        # ================================== #

        def _load_batch(self, a_idx, p_idx, n_idx):
            with torch.no_grad():
                # get all indices
                all_indices = np.stack([
                    a_idx.detach().cpu().numpy(),
                    p_idx.detach().cpu().numpy(),
                    n_idx.detach().cpu().numpy(),
                ], axis=0)
                # find unique values
                param_idxs, inverse_indices = np.unique(all_indices.flatten(), return_inverse=True)
                inverse_indices = inverse_indices.reshape(all_indices.shape)
                # load data with values & move to gpu
                # - for batch size (256*3, 3, 64, 64) with num_workers=0, this is 5% faster
                #   than .to(device=self.device, dtype=DST_DTYPE) in one call, as we reduce
                #   the memory overhead in the transfer. This does slightly increase the
                #   memory usage on the target device.
                # - for batch size (1024*3, 3, 64, 64) with num_workers=12, this is 15% faster
                #   but consumes slightly more memory: 2492MiB vs. 2510MiB
                params = self.array[param_idxs].to(device=self.device).to(dtype=DST_DTYPE)
            # make params and optimizer
            params = torch.nn.Parameter(params, requires_grad=True)
            optimizer = self._make_optimizer(params)
            # get batches -- it is ok to index by a numpy array without conversion
            a_x = params[inverse_indices[0, :]]
            p_x = params[inverse_indices[1, :]]
            n_x = params[inverse_indices[2, :]]
            # return values
            return (a_x, p_x, n_x), (params, param_idxs, optimizer)

        def _update_with_batch(self, loss, params, param_idxs, optimizer):
            # backprop
            H.step_optimizer(optimizer, loss)
            # save values to dataset
            with torch.no_grad():
                self.array[param_idxs] = params.detach().cpu().to(SRC_DTYPE)

    return AdversarialModel


# ========================================================================= #
# Run Hydra                                                                 #
# ========================================================================= #


# ROOT_DIR = os.path.abspath(__file__ + '/../../../..')
#
#
# @hydra.main(config_path=os.path.join(ROOT_DIR, 'experiment/config'), config_name="config_adversarial_dataset")
# def run_gen_adversarial_dataset(cfg):
#     # ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~ #
#     cfg = make_non_strict(cfg)
#     # - - - - - - - - - - - - - - - #
#     # check CUDA setting
#     cfg.trainer.setdefault('cuda', 'try_cuda')
#     hydra_check_cuda(cfg)
#     # create logger
#     logger = hydra_make_logger(cfg)
#     # create callbacks
#     callbacks = []
#     hydra_append_progress_callback(callbacks, cfg)
#     # - - - - - - - - - - - - - - - #
#     # get the logger and initialize
#     if logger is not None:
#         wandb_experiment = logger.experiment  # initialize
#         logger.log_hyperparams(cfg)
#     # print the final config!
#     log.info('Final Config' + make_box_str(OmegaConf.to_yaml(cfg)))
#     # ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~ #
#     # | | | | | | | | | | | | | | | #
#     seed(disent.util.seeds.seed)
#     # | | | | | | | | | | | | | | | #
#     # ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~ #
#
#
#
#
#     # train
#     trainer = pl.Trainer(
#         log_every_n_steps=cfg.logging.setdefault('log_every_n_steps', 50),
#         flush_logs_every_n_steps=cfg.logging.setdefault('flush_logs_every_n_steps', 100),
#         logger=logger,
#         callbacks=callbacks,
#         gpus=1 if cfg.trainer.cuda else 0,
#         max_epochs=cfg.trainer.setdefault('epochs', None),
#         max_steps=cfg.trainer.setdefault('steps', 10000),
#         progress_bar_refresh_rate=0,  # ptl 0.9
#         terminate_on_nan=True,  # we do this here so we don't run the final metrics
#         checkpoint_callback=False,
#     )
#     trainer.fit(framework, dataloader)
#     # ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~ #
#     # save kernel
#     if cfg.exp.rel_save_dir is not None:
#         assert not os.path.isabs(cfg.exp.rel_save_dir), f'rel_save_dir must be relative: {repr(cfg.exp.rel_save_dir)}'
#         save_dir = os.path.join(ROOT_DIR, cfg.exp.rel_save_dir)
#         assert os.path.isabs(save_dir), f'save_dir must be absolute: {repr(save_dir)}'
#         # save kernel
#         H.torch_write(os.path.join(save_dir, cfg.exp.save_name), framework.model._kernel)
#
#
# # ========================================================================= #
# # Entry Point                                                               #
# # ========================================================================= #
#
#
# if __name__ == '__main__':
#     # EXP ARGS:
#     # $ ... -m dataset=smallnorb,shapes3d
#     run_gen_adversarial_dataset()


if __name__ == '__main__':

    batch_optimizer, gpu, fp16 = True, True, True

    # BENCHMARK (batch_size=256, optimizer=sgd, lr=1e-2, dataset_num_workers=0):
    # - batch_optimizer=False, gpu=True,  fp16=True   : [3168MiB/5932MiB, 3.32/11.7G, 5.52it/s]
    # - batch_optimizer=False, gpu=True,  fp16=False  : [5248MiB/5932MiB, 3.72/11.7G, 4.84it/s]
    # - batch_optimizer=False, gpu=False, fp16=True   : [same as fp16=False]
    # - batch_optimizer=False, gpu=False, fp16=False  : [0003MiB/5932MiB, 4.60/11.7G, 1.05it/s]

    # - batch_optimizer=True,  gpu=True,  fp16=True   : [1284MiB/5932MiB, 3.45/11.7G, 4.31it/s]
    # - batch_optimizer=True,  gpu=True,  fp16=False  : [1284MiB/5932MiB, 3.72/11.7G, 4.31it/s]
    # - batch_optimizer=True,  gpu=False, fp16=True   : [same as fp16=False]
    # - batch_optimizer=True,  gpu=False, fp16=False  : [0003MiB/5932MiB, 1.80/11.7G, 4.18it/s]

    # BENCHMARK (batch_size=1024, optimizer=sgd, lr=1e-2, dataset_num_workers=12):
    # - batch_optimizer=True,  gpu=True,  fp16=True   : [2510MiB/5932MiB, 4.10/11.7G, 4.75it/s, 20% gpu util] (to(device).to(dtype))
    # - batch_optimizer=True,  gpu=True,  fp16=True   : [2492MiB/5932MiB, 4.10/11.7G, 4.12it/s, 19% gpu util] (to(device, dtype))

    AdversarialModel = make_adversarial_class(batch_optimizer=batch_optimizer, gpu=gpu, fp16=fp16)
    framework = AdversarialModel(optimizer_lr=1e-2, optimizer_name='sgd', dataset_batch_size=1024, dataset_num_workers=12)

    trainer = pl.Trainer(
        log_every_n_steps=50,
        flush_logs_every_n_steps=100,
        logger=False,
        callbacks=None,
        gpus=1 if gpu else 0,
        max_epochs=None,
        max_steps=10000,
        # progress_bar_refresh_rate=0,  # ptl 0.9
        terminate_on_nan=False,  # we do this here so we don't run the final metrics
        checkpoint_callback=False,
    )

    trainer.fit(framework)
