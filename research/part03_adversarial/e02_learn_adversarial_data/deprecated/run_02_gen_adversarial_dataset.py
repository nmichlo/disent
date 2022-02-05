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

"""
Generate an adversarial dataset
- images are directly optimized against each other, could decay in some cases?
- All data is stored in memory, with minibatches taken and optimized.
"""

import logging
import os
import warnings
from datetime import datetime
from typing import Iterator
from typing import List
from typing import Optional
from typing import Sequence

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
import wandb
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch.utils.data import IterableDataset
from torch.utils.data.dataset import T_co

import research.code.util as H
from disent.dataset import DisentDataset
from disent.dataset.sampling import BaseDisentSampler
from disent.dataset.util.hdf5 import H5Builder
from disent.util import to_numpy
from disent.util.deprecate import deprecated
from disent.util.inout.paths import ensure_parent_dir_exists
from disent.util.lightning.callbacks import BaseCallbackPeriodic
from disent.util.lightning.callbacks import LoggerProgressCallback
from disent.util.lightning.logger_util import wb_log_metrics
from disent.util.math.random import random_choice_prng
from disent.util.seeds import seed
from disent.util.seeds import TempNumpySeed
from disent.util.strings.fmt import bytes_to_human
from disent.util.strings.fmt import make_box_str
from disent.util.visualize.vis_util import make_image_grid
from experiment.run import hydra_get_callbacks
from experiment.run import hydra_get_gpus
from experiment.run import hydra_make_logger
from experiment.util.hydra_utils import make_non_strict
from experiment.util.run_utils import log_error_and_exit
from research.part03_adversarial.e02_learn_adversarial_data.util_gen_adversarial_dataset import adversarial_loss
from research.part03_adversarial.e02_learn_adversarial_data.util_gen_adversarial_dataset import make_adversarial_sampler


log = logging.getLogger(__name__)


# ========================================================================= #
# adversarial dataset generator                                             #
# ========================================================================= #


class AdversarialModel(pl.LightningModule):

    def __init__(
        self,
        # optimizer options
            optimizer_name: str = 'sgd',
            optimizer_lr: float = 5e-2,
            optimizer_kwargs: Optional[dict] = None,
        # dataset config options
            dataset_name: str = 'cars3d',
            dataset_num_workers: int = min(os.cpu_count(), 16),
            dataset_batch_size: int = 1024,  # approx
            data_root: str = 'data/dataset',
            # data_load_into_memory: bool = False,
        # adversarial loss options
            adversarial_mode: str = 'self',
            adversarial_swapped: bool = False,
            adversarial_masking: bool = False,
            adversarial_top_k: Optional[int] = None,
            pixel_loss_mode: str = 'mse',
        # loss extras
            # loss_adversarial_weight: Optional[float] = 1.0,
            # loss_same_stats_weight: Optional[float] = 0.0,
            # loss_similarity_weight: Optional[float] = 0.0,
            # loss_out_of_bounds_weight: Optional[float] = 0.0,
        # sampling config
            sampler_name: str = 'close_far',
        # train options
            train_batch_optimizer: bool = True,
            train_dataset_fp16: bool = True,
            train_is_gpu: bool = False,
        # logging settings
            # logging_scale_imgs: bool = False,
    ):
        super().__init__()
        # check values
        if train_dataset_fp16 and (not train_is_gpu):
            warnings.warn('`train_dataset_fp16=True` is not supported on CPU, overriding setting to `False`')
            train_dataset_fp16 = False
        self._dtype_dst = torch.float32
        self._dtype_src = torch.float16 if train_dataset_fp16 else torch.float32
        # modify hparams
        if optimizer_kwargs is None:
            optimizer_kwargs = {}
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
        self.dataset = H.make_dataset(self.hparams.dataset_name, load_into_memory=True, load_memory_dtype=self._dtype_src, data_root=self.hparams.data_root)
        # load dataset into memory as fp16
        if self.hparams.train_batch_optimizer:
            self.array = self.dataset.gt_data.array
        else:
            self.array = torch.nn.Parameter(self.dataset.gt_data.array, requires_grad=True)  # move with model to correct device
        # create sampler
        self.sampler = make_adversarial_sampler(self.hparams.sampler_name)
        self.sampler.init(self.dataset.gt_data)

    def _make_optimizer(self, params):
        return H.make_optimizer(
            params,
            name=self.hparams.optimizer_name,
            lr=self.hparams.optimizer_lr,
            **self.hparams.optimizer_kwargs,
        )

    def configure_optimizers(self):
        if self.hparams.train_batch_optimizer:
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
        if self.hparams.train_batch_optimizer:
            (a_x, p_x, n_x), (params, param_idxs, optimizer) = self._load_batch(a_idx, p_idx, n_idx)
        else:
            a_x = self.array[a_idx]
            p_x = self.array[p_idx]
            n_x = self.array[n_idx]
        # compute loss
        loss = adversarial_loss(
            ys=(a_x, p_x, n_x),
            xs=None,
            adversarial_mode=self.hparams.adversarial_mode,
            adversarial_swapped=self.hparams.adversarial_swapped,
            adversarial_masking=self.hparams.adversarial_masking,
            adversarial_top_k=self.hparams.adversarial_top_k,
            pixel_loss_mode=self.hparams.pixel_loss_mode,
        )
        # log results
        self.log_dict({
            'loss': loss,
            'adv_loss': loss,
        }, prog_bar=True)
        # done!
        if self.hparams.train_batch_optimizer:
            self._update_with_batch(loss, params, param_idxs, optimizer)
            return None
        else:
            return loss

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
            params = self.array[param_idxs].to(device=self.device).to(dtype=self._dtype_dst)
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
        with TempNumpySeed(777):
            std, mean = torch.std_mean(self.array[np.random.randint(0, len(self.array), size=128)])
            std, mean = std.cpu().numpy().tolist(), mean.cpu().numpy().tolist()
            self.log_dict({'approx_mean': mean, 'approx_std': std}, prog_bar=True)
        # backprop
        H.step_optimizer(optimizer, loss)
        # save values to dataset
        with torch.no_grad():
            self.array[param_idxs] = params.detach().cpu().to(self._dtype_src)

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

    def make_train_periodic_callbacks(self, cfg) -> Sequence[pl.Callback]:
        class ImShowCallback(BaseCallbackPeriodic):
            def do_step(this, trainer: pl.Trainer, pl_module: pl.LightningModule):
                if self.dataset is None:
                    log.warning('dataset not initialized, skipping visualisation')
                # get dataset images
                with TempNumpySeed(777):
                    # get scaling values
                    samples = self.dataset.dataset_sample_batch(num_samples=128, mode='raw').to(torch.float32)
                    m, M = float(torch.amin(samples)), float(torch.amax(samples))
                    # add transform to dataset
                    self.dataset._transform = lambda x: H.to_img((x.to(torch.float32) - m) / (M - m))  # this is hacky, scale values to [0, 1] then to [0, 255]
                    # get images
                    image = make_image_grid(self.dataset.dataset_sample_batch(num_samples=16, mode='input'))
                # get augmented traversals
                with torch.no_grad():
                    wandb_image, wandb_animation = H.visualize_dataset_traversal(self.dataset, data_mode='input', output_wandb=True)
                # log images to WANDB
                wb_log_metrics(trainer.logger, {
                    'random_images': wandb.Image(image),
                    'traversal_image': wandb_image, 'traversal_animation': wandb_animation,
                })
        return [ImShowCallback(every_n_steps=cfg.exp.show_every_n_steps, begin_first_step=True)]


# ========================================================================= #
# Run Hydra                                                                 #
# ========================================================================= #


ROOT_DIR = os.path.abspath(__file__ + '/../../../..')


@deprecated('Replaced with run_02_gen_adversarial_dataset_approx')
def run_gen_adversarial_dataset(cfg):
    time_string = datetime.today().strftime('%Y-%m-%d--%H-%M-%S')
    log.info(f'Starting run at time: {time_string}')
    # ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~ #
    # cleanup from old runs:
    try:
        wandb.finish()
    except:
        pass
    # ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~ #
    cfg = make_non_strict(cfg)
    # - - - - - - - - - - - - - - - #
    # check CUDA setting
    gpus = hydra_get_gpus(cfg)
    # create logger
    logger = hydra_make_logger(cfg)
    # create callbacks
    callbacks: List[pl.Callback] = [c for c in hydra_get_callbacks(cfg) if isinstance(c, LoggerProgressCallback)]
    # - - - - - - - - - - - - - - - #
    # check save dirs
    assert not os.path.isabs(cfg.settings.exp.rel_save_dir), f'rel_save_dir must be relative: {repr(cfg.settings.exp.rel_save_dir)}'
    save_dir = os.path.join(ROOT_DIR, cfg.settings.exp.rel_save_dir)
    assert os.path.isabs(save_dir), f'save_dir must be absolute: {repr(save_dir)}'
    # - - - - - - - - - - - - - - - #
    # get the logger and initialize
    if logger is not None:
        logger.log_hyperparams(cfg)
    # print the final config!
    log.info('Final Config' + make_box_str(OmegaConf.to_yaml(cfg)))
    # ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~ #
    # | | | | | | | | | | | | | | | #
    seed(cfg.settings.job.seed)
    # | | | | | | | | | | | | | | | #
    # ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~ #
    # make framework
    framework = AdversarialModel(train_is_gpu=cfg.trainer.cuda, **cfg.adv_system)
    callbacks.extend(framework.make_train_periodic_callbacks(cfg))
    # train
    trainer = pl.Trainer(
        logger=logger,
        callbacks=callbacks,
        # cfg.dsettings.trainer
        gpus=gpus,
        # cfg.trainer
        max_epochs=cfg.trainer.max_epochs,
        max_steps=cfg.trainer.max_steps,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        flush_logs_every_n_steps=cfg.trainer.flush_logs_every_n_steps,
        progress_bar_refresh_rate=cfg.trainer.progress_bar_refresh_rate,
        prepare_data_per_node=cfg.trainer.prepare_data_per_node,
        # we do this here so we don't run the final metrics
        terminate_on_nan=True,
        checkpoint_callback=False,
    )
    trainer.fit(framework)
    # ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~ #
    # get save paths
    save_prefix = f'{cfg.settings.exp.save_prefix}_' if cfg.settings.exp.save_prefix else ''
    save_path_data = os.path.join(save_dir, f'{save_prefix}{time_string}_{cfg.settings.job.name}', f'data.h5')
    # create directories
    if cfg.settings.exp.save_data: ensure_parent_dir_exists(save_path_data)
    # ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~ #
    # compute standard deviation when saving and scale so
    # that we have mean=0 and std=1 of the saved data!
    with TempNumpySeed(777):
        std, mean = torch.std_mean(framework.array[random_choice_prng(len(framework.array), size=2048, replace=False)])
        std, mean = float(std), float(mean)
        log.info(f'normalizing saved dataset of shape: {tuple(framework.array.shape)} and dtype: {framework.array.dtype} with mean: {repr(mean)} and std: {repr(std)}')
    # ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~ #
    # save adversarial dataset
    if cfg.settings.exp.save_data:
        log.info(f'saving data to path: {repr(save_path_data)}')
        # transfer to GPU
        if torch.cuda.is_available():
            framework = framework.cuda()
        # create new h5py file -- TODO: use this in other places!
        with H5Builder(path=save_path_data, mode='atomic_w') as builder:
            # this dataset is self-contained and can be loaded by SelfContainedHdf5GroundTruthData
            # we normalize the values to have approx mean of 0 and std of 1
            builder.add_dataset_from_gt_data(
                data=framework.dataset,  # produces tensors
                mutator=lambda x: np.moveaxis((to_numpy(x).astype('float32') - mean) / std, -3, -1).astype('float16'),  # consumes tensors -> np.ndarrays
                img_shape=(64, 64, None),
                compression_lvl=9,
                batch_size=32,
                dtype='float16',
                attrs=dict(
                    norm_mean=mean,
                    norm_std=std,
                )
            )
        log.info(f'saved data size: {bytes_to_human(os.path.getsize(save_path_data))}')
    # ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~ #


# ========================================================================= #
# Entry Point                                                               #
# ========================================================================= #


if __name__ == '__main__':

    # BENCHMARK (batch_size=256, optimizer=sgd, lr=1e-2, dataset_num_workers=0):
    # - batch_optimizer=False, gpu=True,  fp16=True   : [3168MiB/5932MiB, 3.32/11.7G, 5.52it/s]
    # - batch_optimizer=False, gpu=True,  fp16=False  : [5248MiB/5932MiB, 3.72/11.7G, 4.84it/s]
    # - batch_optimizer=False, gpu=False, fp16=True   : [same as fp16=False]
    # - batch_optimizer=False, gpu=False, fp16=False  : [0003MiB/5932MiB, 4.60/11.7G, 1.05it/s]
    # ---------
    # - batch_optimizer=True,  gpu=True,  fp16=True   : [1284MiB/5932MiB, 3.45/11.7G, 4.31it/s]
    # - batch_optimizer=True,  gpu=True,  fp16=False  : [1284MiB/5932MiB, 3.72/11.7G, 4.31it/s]
    # - batch_optimizer=True,  gpu=False, fp16=True   : [same as fp16=False]
    # - batch_optimizer=True,  gpu=False, fp16=False  : [0003MiB/5932MiB, 1.80/11.7G, 4.18it/s]

    # BENCHMARK (batch_size=1024, optimizer=sgd, lr=1e-2, dataset_num_workers=12):
    # - batch_optimizer=True,  gpu=True,  fp16=True   : [2510MiB/5932MiB, 4.10/11.7G, 4.75it/s, 20% gpu util] (to(device).to(dtype))
    # - batch_optimizer=True,  gpu=True,  fp16=True   : [2492MiB/5932MiB, 4.10/11.7G, 4.12it/s, 19% gpu util] (to(device, dtype))

    @hydra.main(config_path=os.path.join(ROOT_DIR, 'experiment/config'), config_name="config_adversarial_dataset")
    def main(cfg):
        try:
            run_gen_adversarial_dataset(cfg)
        except Exception as e:
            # truncate error
            err_msg = str(e)
            err_msg = err_msg[:244] + ' <TRUNCATED>' if len(err_msg) > 244 else err_msg
            # log something at least
            log.error(f'exiting: experiment error | {err_msg}', exc_info=True)

    # EXP ARGS:
    # $ ... -m dataset=smallnorb,shapes3d
    try:
        main()
    except KeyboardInterrupt as e:
        log_error_and_exit(err_type='interrupted', err_msg=str(e), exc_info=False)
    except Exception as e:
        log_error_and_exit(err_type='hydra error', err_msg=str(e))
