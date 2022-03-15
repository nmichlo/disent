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
Generate an adversarial dataset by approximating the difference between
the dataset and the target adversarial images using a model.
    adv = obs + diff(obs)
"""

import logging
import os
from datetime import datetime
from typing import List
from typing import Optional
from typing import Union

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import wandb
from omegaconf import OmegaConf

import research.code.util as H
from disent.dataset import DisentDataset
from disent.dataset.sampling import BaseDisentSampler
from disent.dataset.util.hdf5 import H5Builder
from disent.nn.modules import DisentModule
from disent.nn.weights import init_model_weights
from disent.util.function import wrapped_partial
from disent.util.inout.paths import ensure_parent_dir_exists
from disent.util.lightning.callbacks import LoggerProgressCallback
from disent.util.seeds import seed
from disent.util.strings.fmt import bytes_to_human
from disent.util.strings.fmt import make_box_str
from experiment.run import hydra_get_gpus
from experiment.run import hydra_get_callbacks
from experiment.run import hydra_make_logger
from experiment.util.hydra_main import hydra_main
from experiment.util.hydra_utils import make_non_strict
from research.part03_learnt_overlap.e01_learn_to_disentangle.run_03_train_disentangle_kernel import disentangle_loss
from research.part03_learnt_overlap.e02_learn_adversarial_data.run_02_gen_adversarial_dataset_approx import AdversarialAugmentModel
from research.part03_learnt_overlap.e02_learn_adversarial_data.run_02_gen_adversarial_dataset_approx import AdversarialModel
from research.part03_learnt_overlap.e02_learn_adversarial_data.run_02_gen_adversarial_dataset_approx import gen_approx_dataset_mask


log = logging.getLogger(__name__)


# ========================================================================= #
# adversarial dataset generator                                             #
# ========================================================================= #


class DisentangleModel(AdversarialModel):

    def __init__(
        self,
        # optimizer options
            optimizer_name: str = 'sgd',
            optimizer_lr: float = 5e-2,
            optimizer_kwargs: Optional[dict] = None,
        # dataset config options
            dataset_name: str = 'cars3d',
            dataset_num_workers: int = min(os.cpu_count(), 16),
            dataset_batch_size: int = 256,
            data_root: str = 'data/dataset',
            data_load_into_memory: bool = False,
        # disentangle loss options
            disentangle_mode: str = 'improve',
            disentangle_pairs_ratio: float = 8.0,
            disentangle_factors: Optional[List[Union[str, int]]] = None,
            disentangle_loss: str = 'mse',
            disentangle_reg_strength: float = 1.0,
            disentangle_scale_dists: bool = True,
            disentangle_combined_loss: bool = True,
        # loss extras
            loss_disentangle_weight: Optional[float] = 1.0,
            loss_stats_mean_weight: Optional[float] = 0.0,
            loss_stats_var_weight: Optional[float] = 0.0,
            loss_similarity_weight: Optional[float] = 0.0,
            loss_out_of_bounds_weight: Optional[float] = 0.0,
        # model settings
            model_type: str = 'ae_linear',
            model_mask_mode: Optional[str] = 'none',
            model_weight_init: str = 'xavier_normal',
        # logging settings
            logging_scale_imgs: bool = False,
    ):
        # some hparams from here are not actually used!
        # we should delete them ...
        super().__init__()
        # modify hparams
        if optimizer_kwargs is None:
            optimizer_kwargs = {}  # value used by save_hyperparameters
        # save hparams
        self.save_hyperparameters()
        # variables
        self.dataset: DisentDataset = None
        self.model: DisentModule = None
        self._disentangle_idxs: np.array = None

    # ================================== #
    # setup                              #
    # ================================== #

    def prepare_data(self) -> None:
        # create dataset
        self.dataset = H.make_dataset(
            self.hparams.dataset_name,
            load_into_memory=self.hparams.data_load_into_memory,
            load_memory_dtype=torch.float32,
            data_root=self.hparams.data_root,
            factors=True,
        )
        # normalize factors
        self._disentangle_idxs = self.dataset.gt_data.normalise_factor_idxs(self.hparams.disentangle_factors)
        self.hparams.disentangle_factors = tuple(self.dataset.gt_data.factor_names[i] for i in self._disentangle_idxs)
        log.info('Disentangling Factors:')
        for factor_name in self.hparams.disentangle_factors:
            log.info(f'* {factor_name}')
        # make the model
        self.model = AdversarialAugmentModel(
            model_type=self.hparams.model_type,
            x_shape=(self.dataset.gt_data.img_channels, 64, 64),
            mask=gen_approx_dataset_mask(dataset=self.dataset, model_mask_mode=self.hparams.model_mask_mode),
            # if we save the model we can restore things!
            meta=dict(
                dataset_name=self.hparams.dataset_name,
                dataset_factor_sizes=self.dataset.gt_data.factor_sizes,
                dataset_factor_names=self.dataset.gt_data.factor_names,
                sampler_name=self.hparams.sampler_name,
                hparams=dict(self.hparams)
            ),
        )
        # initialize model
        self.model = init_model_weights(self.model, mode=self.hparams.model_weight_init)

    # ================================== #
    # train step                         #
    # ================================== #

    def training_step(self, batch, batch_idx):
        (x,) = batch['x_targ']
        (f,) = batch['factors']
        # feed forward
        y = self.model(x)
        # compute loss
        loss_dis = 0
        if (self.hparams.loss_adversarial_weight is not None) and (self.hparams.loss_adversarial_weight > 0):
            loss_dis = self.hparams.loss_disentangle_weight * disentangle_loss(
                batch     = x if self.hparams.disentangle_combined_loss else y,
                aug_batch = y if self.hparams.disentangle_combined_loss else None,
                factors=f,
                num_pairs=int(len(y) * self.hparams.disentangle_pairs_ratio),
                f_idxs=self._disentangle_idxs,
                loss_fn=self.hparams.disentangle_loss,
                corr_mode=self.hparams.disentangle_mode,
                regularization_strength=self.hparams.disentangle_reg_strength,
                factor_sizes=torch.as_tensor(self.dataset.gt_data.factor_sizes, device=f.device) if self.hparams.disentangle_scale_dists else None,
            )
        # additional loss components
        # - keep stats the same
        loss_stats = 0
        if (self.hparams.loss_stats_mean_weight is not None) and (self.hparams.loss_stats_mean_weight > 0):
            img_mean_loss = F.mse_loss(y.mean(dim=0), x.mean(dim=0), reduction='mean')
            loss_stats += self.hparams.loss_stats_mean_weight * img_mean_loss
        if (self.hparams.loss_stats_var_weight is not None) and (self.hparams.loss_stats_var_weight > 0):
            img_std_loss = F.mse_loss(y.std(dim=0), x.std(dim=0), reduction='mean')
            loss_stats += self.hparams.loss_stats_var_weight * img_std_loss
        # - try keep similar to inputs
        loss_sim = 0
        if (self.hparams.loss_similarity_weight is not None) and (self.hparams.loss_similarity_weight > 0):
            loss_sim = self.hparams.loss_similarity_weight * F.mse_loss(y, x, reduction='mean')
        # - regularize if out of bounds
        loss_out = 0
        if (self.hparams.loss_out_of_bounds_weight is not None) and (self.hparams.loss_out_of_bounds_weight > 0):
            zeros = torch.zeros_like(y)
            gt_loss = torch.where(y < 0, -y, zeros).mean()
            lt_loss = torch.where(y > 1, y-1, zeros).mean()
            loss_out = self.hparams.loss_out_of_bounds_weight * (gt_loss + lt_loss)
        # final loss
        loss = loss_dis + loss_stats + loss_sim + loss_out
        # log everything
        self.log_dict({
            'loss': loss,
            'dis': loss_dis,
            'sta': loss_stats,
            'out': loss_out,
            'sim': loss_sim,
        }, prog_bar=True)
        # done!
        return loss


# ========================================================================= #
# Run Hydra                                                                 #
# ========================================================================= #


ROOT_DIR = os.path.abspath(__file__ + '/../../..')


def run_gen_dataset(cfg):
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
    framework = DisentangleModel(**cfg.dis_system)
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
        enable_progress_bar=cfg.trainer.enable_progress_bar,
        # prepare_data_per_node=cfg.trainer.prepare_data_per_node,  # TODO: moved into data module / framework !
        # we do this here so we don't run the final metrics
        detect_anomaly=False,  # this should only be enabled for debugging torch and finding NaN values, slows down execution, not by much though?
        enable_checkpointing=False,
    )
    trainer.fit(framework)
    # ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~ #
    # get save paths
    save_prefix = f'{cfg.settings.exp.save_prefix}_' if cfg.settings.exp.save_prefix else ''
    save_path_model = os.path.join(save_dir, f'{save_prefix}{time_string}_{cfg.settings.job.name}', f'model.pt')
    save_path_data = os.path.join(save_dir, f'{save_prefix}{time_string}_{cfg.settings.job.name}', f'data.h5')
    # create directories
    if cfg.settings.exp.save_model: ensure_parent_dir_exists(save_path_model)
    if cfg.settings.exp.save_data: ensure_parent_dir_exists(save_path_data)
    # ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~ #
    # save adversarial model
    if cfg.settings.exp.save_model:
        log.info(f'saving model to path: {repr(save_path_model)}')
        torch.save(framework.model, save_path_model)
        log.info(f'saved model size: {bytes_to_human(os.path.getsize(save_path_model))}')
    # ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~ #
    # save adversarial dataset
    if cfg.settings.exp.save_data:
        log.info(f'saving data to path: {repr(save_path_data)}')
        # transfer to GPU
        if torch.cuda.is_available():
            framework = framework.cuda()
        # create new h5py file -- TODO: use this in other places!
        with H5Builder(path=save_path_data, mode='atomic_w') as builder:
            # set the transform -- TODO: we should not need to do this!
            assert framework.dataset.gt_data._transform is None
            framework.dataset.gt_data._transform = framework.dataset.transform
            # this dataset is self-contained and can be loaded by SelfContainedHdf5GroundTruthData
            builder.add_dataset_from_gt_data(
                data=framework.dataset.gt_data,  # produces raw
                mutator=wrapped_partial(framework.batch_to_adversarial_imgs, mode=cfg.settings.exp.save_dtype),  # consumes tensors -> np.ndarrays
                img_shape=(64, 64, None),
                compression_lvl=4,
                dtype=cfg.settings.exp.save_dtype,
                batch_size=32,
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

    CONFIGS_THIS_EXP = os.path.abspath(os.path.join(__file__, '..', 'config'))
    CONFIGS_RESEARCH = os.path.abspath(os.path.join(__file__, '../../..', 'config'))

    # launch the action
    hydra_main(
        callback=run_gen_dataset,
        config_name='config_disentangle_dataset_approx',
        search_dirs_prepend=[CONFIGS_THIS_EXP, CONFIGS_RESEARCH],
        log_level=logging.INFO,
    )
