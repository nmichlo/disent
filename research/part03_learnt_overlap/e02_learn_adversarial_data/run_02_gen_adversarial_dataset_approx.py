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
from typing import Sequence
from typing import Tuple

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import wandb
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

import research.code.util as H
from disent import registry
from disent.dataset import DisentDataset
from disent.dataset.sampling import BaseDisentSampler
from disent.dataset.util.hdf5 import H5Builder
from disent.model import AutoEncoder
from disent.nn.activations import Swish
from disent.nn.modules import DisentModule
from disent.nn.weights import init_model_weights
from disent.util import to_numpy
from disent.util.function import wrapped_partial
from disent.util.inout.paths import ensure_parent_dir_exists
from disent.util.lightning.callbacks import BaseCallbackPeriodic
from disent.util.lightning.callbacks import LoggerProgressCallback
from disent.util.lightning.logger_util import wb_has_logger
from disent.util.lightning.logger_util import wb_log_metrics
from disent.util.seeds import seed
from disent.util.seeds import TempNumpySeed
from disent.util.strings.fmt import bytes_to_human
from disent.util.strings.fmt import make_box_str
from disent.util.visualize.vis_util import make_image_grid
from experiment.run import hydra_get_gpus
from experiment.run import hydra_get_callbacks
from experiment.run import hydra_make_logger
from experiment.util.hydra_utils import make_non_strict
from experiment.util.run_utils import log_error_and_exit
from research.part03_learnt_overlap.e02_learn_adversarial_data.util_gen_adversarial_dataset import adversarial_loss
from research.part03_learnt_overlap.e02_learn_adversarial_data.util_gen_adversarial_dataset import make_adversarial_sampler
from research.part03_learnt_overlap.e02_learn_adversarial_data.util_gen_adversarial_dataset import sort_samples


log = logging.getLogger(__name__)


# ========================================================================= #
# Dataset Mask                                                              #
# ========================================================================= #

@torch.no_grad()
def _sample_stacked_batch(dataset: DisentDataset) -> torch.Tensor:
    batch = next(iter(DataLoader(dataset, batch_size=1024, num_workers=0, shuffle=True)))
    batch = torch.cat(batch['x_targ'], dim=0)
    return batch

@torch.no_grad()
def gen_approx_dataset_mask(dataset: DisentDataset, model_mask_mode: Optional[str]) -> Optional[torch.Tensor]:
    if model_mask_mode in ('none', None):
        mask = None
    elif model_mask_mode == 'diff':
        batch = _sample_stacked_batch(dataset)
        mask = ~torch.all(batch[1:] == batch[0:1], dim=0)
    elif model_mask_mode == 'std':
        batch = _sample_stacked_batch(dataset)
        mask = torch.std(batch, dim=0)
        m, M = torch.amin(mask), torch.amax(mask)
        mask = (mask - m) / (M - m)
    else:
        raise KeyError(f'invalid `model_mask_mode`: {repr(model_mask_mode)}')
    # done
    return mask


# ========================================================================= #
# adversarial dataset generator                                             #
# ========================================================================= #


class AeModel(AutoEncoder):
    def forward(self, x):
        return self.decode(self.encode(x))


def make_delta_model(model_type: str, x_shape: Tuple[int, ...]):
    C, H, W = x_shape
    # get model
    if model_type.startswith('ae_'):
        return AeModel(
            encoder=registry.MODELS[f'encoder_{model_type[len("ae_"):]}'](x_shape=x_shape, z_size=64, z_multiplier=1),
            decoder=registry.MODELS[f'decoder_{model_type[len("ae_"):]}'](x_shape=x_shape, z_size=64, z_multiplier=1),
        )
    elif model_type == 'fcn_small':
        return torch.nn.Sequential(
            torch.nn.ReflectionPad2d(1), torch.nn.Conv2d(in_channels=C, out_channels=5, kernel_size=3), Swish(),
            torch.nn.ReflectionPad2d(1), torch.nn.Conv2d(in_channels=5, out_channels=7, kernel_size=3), Swish(),
            torch.nn.ReflectionPad2d(1), torch.nn.Conv2d(in_channels=7, out_channels=9, kernel_size=3), Swish(),
            torch.nn.ReflectionPad2d(1), torch.nn.Conv2d(in_channels=9, out_channels=7, kernel_size=3), Swish(),
            torch.nn.ReflectionPad2d(1), torch.nn.Conv2d(in_channels=7, out_channels=5, kernel_size=3), Swish(),
            torch.nn.ReflectionPad2d(1), torch.nn.Conv2d(in_channels=5, out_channels=C, kernel_size=3),
        )
    else:
        raise KeyError(f'invalid model type: {repr(model_type)}')


class AdversarialAugmentModel(DisentModule):

    def __init__(self, model_type: str, x_shape=(3, 64, 64), mask=None, meta: dict = None):
        super().__init__()
        # make layers
        self.delta_model = make_delta_model(model_type=model_type, x_shape=x_shape)
        self.meta = meta if meta else {}
        # mask
        if mask is not None:
            self.register_buffer('mask', mask[None, ...])
            assert self.mask.ndim == 4  # (1, C, H, W)

    def forward(self, x):
        assert x.ndim == 4
        # compute
        if hasattr(self, 'mask'):
            return x + self.delta_model(x) * self.mask
        else:
            return x + self.delta_model(x)


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
            data_load_into_memory: bool = False,
        # adversarial loss options
            adversarial_mode: str = 'self',
            adversarial_swapped: bool = False,
            adversarial_masking: bool = False,
            adversarial_top_k: Optional[int] = None,
            pixel_loss_mode: str = 'mse',
        # loss extras
            loss_adversarial_weight: Optional[float] = 1.0,
            loss_same_stats_weight: Optional[float] = 0.0,
            loss_similarity_weight: Optional[float] = 0.0,
            loss_out_of_bounds_weight: Optional[float] = 0.0,
        # sampling config
            sampler_name: str = 'close_far',
            samples_sort_mode: str = 'none',
        # model settings
            model_type: str = 'ae_linear',
            model_mask_mode: Optional[str] = 'none',
            model_weight_init: str = 'xavier_normal',
        # logging settings
            logging_scale_imgs: bool = False,
            # log_wb_stats_table: bool = True,
    ):
        super().__init__()
        # modify hparams
        if optimizer_kwargs is None:
            optimizer_kwargs = {}  # value used by save_hyperparameters
        # save hparams
        self.save_hyperparameters()
        # variables
        self.dataset: DisentDataset = None
        self.sampler: BaseDisentSampler = None
        self.model: DisentModule = None

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
            sampler=make_adversarial_sampler(self.hparams.sampler_name),
        )
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

    def train_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.hparams.dataset_batch_size,
            num_workers=self.hparams.dataset_num_workers,
            shuffle=True,
        )

    def configure_optimizers(self):
        return H.make_optimizer(
            self.model,
            name=self.hparams.optimizer_name,
            lr=self.hparams.optimizer_lr,
            **self.hparams.optimizer_kwargs,
        )

    # ================================== #
    # train step                         #
    # ================================== #

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        (a_x, p_x, n_x) = batch['x_targ']
        # sort inputs
        a_x, p_x, n_x = sort_samples(a_x, p_x, n_x, sort_mode=self.hparams.samples_sort_mode, pixel_loss_mode=self.hparams.pixel_loss_mode)
        # feed forward
        a_y = self.model(a_x)
        p_y = self.model(p_x)
        n_y = self.model(n_x)
        # compute loss
        loss_adv = 0
        if (self.hparams.loss_adversarial_weight is not None) and (self.hparams.loss_adversarial_weight > 0):
            loss_adv, loss_adv_stats = adversarial_loss(
                ys=(a_y, p_y, n_y),
                xs=(a_x, p_x, n_x),
                adversarial_mode=self.hparams.adversarial_mode,
                adversarial_swapped=self.hparams.adversarial_swapped,
                adversarial_masking=self.hparams.adversarial_masking,
                adversarial_top_k=self.hparams.adversarial_top_k,
                pixel_loss_mode=self.hparams.pixel_loss_mode,
                return_stats=True,
            )
            loss_adv *= self.hparams.loss_adversarial_weight
            self.log_dict(loss_adv_stats)
        # additional loss components
        # - keep stats the same
        loss_stats = 0
        if (self.hparams.loss_same_stats_weight is not None) and (self.hparams.loss_same_stats_weight > 0):
            loss_stats += (self.hparams.loss_same_stats_weight/3) * ((
                F.mse_loss(a_y.mean(dim=[-3, -2, -1]), a_x.mean(dim=[-3, -2, -1]), reduction='mean') +
                F.mse_loss(p_y.mean(dim=[-3, -2, -1]), p_x.mean(dim=[-3, -2, -1]), reduction='mean') +
                F.mse_loss(n_y.mean(dim=[-3, -2, -1]), n_x.mean(dim=[-3, -2, -1]), reduction='mean')
            ) + (
                F.mse_loss(a_y.std(dim=[-3, -2, -1]), a_x.std(dim=[-3, -2, -1]), reduction='mean') +
                F.mse_loss(p_y.std(dim=[-3, -2, -1]), p_x.std(dim=[-3, -2, -1]), reduction='mean') +
                F.mse_loss(n_y.std(dim=[-3, -2, -1]), n_x.std(dim=[-3, -2, -1]), reduction='mean')
            ))
        # - try keep similar to inputs
        loss_sim = 0
        if (self.hparams.loss_similarity_weight is not None) and (self.hparams.loss_similarity_weight > 0):
            loss_sim = (self.hparams.loss_similarity_weight / 3) * (
                F.mse_loss(a_y, a_x, reduction='mean') +
                F.mse_loss(p_y, p_x, reduction='mean') +
                F.mse_loss(n_y, n_x, reduction='mean')
            )
        # - regularize if out of bounds
        loss_out = 0
        if (self.hparams.loss_out_of_bounds_weight is not None) and (self.hparams.loss_out_of_bounds_weight > 0):
            zeros = torch.zeros_like(a_y)
            loss_out = (self.hparams.loss_out_of_bounds_weight / 6) * (
                torch.where(a_y < 0, -a_y, zeros).mean() + torch.where(a_y > 1, a_y-1, zeros).mean() +
                torch.where(p_y < 0, -p_y, zeros).mean() + torch.where(p_y > 1, p_y-1, zeros).mean() +
                torch.where(n_y < 0, -n_y, zeros).mean() + torch.where(n_y > 1, n_y-1, zeros).mean()
            )
        # final loss
        loss = loss_adv + loss_sim + loss_out
        # log everything
        self.log_dict({
            'loss': loss,
            'loss_stats': loss_stats,
            'loss_adv': loss_adv,
            'loss_out': loss_out,
            'loss_sim': loss_sim,
        }, prog_bar=True)
        # done!
        return loss

    # ================================== #
    # dataset                            #
    # ================================== #

    @torch.no_grad()
    def batch_to_adversarial_imgs(self, batch: torch.Tensor, m=0, M=1, mode='uint8') -> np.ndarray:
        batch = batch.to(device=self.device, dtype=torch.float32)
        batch = self.model(batch)
        batch = (batch - m) / (M - m)
        if mode == 'uint8': return H.to_imgs(batch).numpy()
        elif mode == 'float32': return torch.moveaxis(batch, -3, -1).to(torch.float32).cpu().numpy()
        elif mode == 'float16': return torch.moveaxis(batch, -3, -1).to(torch.float16).cpu().numpy()
        else: raise KeyError(f'invalid output mode: {repr(mode)}')

    def make_train_periodic_callbacks(self, cfg) -> Sequence[BaseCallbackPeriodic]:

        # dataset transform helper
        @TempNumpySeed(42)
        @torch.no_grad()
        def make_scale_uint8_transform():
            # get scaling values
            if self.hparams.logging_scale_imgs:
                samples = self.dataset.dataset_sample_batch(num_samples=128, mode='raw').to(torch.float32)
                samples = self.model(samples.to(self.device)).cpu()
                m, M = float(torch.amin(samples)), float(torch.amax(samples))
            else:
                m, M = 0, 1
            return lambda x: self.batch_to_adversarial_imgs(x[None, ...], m=m, M=M)[0]

        # show image callback
        class _BaseDatasetCallback(BaseCallbackPeriodic):
            @TempNumpySeed(777)
            @torch.no_grad()
            def do_step(this, trainer: pl.Trainer, system: AdversarialModel):
                if not wb_has_logger(trainer.logger):
                    log.warning(f'no wandb logger found, skipping visualisation: {system.__class__.__name__}')
                    return
                if system.dataset is None:
                    log.warning(f'dataset not initialized, skipping visualisation: {system.__class__.__name__}')
                    return
                log.info(f'visualising: {this.__class__.__name__}')
                try:
                    this._do_step(trainer, system)
                except:
                    log.error('Failed to do visualise callback step!', exc_info=True)

            # override this
            def _do_step(this, trainer: pl.Trainer, system: AdversarialModel):
                raise NotImplementedError

        # show image callback
        class ImShowCallback(_BaseDatasetCallback):
            def _do_step(this, trainer: pl.Trainer, system: AdversarialModel):
                # make dataset with required transform
                # -- this is inefficient for multiple subclasses of this class, we need to recompute the transform each time
                dataset = system.dataset.shallow_copy(transform=make_scale_uint8_transform())
                # get images & traversal
                image = make_image_grid(dataset.dataset_sample_batch(num_samples=16, mode='input'))
                wandb_image, wandb_animation = H.visualize_dataset_traversal(dataset, data_mode='input', output_wandb=True)
                # log images to WANDB
                wb_log_metrics(trainer.logger, {
                    'random_images': wandb.Image(image),
                    'traversal_image': wandb_image,
                    'traversal_animation': wandb_animation,
                })

        # factor distances callback
        class DistsPlotCallback(_BaseDatasetCallback):
            def _do_step(this, trainer: pl.Trainer, system: AdversarialModel):
                from disent.util.lightning.callbacks._callbacks_vae import compute_factor_distances, plt_factor_distances

                # make distances function
                def dists_fn(xs_a, xs_b):
                    dists = H.pairwise_loss(xs_a, xs_b, mode=system.hparams.pixel_loss_mode, mean_dtype=torch.float32, mask=None)
                    return [dists]

                def transform_batch(batch):
                    return system.model(batch.to(device=system.device))

                # compute various distances matrices for each factor
                dists_names, f_grid = compute_factor_distances(
                    dataset=system.dataset,
                    dists_fn=dists_fn,
                    dists_names=['dists'],
                    traversal_repeats=100,
                    batch_size=system.hparams.dataset_batch_size,
                    include_gt_factor_dists=True,
                    transform_batch=transform_batch,
                    seed=777,
                    data_mode='input',
                )
                # plot these results
                fig, axs = plt_factor_distances(
                    gt_data=system.dataset.gt_data,
                    f_grid=f_grid,
                    dists_names=dists_names,
                    title=f'{system.hparams.model_type.capitalize()}: {system.hparams.dataset_name.capitalize()} Distances',
                    plt_block_size=1.25,
                    plt_transpose=True,
                    plt_cmap='Blues',
                )
                # recolour dists axis
                for ax in axs[-1, :]:
                    ax.images[0].set_cmap('Reds')
                # generate image & close matplotlib instace
                from matplotlib import pyplot as plt
                img = wandb.Image(fig)
                plt.close()
                # log the plot to wandb
                if True:
                    wb_log_metrics(trainer.logger, {
                        'factor_distances': img
                    })

        # show stats callback
        class StatsShowCallback(_BaseDatasetCallback):
            def _do_step(this, trainer: pl.Trainer, system: AdversarialModel):
                # make dataset with required transform
                # -- this is inefficient for multiple subclasses of this class, we need to recompute the transform each time
                dataset = system.dataset.shallow_copy(transform=make_scale_uint8_transform())
                # get batches
                batch, factors = dataset.dataset_sample_batch_with_factors(num_samples=512, mode='input')
                batch = batch.to(torch.float32)
                a_idx = torch.randint(0, len(batch), size=[4*len(batch)])
                b_idx = torch.randint(0, len(batch), size=[4*len(batch)])
                mask = (a_idx != b_idx)
                # TODO: check that this is deterministic
                # compute distances
                deltas = to_numpy(H.pairwise_overlap(batch[a_idx[mask]], batch[b_idx[mask]], mode='mse'))
                fdists = to_numpy(torch.abs(factors[a_idx[mask]] - factors[b_idx[mask]]).sum(dim=-1))
                sdists = to_numpy((torch.abs(factors[a_idx[mask]] - factors[b_idx[mask]]) / to_numpy(dataset.gt_data.factor_sizes)[None, :]).sum(dim=-1))
                # log to wandb
                from matplotlib import pyplot as plt
                plt.scatter(fdists, deltas); img_fdists = wandb.Image(plt); plt.close()
                plt.scatter(sdists, deltas); img_sdists = wandb.Image(plt); plt.close()
                wb_log_metrics(trainer.logger, {
                    'fdists_vs_overlap': img_fdists,
                    'sdists_vs_overlap': img_sdists,
                })

        # done!
        return [
            ImShowCallback(every_n_steps=cfg.settings.exp.show_every_n_steps, begin_first_step=True),
            DistsPlotCallback(every_n_steps=cfg.settings.exp.show_every_n_steps, begin_first_step=True),
            StatsShowCallback(every_n_steps=cfg.settings.exp.show_every_n_steps, begin_first_step=True),
        ]


# ========================================================================= #
# Run Hydra                                                                 #
# ========================================================================= #


ROOT_DIR = os.path.abspath(__file__ + '/../../..')


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
    framework = AdversarialModel(**cfg.adv_system)
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
        detect_anomaly=True,
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
            # this dataset is self-contained and can be loaded by SelfContainedHdf5GroundTruthData
            builder.add_dataset_from_gt_data(
                data=framework.dataset,  # produces tensors
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

    @hydra.main(config_path=os.path.join(ROOT_DIR, 'experiment/config'), config_name="config_adversarial_dataset_approx")
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
