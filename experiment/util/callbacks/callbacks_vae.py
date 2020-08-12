import logging

import wandb
import numpy as np
import pytorch_lightning as pl

from disent.frameworks.unsupervised.vae import Vae
from disent.util import TempNumpySeed
from disent.visualize.visualize_model import latent_cycle
from disent.visualize.visualize_util import gridify_animation, reconstructions_to_images

from experiment.util.callbacks.callbacks_base import _PeriodicCallback


log = logging.getLogger(__name__)


# ========================================================================= #
# Vae Framework Callbacks                                                   #
# ========================================================================= #


class VaeLatentCycleLoggingCallback(_PeriodicCallback):
    
    def __init__(self, seed=7777, every_n_steps=None, begin_first_step=False):
        super().__init__(every_n_steps, begin_first_step)
        self.seed = seed
    
    def do_step(self, trainer: pl.Trainer, pl_module: Vae):
        assert isinstance(pl_module, Vae), 'pl_module is not an instance of the Vae framework'
        
        # get random sample of z_means and z_logvars for computing the range of values for the latent_cycle
        with TempNumpySeed(self.seed):
            obs = trainer.datamodule.dataset.sample_observations(64).to(pl_module.device)
        z_means, z_logvars = pl_module.encode_gaussian(obs)
        
        # produce latent cycle animation & merge frames
        animation = latent_cycle(pl_module.decode, z_means, z_logvars, mode='fitted_gaussian_cycle', num_animations=1, num_frames=21)
        animation = reconstructions_to_images(animation, mode='int', moveaxis=False)  # axis already moved above
        frames = np.transpose(gridify_animation(animation[0], padding_px=4, value=64), [0, 3, 1, 2])
        
        # check and add missing channel if needed (convert greyscale to rgb images)
        assert frames.shape[1] in {1, 3}, f'Invalid number of image channels: {animation.shape} -> {frames.shape}'
        if frames.shape[1] == 1:
            frames = np.repeat(frames, 3, axis=1)
        
        # log video
        trainer.log_metrics({
            'fitted_gaussian_cycle': wandb.Video(frames, fps=5, format='mp4'),
        }, {})


class VaeDisentanglementLoggingCallback(_PeriodicCallback):
    
    def __init__(self, step_end_metrics=None, train_end_metrics=None, every_n_steps=None, begin_first_step=False):
        super().__init__(every_n_steps, begin_first_step)
        self.step_end_metrics = step_end_metrics if step_end_metrics else []
        self.train_end_metrics = train_end_metrics if train_end_metrics else []
        assert isinstance(self.step_end_metrics, list)
        assert isinstance(self.train_end_metrics, list)
        assert self.step_end_metrics or self.train_end_metrics, 'No metrics given to step_end_metrics or train_end_metrics'
    
    def _compute_metrics_and_log(self, trainer: pl.Trainer, pl_module: Vae, metrics: list, is_final=False):
        assert isinstance(pl_module, Vae)
        # compute all metrics
        for metric in metrics:
            scores = metric(trainer.datamodule.dataset, lambda x: pl_module.encode(x.to(pl_module.device)))
            log.info(f'metric (step: {trainer.global_step}): {scores}')
            trainer.log_metrics({'final_metric' if is_final else 'epoch_metric': scores}, {})
    
    def do_step(self, trainer: pl.Trainer, pl_module: Vae):
        if self.step_end_metrics:
            log.info('Computing epoch metrics:')
            self._compute_metrics_and_log(trainer, pl_module, metrics=self.step_end_metrics, is_final=False)
    
    def on_train_end(self, trainer: pl.Trainer, pl_module: Vae):
        if self.train_end_metrics:
            log.info('Computing final training run metrics:')
            self._compute_metrics_and_log(trainer, pl_module, metrics=self.train_end_metrics, is_final=True)


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
