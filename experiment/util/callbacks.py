import logging
import time

import numpy as np
import pytorch_lightning as pl
import wandb

from disent.util import TempNumpySeed
from disent.visualize.visualize_model import latent_cycle
from disent.visualize.visualize_util import gridify_animation, reconstructions_to_images

log = logging.getLogger(__name__)


# ========================================================================= #
# callbacks                                                                   #
# ========================================================================= #


class LatentCycleLoggingCallback(pl.Callback):
    
    def __init__(self, seed=7777):
        self.seed = seed
    
    def on_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        # VISUALISE!
        # generate and log latent traversals
        # TODO: reenable
        # assert isinstance(pl_module, HydraLightningModule)
        # get random sample of z_means and z_logvars for computing the range of values for the latent_cycle
        with TempNumpySeed(self.seed):
            log.debug(dir(pl_module))
            log.debug(dir(trainer))
            obs = pl_module.datamodule.dataset.sample_observations(64).to(pl_module.device)
        z_means, z_logvars = pl_module.model.encode_gaussian(obs)
        # produce latent cycle animation & merge frames
        animation = latent_cycle(pl_module.model.reconstruct, z_means, z_logvars, mode='fitted_gaussian_cycle',
                                 num_animations=1, num_frames=21)
        animation = reconstructions_to_images(animation, mode='int', moveaxis=False)  # axis already moved above
        frames = np.transpose(gridify_animation(animation[0], padding_px=4, value=64), [0, 3, 1, 2])
        # check and add missing channel if needed (convert greyscale to rgb images)
        assert frames.shape[1] in {1, 3}, f'Invalid number of image channels: {animation.shape} -> {frames.shape}'
        if frames.shape[1] == 1:
            frames = np.repeat(frames, 3, axis=1)
        # log video
        trainer.log_metrics({
            'epoch': trainer.current_epoch,
            'fitted_gaussian_cycle': wandb.Video(frames, fps=5, format='mp4'),
        }, {})


class DisentanglementLoggingCallback(pl.Callback):
    
    def __init__(self, epoch_end_metrics=None, train_end_metrics=None, every_n_epochs=2, begin_first_epoch=True):
        self.begin_first_epoch = begin_first_epoch
        self.epoch_end_metrics = epoch_end_metrics if epoch_end_metrics else []
        self.train_end_metrics = train_end_metrics if train_end_metrics else []
        self.every_n_epochs = every_n_epochs
        assert isinstance(self.epoch_end_metrics, list)
        assert isinstance(self.train_end_metrics, list)
        assert self.epoch_end_metrics or self.train_end_metrics, 'No metrics given to epoch_end_metrics or train_end_metrics'
    
    def _compute_metrics_and_log(self, trainer: pl.Trainer, pl_module: pl.LightningModule, metrics, is_final=False):
        # checks
        # TODO: reenable
        # assert isinstance(pl_module, HydraLightningModule)
        # compute all metrics
        for metric in metrics:
            log.info(dir(pl_module))
            log.info(dir(pl_module))
            log.info(dir(pl_module))
            log.info(pl_module.datamodule)
            log.info(pl_module.train_dataloader)
            scores = metric(pl_module.datamodule.dataset,
                            lambda x: pl_module.model.encode_deterministic(x.to(pl_module.device)))
            log.info(f'metric (epoch: {trainer.current_epoch}): {scores}')
            # log to wandb if it exists
            trainer.log_metrics({
                'epoch': trainer.current_epoch,
                'final_metric' if is_final else 'epoch_metric': scores,
            }, {})
    
    def on_epoch_end(self, trainer, pl_module):
        if self.epoch_end_metrics:
            # first epoch is 0, if we dont want the first one to be run we need to increment by 1
            if 0 == (trainer.current_epoch + int(not self.begin_first_epoch)) % self.every_n_epochs:
                self._compute_metrics_and_log(trainer, pl_module, metrics=self.epoch_end_metrics)
    
    def on_train_end(self, trainer, pl_module):
        if self.train_end_metrics:
            self._compute_metrics_and_log(trainer, pl_module, metrics=self.train_end_metrics, is_final=True)


class LoggerProgressCallback(pl.Callback):
    def __init__(self, time_step=10):
        self.start_time = None
        self.last_time = 0
        self.time_step = time_step
        self.start_time = time.time()
    
    def on_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        current_time = time.time()
        step_delta = current_time - self.last_time
        
        if self.time_step < step_delta:
            self.last_time = time.time()
            # vars
            step, max_steps = trainer.batch_idx + 1, trainer.num_training_batches
            epoch, max_epoch = trainer.current_epoch + 1, trainer.max_epochs
            # computed
            epoch_pct = epoch / max_epoch
            step_pct = step / max_steps
            step_len = len(str(max_steps))
            # completion
            epoch_delta = current_time - self.start_time
            epoch_remain_time = step_delta / step_pct - step_delta
            train_remain_time = epoch_delta / epoch_pct - epoch_delta
            # log
            log.info(f'EPOCH: {epoch}/{max_epoch} [{int(epoch_pct * 100):3d}%, {int(epoch_remain_time)}s] '
                     f'STEP: {step:{step_len}d}/{max_steps} [{int(step_pct * 100):3d}%] '
                     f'[GLOBAL STEP: {trainer.global_step}, {int(train_remain_time)}s]')

# ========================================================================= #
# END                                                                       #
# ========================================================================= #
