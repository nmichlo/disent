import logging
import time

import numpy as np
import pytorch_lightning as pl
import wandb

from disent.frameworks.unsupervised.vae import Vae
from disent.util import TempNumpySeed
from disent.visualize.visualize_model import latent_cycle
from disent.visualize.visualize_util import gridify_animation, reconstructions_to_images

log = logging.getLogger(__name__)


# ========================================================================= #
# HELPER CALLBACKS                                                          #
# ========================================================================= #


class _PeriodicCallback(pl.Callback):
    
    def __init__(self, every_n_steps=None, begin_first_step=False):
        self.every_n_steps = every_n_steps
        self.begin_first_step = begin_first_step

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if self.every_n_steps is None:
            # number of steps/batches in an epoch
            self.every_n_steps = trainer.num_training_batches
            
    def on_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        # skip if need be
        if 0 == (trainer.global_step + int(not self.begin_first_step)) % self.every_n_steps:
            self.do_step(trainer, pl_module)

    def do_step(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        raise NotImplementedError


class _TimedCallback(pl.Callback):
    
    def __init__(self, interval=10):
        self._last_time = 0
        self._interval = interval
        self._start_time = time.time()

    def on_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        current_time = time.time()
        step_delta = current_time - self._last_time
        # only run every few seconds
        if self._interval < step_delta:
            self._last_time = current_time
            self.do_interval(trainer, pl_module, current_time, self._start_time)

    def do_interval(self, trainer: pl.Trainer, pl_module: pl.LightningModule, current_time, start_time):
        raise NotImplementedError


# ========================================================================= #
# MAIN CALLBACKS                                                            #
# ========================================================================= #


class LatentCycleLoggingCallback(_PeriodicCallback):

    def __init__(self, seed=7777, every_n_steps=None, begin_first_step=False):
        super().__init__(every_n_steps, begin_first_step)
        self.seed = seed

    def do_step(self, trainer: pl.Trainer, pl_module: Vae):
        # VISUALISE - generate and log latent traversals
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


class DisentanglementLoggingCallback(_PeriodicCallback):
    
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


class LoggerProgressCallback(_TimedCallback):
    
    def sig(self, num, digits):
        return f'{num:.{digits}g}'
    
    def do_interval(self, trainer: pl.Trainer, pl_module: pl.LightningModule, current_time, start_time):
        # vars
        batch, max_batches = trainer.batch_idx + 1, trainer.num_training_batches
        epoch, max_epoch = trainer.current_epoch + 1, min(trainer.max_epochs, (trainer.max_steps + max_batches - 1) // max_batches)
        global_step, global_steps = trainer.global_step + 1, min(trainer.max_epochs * max_batches, trainer.max_steps)
        # computed
        train_pct = global_step / global_steps
        # completion
        train_remain_time = (current_time - start_time) * (1 - train_pct) / train_pct
        # info dict
        info_dict = {k: f'{self.sig(v, 4)}' if isinstance(v, (int, float)) else f'{v}' for k, v in trainer.progress_bar_dict.items() if k != 'v_num'}
        sorted_k = sorted(info_dict.keys(), key=lambda k: ('loss' != k.lower(), 'loss' not in k.lower(), k))
        # log
        log.info(
            f'EPOCH: {epoch}/{max_epoch} - {global_step:0{len(str(global_steps))}d}/{global_steps} '
            f'({int(train_pct * 100):02d}%) [{int(train_remain_time)}s] '
            f'STEP: {batch:{len(str(max_batches))}d}/{max_batches} ({int(batch / max_batches * 100):02d}%) '
            f'| {" ".join(f"{k}={info_dict[k]}" for k in sorted_k)}'
        )

# ========================================================================= #
# END                                                                       #
# ========================================================================= #
