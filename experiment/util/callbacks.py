import logging
import time

import numpy as np
import pytorch_lightning as pl
import wandb

from disent.frameworks.weaklysupervised.msp_adavae import MspAdaVae
from disent.util import TempNumpySeed
from disent.visualize.visualize_model import latent_cycle
from disent.visualize.visualize_util import gridify_animation, reconstructions_to_images

log = logging.getLogger(__name__)


# ========================================================================= #
# callbacks                                                                   #
# ========================================================================= #


class LatentCycleLoggingCallback(pl.Callback):
    
    def __init__(self, seed=7777, every_n_steps=250, begin_first_step=False):
        self.seed = seed
        self.every_n_steps = every_n_steps
        self.begin_first_step = begin_first_step
    
    def on_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        # skip if need be
        if 0 != (trainer.global_step + int(not self.begin_first_step)) % self.every_n_steps:
            return
        
        # VISUALISE - generate and log latent traversals
        # TODO: re-enable
        # assert isinstance(pl_module, HydraSystem)
        
        # get random sample of z_means and z_logvars for computing the range of values for the latent_cycle
        with TempNumpySeed(self.seed):
            obs = trainer.datamodule.dataset.sample_observations(64).to(pl_module.device)
        z_means, z_logvars = pl_module.encode_gaussian(obs)

        # produce latent cycle animation & merge frames
        if isinstance(pl_module, MspAdaVae):
            animation = latent_cycle(
                lambda y: pl_module.decode(pl_module._msp.mutated_z(z_means[0], y, 1)),
                pl_module._msp.latent_to_labels(z_means),
                pl_module._msp.latent_to_labels(z_logvars),
                mode='fitted_gaussian_cycle',
                num_animations=1,
                num_frames=21
            )
        else:
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


class DisentanglementLoggingCallback(pl.Callback):
    
    def __init__(self, step_end_metrics=None, train_end_metrics=None, every_n_steps=1000, begin_first_step=False):
        self.begin_first_step = begin_first_step
        self.step_end_metrics = step_end_metrics if step_end_metrics else []
        self.train_end_metrics = train_end_metrics if train_end_metrics else []
        self.every_n_steps = every_n_steps
        assert isinstance(self.step_end_metrics, list)
        assert isinstance(self.train_end_metrics, list)
        assert self.step_end_metrics or self.train_end_metrics, 'No metrics given to step_end_metrics or train_end_metrics'

    def _compute_metrics_and_log(self, trainer: pl.Trainer, pl_module: pl.LightningModule, metrics: list, is_final=False):
        # TODO: re-enable
        # assert isinstance(pl_module, HydraSystem)
    
        # compute all metrics
        for metric in metrics:
            scores = metric(trainer.datamodule.dataset, lambda x: pl_module.encode(x.to(pl_module.device)))
            log.info(f'metric (step: {trainer.global_step}): {scores}')
            # log to wandb if it exists
            trainer.log_metrics({
                'final_metric' if is_final else 'epoch_metric': scores,
            }, {})
    
    def on_batch_end(self, trainer, pl_module):
        if self.step_end_metrics:
            # first epoch is 0, if we dont want the first one to be run we need to increment by 1
            if 0 == (trainer.global_step + int(not self.begin_first_step)) % self.every_n_steps:
                log.info('Computing epoch metrics:')
                self._compute_metrics_and_log(trainer, pl_module, metrics=self.step_end_metrics)
    
    def on_train_end(self, trainer, pl_module):
        if self.train_end_metrics:
            log.info('Computing final training run metrics:')
            self._compute_metrics_and_log(trainer, pl_module, metrics=self.train_end_metrics)


class LoggerProgressCallback(pl.Callback):
    def __init__(self, time_step=10):
        self.start_time = None
        self.last_time = 0
        self.time_step = time_step
        self.start_time = time.time()
    
    def sig(self, num, digits):
        return f'{num:.{digits}g}'
    
    def on_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        current_time = time.time()
        step_delta = current_time - self.last_time
        # only print every few seconds
        if self.time_step < step_delta:
            self.last_time = current_time
            # vars
            step, max_steps = trainer.batch_idx + 1, trainer.num_training_batches
            epoch, max_epoch = trainer.current_epoch + 1, trainer.max_epochs
            global_step, global_steps = trainer.global_step + 1, max_epoch * max_steps
            # computed
            train_pct = global_step / global_steps
            step_len, global_step_len = len(str(max_steps)), len(str(global_steps))
            # completion
            train_remain_time = (current_time - self.start_time) * (1 - train_pct) / train_pct
            # info dict
            info_dict = {k: f'{self.sig(v, 4)}' if isinstance(v, (int, float)) else f'{v}' for k, v in trainer.progress_bar_dict.items() if k != 'v_num'}
            sorted_k = sorted(info_dict.keys(), key=lambda k: ('loss' != k.lower(), 'loss' not in k.lower(), k))
            # log
            log.info(
                f'EPOCH: {epoch}/{max_epoch} - {global_step:0{global_step_len}d}/{global_steps} '
                f'({int(train_pct * 100):02d}%) [{int(train_remain_time)}s] '
                f'STEP: {step:{step_len}d}/{max_steps} ({int(step / max_steps * 100):02d}%) '
                f'| {" ".join(f"{k}={info_dict[k]}" for k in sorted_k)}'
            )

# ========================================================================= #
# END                                                                       #
# ========================================================================= #
