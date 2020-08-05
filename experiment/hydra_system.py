import os
import logging
import time

from omegaconf import DictConfig
import hydra
import numpy as np
import pytorch_lightning as pl
import torch
import torch.utils.data
import torchvision
import wandb
from pytorch_lightning.loggers import WandbLogger

from disent.dataset.single import GroundTruthDataset
from disent.frameworks.addon.msp import MatrixSubspaceProjection
from disent.frameworks.framework import BaseFramework
from disent.metrics import compute_dci, compute_factor_vae
from disent.model import GaussianAutoEncoder
from disent.util import TempNumpySeed, make_box_str
from disent.visualize.visualize_model import latent_cycle
from disent.visualize.visualize_util import gridify_animation, reconstructions_to_images


# ========================================================================= #
# runner                                                                    #
# ========================================================================= #


class HydraSystem(pl.LightningModule):

    def __init__(self, hparams: DictConfig):
        super().__init__()
        # hyper-parameters
        self.hparams = hparams
        # vae model
        self.model = GaussianAutoEncoder(
            hydra.utils.instantiate(self.hparams.model.encoder.cls),
            hydra.utils.instantiate(self.hparams.model.decoder.cls)
        )
        # TODO: THIS NEEDS TO BE MOVED ELSEWHERE:
        self.msp = MatrixSubspaceProjection(y_size=hparams.model.y_size, x_shape=hparams.dataset.x_shape, z_size=hparams.model.z_size)
        # framework
        try:
            self.framework: BaseFramework = hydra.utils.instantiate(self.hparams.framework.cls, msp=self.msp)
        except:
            self.framework: BaseFramework = hydra.utils.instantiate(self.hparams.framework.cls)
            self.msp = None
        # data
        self.dataset = None
        self.dataset_train = None

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
    # Initialisation                                                        #
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

    def prepare_data(self) -> None:
        # *NB* Do not set model parameters here.
        # - Instantiate data once to download and prepare if needed.
        # - trainer.prepare_data_per_node affects this functions behavior per node.
        if 'in_memory' in self.hparams.dataset.cls.params:
            hydra.utils.instantiate(self.hparams.dataset.cls, in_memory=False)
        else:
            hydra.utils.instantiate(self.hparams.dataset.cls)

    def setup(self, stage: str):
        # single observations
        self.dataset = GroundTruthDataset(
            ground_truth_data=hydra.utils.instantiate(self.hparams.dataset.cls),
            transform=torchvision.transforms.Compose([
                hydra.utils.instantiate(transform_cls)
                for transform_cls in self.hparams.dataset.transforms
            ])
        )
        # augment dataset if the framework requires
        self.dataset_train = self.dataset
        if 'augment_dataset' in self.hparams.framework:
            self.dataset_train = hydra.utils.instantiate(self.hparams.framework.augment_dataset.cls, self.dataset)

    def configure_optimizers(self):
        return hydra.utils.instantiate(
            self.hparams.optimizer.cls,
            self.model.parameters()
        )

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
    # Output                                                                #
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

    def forward(self, x):
        return self.model.forward(x)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
    # Training & Evaluation                                                 #
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

    def training_step(self, batch, batch_idx):
        loss_dict = self.framework.training_step(self.model, batch)
        # log & train
        return {
            'loss': loss_dict['train_loss'],
            'log': loss_dict,
            'progress_bar': loss_dict
        }

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
    # Training Dataset:
    #     The sample of data used to fit the model.
    # Validation Dataset:
    #     Data used to provide an unbiased evaluation of a model fit on the
    #     training dataset while tuning model hyperparameters. The
    #     evaluation becomes more biased as skill on the validation
    #     dataset is incorporated into the model configuration.
    # Test Dataset:
    #     The sample of data used to provide an unbiased evaluation of a
    #     final model fit on the training dataset.
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

    def train_dataloader(self):
        # Sample of data used to fit the model.
        return torch.utils.data.DataLoader(
            self.dataset_train,
            batch_size=self.hparams.dataset.batch_size,
            num_workers=self.hparams.dataset.num_workers,
            shuffle=True
        )


# ========================================================================= #
# CALLBACKS                                                                 #
# ========================================================================= #


class LatentCycleLoggingCallback(pl.Callback):

    def __init__(self, seed=7777, every_n_epochs=1, begin_first_epoch=False):
        self.seed = seed
        self.every_n_epochs = every_n_epochs
        self.begin_first_epoch = begin_first_epoch

    def on_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        # skip if need be
        if 0 != (trainer.current_epoch + int(not self.begin_first_epoch)) % self.every_n_epochs:
            return
        # VISUALISE!
        # generate and log latent traversals
        assert isinstance(pl_module, HydraSystem)
        # get random sample of z_means and z_logvars for computing the range of values for the latent_cycle
        with TempNumpySeed(self.seed):
            obs = pl_module.dataset.sample_observations(64).to(pl_module.device)
        z_means, z_logvars = pl_module.model.encode_gaussian(obs)
        # produce latent cycle animation & merge frames
        animation = latent_cycle(pl_module.model.reconstruct, z_means, z_logvars, mode='fitted_gaussian_cycle', num_animations=1, num_frames=21)
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
    
    def __init__(self, epoch_end_metrics=None, train_end_metrics=None, every_n_epochs=2, begin_first_epoch=False):
        self.begin_first_epoch = begin_first_epoch
        self.epoch_end_metrics = epoch_end_metrics if epoch_end_metrics else []
        self.train_end_metrics = train_end_metrics if train_end_metrics else []
        self.every_n_epochs = every_n_epochs
        assert isinstance(self.epoch_end_metrics, list)
        assert isinstance(self.train_end_metrics, list)
        assert self.epoch_end_metrics or self.train_end_metrics, 'No metrics given to epoch_end_metrics or train_end_metrics'

    def _compute_metrics_and_log(self, trainer, pl_module, metrics, is_final=False):
        # checks
        assert isinstance(pl_module, HydraSystem)
        # compute all metrics
        for metric in metrics:
            scores = metric(pl_module.dataset, lambda x: pl_module.model.encode_deterministic(x.to(pl_module.device)))
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
            global_step, global_steps = trainer.global_step + 1, max_epoch*max_steps
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
                f'({int(train_pct*100):02d}%) [{int(train_remain_time)}s] '
                f'STEP: {step:{step_len}d}/{max_steps} ({int(step/max_steps*100):02d}%) '
                f'| {" ".join(f"{k}={info_dict[k]}" for k in sorted_k)}'
            )

# ========================================================================= #
# RUNNER                                                                    #
# ========================================================================= #

@hydra.main(config_path='hydra_config', config_name="config")
def main(cfg: DictConfig):
    # print useful info
    log.info(make_box_str(cfg.pretty()))
    log.info(f"Current working directory : {os.getcwd()}")
    log.info(f"Orig working directory    : {hydra.utils.get_original_cwd()}")

    # warn about CUDA
    cuda = cfg.trainer.get('cuda', torch.cuda.is_available())
    if not torch.cuda.is_available():
        if cuda:
            log.error('trainer.cuda=True but CUDA is not available on this machine!')
            exit()
        else:
            log.warning('CUDA is not available on this machine!')
    else:
        if not cuda:
            log.warning('CUDA is available but is not being used!')

    # create trainer loggers & callbacks
    logger, callbacks = None, []
    if cfg.logging.wandb.enabled:
        log.info(f'wandb log directory: {os.path.abspath("wandb")}')
        logger = WandbLogger(
            name=cfg.logging.wandb.name,
            project=cfg.logging.wandb.project,
            # group=cfg.logging.wandb.get('group', None),
            tags=cfg.logging.wandb.get('tags', None),
            entity=cfg.logging.get('entity', None),
            save_dir=hydra.utils.to_absolute_path(cfg.logging.logs_dir),  # relative to hydra's original cwd
            offline=cfg.logging.wandb.get('offline', False),
        )
        if cfg.callbacks.latent_cycle.enabled:
            # Log the latent cycle visualisations to wandb
            callbacks.append(LatentCycleLoggingCallback(
                seed=cfg.callbacks.latent_cycle.seed,
                every_n_epochs=cfg.callbacks.latent_cycle.every_n_epochs,
                begin_first_epoch=False,
            ))

    # Log metric scores
    if cfg.callbacks.metrics.enabled:
        callbacks.append(DisentanglementLoggingCallback(
            every_n_epochs=cfg.callbacks.metrics.every_n_epochs,
            begin_first_epoch=False,
            epoch_end_metrics=[
                lambda dat, fn: compute_dci(dat, fn, 1000, 500, boost_mode='sklearn'),
                lambda dat, fn: compute_factor_vae(dat, fn, num_train=1000, num_eval=500, num_variance_estimate=1000),
            ],
            train_end_metrics=[
                compute_dci,
                compute_factor_vae
            ],
        ))
    
    # custom progress bar
    if cfg.callbacks.progress.enabled:
        callbacks.append(LoggerProgressCallback(
            time_step=cfg.callbacks.progress.step_time
        ))

    # check data preparation
    prepare_data_per_node = cfg.trainer.get('prepare_data_per_node', True)
    if not os.path.isabs(cfg.dataset.data_dir):
        log.warning(
            f'A relative path was specified for dataset.data_dir={repr(cfg.dataset.data_dir)}.'
            f' This is probably an error! Using relative paths can have unintended consequences'
            f' and performance drawbacks if the current working directory is on a shared/network drive.'
            f' Hydra config also uses a new working directory for each run of the program, meaning'
            f' data will be repeatedly downloaded.'
        )
        if prepare_data_per_node:
            log.error(
                f'trainer.prepare_data_per_node={repr(prepare_data_per_node)} but  dataset.data_dir='
                f'{repr(cfg.dataset.data_dir)} is a relative path which may be an error! Try specifying an'
                f' absolute path that is guaranteed to be unique from each node, eg. dataset.data_dir=/tmp/dataset'
            )
        raise RuntimeError('dataset.data_dir={repr(cfg.dataset.data_dir)} is a relative path!')

    # TRAIN

    system = HydraSystem(cfg)
    trainer = pl.Trainer(
        row_log_interval=cfg.logging.get('log_interval', 50),
        log_save_interval=cfg.logging.get('save_log_interval', 100),
        logger=logger,
        callbacks=callbacks,
        gpus=1 if cuda else 0,
        max_epochs=cfg.trainer.get('epochs', 100),
        max_steps=cfg.trainer.get('steps', None),
        prepare_data_per_node=prepare_data_per_node,
        progress_bar_refresh_rate=0,  # ptl 0.9
    )
    trainer.fit(system)
    # trainer resets the systems device

# ========================================================================= #
# LOGGING                                                                   #
# ========================================================================= #


log = logging.getLogger(os.path.splitext(os.path.basename(__file__))[0])


# ========================================================================= #
# MAIN                                                                      #
# ========================================================================= #

if __name__ == '__main__':
    try:
        main()
    except:
        log.error('A critical error occurred! Exiting safely...', exc_info=True)

# ========================================================================= #
# END                                                                       #
# ========================================================================= #
