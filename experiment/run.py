import os
import logging
from omegaconf import DictConfig, OmegaConf
import hydra

import pytorch_lightning as pl
import torch
import torch.utils.data
from pytorch_lightning.loggers import WandbLogger, CometLogger

from disent import metrics
from disent.model.ae.base import GaussianAutoEncoder
from disent.util import make_box_str

from experiment.hydra_data import HydraDataModule
from experiment.util.callbacks import VaeDisentanglementLoggingCallback, VaeLatentCycleLoggingCallback, LoggerProgressCallback
from experiment.util.callbacks.callbacks_vae import VaeLatentCorrelationLoggingCallback


log = logging.getLogger(__name__)


# ========================================================================= #
# HYDRA CONFIG HELPERS                                                      #
# ========================================================================= #


def hydra_check_cuda(cuda):
    if not torch.cuda.is_available():
        if cuda:
            log.error('trainer.cuda=True but CUDA is not available on this machine!')
            exit()
        else:
            log.warning('CUDA is not available on this machine!')
    else:
        if not cuda:
            log.warning('CUDA is available but is not being used!')


def hydra_check_datadir(prepare_data_per_node, cfg):
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


def hydra_make_logger(cfg):
    loggers = []
    if ('wandb' in cfg.logging) and cfg.logging.wandb.get('enabled', True):
        loggers.append(WandbLogger(
            offline=cfg.logging.wandb.get('offline', False),
            entity=cfg.logging.wandb.get('entity', None),  # cometml: workspace
            project=cfg.logging.wandb.project,             # cometml: project_name
            name=cfg.logging.wandb.name,                   # cometml: experiment_name
            group=cfg.logging.wandb.get('group', None),    # experiment group
            tags=cfg.logging.wandb.get('tags', None),      # experiment tags
            save_dir=hydra.utils.to_absolute_path(cfg.logging.logs_dir),  # relative to hydra's original cwd
        ))
    if ('cometml' in cfg.logging) and cfg.logging.cometml.get('enabled', True):
        loggers.append(CometLogger(
            offline=cfg.logging.cometml.get('offline', False),
            workspace=cfg.logging.cometml.get('workspace', None),  # wandb: entity
            project_name=cfg.logging.cometml.project,              # wandb: project
            experiment_name=cfg.logging.cometml.name,              # wandb: name
            api_key=os.environ['COMET_API_KEY'],                   # TODO: use dotenv
            save_dir=hydra.utils.to_absolute_path(cfg.logging.logs_dir),  # relative to hydra's original cwd
        ))
    return loggers if loggers else None  # lists are turned into a LoggerCollection by pl


def hydra_append_progress_callback(callbacks, cfg):
    if 'progress' in cfg.callbacks:
        callbacks.append(LoggerProgressCallback(
            interval=cfg.callbacks.progress.interval
        ))


def hydra_append_latent_cycle_logger_callback(callbacks, cfg):
    if 'latent_cycle' in cfg.callbacks:
        if cfg.logging.wandb.enabled:
            # this currently only supports WANDB logger
            callbacks.append(VaeLatentCycleLoggingCallback(
                seed=cfg.callbacks.latent_cycle.seed,
                every_n_steps=cfg.callbacks.latent_cycle.every_n_steps,
                begin_first_step=False,
                mode=cfg.callbacks.latent_cycle.mode,
            ))
        else:
            log.warning('latent_cycle callback is not being used because wandb is not enabled!')


def hydra_append_metric_callback(callbacks, cfg):
    if 'metrics' in cfg.callbacks:
        callbacks.append(VaeDisentanglementLoggingCallback(
            every_n_steps=cfg.callbacks.metrics.every_n_steps,
            begin_first_step=False,
            step_end_metrics=[
                # TODO: this needs to be configurable from the config
                lambda dat, fn: metrics.compute_dci(dat, fn, num_train=1000, num_test=500, boost_mode='sklearn'),
                lambda dat, fn: metrics.compute_factor_vae(dat, fn, num_train=1000, num_eval=500, num_variance_estimate=1000),
                lambda dat, fn: metrics.compute_mig(dat, fn, num_train=1000),
                lambda dat, fn: metrics.compute_sap(dat, fn, num_train=1000, num_test=500),
                lambda dat, fn: metrics.compute_unsupervised(dat, fn, num_train=1000),
            ],
            train_end_metrics=[
                # TODO: this needs to be configurable from the config
                metrics.compute_dci,
                metrics.compute_factor_vae,
                metrics.compute_mig,
                metrics.compute_sap,
                metrics.compute_unsupervised,
            ],
        ))


def hydra_append_correlation_callback(callbacks, cfg):
    if 'correlation' in cfg.callbacks:
        callbacks.append(VaeLatentCorrelationLoggingCallback(
            repeats_per_factor=cfg.callbacks.correlation.repeats_per_factor,
            every_n_steps=cfg.callbacks.correlation.every_n_steps,
            begin_first_step=False,
        ))


# ========================================================================= #
# RUNNER                                                                    #
# ========================================================================= #


def run(cfg: DictConfig):
    # print useful info
    log.info(make_box_str(OmegaConf.to_yaml(cfg)))
    log.info(f"Current working directory : {os.getcwd()}")
    log.info(f"Orig working directory    : {hydra.utils.get_original_cwd()}")

    # check CUDA setting
    cuda = cfg.trainer.get('cuda', torch.cuda.is_available())
    hydra_check_cuda(cuda)

    # check data preparation
    prepare_data_per_node = cfg.trainer.get('prepare_data_per_node', True)
    hydra_check_datadir(prepare_data_per_node, cfg)

    # create trainer loggers & callbacks
    logger = hydra_make_logger(cfg)

    # TRAINER CALLBACKS
    callbacks = []
    hydra_append_progress_callback(callbacks, cfg)
    hydra_append_latent_cycle_logger_callback(callbacks, cfg)
    hydra_append_metric_callback(callbacks, cfg)
    hydra_append_correlation_callback(callbacks, cfg)

    # DATA
    datamodule = HydraDataModule(cfg)

    # FRAMEWORK
    framework: pl.LightningModule = hydra.utils.instantiate(
        cfg.framework.module,
        make_optimizer_fn=lambda params: hydra.utils.instantiate(cfg.optimizer.cls, params),
        make_model_fn=lambda: GaussianAutoEncoder(
            encoder=hydra.utils.instantiate(cfg.model.encoder),
            decoder=hydra.utils.instantiate(cfg.model.decoder)
        ),
        # apply augmentations to batch on GPU which can be faster than via the dataloader
        batch_augment=datamodule.batch_augment
    )

    # Setup Trainer
    trainer = pl.Trainer(
        log_every_n_steps=cfg.logging.get('log_every_n_steps', 50),
        flush_logs_every_n_steps=cfg.logging.get('flush_logs_every_n_steps', 100),
        logger=logger,
        callbacks=callbacks,
        gpus=1 if cuda else 0,
        max_epochs=cfg.trainer.get('epochs', 100),
        max_steps=cfg.trainer.get('steps', None),
        prepare_data_per_node=prepare_data_per_node,
        progress_bar_refresh_rate=0,  # ptl 0.9
    )

    # save hparams TODO: I think this is a pytorch lightning bug... The trainer should automatically save these if hparams is set.
    framework.hparams = cfg
    trainer.logger.log_hyperparams(framework.hparams)

    # fit the model
    trainer.fit(framework, datamodule=datamodule)


@hydra.main(config_path='config', config_name="config")
def main(cfg: DictConfig):
    try:
        run(cfg)
    except:
        log.error('A critical error occurred!', exc_info=True)


# ========================================================================= #
# MAIN                                                                      #
# ========================================================================= #


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt as e:
        log.warning('Interrupted - Exited early!')


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
