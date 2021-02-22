import dataclasses
import os
import logging

import signal
from omegaconf import DictConfig, OmegaConf
import hydra

import pytorch_lightning as pl
import torch
import torch.utils.data
from pytorch_lightning.loggers import WandbLogger, CometLogger, LoggerCollection

from disent import metrics
from disent.model.ae.base import AutoEncoder
from disent.model.init import init_model_weights
from disent.util import make_box_str, wrapped_partial

from experiment.util.hydra_data import HydraDataModule
from experiment.util.callbacks import VaeDisentanglementLoggingCallback, VaeLatentCycleLoggingCallback, LoggerProgressCallback
from experiment.util.callbacks.callbacks_vae import VaeLatentCorrelationLoggingCallback
from experiment.util.hydra_utils import merge_specializations, make_non_strict
from experiment.util.run_utils import set_debug_logger, set_debug_trainer, log_error_and_exit


log = logging.getLogger(__name__)


# ========================================================================= #
# HYDRA CONFIG HELPERS                                                      #
# ========================================================================= #


def hydra_check_cuda(cfg):
    # set cuda
    if cfg.trainer.cuda in {'try_cuda', None}:
        cfg.trainer.cuda = torch.cuda.is_available()
        if not cfg.trainer.cuda:
            log.warning('CUDA was requested, but not found on this system... CUDA has been disabled!')
    else:
        if not torch.cuda.is_available():
            if cfg.trainer.cuda:
                log.error('trainer.cuda=True but CUDA is not available on this machine!')
                raise RuntimeError('CUDA not available!')
            else:
                log.warning('CUDA is not available on this machine!')
        else:
            if not cfg.trainer.cuda:
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

    # initialise logging dict
    cfg.setdefault('loggings', {})

    if ('wandb' in cfg.logging) and cfg.logging.wandb.setdefault('enabled', True):
        loggers.append(WandbLogger(
            offline=cfg.logging.wandb.setdefault('offline', False),
            entity=cfg.logging.wandb.setdefault('entity', None),  # cometml: workspace
            project=cfg.logging.wandb.project,                    # cometml: project_name
            name=cfg.logging.wandb.name,                          # cometml: experiment_name
            group=cfg.logging.wandb.setdefault('group', None),    # experiment group
            tags=cfg.logging.wandb.setdefault('tags', None),      # experiment tags
            save_dir=hydra.utils.to_absolute_path(cfg.logging.logs_dir),  # relative to hydra's original cwd
        ))
    else:
        cfg.logging.setdefault('wandb', dict(enabled=False))

    if ('cometml' in cfg.logging) and cfg.logging.cometml.setdefault('enabled', True):
        loggers.append(CometLogger(
            offline=cfg.logging.cometml.setdefault('offline', False),
            workspace=cfg.logging.cometml.setdefault('workspace', None),  # wandb: entity
            project_name=cfg.logging.cometml.project,                     # wandb: project
            experiment_name=cfg.logging.cometml.name,                     # wandb: name
            api_key=os.environ['COMET_API_KEY'],                          # TODO: use dotenv
            save_dir=hydra.utils.to_absolute_path(cfg.logging.logs_dir),  # relative to hydra's original cwd
        ))
    else:
        cfg.logging.setdefault('cometml', dict(enabled=False))

    return LoggerCollection(loggers) if loggers else None  # lists are turned into a LoggerCollection by pl


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
                wrapped_partial(metrics.metric_dci, num_train=1000, num_test=500, boost_mode='sklearn'),
                wrapped_partial(metrics.metric_factor_vae, num_train=1000, num_eval=500, num_variance_estimate=1000),
                wrapped_partial(metrics.metric_mig, num_train=2000),
                wrapped_partial(metrics.metric_sap, num_train=2000, num_test=1000),
                wrapped_partial(metrics.metric_unsupervised, num_train=2000),
            ],
            train_end_metrics=[
                # TODO: this needs to be configurable from the config
                metrics.metric_dci,
                metrics.metric_factor_vae,
                metrics.metric_mig,
                metrics.metric_sap,
                metrics.metric_unsupervised,
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
    cfg = make_non_strict(cfg)

    # -~-~-~-~-~-~-~-~-~-~-~-~- #
    # INITIALISE & SETDEFAULT IN CONFIG
    # -~-~-~-~-~-~-~-~-~-~-~-~- #

    # create trainer loggers & callbacks & initialise error messages
    logger = set_debug_logger(hydra_make_logger(cfg))

    # print useful info
    log.info(f"Current working directory : {os.getcwd()}")
    log.info(f"Orig working directory    : {hydra.utils.get_original_cwd()}")

    # hydra config does not support variables in defaults lists, we handle this manually
    cfg = merge_specializations(cfg, CONFIG_PATH, run)
    # create framework config - this is also kinda hacky
    framework_cfg = hydra.utils.instantiate({**cfg.framework.module, **dict(_target_=cfg.framework.module._target_ + '.cfg')})
    # update config params in case we missed variables in the cfg
    cfg.framework.module.update(dataclasses.asdict(framework_cfg))

    # check CUDA setting
    cfg.trainer.setdefault('cuda', 'try_cuda')
    hydra_check_cuda(cfg)

    # check data preparation
    prepare_data_per_node = cfg.trainer.setdefault('prepare_data_per_node', True)
    hydra_check_datadir(prepare_data_per_node, cfg)

    # TRAINER CALLBACKS
    callbacks = []
    hydra_append_progress_callback(callbacks, cfg)
    hydra_append_latent_cycle_logger_callback(callbacks, cfg)
    hydra_append_metric_callback(callbacks, cfg)
    hydra_append_correlation_callback(callbacks, cfg)

    # DATA
    datamodule = HydraDataModule(cfg)

    # FRAMEWORK - this is kinda hacky
    framework: pl.LightningModule = hydra.utils.instantiate(
        dict(_target_=cfg.framework.module._target_),
        make_optimizer_fn=lambda params: hydra.utils.instantiate(cfg.optimizer.cls, params),
        make_model_fn=lambda: init_model_weights(AutoEncoder(
            encoder=hydra.utils.instantiate(cfg.model.encoder),
            decoder=hydra.utils.instantiate(cfg.model.decoder)
        ), mode=cfg.model.weight_init),
        # apply augmentations to batch on GPU which can be faster than via the dataloader
        batch_augment=datamodule.batch_augment,
        cfg=framework_cfg
    )

    # Setup Trainer
    trainer = set_debug_trainer(pl.Trainer(
        log_every_n_steps=cfg.logging.setdefault('log_every_n_steps', 50),
        flush_logs_every_n_steps=cfg.logging.setdefault('flush_logs_every_n_steps', 100),
        logger=logger,
        callbacks=callbacks,
        gpus=1 if cfg.trainer.cuda else 0,
        max_epochs=cfg.trainer.setdefault('epochs', 100),
        max_steps=cfg.trainer.setdefault('steps', None),
        prepare_data_per_node=prepare_data_per_node,
        progress_bar_refresh_rate=0,  # ptl 0.9
        terminate_on_nan=True,  # we do this here so we don't run the final metrics
    ))

    # -~-~-~-~-~-~-~-~-~-~-~-~- #
    # BEGIN TRAINING
    # -~-~-~-~-~-~-~-~-~-~-~-~- #

    # print the config
    log.info(f'Final Config Is:\n{make_box_str(OmegaConf.to_yaml(cfg))}')

    # save hparams TODO: I think this is a pytorch lightning bug... The trainer should automatically save these if hparams is set.
    framework.hparams = cfg
    if trainer.logger:
        trainer.logger.log_hyperparams(framework.hparams)

    # fit the model
    # -- if an error/signal occurs while pytorch lightning is
    #    initialising the training process we cannot capture it!
    trainer.fit(framework, datamodule=datamodule)


# ========================================================================= #
# MAIN                                                                      #
# ========================================================================= #


if __name__ == '__main__':
    CONFIG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 'config'))
    CONFIG_NAME = 'config'

    @hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME)
    def main(cfg: DictConfig):
        try:
            run(cfg)
        except Exception as e:
            log_error_and_exit(err_type='experiment error', err_msg=str(e))

    try:
        main()
    except KeyboardInterrupt as e:
        log_error_and_exit(err_type='interrupted', err_msg=str(e), exc_info=False)
    except Exception as e:
        log_error_and_exit(err_type='hydra error', err_msg=str(e))


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
