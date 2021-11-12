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

import logging
import os
from datetime import datetime

import hydra
import pytorch_lightning as pl
import torch
import torch.utils.data
import wandb
from omegaconf import DictConfig
from omegaconf import ListConfig
from omegaconf import OmegaConf
from pytorch_lightning.loggers import WandbLogger

from disent import metrics
from disent.frameworks import DisentConfigurable
from disent.frameworks import DisentFramework
from disent.model import AutoEncoder
from disent.nn.weights import init_model_weights
from disent.util.seeds import seed
from disent.util.strings.fmt import make_box_str
from disent.util.strings import colors as c
from disent.util.lightning.callbacks import LoggerProgressCallback
from disent.util.lightning.callbacks import VaeMetricLoggingCallback
from disent.util.lightning.callbacks import VaeLatentCycleLoggingCallback
from disent.util.lightning.callbacks import VaeGtDistsLoggingCallback
from experiment.util.hydra_data import HydraDataModule
from experiment.util.run_utils import log_error_and_exit
from experiment.util.run_utils import safe_unset_debug_logger
from experiment.util.run_utils import safe_unset_debug_trainer
from experiment.util.run_utils import set_debug_logger
from experiment.util.run_utils import set_debug_trainer


log = logging.getLogger(__name__)


# ========================================================================= #
# HYDRA CONFIG HELPERS                                                      #
# ========================================================================= #


def hydra_get_gpus(cfg) -> int:
    use_cuda = cfg.dsettings.trainer.cuda
    # check cuda values
    if use_cuda in {'try_cuda', None}:
        use_cuda = torch.cuda.is_available()
        if not use_cuda:
            log.warning('CUDA was requested, but not found on this system... CUDA has been disabled!')
    elif use_cuda:
        if not torch.cuda.is_available():
            log.error('`dsettings.trainer.cuda=True` but CUDA is not available on this machine!')
            raise RuntimeError('CUDA not available!')
    else:
        if not torch.cuda.is_available():
            log.info('CUDA is not available on this machine!')
        else:
            log.warning('CUDA is available but is not being used!')
    # get number of gpus to use
    return (1 if use_cuda else 0)


def hydra_check_data_paths(cfg):
    prepare_data_per_node = cfg.trainer.prepare_data_per_node
    data_root             = cfg.dsettings.storage.data_root
    # check relative paths
    if not os.path.isabs(data_root):
        log.warning(
            f'A relative path was specified for dsettings.storage.data_root={repr(data_root)}.'
            f' This is probably an error! Using relative paths can have unintended consequences'
            f' and performance drawbacks if the current working directory is on a shared/network drive.'
            f' Hydra config also uses a new working directory for each run of the program, meaning'
            f' data will be repeatedly downloaded.'
        )
        if prepare_data_per_node:
            log.error(
                f'trainer.prepare_data_per_node={repr(prepare_data_per_node)} but dsettings.storage.data_root='
                f'{repr(data_root)} is a relative path which may be an error! Try specifying an'
                f' absolute path that is guaranteed to be unique from each node, eg. default_settings.storage.data_root=/tmp/dataset'
            )
        raise RuntimeError(f'default_settings.storage.data_root={repr(data_root)} is a relative path!')


def hydra_make_logger(cfg):
    # make wandb logger
    backend = cfg.logging.wandb
    if backend.enabled:
        log.info('Initialising Weights & Biases Logger')
        return WandbLogger(
            offline=backend.offline,
            entity=backend.entity,    # cometml: workspace
            project=backend.project,  # cometml: project_name
            name=backend.name,        # cometml: experiment_name
            group=backend.group,      # experiment group
            tags=backend.tags,        # experiment tags
            save_dir=hydra.utils.to_absolute_path(cfg.dsettings.storage.logs_dir),  # relative to hydra's original cwd
        )
    # don't return a logger
    return None  # LoggerCollection([...]) OR DummyLogger(...)


def _callback_make_progress(cfg, callback_cfg):
    return LoggerProgressCallback(
        interval=callback_cfg.interval
    )


def _callback_make_latent_cycle(cfg, callback_cfg):
    if cfg.logging.wandb.enabled:
        # checks
        if not (('vis_min' in cfg.dataset and 'vis_max' in cfg.dataset) or ('vis_mean' in cfg.dataset and 'vis_std' in cfg.dataset)):
            log.warning('dataset does not have visualisation ranges specified, set `vis_min` & `vis_max` OR `vis_mean` & `vis_std`')
        # this currently only supports WANDB logger
        return VaeLatentCycleLoggingCallback(
            seed             = callback_cfg.seed,
            every_n_steps    = callback_cfg.every_n_steps,
            begin_first_step = callback_cfg.begin_first_step,
            mode             = callback_cfg.mode,
            # recon_min        = cfg.data.meta.vis_min,
            # recon_max        = cfg.data.meta.vis_max,
            recon_mean       = cfg.dataset.meta.vis_mean,
            recon_std        = cfg.dataset.meta.vis_std,
        )
    else:
        log.warning('latent_cycle callback is not being used because wandb is not enabled!')
        return None


def _callback_make_gt_dists(cfg, callback_cfg):
    return VaeGtDistsLoggingCallback(
        seed                 = callback_cfg.seed,
        every_n_steps        = callback_cfg.every_n_steps,
        traversal_repeats    = callback_cfg.traversal_repeats,
        begin_first_step     = callback_cfg.begin_first_step,
        plt_block_size       = 1.25,
        plt_show             = False,
        plt_transpose        = False,
        log_wandb            = True,
        batch_size           = cfg.settings.dataset.batch_size,
        include_factor_dists = True,
    )


_CALLBACK_MAKERS = {
    'progress': _callback_make_progress,
    'latent_cycle': _callback_make_latent_cycle,
    'gt_dists': _callback_make_gt_dists,
}


def hydra_get_callbacks(cfg) -> list:
    callbacks = []
    # add all callbacks
    for name, item in cfg.callbacks.items():
        # custom callback handling vs instantiation
        if '_target_' in item:
            name = f'{name} ({item._target_})'
            callback = hydra.utils.instantiate(item)
        else:
            callback = _CALLBACK_MAKERS[name](cfg, item)
        # add to callbacks list
        if callback is not None:
            log.info(f'made callback: {name}')
            callbacks.append(callback)
        else:
            log.info(f'skipped callback: {name}')
    return callbacks


def hydra_get_metric_callbacks(cfg) -> list:
    callbacks = []
    # set default values used later
    default_every_n_steps    = cfg.metrics.default_every_n_steps
    default_on_final         = cfg.metrics.default_on_final
    default_on_train         = cfg.metrics.default_on_train
    default_begin_first_step = cfg.metrics.default_begin_first_step
    # get metrics
    metric_list = cfg.metrics.metric_list
    assert isinstance(metric_list, (list, ListConfig)), f'`metrics.metric_list` is not a list, got: {type(metric_list)}'
    # get metrics
    for metric in metric_list:
        assert isinstance(metric, (dict, DictConfig)), f'entry in metric list is not a dictionary, got type: {type(metric)} or value: {repr(metric)}'
        # fix the values
        if isinstance(metric, str):
            metric = {metric: {}}
        ((name, settings),) = metric.items()
        # check values
        assert isinstance(metric, (dict, DictConfig)), f'settings for entry in metric list is not a dictionary, got type: {type(settings)} or value: {repr(settings)}'
        # make metrics
        train_metric = [metrics.FAST_METRICS[name]]    if settings.get('on_train', default_on_train) else None
        final_metric = [metrics.DEFAULT_METRICS[name]] if settings.get('on_final', default_on_final) else None
        # add the metric callback
        if final_metric or train_metric:
            callbacks.append(VaeMetricLoggingCallback(
                step_end_metrics  = train_metric,
                train_end_metrics = final_metric,
                every_n_steps     = settings.get('every_n_steps', default_every_n_steps),
                begin_first_step  = settings.get('begin_first_step', default_begin_first_step),
            ))
    return callbacks


def hydra_register_schedules(module: DisentFramework, cfg):
    # check the type
    schedule_items = cfg.schedule.schedule_items
    assert isinstance(schedule_items, (dict, DictConfig)), f'`schedule.schedule_items` must be a dictionary, got type: {type(schedule_items)} with value: {repr(schedule_items)}'
    # add items
    if schedule_items:
        log.info(f'Registering Schedules:')
        for target, schedule in schedule_items.items():
            module.register_schedule(target, hydra.utils.instantiate(schedule), logging=True)


def hydra_create_and_update_framework_config(cfg) -> DisentConfigurable.cfg:
    # create framework config - this is also kinda hacky
    # - we need instantiate_recursive because of optimizer_kwargs,
    #   otherwise the dictionary is left as an OmegaConf dict
    framework_cfg: DisentConfigurable.cfg = hydra.utils.instantiate(cfg.framework.cfg)
    # warn if some of the cfg variables were not overridden
    missing_keys = sorted(set(framework_cfg.get_keys()) - (set(cfg.framework.cfg.keys())))
    if missing_keys:
        log.warning(f'{c.RED}Framework {repr(cfg.framework.name)} is missing config keys for:{c.RST}')
        for k in missing_keys:
            log.warning(f'{c.RED}{repr(k)}{c.RST}')
    # return config
    return framework_cfg


def hydra_create_framework(framework_cfg: DisentConfigurable.cfg, datamodule, cfg):
    # specific handling for experiment, this is HACKY!
    # - not supported normally, we need to instantiate to get the class (is there hydra support for this?)
    framework_cfg.optimizer        = hydra.utils.get_class(framework_cfg.optimizer)
    framework_cfg.optimizer_kwargs = dict(framework_cfg.optimizer_kwargs)
    # get framework path
    assert str.endswith(cfg.framework.cfg._target_, '.cfg'), f'`cfg.framework.cfg._target_` does not end with ".cfg", got: {repr(cfg.framework.cfg._target_)}'
    framework_cls = hydra.utils.get_class(cfg.framework.cfg._target_[:-len(".cfg")])
    # create model
    model = AutoEncoder(
        encoder=hydra.utils.instantiate(cfg.model.encoder_cls),
        decoder=hydra.utils.instantiate(cfg.model.decoder_cls),
    )
    # initialise the model
    model = init_model_weights(model, mode=cfg.settings.model.weight_init)
    # create framework
    return framework_cls(
        model=model,
        cfg=framework_cfg,
        batch_augment=datamodule.batch_augment,  # apply augmentations to batch on GPU which can be faster than via the dataloader
    )


# ========================================================================= #
# ACTIONS                                                                   #
# ========================================================================= #


def action_prepare_data(cfg: DictConfig):
    # get the time the run started
    time_string = datetime.today().strftime('%Y-%m-%d--%H-%M-%S')
    log.info(f'Starting run at time: {time_string}')
    # deterministic seed
    seed(cfg.settings.job.seed)
    # print useful info
    log.info(f"Current working directory : {os.getcwd()}")
    log.info(f"Orig working directory    : {hydra.utils.get_original_cwd()}")
    # check data preparation
    hydra_check_data_paths(cfg)
    # print the config
    log.info(f'Dataset Config Is:\n{make_box_str(OmegaConf.to_yaml({"dataset": cfg.dataset}))}')
    # prepare data
    datamodule = HydraDataModule(cfg)
    datamodule.prepare_data()


def action_train(cfg: DictConfig):

    # get the time the run started
    time_string = datetime.today().strftime('%Y-%m-%d--%H-%M-%S')
    log.info(f'Starting run at time: {time_string}')

    # -~-~-~-~-~-~-~-~-~-~-~-~- #

    # cleanup from old runs:
    try:
        safe_unset_debug_trainer()
        safe_unset_debug_logger()
        wandb.finish()
    except:
        pass

    # -~-~-~-~-~-~-~-~-~-~-~-~- #

    # deterministic seed
    seed(cfg.settings.job.seed)

    # -~-~-~-~-~-~-~-~-~-~-~-~- #
    # INITIALISE & SETDEFAULT IN CONFIG
    # -~-~-~-~-~-~-~-~-~-~-~-~- #

    # create trainer loggers & callbacks & initialise error messages
    logger = set_debug_logger(hydra_make_logger(cfg))

    # print useful info
    log.info(f"Current working directory : {os.getcwd()}")
    log.info(f"Orig working directory    : {hydra.utils.get_original_cwd()}")

    # check CUDA setting
    gpus = hydra_get_gpus(cfg)

    # check data preparation
    hydra_check_data_paths(cfg)

    # TRAINER CALLBACKS
    callbacks = [
        *hydra_get_callbacks(cfg),
        *hydra_get_metric_callbacks(cfg),
    ]

    # HYDRA MODULES
    datamodule = HydraDataModule(cfg)
    framework_cfg = hydra_create_and_update_framework_config(cfg)
    framework = hydra_create_framework(framework_cfg, datamodule, cfg)

    # register schedules
    hydra_register_schedules(framework, cfg)

    # Setup Trainer
    trainer = set_debug_trainer(pl.Trainer(
        logger=logger,
        callbacks=callbacks,
        gpus=gpus,
        # we do this here too so we don't run the final
        # metrics, even through we check for it manually.
        terminate_on_nan=True,
        # TODO: re-enable this in future... something is not compatible
        #       with saving/checkpointing models + allow enabling from the
        #       config. Seems like something cannot be pickled?
        checkpoint_callback=False,
        # additional trainer kwargs
        **cfg.trainer,
    ))

    # -~-~-~-~-~-~-~-~-~-~-~-~- #
    # BEGIN TRAINING
    # -~-~-~-~-~-~-~-~-~-~-~-~- #

    # get config sections
    print_cfg, boxed_pop = dict(cfg), lambda *keys: make_box_str(OmegaConf.to_yaml({k: print_cfg.pop(k) for k in keys} if keys else print_cfg))
    cfg_str_logging  = boxed_pop('logging', 'callbacks', 'metrics')
    cfg_str_dataset  = boxed_pop('dataset', 'sampling', 'augment')
    cfg_str_system   = boxed_pop('framework', 'model', 'schedule')
    cfg_str_settings = boxed_pop('dsettings', 'settings')
    cfg_str_other    = boxed_pop()
    # print config sections
    log.info(f'Final Config For Action: {cfg.action}\n\nLOGGING:{cfg_str_logging}\nDATASET:{cfg_str_dataset}\nSYSTEM:{cfg_str_system}\nTRAINER:{cfg_str_other}\nSETTINGS:{cfg_str_settings}')

    # save hparams TODO: is this a pytorch lightning bug? The trainer should automatically save these if hparams is set?
    framework.hparams.update(cfg)
    if trainer.logger:
        trainer.logger.log_hyperparams(framework.hparams)

    # fit the model
    # -- if an error/signal occurs while pytorch lightning is
    #    initialising the training process we cannot capture it!
    trainer.fit(framework, datamodule=datamodule)

    # -~-~-~-~-~-~-~-~-~-~-~-~- #

    # cleanup this run
    try:
        wandb.finish()
    except:
        pass

    # -~-~-~-~-~-~-~-~-~-~-~-~- #


# available actions
ACTIONS = {
    'prepare_data': action_prepare_data,
    'train': action_train,
    'skip': lambda *args, **kwargs: None,
}


def run_action(cfg: DictConfig):
    action_key = cfg.action
    # get the action
    if action_key not in ACTIONS:
        raise KeyError(f'The given action: {repr(action_key)} is invalid, must be one of: {sorted(ACTIONS.keys())}')
    action = ACTIONS[action_key]
    # run the action
    action(cfg)


# ========================================================================= #
# MAIN                                                                      #
# ========================================================================= #


# path to root directory containing configs
CONFIG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 'config'))
# root config existing inside `CONFIG_ROOT`, with '.yaml' appended.
CONFIG_NAME = 'config'


if __name__ == '__main__':

    # register a custom OmegaConf resolver that allows us to put in a ${exit:msg} that exits the program
    # - if we don't register this, the program will still fail because we have an unknown
    #   resolver. This just prettifies the output.
    class ConfigurationError(Exception):
        pass

    def _error_resolver(msg: str):
        raise ConfigurationError(msg)

    OmegaConf.register_new_resolver('exit', _error_resolver)

    @hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME)
    def hydra_main(cfg: DictConfig):
        try:
            run_action(cfg)
        except Exception as e:
            log_error_and_exit(err_type='experiment error', err_msg=str(e), exc_info=True)
        except:
            log_error_and_exit(err_type='experiment error', err_msg='<UNKNOWN>', exc_info=True)

    try:
        hydra_main()
    except KeyboardInterrupt as e:
        log_error_and_exit(err_type='interrupted', err_msg=str(e), exc_info=False)
    except Exception as e:
        log_error_and_exit(err_type='hydra error', err_msg=str(e), exc_info=True)
    except:
        log_error_and_exit(err_type='hydra error', err_msg='<UNKNOWN>', exc_info=True)


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
