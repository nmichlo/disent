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
from typing import Callable
from typing import List
from typing import NoReturn
from typing import Optional
from typing import Union

import hydra
import pytorch_lightning as pl
import torch
import torch.utils.data
import wandb
from omegaconf import DictConfig
from omegaconf import ListConfig
from omegaconf import OmegaConf
from pytorch_lightning import Callback
from pytorch_lightning.callbacks import ModelSummary
from pytorch_lightning.loggers import LightningLoggerBase

import disent.registry as R
from disent.frameworks import DisentFramework
from disent.util.lightning.callbacks import VaeMetricLoggingCallback
from disent.util.seeds import seed
from disent.util.strings import colors as c
from disent.util.strings.fmt import make_box_str

from experiment.util.hydra_data import HydraDataModule
from experiment.util.hydra_main import EXP_CONFIG_DIR
from experiment.util.hydra_main import hydra_main
from experiment.util.run_utils import safe_unset_debug_logger
from experiment.util.run_utils import safe_unset_debug_trainer
from experiment.util.run_utils import set_debug_logger
from experiment.util.run_utils import set_debug_trainer


log = logging.getLogger(__name__)


# ========================================================================= #
# HYDRA CONFIG HELPERS                                                      #
# ========================================================================= #


def hydra_register_disent_plugins(cfg):
    # TODO: there should be a plugin mechanism for disent?
    if cfg.experiment.plugins:
        log.info('Running experiment plugins:')
        for plugin in cfg.experiment.plugins:
            log.info(f'* registering: {plugin}')
            hydra.utils.instantiate(dict(_target_=plugin))
    else:
        log.info('No experiment plugins were listed. Register these under the `experiment.plugins` in the config, which lists targets of functions.')


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
    prepare_data_per_node = cfg.datamodule.prepare_data_per_node
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
                f'datamodule.prepare_data_per_node={repr(prepare_data_per_node)} but dsettings.storage.data_root='
                f'{repr(data_root)} is a relative path which may be an error! Try specifying an'
                f' absolute path that is guaranteed to be unique from each node, eg. default_settings.storage.data_root=/tmp/dataset'
            )
        raise RuntimeError(f'default_settings.storage.data_root={repr(data_root)} is a relative path!')


def hydra_check_data_meta(cfg):
    # checks
    if (cfg.dataset.meta.vis_mean is None) or (cfg.dataset.meta.vis_std is None):
        log.warning(f'Dataset has no normalisation values... Are you sure this is correct?')
        log.warning(f'* dataset.meta.vis_mean: {cfg.dataset.meta.vis_mean}')
        log.warning(f'* dataset.meta.vis_std:  {cfg.dataset.meta.vis_std}')
    else:
        log.info(f'Dataset has normalisation values!')
        log.info(f'* dataset.meta.vis_mean: {cfg.dataset.meta.vis_mean}')
        log.info(f'* dataset.meta.vis_std:  {cfg.dataset.meta.vis_std}')


def hydra_make_logger(cfg) -> Optional[LightningLoggerBase]:
    logger = hydra.utils.instantiate(cfg.logging.logger)
    if logger:
        log.info(f'Initialised Logger: {logger}')
    else:
        log.warning(f'No Logger Utilised!')
    return logger


def hydra_get_callbacks(cfg) -> list:
    callbacks = []
    # add all callbacks
    for name, item in cfg.callbacks.items():
        # custom callback handling vs instantiation
        callback = hydra.utils.instantiate(item)
        assert isinstance(callback, Callback), f'instantiated callback is not an instance of {Callback}, got: {callback}'
        # add to callbacks list
        log.info(f'made callback: {name} ({item._target_})')
        callbacks.append(callback)
    return callbacks


def hydra_get_metric_callbacks(cfg) -> list:
    # TODO: simplify this, make better use of the config!
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
        train_metric = [R.METRICS[name].compute_fast] if settings.get('on_train', default_on_train) else None
        final_metric = [R.METRICS[name].compute]      if settings.get('on_final', default_on_final) else None
        # add the metric callback
        if final_metric or train_metric:
            callbacks.append(VaeMetricLoggingCallback(
                step_end_metrics  = train_metric,
                train_end_metrics = final_metric,
                every_n_steps     = settings.get('every_n_steps', default_every_n_steps),
                begin_first_step  = settings.get('begin_first_step', default_begin_first_step),
            ))
    return callbacks


def hydra_create_framework(cfg, gpu_batch_augment: Optional[Callable[[torch.Tensor], torch.Tensor]] = None) -> DisentFramework:
    # create framework
    assert str.endswith(cfg.framework.cfg['_target_'], '.cfg'), f'`cfg.framework.cfg._target_` does not end with ".cfg", got: {repr(cfg.framework.cfg["_target_"])}'
    framework_cls = hydra.utils.get_class(cfg.framework.cfg['_target_'][:-len(".cfg")])
    framework: DisentFramework = framework_cls(
        model=hydra.utils.instantiate(cfg.model.model_cls),
        cfg=hydra.utils.instantiate(cfg.framework.cfg, _convert_='all'),  # DisentConfigurable -- convert all OmegaConf objects to python equivalents, eg. DictConfig -> dict
        batch_augment=gpu_batch_augment,
    )

    # check if some cfg variables were not overridden
    missing_keys = sorted(set(framework.cfg.get_keys()) - (set(cfg.framework.cfg.keys())))
    if missing_keys:
        log.warning(f'{c.RED}Framework {repr(cfg.framework.name)} is missing config keys for:{c.RST}')
        for k in missing_keys:
            log.warning(f'{c.RED}{repr(k)}{c.RST}')

    # register schedules to the framework
    schedule_items = cfg.schedule.schedule_items
    assert isinstance(schedule_items, (dict, DictConfig)), f'`schedule.schedule_items` must be a dictionary, got type: {type(schedule_items)} with value: {repr(schedule_items)}'
    if schedule_items:
        log.info(f'Registering Schedules:')
        for target, schedule in schedule_items.items():
            framework.register_schedule(target, hydra.utils.instantiate(schedule), logging=True)

    return framework


def hydra_make_datamodule(cfg):
    return HydraDataModule(
        data                  = cfg.dataset.data,                    # from: dataset
        transform             = cfg.dataset.transform,               # from: dataset
        augment               = cfg.augment.augment_cls,             # from: augment
        sampler               = cfg.sampling._sampler_.sampler_cls,  # from: sampling
        # from: run_location
        using_cuda            = cfg.dsettings.trainer.cuda,
        dataloader_kwargs     = cfg.datamodule.dataloader,
        augment_on_gpu        = cfg.datamodule.gpu_augment,
        prepare_data_per_node = cfg.datamodule.prepare_data_per_node,
        # from: framework.meta
        return_indices        = cfg.framework.meta.get('requires_indices', False),
        return_factors        = cfg.framework.meta.get('requires_factors', False),
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
    # register plugins
    hydra_register_disent_plugins(cfg)
    # print useful info
    log.info(f"Current working directory : {os.getcwd()}")
    log.info(f"Orig working directory    : {hydra.utils.get_original_cwd()}")
    # check data preparation
    hydra_check_data_paths(cfg)
    hydra_check_data_meta(cfg)
    # print the config
    log.info(f'Dataset Config Is:\n{make_box_str(OmegaConf.to_yaml({"dataset": cfg.dataset}))}')
    # prepare data
    datamodule = hydra_make_datamodule(cfg)
    datamodule.prepare_data()


def action_train(cfg: DictConfig):

    # get the time the run started
    time_string = datetime.today().strftime('%Y-%m-%d--%H-%M-%S')
    log.info(f'Starting run at time: {time_string}')

    # -~-~-~-~-~-~-~-~-~-~-~-~- #
    # cleanup from old runs:
    # -~-~-~-~-~-~-~-~-~-~-~-~- #

    try:
        safe_unset_debug_trainer()
        safe_unset_debug_logger()
        wandb.finish()
    except:
        pass

    # -~-~-~-~-~-~-~-~-~-~-~-~- #
    # SETUP
    # -~-~-~-~-~-~-~-~-~-~-~-~- #

    # create trainer loggers & callbacks & initialise error messages
    logger = set_debug_logger(hydra_make_logger(cfg))

    # deterministic seed
    seed(cfg.settings.job.seed)
    # register plugins
    hydra_register_disent_plugins(cfg)
    # print useful info
    log.info(f"Current working directory : {os.getcwd()}")
    log.info(f"Orig working directory    : {hydra.utils.get_original_cwd()}")
    # checks
    gpus = hydra_get_gpus(cfg)
    hydra_check_data_paths(cfg)
    hydra_check_data_meta(cfg)

    # -~-~-~-~-~-~-~-~-~-~-~-~- #
    # INITIALISE
    # -~-~-~-~-~-~-~-~-~-~-~-~- #

    # HYDRA MODULES
    datamodule = hydra_make_datamodule(cfg)
    framework = hydra_create_framework(cfg, gpu_batch_augment=datamodule.gpu_batch_augment)

    # trainer default kwargs
    # Setup Trainer
    trainer = set_debug_trainer(pl.Trainer(
        # cannot override these
        logger=logger,
        gpus=gpus,
        callbacks=[
            *hydra_get_callbacks(cfg),
            *hydra_get_metric_callbacks(cfg),
            ModelSummary(max_depth=2),  # override default ModelSummary
        ],
        # additional kwargs from the config
        **{
            **dict(
                detect_anomaly=False,        # this should only be enabled for debugging torch and finding NaN values, slows down execution, not by much though?
                enable_checkpointing=False,  # TODO: enable this in future
            ),
            **cfg.trainer,  # overrides
        }
    ))

    # -~-~-~-~-~-~-~-~-~-~-~-~- #
    # DEBUG
    # -~-~-~-~-~-~-~-~-~-~-~-~- #

    # get config sections
    print_cfg, boxed_pop = dict(cfg), lambda *keys: make_box_str(OmegaConf.to_yaml({k: print_cfg.pop(k) for k in keys} if keys else print_cfg))
    cfg_str_exp      = boxed_pop('action', 'experiment')
    cfg_str_logging  = boxed_pop('logging', 'callbacks', 'metrics')
    cfg_str_dataset  = boxed_pop('dataset', 'datamodule', 'sampling', 'augment')
    cfg_str_system   = boxed_pop('framework', 'model', 'schedule')
    cfg_str_settings = boxed_pop('dsettings', 'settings')
    cfg_str_other    = boxed_pop()
    # print config sections
    log.info(f'Final Config For Action: {cfg.action}\n\nEXPERIMENT:{cfg_str_exp}\nLOGGING:{cfg_str_logging}\nDATASET:{cfg_str_dataset}\nSYSTEM:{cfg_str_system}\nTRAINER:{cfg_str_other}\nSETTINGS:{cfg_str_settings}')

    # -~-~-~-~-~-~-~-~-~-~-~-~- #
    # BEGIN TRAINING
    # -~-~-~-~-~-~-~-~-~-~-~-~- #

    # save hparams
    framework.hparams.update(cfg)
    if trainer.logger:
        trainer.logger.log_hyperparams(framework.hparams)  # TODO: is this a pytorch lightning bug? The trainer should automatically save these if hparams is set?

    # fit the model
    # -- if an error/signal occurs while pytorch lightning is
    #    initialising the training process we cannot capture it!
    trainer.fit(framework, datamodule=datamodule)

    # -~-~-~-~-~-~-~-~-~-~-~-~- #
    # cleanup this run
    # -~-~-~-~-~-~-~-~-~-~-~-~- #

    try:
        wandb.finish()
    except:
        pass


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


def hydra_experiment(
    callback: Callable[[DictConfig], NoReturn] = run_action,
    config_name: str = 'config',
    # config search path
    search_dir_main: str = EXP_CONFIG_DIR,
    search_dirs_prepend: Optional[Union[str, List[str]]] = None,
    search_dirs_append:  Optional[Union[str, List[str]]] = None,
    # logging
    log_level: Optional[int] = logging.INFO,
    log_exc_info_callback: bool = True,
    log_exc_info_hydra: bool = False,
):
    """
    Same as `hydra_main`, but with defaults to use
    with this experiment directory!
    """
    hydra_main(
        callback=callback,
        config_name=config_name,
        search_dir_main=search_dir_main,
        search_dirs_prepend=search_dirs_prepend,
        search_dirs_append=search_dirs_append,
        log_level=log_level,
        log_exc_info_callback=log_exc_info_callback,
        log_exc_info_hydra=log_exc_info_hydra,
    )


# launch the action
if __name__ == '__main__':
    hydra_experiment()


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
