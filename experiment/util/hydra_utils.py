import hydra
from omegaconf import ListConfig, DictConfig, OmegaConf
import logging

log = logging.getLogger(__name__)


# ========================================================================= #
# Recursive Hydra Instantiation                                             #
# TODO: use https://github.com/facebookresearch/hydra/pull/989              #
#       I think this is quicker? Just doesn't perform checks...             #
# ========================================================================= #


def call_recursive(config):
    if isinstance(config, (dict, DictConfig)):
        c = {k: call_recursive(v) for k, v in config.items() if k != '_target_'}
        if '_target_' in config:
            config = hydra.utils.instantiate({'_target_': config['_target_']}, **c)
    elif isinstance(config, (tuple, list, ListConfig)):
        config = [call_recursive(v) for v in config]
    return config

instantiate_recursive = call_recursive


# ========================================================================= #
# Better Specializations                                                    #
# TODO: this might be replaced by recursive instantiation                   #
#       https://github.com/facebookresearch/hydra/pull/1044                 #
# ========================================================================= #


def merge_specializations(cfg: DictConfig, config_path: str, main_fn: callable):
    # TODO: this should eventually be replaced with hydra recursive defaults

    # skip if we do not have any specializations
    if 'specializations' not in cfg:
        return

    # imports
    import os
    from hydra._internal.utils import detect_calling_file_or_module_from_task_function

    # get hydra config root
    calling_file, _, _ = detect_calling_file_or_module_from_task_function(main_fn)
    config_root = os.path.join(os.path.dirname(calling_file), config_path)

    # set and update specializations
    for group, specialization in cfg.specializations.items():
        assert group not in cfg, f'{group=} already exists on cfg, merging is not yet supported!'
        log.info(f'merging specialization: {repr(specialization)}')
        # load specialization config
        specialization_cfg = OmegaConf.load(os.path.join(config_root, group, f'{specialization}.yaml'))
        # create new config
        cfg = OmegaConf.create({**cfg, group: specialization_cfg})

    # remove specializations key
    del cfg['specializations']

    # done
    return cfg


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
