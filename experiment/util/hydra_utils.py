import hydra
from omegaconf import ListConfig, DictConfig

# ========================================================================= #
# Recursive Hydra Instantiation                                             #
# TODO: use https://github.com/facebookresearch/hydra/pull/989              #
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
# END                                                                       #
# ========================================================================= #
