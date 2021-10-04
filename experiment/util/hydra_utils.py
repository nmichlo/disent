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
from copy import deepcopy

import hydra
from omegaconf import DictConfig
from omegaconf import ListConfig
from omegaconf import OmegaConf

from disent.util.deprecate import deprecated


log = logging.getLogger(__name__)


# ========================================================================= #
# Recursive Hydra Instantiation                                             #
# TODO: use https://github.com/facebookresearch/hydra/pull/989              #
#       I think this is quicker? Just doesn't perform checks...             #
# ========================================================================= #


@deprecated('replace with hydra 1.1')
def call_recursive(config):
    # recurse
    def _call_recursive(config):
        if isinstance(config, (dict, DictConfig)):
            c = {k: _call_recursive(v) for k, v in config.items() if k != '_target_'}
            if '_target_' in config:
                config = hydra.utils.instantiate({'_target_': config['_target_']}, **c)
        elif isinstance(config, (tuple, list, ListConfig)):
            config = [_call_recursive(v) for v in config]
        return config
    return _call_recursive(config)


# alias
@deprecated('replace with hydra 1.1')
def instantiate_recursive(config):
    return call_recursive(config)


@deprecated('replace with hydra 1.1')
def instantiate_object_if_needed(config_or_object):
    if isinstance(config_or_object, dict):
        return instantiate_recursive(config_or_object)
    else:
        return config_or_object


# ========================================================================= #
# Better Specializations                                                    #
# TODO: this might be replaced by recursive instantiation                   #
#       https://github.com/facebookresearch/hydra/pull/1044                 #
# ========================================================================= #


@deprecated('replace with hydra 1.1')
def make_non_strict(cfg: DictConfig):
    cfg = deepcopy(cfg)
    return OmegaConf.create({**cfg})


@deprecated('replace with hydra 1.1')
def merge_specializations(cfg: DictConfig, config_path: str, strict=True):
    import os

    # TODO: this should eventually be replaced with hydra recursive defaults
    # TODO: this makes config non-strict, allows setdefault to work even if key does not exist in config

    assert os.path.isabs(config_path), f'config_path cannot be relative for merge_specializations: {repr(config_path)}, current working directory: {repr(os.getcwd())}'

    # skip if we do not have any specializations
    if 'specializations' not in cfg:
        log.warning('`specializations` key not found in `cfg`, skipping merging specializations')
        return

    # we allow overwrites & missing values to be inserted
    if not strict:
        cfg = make_non_strict(cfg)

    # set and update specializations
    for group, specialization in cfg.specializations.items():
        assert group not in cfg, f'group={repr(group)} already exists on cfg, specialization merging is not supported!'
        log.info(f'merging specialization: {repr(specialization)}')
        # load specialization config
        specialization_cfg = OmegaConf.load(os.path.join(config_path, group, f'{specialization}.yaml'))
        # create new config
        cfg = OmegaConf.merge(cfg, {group: specialization_cfg})

    # remove specializations key
    del cfg['specializations']

    # done
    return cfg


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
