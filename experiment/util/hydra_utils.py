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
import inspect
import logging

import hydra
from omegaconf import DictConfig
from omegaconf import ListConfig
from omegaconf import OmegaConf


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


def make_non_strict(cfg: DictConfig):
    return OmegaConf.create({**cfg})


def merge_specializations(cfg: DictConfig, config_path: str, main_fn: callable, strict=True):
    # TODO: this should eventually be replaced with hydra recursive defaults
    # TODO: this makes config non-strict, allows setdefault to work even if key does not exist in config

    # skip if we do not have any specializations
    if 'specializations' not in cfg:
        return

    if not strict:
        # we allow overwrites & missing values to be inserted
        cfg = make_non_strict(cfg)

    # imports
    import os
    from hydra._internal.utils import detect_calling_file_or_module_from_task_function

    # get hydra config root
    calling_file, _, _ = detect_calling_file_or_module_from_task_function(main_fn)
    config_root = os.path.join(os.path.dirname(calling_file), config_path)

    # set and update specializations
    for group, specialization in cfg.specializations.items():
        assert group not in cfg, f'group={repr(group)} already exists on cfg, specialization merging is not supported!'
        log.info(f'merging specialization: {repr(specialization)}')
        # load specialization config
        specialization_cfg = OmegaConf.load(os.path.join(config_root, group, f'{specialization}.yaml'))
        # create new config
        cfg = OmegaConf.merge(cfg, {group: specialization_cfg})

    # remove specializations key
    del cfg['specializations']

    # done
    return cfg


# ========================================================================= #
# Make Target Helper                                                        #
# - from https://github.com/nmichlo/eunomia                                 #
# ========================================================================= #


def _fn_get_kwargs(func) -> dict:
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }


def _fn_get_args(func) -> list:
    signature = inspect.signature(func)
    return [
        k
        for k, v in signature.parameters.items()
        if v.default is inspect.Parameter.empty
    ]


def _fn_get_all_args(func) -> list:
    signature = inspect.signature(func)
    return list(signature.parameters.keys())


def _fn_get_module_path(obj):
    # get package search paths
    import os
    import sys
    # get module path
    path = os.path.abspath(inspect.getmodule(obj).__file__)
    # return the shortest relative path from all the packages
    rel_paths = []
    for site in sys.path:
        site = os.path.abspath(site)
        if os.path.commonprefix([site, path]) == site:
            rel_paths.append(os.path.relpath(path, site))
    # get shortest rel path
    rel_paths = sorted(rel_paths, key=str.__len__)
    assert len(rel_paths) > 0, f'no valid path found to: {obj}'
    # get the correct path
    path = rel_paths[0]
    assert path.endswith('.py')
    path = path[:-len('.py')]
    return os.path.normpath(path)


def _fn_get_import_path(obj):
    module_path = '.'.join(_fn_get_module_path(obj).split('/'))
    return f'{module_path}.{obj.__name__}'


# ========================================================================= #
# Make Target                                                               #
# - from https://github.com/nmichlo/eunomia                                 #
# ========================================================================= #


def make_target_dict_verbose(
        func,
        # target function
        params: dict = None,
        target: str = None,
        mode: str = 'any',
        keep_defaults: bool = True,
) -> dict:
    # get default values
    target = _fn_get_import_path(func) if target is None else target
    overrides = {} if params is None else params

    # if we should include all the non-overridden default
    # parameters in the final config
    if keep_defaults:
        defaults = _fn_get_kwargs(func)
    else:
        defaults = {}

    # get the parameters that can be overridden
    args, kwargs, all_args = _fn_get_args(func), _fn_get_kwargs(func), _fn_get_all_args(func)
    if mode == 'kwargs':
        allowed_overrides = set(kwargs)
    elif mode == 'any':
        allowed_overrides = set(args) | set(kwargs)
    elif mode == 'all':
        allowed_overrides = set(args) | set(kwargs)
    elif mode == 'unchecked':
        allowed_overrides = None
    else:
        raise KeyError(f'Invalid override mode: {repr(mode)}')

    # generate final list of parameter overrides
    if mode == 'unchecked':
        defaults.update(overrides)
        overrides = {}
    else:
        for k in all_args:  # sorted
            if k in allowed_overrides:
                if k in overrides:
                    defaults[k] = overrides.pop(k)

    # check that no extra unused overrides exist
    # and that _target_ is not a parameter name
    assert not overrides, f'cannot override params: {list(overrides.keys())}'
    assert '_target_' not in defaults, f'object {target} has conflicting optional parameter: "_target_"'

    # check that nothing is left over according to the override mode
    if mode == 'all':
        missed_args = set(args) - set(defaults)
        if missed_args:
            missed_args = [a for a in all_args if a in missed_args]  # sorted
            raise AssertionError(f'all non-default parameters require an override: {missed_args}')

    # TODO: this should be added to eunomia
    # re-sort defaults
    target_dict = {'_target_': target}
    # add args and kwargs
    for k in all_args:
        if k in defaults:
            assert k not in target_dict
            target_dict[k] = defaults.pop(k)
    # add unchecked kwargs
    for k, v in defaults.items():
        assert k not in target_dict
        target_dict[k] = v

    # return final dictionary
    return target_dict


def make_target_dict(func, **params):
    return make_target_dict_verbose(func, params=params, target=None, mode='any', keep_defaults=True)


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
