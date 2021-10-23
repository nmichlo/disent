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
from typing import Optional
from typing import Sequence

import hydra
from omegaconf import DictConfig
from omegaconf import ListConfig
from omegaconf import OmegaConf

from disent.util.deprecate import deprecated


log = logging.getLogger(__name__)


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
def merge_specializations(cfg: DictConfig, config_path: str, strict=True, delete_key: bool = False, required: Optional[Sequence[str]] = None, force_package_mode: str = 'global'):
    import os

    # force the package mode
    # -- auto should be obtained from the config header, but this is not yet supported.
    assert force_package_mode in ['global'], f'invalid force_package_mode, must one of ["global"], got: {repr(force_package_mode)}. "auto" and "group" are not yet supported.'

    # TODO: this should eventually be replaced with hydra recursive defaults
    # TODO: this makes config non-strict, allows setdefault to work even if key does not exist in config

    assert os.path.isabs(config_path), f'config_path cannot be relative for merge_specializations: {repr(config_path)}, current working directory: {repr(os.getcwd())}'

    # skip if we do not have any specializations or handle requirements
    if required is not None:
        if 'specializations' not in cfg:
            raise RuntimeError(f'config does not contain the "specializations" key, required specializations include: {sorted(required)}')
        missing = set(required) - set(cfg.specializations.keys())
        if missing:
            raise RuntimeError(f'config does not contain the required specializations, missing keys for: {sorted(missing)}')
    else:
        if 'specializations' not in cfg:
            log.warning('`specializations` key not found in `cfg`, skipping merging specializations')
            return

    # we allow overwrites & missing values to be inserted
    if not strict:
        cfg = make_non_strict(cfg)

    # set and update specializations
    for group, specialization in cfg.specializations.items():
        log.info(f'merging specialization: {repr(specialization)}')
        # load specialization config
        specialization_cfg = OmegaConf.load(os.path.join(config_path, group, f'{specialization}.yaml'))
        # merge warnings
        conflicts = set(cfg.keys()) & set(specialization_cfg.keys())
        if conflicts:
            log.warning(f'- merging specialization has conflicting keys: {sorted(conflicts)}. This is probably an error!')
        # create new config
        cfg = OmegaConf.merge(cfg, specialization_cfg)

    # remove specializations key
    if delete_key:
        del cfg['specializations']

    # done
    return cfg


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
