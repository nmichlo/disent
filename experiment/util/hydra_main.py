#  ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~
#  MIT License
#
#  Copyright (c) 2022 Nathan Juraj Michlo
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
import subprocess
import sys
from pathlib import Path
from typing import Callable
from typing import List
from typing import NoReturn
from typing import Optional
from typing import Union

import hydra
from omegaconf import OmegaConf
from omegaconf import DictConfig

from experiment.util.path_utils import get_current_experiment_number
from experiment.util.path_utils import make_current_experiment_dir
from experiment.util.run_utils import log_error_and_exit


log = logging.getLogger(__name__)


# ========================================================================= #
# VARS                                                                      #
# ========================================================================= #


# experiment/util/_hydra_searchpath_plugin_
PLUGIN_NAMESPACE = os.path.abspath(os.path.join(__file__, '..', '_hydra_searchpath_plugin_'))

# experiment/config
EXP_CONFIG_DIR = os.path.abspath(os.path.join(__file__, '../..', 'config'))

# list of configs
_DISENT_CONFIG_DIRS: List[str] = None


# ========================================================================= #
# PATCHING                                                                  #
# ========================================================================= #


def register_searchpath_plugin(
    search_dir_main: str = EXP_CONFIG_DIR,
    search_dirs_prepend: Optional[Union[str, List[str]]] = None,
    search_dirs_append:  Optional[Union[str, List[str]]] = None,
):
    """
    Patch Hydra:
    1. sets the default search path to `experiment/config`
    2. add to the search path with the `DISENT_CONFIGS_PREPEND` and `DISENT_CONFIGS_APPEND` environment variables
   NOTE: --config-dir has lower priority than all these, --config-path has higher priority.

    This function can safely be called multiple times
        - unless other functions modify these same libs which is unlikely!
    """
    # normalise the config paths
    if search_dirs_prepend is None: search_dirs_prepend = []
    if search_dirs_append is None:  search_dirs_append = []
    if isinstance(search_dirs_prepend, str): search_dirs_prepend = [search_dirs_prepend]
    if isinstance(search_dirs_append, str): search_dirs_append = [search_dirs_append]
    assert isinstance(search_dirs_prepend, (tuple, list)) and all((isinstance(d, str) and d) for d in search_dirs_prepend), f'`search_dirs_prepend` must be a list or tuple of non-empty path strings to directories, got: {repr(search_dirs_prepend)}'
    assert isinstance(search_dirs_append, (tuple, list)) and all((isinstance(d, str) and d) for d in search_dirs_append), f'`search_dirs_append` must be a list or tuple of non-empty path strings to directories, got: {repr(search_dirs_append)}'
    assert isinstance(search_dir_main, str) and search_dir_main, f'`search_dir_main` must be a non-empty path string to a directory, got: {repr(search_dir_main)}'
    # get dirs
    config_dirs = [*search_dirs_prepend, search_dir_main, *search_dirs_append]

    # check that it is the same as what has previously been registered, otherwise set the directories!
    global _DISENT_CONFIG_DIRS
    if _DISENT_CONFIG_DIRS is None:
        _DISENT_CONFIG_DIRS = config_dirs
    else:
        assert _DISENT_CONFIG_DIRS == config_dirs, f'Config dirs have already been registered, on additional calls, registered dirs must be the same as previously values!\n- existing: {_DISENT_CONFIG_DIRS}\n- registered: {config_dirs}'

    # register the experiment's search path plugin with disent, using hydras auto-detection
    # of folders named `hydra_plugins` contained insided `namespace packages` or rather
    # packages that are in the `PYTHONPATH` or `sys.path`
    #   1. sets the default search path to those registered above
    #   2. add to the search path with the `DISENT_CONFIGS_PREPEND` and `DISENT_CONFIGS_APPEND` environment variables
    #      NOTE: --config-dir has lower priority than all these, --config-path has higher priority.
    if PLUGIN_NAMESPACE not in sys.path:
        sys.path.insert(0, PLUGIN_NAMESPACE)


def register_hydra_resolvers():
    """
    Patch OmegaConf, enabling various config resolvers:
        - enable the ${exit:<msg>} resolver for omegaconf/hydra
        - enable the ${exp_num:<root_dir>} and ${exp_dir:<root_dir>,<name>} resolvers to detect the experiment number
        - enable the ${fmt:<strfmt>,<kwargs...>} resolver that wraps `str.format`
        - enable the ${abspath:<path>} resolver that wraps `hydra.utils.to_absolute_path` formatting relative paths in relation to the original working directory
        - enable the ${rsync_dir:<src>/<name>,<dst>/<name>} resolver that returns `<dst>/<name>`, but first rsync's the two directories!

    This function can safely be called multiple times
        - unless other functions modify these same libs which is unlikely!
    """


    # register a custom OmegaConf resolver that allows us to put in a ${exit:msg} that exits the program
    # - if we don't register this, the program will still fail because we have an unknown
    #   resolver. This just prettifies the output.
    if not OmegaConf.has_resolver('exit'):
        class ConfigurationError(Exception):
            pass
        # resolver function
        def _error_resolver(msg: str):
            raise ConfigurationError(msg)
        # patch omegaconf for hydra
        OmegaConf.register_new_resolver('exit', _error_resolver)

    # register a custom OmegaConf resolver that allows us to get the next experiment number from a directory
    # - ${run_num:<root_dir>} returns the current experiment number
    if not OmegaConf.has_resolver('exp_num'):
        OmegaConf.register_new_resolver('exp_num', get_current_experiment_number)
    # - ${run_dir:<root_dir>,<name>} returns the current experiment folder with the name appended
    if not OmegaConf.has_resolver('exp_dir'):
        OmegaConf.register_new_resolver('exp_dir', make_current_experiment_dir)

    # register a function that pads an integer to a specified length
    # - ${fmt:"{:04d}",42} -> "0042"
    if not OmegaConf.has_resolver('fmt'):
        OmegaConf.register_new_resolver('fmt', str.format)

    # register hydra helper functions
    # - ${abspath:<rel_path>} convert a relative path to an abs path using the original hydra working directory, not the changed experiment dir.
    if not OmegaConf.has_resolver('abspath'):
        OmegaConf.register_new_resolver('abspath', hydra.utils.to_absolute_path)

    # registry copy directory function
    # - useful if datasets are already prepared on a shared drive and need to be copied to a temp drive for example!
    if not OmegaConf.has_resolver('rsync_dir'):
        def rsync_dir(src: str, dst: str) -> str:
            src, dst = Path(src), Path(dst)
            # checks
            assert src.name and src.is_absolute(), f'src path must be absolute and not the root: {repr(str(src))}'
            assert dst.name and dst.is_absolute(), f'dst path must be absolute and not the root: {repr(str(dst))}'
            assert src.name == dst.name, f'src and dst paths must point to dirs with the same names: src.name={repr(src.name)}, dst.name={repr(dst.name)}'
            # synchronize dirs
            logging.info(f'rsync files:\n- src={repr(str(src))}\n- dst={repr(str(dst))}')
            # create the parent dir and copy files into the parent
            dst.parent.mkdir(parents=True, exist_ok=True)
            returncode = subprocess.Popen(['rsync', '-avh', str(src), str(dst.parent)]).wait()
            if returncode != 0:
                raise RuntimeError('Failed to rsync files!')
            # return the destination dir
            return str(dst)
        # REGISTER
        OmegaConf.register_new_resolver('rsync_dir', rsync_dir)


# ========================================================================= #
# RUN HYDRA                                                                 #
# ========================================================================= #


def patch_hydra(
    # config search path
    search_dir_main: str = EXP_CONFIG_DIR,
    search_dirs_prepend: Optional[Union[str, List[str]]] = None,
    search_dirs_append: Optional[Union[str, List[str]]] = None,
):
    # Patch Hydra and OmegaConf:
    register_searchpath_plugin(search_dir_main=search_dir_main, search_dirs_prepend=search_dirs_prepend, search_dirs_append=search_dirs_append)
    register_hydra_resolvers()


def hydra_main(
    callback: Callable[[DictConfig], NoReturn],
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
    # manually set log level before hydra initialises!
    if log_level is not None:
        logging.basicConfig(level=log_level)

    # Patch Hydra and OmegaConf:
    patch_hydra(search_dir_main=search_dir_main, search_dirs_prepend=search_dirs_prepend, search_dirs_append=search_dirs_append)

    @hydra.main(config_path=None, config_name=config_name)
    def _hydra_main(cfg: DictConfig):
        try:
            callback(cfg)
        except Exception as e:
            log_error_and_exit(err_type='experiment error', err_msg=str(e), exc_info=log_exc_info_callback)
        except:
            log_error_and_exit(err_type='experiment error', err_msg='<UNKNOWN>', exc_info=log_exc_info_callback)

    try:
        _hydra_main()
    except KeyboardInterrupt as e:
        log_error_and_exit(err_type='interrupted', err_msg=str(e), exc_info=False)
    except Exception as e:
        log_error_and_exit(err_type='hydra error', err_msg=str(e), exc_info=log_exc_info_hydra)
    except:
        log_error_and_exit(err_type='hydra error', err_msg='<UNKNOWN>', exc_info=log_exc_info_hydra)


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
