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
from typing import List
from typing import Optional

from hydra.core.config_search_path import ConfigSearchPath
from hydra.plugins.search_path_plugin import SearchPathPlugin


log = logging.getLogger(__name__)


ENV_KEY = 'DISENT_CONFIG_ROOTS'


def _new__create_config_search_path(search_path_dir: Optional[str]) -> ConfigSearchPath:
    """
    This function is copied from: hydra._internal.utils.create_config_search_path
    -- we inject search paths before the main path so that they override what is there!

    THIS IS HACKEY! hydra should support this by default!
    -- maybe submit bug report or feature request!
    """
    from hydra.core.plugins import Plugins
    from hydra.plugins.search_path_plugin import SearchPathPlugin
    from hydra._internal.config_search_path_impl import ConfigSearchPathImpl

    search_path = ConfigSearchPathImpl()
    search_path.append("hydra", "pkg://hydra.conf")

    # CUSTOM ADDITION: inject our search path before the main path so that
    #                  anything that exists there overrides the experiment configs!
    logging.info(f'[hydra search path] * loading custom dirs from environment: {repr(ENV_KEY)}')
    custom_paths = os.environ.get(ENV_KEY, None)
    custom_paths = custom_paths.split(';') if custom_paths else []
    for path in custom_paths:
        path = os.path.abspath(path)
        logging.info(f'[hydra search path] - custom dir: {repr(path)}')
        search_path.append("disent", path)

    # TODO: traverse the directories and list overridden files?

    # add the main config
    logging.info(f'[hydra search path] - main dir:   {repr(search_path_dir)}')
    if search_path_dir is not None:
        search_path.append("main", search_path_dir)

    # add the plugins
    search_path_plugins = Plugins.instance().discover(SearchPathPlugin)
    for spp in search_path_plugins:
        plugin = spp()
        assert isinstance(plugin, SearchPathPlugin)
        plugin.manipulate_search_path(search_path)

    # add the schema
    search_path.append("schema", "structured://")

    return search_path


def inject_disent_search_path_finder():
    import hydra._internal.utils
    hydra._internal.utils.create_config_search_path = _new__create_config_search_path
    log.info('injected disent search path utility into hydra! This is hacky... submit a feature request!')
