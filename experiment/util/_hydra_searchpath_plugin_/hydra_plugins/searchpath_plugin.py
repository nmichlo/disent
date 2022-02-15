"""
This file is currently hacky, and hopefully temporary! See:
https://github.com/facebookresearch/hydra/issues/2001
"""

import logging
import os

from hydra.core.config_search_path import ConfigSearchPath
from hydra.plugins.search_path_plugin import SearchPathPlugin


log = logging.getLogger(__name__)


class DisentExperimentSearchPathPlugin(SearchPathPlugin):
    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        # find paths
        paths = [
            *os.environ.get('DISENT_CONFIGS_PREPEND', '').split(';'),
            os.path.abspath(os.path.join(__file__, '../../../..', 'config')),
            *os.environ.get('DISENT_CONFIGS_APPEND', '').split(';'),
        ]
        # add paths
        for path in paths:
            if path:
                log.info(f' [disent-search-path]: {path}')
                search_path.append(provider='disent-searchpath-plugin', path=os.path.abspath(path))
