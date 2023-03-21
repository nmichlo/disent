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
        from experiment.util.hydra_main import _DISENT_CONFIG_DIRS

        # find paths
        paths = [
            *os.environ.get("DISENT_CONFIGS_PREPEND", "").split(";"),
            *_DISENT_CONFIG_DIRS,
            *os.environ.get("DISENT_CONFIGS_APPEND", "").split(";"),
        ]
        # print information
        log.info(f" [disent-search-path-plugin]: Activated hydra plugin: {self.__class__.__name__}")
        log.info(
            f" [disent-search-path-plugin]: To register more search paths, adjust the `DISENT_CONFIGS_PREPEND` and `DISENT_CONFIGS_APPEND` environment variables!"
        )
        # add paths
        for path in paths:
            if path:
                log.info(f" [disent-search-path] - {repr(path)}")
                search_path.append(provider="disent-searchpath-plugin", path=os.path.abspath(path))
