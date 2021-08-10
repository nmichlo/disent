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
import warnings
from typing import Iterable
from typing import Optional

from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.loggers import LoggerCollection
from pytorch_lightning.loggers import WandbLogger


log = logging.getLogger(__name__)


# ========================================================================= #
# Logger Utils                                                              #
# ========================================================================= #


def log_metrics(logger: Optional[LightningLoggerBase], metrics_dct: dict):
    """
     Log the given values to the given logger.
     - warn the user if something goes wrong
    """
    if logger:
        try:
            logger.log_metrics(metrics_dct)
        except:
            warnings.warn(f'Failed to log metrics: {repr(metrics_dct)}')
    else:
        warnings.warn('no trainer.logger found!')


# ========================================================================= #
# W&B Logger Utils                                                          #
# ========================================================================= #


def wb_yield_loggers(logger: Optional[LightningLoggerBase]) -> Iterable[WandbLogger]:
    """
    Recursively yield all the loggers or sub-loggers that are an instance of WandbLogger
    """
    if logger:
        if isinstance(logger, WandbLogger):
            yield logger
        elif isinstance(logger, LoggerCollection):
            for l in logger:
                yield from wb_yield_loggers(l)


def wb_has_logger(logger: Optional[LightningLoggerBase]) -> bool:
    for l in wb_yield_loggers(logger):
        return True
    return False


def wb_log_metrics(logger: Optional[LightningLoggerBase], metrics_dct: dict):
    """
    Log the given values only to loggers that are an instance of WandbLogger
    """
    wb_logger = None
    # iterate over loggers & update metrics
    for wb_logger in wb_yield_loggers(logger):
        wb_logger.log_metrics(metrics_dct)
    # warn if nothing logged
    if wb_logger is None:
        warnings.warn('no wandb logger found to log metrics to!')


_SUMMARY_REDICTIONS = {
    'min': min,
    'max': max,
    'recent': lambda prev, current: current,
}


def wb_log_reduced_summaries(logger: Optional[LightningLoggerBase], summary_dct: dict, reduction='max'):
    """
    Aggregate the given values only to loggers that are an instance of WandbLogger
    - supported reduction modes are `"max"` and `"min"`
    """
    reduce_fn = _SUMMARY_REDICTIONS[reduction]
    wb_logger = None
    # iterate over loggers & update summaries
    for wb_logger in wb_yield_loggers(logger):
        for key, val_current in summary_dct.items():
            key = f'{key}.{reduction}'
            try:
                val_prev = wb_logger.experiment.summary.get(key, val_current)
                val_next = reduce_fn(val_prev, val_current)
                wb_logger.experiment.summary[key] = val_next
            except:
                log.error(f'W&B failed to update summary for: {repr(key)}', exc_info=True)
    # warn if nothing logged!
    if wb_logger is None:
        warnings.warn('no wandb logger found to log metrics to!')


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
