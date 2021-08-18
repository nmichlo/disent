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

import sys
from multiprocessing import current_process
from typing import Optional

import signal
import logging

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import LoggerCollection

from disent.util.lightning.logger_util import wb_yield_loggers

log = logging.getLogger(__name__)


# ========================================================================= #
# HYDRA CONFIG HELPERS                                                      #
# ========================================================================= #

_PL_SIGNALS_OLD_HANDLERS = {}
_PL_SIGNALS = (      # we can't capture SIGKILL
    signal.SIGINT,   # interrupted from the dialogue station
    signal.SIGTERM,  # terminate the process in a soft way
    signal.SIGABRT,  # abnormal termination
    signal.SIGSEGV,  # segmentation fault
)

_PL_LOGGER: Optional[LoggerCollection] = None
_PL_TRAINER: Optional[Trainer] = None


def safe_unset_debug_trainer():
    global _PL_TRAINER
    if _PL_TRAINER is not None:
        _PL_TRAINER = None


def set_debug_trainer(trainer: Optional[Trainer]):
    global _PL_TRAINER
    assert _PL_TRAINER is None, 'debug trainer has already been set'
    _PL_TRAINER = trainer
    return trainer


def _signal_handler_log_and_exit(signal_number, frame):
    # call in all the child processes for the best chance of clearing this...
    # remove callbacks from trainer so we aren't stuck running forever!
    # TODO: this is a hack... there must be a better way to do this... could it be a pl bug?
    #       this logic is duplicated in the framework training_step
    if _PL_TRAINER and _PL_TRAINER.callbacks:
        _PL_TRAINER.callbacks.clear()

    # make sure that we only exit in the parent process
    if current_process().name != 'MainProcess':
        log.debug('Skipping signal handling for child process!')
        return
    # get the signal name
    numbers_to_names = dict((k, v) for v, k in reversed(sorted(signal.__dict__.items())) if v.startswith('SIG') and not v.startswith('SIG_'))
    signal_name = numbers_to_names.get(signal_number, signal_number)
    # log everything!
    log_error_and_exit(
        err_type=f'received exit signal',
        err_msg=f'{signal_name}',
        exit_code=signal_number,
        exc_info=False,
    )


def safe_unset_debug_logger():
    global _PL_LOGGER
    # unset logger
    if _PL_LOGGER is not None:
        _PL_LOGGER = None
        # return control to original handlers
        for signal_type in _PL_SIGNALS:
            if signal_type in _PL_SIGNALS_OLD_HANDLERS:
                handler = _PL_SIGNALS_OLD_HANDLERS.pop(signal_type)
                signal.signal(signal_type, handler)


def set_debug_logger(logger: Optional[LoggerCollection]):
    global _PL_LOGGER
    assert _PL_LOGGER is None, 'debug logger has already been set'
    _PL_LOGGER = logger
    # set initial messages
    if _PL_LOGGER is not None:
        _PL_LOGGER.log_metrics({
            'error_type': 'N/A',
            'error_msg': 'N/A',
            'error_occurred': False,
        })
    # register signal listeners
    for signal_type in _PL_SIGNALS:
        # save the old handler
        _PL_SIGNALS_OLD_HANDLERS[signal_type] = signal.getsignal(signal_type)
        # update the handler
        signal.signal(signal_type, _signal_handler_log_and_exit)
    # return the logger
    return logger


def log_error_and_exit(err_type: str, err_msg: str, exit_code: int = 1, exc_info=True):
    # truncate error
    err_msg = err_msg[:244] + ' <TRUNCATED>' if len(err_msg) > 244 else err_msg
    # log something at least
    log.error(f'exiting: {err_type} | {err_msg}', exc_info=exc_info)
    # try log to pytorch lightning & wandb
    if _PL_LOGGER is not None:
        _PL_LOGGER.log_metrics({
            'error_type': err_type,
            'error_msg': err_msg,
            'error_occurred': True,
        })
        for wb_logger in wb_yield_loggers(_PL_LOGGER):
            # so I dont have to scroll up... I'm lazy...
            run_url = wb_logger.experiment._get_run_url()
            project_url = wb_logger.experiment._get_project_url()
            log.error(f'wandb: run url: {run_url if run_url else "N/A"}')
            log.error(f'wandb: project url: {project_url if run_url else "N/A"}')
            # make sure we log everything online!
            wb_logger.experiment.finish(exit_code=exit_code)
    # EXIT!
    sys.exit(exit_code)


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
