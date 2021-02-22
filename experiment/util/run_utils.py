from typing import Optional

import signal
import logging

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import LoggerCollection, WandbLogger, CometLogger


log = logging.getLogger(__name__)


# ========================================================================= #
# HYDRA CONFIG HELPERS                                                      #
# ========================================================================= #

_MISSING = object()

_PL_LOGGER: LoggerCollection = _MISSING
_PL_TRAINER: Trainer = _MISSING
_PL_CAPTURED_SIGNAL: bool = False


def set_debug_trainer(trainer: Optional[Trainer]):
    global _PL_TRAINER
    assert _PL_TRAINER is _MISSING, 'debug trainer has already been set'
    _PL_TRAINER = trainer
    return trainer


def _debug_logger_signal_handler(signal_number, frame):
    global _PL_CAPTURED_SIGNAL
    # remove callbacks from trainer so we aren't stuck running forever!
    # TODO: this is a hack! is this not a bug?
    if _PL_TRAINER is not None:
        _PL_TRAINER.callbacks.clear()
    # get the signal name
    numbers_to_names = dict((k, v) for v, k in reversed(sorted(signal.__dict__.items())) if v.startswith('SIG') and not v.startswith('SIG_'))
    signal_name = numbers_to_names.get(signal_number, signal_number)
    # log everything!
    log.error(f'Signal encountered: {signal_name} -- logging if possible!')
    log_debug_error(err_type=f'received signal: {signal_name}', err_msg=f'{signal_name}', err_occurred=True)
    # make sure we can't log another error message
    _PL_CAPTURED_SIGNAL = True
    # go through the various loggers and get urls if possible
    if _PL_LOGGER:
        for logger in _PL_LOGGER:
            print(logger)
            if isinstance(logger, WandbLogger):
                log.info(f'wandb: run url: {logger.experiment._get_run_url()}')
                log.info(f'wandb: project url: {logger.experiment._get_project_url()}')


def set_debug_logger(logger: Optional[LoggerCollection]):
    global _PL_LOGGER
    assert _PL_LOGGER is _MISSING, 'debug logger has already been set'
    _PL_LOGGER = logger
    # set initial message
    log_debug_error(err_type='N/A', err_msg='N/A', err_occurred=False)
    # register signal listeners -- we can't capture SIGKILL!
    signal.signal(signal.SIGINT, _debug_logger_signal_handler)    # interrupted from the dialogue station
    signal.signal(signal.SIGTERM, _debug_logger_signal_handler)   # terminate the process in a soft way
    signal.signal(signal.SIGABRT, _debug_logger_signal_handler)   # abnormal termination
    signal.signal(signal.SIGSEGV, _debug_logger_signal_handler)   # segmentation fault
    return logger


def log_debug_error(err_type: str, err_msg: str, err_occurred: bool):
    if _PL_CAPTURED_SIGNAL:
        log.warning(f'signal already captured, but tried to log error after this: err_type={repr(err_type)}, err_msg={repr(err_msg)}, err_occurred={repr(err_occurred)}')
        return
    if _PL_LOGGER is not None:
        _PL_LOGGER.log_metrics({
            'error_type': err_type,
            'error_msg': err_msg[:244] + ' <TRUNCATED>' if len(err_msg) > 244 else err_msg,
            'error_occurred': err_occurred,
        })


def log_error(logger, msg):
    if _PL_CAPTURED_SIGNAL:
        logger.warning(msg)
    else:
        logger.error(msg, exc_info=True)

# ========================================================================= #
# END                                                                       #
# ========================================================================= #
