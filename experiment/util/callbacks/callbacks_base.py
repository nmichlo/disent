import logging
import time
import pytorch_lightning as pl


log = logging.getLogger(__name__)


# ========================================================================= #
# HELPER CALLBACKS                                                          #
# ========================================================================= #


class _PeriodicCallback(pl.Callback):
    
    def __init__(self, every_n_steps=None, begin_first_step=False):
        self.every_n_steps = every_n_steps
        self.begin_first_step = begin_first_step

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if self.every_n_steps is None:
            # number of steps/batches in an epoch
            self.every_n_steps = trainer.num_training_batches
            
    def on_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        # skip if need be
        if 0 == (trainer.global_step + int(not self.begin_first_step)) % self.every_n_steps:
            self.do_step(trainer, pl_module)

    def do_step(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        raise NotImplementedError


class _TimedCallback(pl.Callback):
    
    def __init__(self, interval=10):
        self._last_time = 0
        self._interval = interval
        self._start_time = time.time()

    def on_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        current_time = time.time()
        step_delta = current_time - self._last_time
        # only run every few seconds
        if self._interval < step_delta:
            self._last_time = current_time
            self.do_interval(trainer, pl_module, current_time, self._start_time)

    def do_interval(self, trainer: pl.Trainer, pl_module: pl.LightningModule, current_time, start_time):
        raise NotImplementedError


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
