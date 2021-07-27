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
import time
import pytorch_lightning as pl


log = logging.getLogger(__name__)


# ========================================================================= #
# HELPER CALLBACKS                                                          #
# ========================================================================= #


class BaseCallbackPeriodic(pl.Callback):
    
    def __init__(self, every_n_steps=None, begin_first_step=False):
        self.every_n_steps = every_n_steps
        self.begin_first_step = begin_first_step

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if self.every_n_steps is None:
            # number of steps/batches in an epoch
            self.every_n_steps = trainer.num_training_batches
            
    def on_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if 0 == trainer.global_step % self.every_n_steps:
            # skip on the first step if required
            if trainer.global_step == 0 and not self.begin_first_step:
                return
            self.do_step(trainer, pl_module)

    def do_step(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        raise NotImplementedError


class BaseCallbackTimed(pl.Callback):
    
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
