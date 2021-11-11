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

import pytorch_lightning as pl

from disent.util.lightning.callbacks._callbacks_base import BaseCallbackTimed


log = logging.getLogger(__name__)


# ========================================================================= #
# General Callbacks that work for any pl.LightningModule                    #
# ========================================================================= #


class LoggerProgressCallback(BaseCallbackTimed):
    
    def do_interval(self, trainer: pl.Trainer, pl_module: pl.LightningModule, current_time, start_time):
        # get missing vars
        trainer_max_epochs = trainer.max_epochs if (trainer.max_epochs is not None) else float('inf')
        trainer_max_steps = trainer.max_steps if (trainer.max_steps is not None) else float('inf')

        # compute vars
        max_batches = trainer.num_training_batches  # can be inf
        max_epochs = min(trainer_max_epochs, (trainer_max_steps + max_batches - 1) // max_batches)
        max_steps = min(trainer_max_epochs * max_batches, trainer_max_steps)
        elapsed_sec = current_time - start_time
        # get vars
        global_step = trainer.global_step + 1
        epoch = trainer.current_epoch + 1
        if hasattr(trainer, 'batch_idx'):
            batch = (trainer.batch_idx + 1)
        else:
            # TODO: re-enable this warning but only ever print once!
            # warnings.warn('batch_idx missing on pl.Trainer')
            batch = global_step % max_batches  # might not be int?
        # completion
        train_pct = global_step / max_steps
        train_remain_time = elapsed_sec * (1 - train_pct) / train_pct  # seconds
        # get speed -- TODO: make this a moving average?
        if global_step >= elapsed_sec:
            step_speed_str = f'{global_step / elapsed_sec:4.2f}it/s'
        else:
            step_speed_str = f'{elapsed_sec / global_step:4.2f}s/it'
        # info dict
        info_dict = {
            k: f'{v:.4g}' if isinstance(v, (int, float)) else f'{v}'
            for k, v in trainer.progress_bar_dict.items()
            if k != 'v_num'
        }
        sorted_k = sorted(info_dict.keys(), key=lambda k: ('loss' != k.lower(), 'loss' not in k.lower(), k))
        # log
        log.info(
            f'[{int(elapsed_sec)}s, {step_speed_str}] '
            + f'EPOCH: {epoch}/{max_epochs} - {int(global_step):0{len(str(max_steps))}d}/{max_steps} '
            + f'({int(train_pct * 100):02d}%) [rem. {int(train_remain_time)}s] '
            + f'STEP: {int(batch):{len(str(max_batches))}d}/{max_batches} ({int(batch / max_batches * 100):02d}%) '
            + f'| {" ".join(f"{k}={info_dict[k]}" for k in sorted_k)}'
        )


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
