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
import pytorch_lightning as pl

from disent.util.lightning.callbacks._callbacks_base import BaseCallbackTimed


log = logging.getLogger(__name__)


# ========================================================================= #
# General Callbacks that work for any pl.LightningModule                    #
# ========================================================================= #


class LoggerProgressCallback(BaseCallbackTimed):
    
    def do_interval(self, trainer: pl.Trainer, pl_module: pl.LightningModule, current_time, start_time):
        # vars
        batch, max_batches = trainer.batch_idx + 1, trainer.num_training_batches
        epoch, max_epoch = trainer.current_epoch + 1, min(trainer.max_epochs, (trainer.max_steps + max_batches - 1) // max_batches)
        global_step, global_steps = trainer.global_step + 1, min(trainer.max_epochs * max_batches, trainer.max_steps)
        # computed
        train_pct = global_step / global_steps
        # completion
        train_remain_time = (current_time - start_time) * (1 - train_pct) / train_pct
        # info dict
        info_dict = {k: f'{v:.4g}' if isinstance(v, (int, float)) else f'{v}' for k, v in trainer.progress_bar_dict.items() if k != 'v_num'}
        sorted_k = sorted(info_dict.keys(), key=lambda k: ('loss' != k.lower(), 'loss' not in k.lower(), k))
        # log
        log.info(
            f'EPOCH: {epoch}/{max_epoch} - {global_step:0{len(str(global_steps))}d}/{global_steps} '
            f'({int(train_pct * 100):02d}%) [{int(train_remain_time)}s] '
            f'STEP: {batch:{len(str(max_batches))}d}/{max_batches} ({int(batch / max_batches * 100):02d}%) '
            f'| {" ".join(f"{k}={info_dict[k]}" for k in sorted_k)}'
        )


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
