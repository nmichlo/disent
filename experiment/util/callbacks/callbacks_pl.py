import logging
from experiment.util.callbacks.callbacks_base import _TimedCallback
import pytorch_lightning as pl


log = logging.getLogger(__name__)


# ========================================================================= #
# General Callbacks that work for any pl.LightningModule                    #
# ========================================================================= #


class LoggerProgressCallback(_TimedCallback):
    
    def sig(self, num, digits):
        return f'{num:.{digits}g}'
    
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
        info_dict = {k: f'{self.sig(v, 4)}' if isinstance(v, (int, float)) else f'{v}' for k, v in trainer.progress_bar_dict.items() if k != 'v_num'}
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
