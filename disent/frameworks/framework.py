from typing import final

import torch
from dataclasses import dataclass
from disent.util import DisentLightningModule, DisentConfigurable


# ========================================================================= #
# framework                                                                 #
# ========================================================================= #


class BaseFramework(DisentConfigurable, DisentLightningModule):

    @dataclass
    class cfg(DisentConfigurable.cfg):
        pass

    def __init__(self, make_optimizer_fn, batch_augment=None, cfg: cfg = None):
        super().__init__(cfg=cfg)
        # optimiser
        assert callable(make_optimizer_fn)
        self._make_optimiser_fn = make_optimizer_fn
        # batch augmentations: not implemented as dataset transforms because we want to apply these on the GPU
        assert (batch_augment is None) or callable(batch_augment)
        self._batch_augment = batch_augment

    @final
    def configure_optimizers(self):
        # return optimizers
        return self._make_optimiser_fn(self.parameters())

    @final
    def training_step(self, batch, batch_idx):
        """This is a pytorch-lightning function that should return the computed loss"""
        try:
            # augment batch with GPU support
            if self._batch_augment is not None:
                batch = self._batch_augment(batch)
            # compute loss
            logs_dict = self.compute_training_loss(batch, batch_idx)
            assert 'loss' not in logs_dict
            # return log loss components & return loss
            self.log_dict(logs_dict)
            train_loss = logs_dict['train_loss']
            # check training loss
            self._assert_valid_loss(train_loss)
            # return loss
            return train_loss
        except Exception as e:
            # call in all the child processes for the best chance of clearing this...
            # remove callbacks from trainer so we aren't stuck running forever!
            # TODO: this is a hack... there must be a better way to do this... could it be a pl bug?
            #       this logic is duplicated in the run_utils
            if self.trainer and self.trainer.callbacks:
                self.trainer.callbacks.clear()
            # continue propagating errors
            raise e

    @final
    def _assert_valid_loss(self, loss):
        if self.trainer.terminate_on_nan:
            if torch.isnan(loss) or torch.isinf(loss):
                raise ValueError('The returned loss is nan or inf')
        if loss > 1e+12:
            raise ValueError(f'The returned loss: {loss:.2e} is out of bounds: > {1e+12:.0e}')

    def forward(self, batch) -> torch.Tensor:
        """this function should return the single final output of the model, including the final activation"""
        raise NotImplementedError

    def compute_training_loss(self, batch, batch_idx) -> dict:
        """
        should return a dictionary of items to log with the key 'train_loss'
        as the variable to minimize
        """
        raise NotImplementedError


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
