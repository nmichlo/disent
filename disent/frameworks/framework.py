import torch
import pytorch_lightning as pl


# ========================================================================= #
# framework                                                                 #
# ========================================================================= #


class BaseFramework(pl.LightningModule):
    
    def __init__(
            self,
            make_optimizer_fn,
            batch_augment=None,
    ):
        super().__init__()
        # optimiser
        assert callable(make_optimizer_fn)
        self._make_optimiser_fn = make_optimizer_fn
        # batch augmentations: not implemented as dataset transforms because we want to apply these on the GPU
        assert (batch_augment is None) or callable(batch_augment)
        self._batch_augment = batch_augment

    def configure_optimizers(self):
        return self._make_optimiser_fn(self.parameters())

    def forward(self, batch) -> torch.Tensor:
        """this function should return the single final output of the model, including the final activation"""
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        """This is a pytorch-lightning function that should return the computed loss"""
        # augment batch with GPU support
        if self._batch_augment is not None:
            batch = self._batch_augment(batch)
        # compute loss
        logs_dict = self.compute_training_loss(batch, batch_idx)
        assert 'loss' not in logs_dict
        # return log loss components & return loss
        self.log_dict(logs_dict)
        return logs_dict['train_loss']

    def compute_training_loss(self, batch, batch_idx) -> dict:
        """
        should return a dictionary of items to log with the key 'train_loss'
        as the variable to minimize
        """
        raise NotImplementedError

    def _forward_unimplemented(self, *args, **kwargs):
        # Annoying fix applied by torch for Module.forward:
        # https://github.com/python/mypy/issues/8795
        raise RuntimeError('This should never run!')


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
