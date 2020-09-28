import torch
import pytorch_lightning as pl

# ========================================================================= #
# framework                                                                 #
# ========================================================================= #


class BaseFramework(pl.LightningModule):
    
    def __init__(self, make_optimizer_fn):
        super().__init__()
        # optimiser
        assert callable(make_optimizer_fn)
        self._make_optimiser_fn = make_optimizer_fn

    def configure_optimizers(self):
        return self._make_optimiser_fn(self.parameters())

    def forward(self, batch) -> torch.Tensor:
        """this function should return the single final output of the model, including the final activation"""
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        """This is a pytorch-lightning function that should return the computed loss"""
        # compute loss
        logs_dict = self.compute_training_loss(batch, batch_idx)
        # return log loss components & return loss
        train_result = pl.TrainResult(minimize=logs_dict['train_loss'])
        train_result.log_dict(logs_dict, on_step=True, on_epoch=False)
        return train_result

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


