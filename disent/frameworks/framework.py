from dataclasses import dataclass
import torch
from disent.util import DisentLightningModule


# ========================================================================= #
# framework                                                                 #
# ========================================================================= #


class BaseFramework(DisentLightningModule):

    @dataclass
    class Config(object):
        pass

    cfg: Config  # type hints

    def __init__(
            self,
            make_optimizer_fn,
            batch_augment=None,
            cfg: Config = Config()
    ):
        super().__init__()
        # optimiser
        assert callable(make_optimizer_fn)
        self._make_optimiser_fn = make_optimizer_fn
        # batch augmentations: not implemented as dataset transforms because we want to apply these on the GPU
        assert (batch_augment is None) or callable(batch_augment)
        self._batch_augment = batch_augment
        # store the config
        assert isinstance(cfg, self.Config), f'{cfg=} ({type(cfg)}) is not an instance of {self.Config}'
        self.cfg: BaseFramework.Config = cfg

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


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
