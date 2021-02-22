from typing import final

import torch
import torch.nn.functional as F
from disent.frameworks.helper.reductions import loss_reduction


# ========================================================================= #
# Reconstruction Loss Base                                                  #
# ========================================================================= #


class ReconstructionLoss(object):

    def activate(self, x):
        """
        The final activation of the model.
        - Never use this in a training loop.
        """
        raise NotImplementedError

    @final
    def training_compute_loss(self, x_partial_recon: torch.Tensor, x_targ: torch.Tensor, reduction: str = 'batch_mean') -> torch.Tensor:
        """
        Takes in an **unactivated** tensor from the model
        as well as an original target from the dataset.
        :return: The computed mean loss
        """
        assert x_partial_recon.shape == x_targ.shape
        batch_loss = self._compute_batch_loss(x_partial_recon, x_targ)
        loss = loss_reduction(batch_loss, reduction=reduction)
        return loss

    def _compute_batch_loss(self, x_partial_recon: torch.Tensor, x_targ: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


# ========================================================================= #
# Reconstruction Losses                                                     #
# ========================================================================= #


class ReconstructionLossMse(ReconstructionLoss):
    """
    MSE loss should be used with continuous targets between [0, 1].
    - using BCE for such targets is a prevalent error in VAE research.
    """

    def activate(self, x):
        # we allow the model output x to generally be in the range [-1, 1] and scale
        # it to the range [0, 1] here to match the targets.
        # - this lets it learn more easily as the output is naturally centered on 1
        # - doing this here directly on the output is easier for visualisation etc.
        # - TODO: the better alternative is that we rather calculate the MEAN and STD over the dataset
        #         and normalise that.
        # - sigmoid is numerically not suitable with MSE
        return 0.5 * (x + 1)

    def _compute_batch_loss(self, x_partial_recon, x_targ):
        return F.mse_loss(self.activate(x_partial_recon), x_targ, reduction='none')

    @staticmethod
    def LEGACY_training_compute_loss(x_recon, x_target, reduction: str = 'batch_mean'):
        raise NotImplementedError('LEGACY mse version does not exist!')


class ReconstructionLossBce(ReconstructionLoss):
    """
    BCE loss should only be used with binary targets {0, 1}.
    - ignoring this and not using MSE is a prevalent error in VAE research.
    """

    def activate(self, x):
        # we allow the model output x to generally be in the range [-1, 1] and scale
        # it to the range [0, 1] here to match the targets.
        return torch.sigmoid(x)

    def _compute_batch_loss(self, x_partial_recon, x_targ):
        return F.binary_cross_entropy_with_logits(x_partial_recon, x_targ, reduction='none')

    @staticmethod
    def LEGACY_training_compute_loss(x_recon, x_target, reduction: str = 'batch_mean'):
        """
        Computes the Bernoulli loss for the sigmoid activation function
        FROM: https://github.com/google-research/disentanglement_lib/blob/76f41e39cdeff8517f7fba9d57b09f35703efca9/disentanglement_lib/methods/shared/losses.py
        """
        assert reduction == 'batch_mean', f'legacy reference implementation of BCE loss only supports reduction="batch_mean", not {repr(reduction)}'
        # x, x_recon = x.view(x.shape[0], -1), x_recon.view(x.shape[0], -1)
        # per_sample_loss = F.binary_cross_entropy_with_logits(x_recon, x, reduction='none').sum(axis=1)  # tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=x_recon, labels=x), axis=1)
        # reconstruction_loss = per_sample_loss.mean()                                                    # tf.reduce_mean(per_sample_loss)
        # ALTERNATIVE IMPLEMENTATION https://github.com/YannDubs/disentangling-vae/blob/master/disvae/models/losses.py
        assert x_recon.shape == x_target.shape
        return F.binary_cross_entropy_with_logits(x_recon, x_target, reduction="sum") / len(x_target)


# ========================================================================= #
# Factory                                                                   #
# ========================================================================= #


def make_reconstruction_loss(name) -> ReconstructionLoss:
    if name == 'mse':
        return ReconstructionLossMse()
    elif name == 'bce':
        return ReconstructionLossBce()
    else:
        raise KeyError(f'Invalid vae reconstruction loss: {name}')


# ========================================================================= #
# END                                                                       #
# ========================================================================= #

