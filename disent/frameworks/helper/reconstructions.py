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

import warnings
from typing import final
from typing import Sequence

import torch
import torch.nn.functional as F

from disent.frameworks.helper.reductions import loss_reduction
from disent.frameworks.helper.util import compute_ave_loss


# ========================================================================= #
# Reconstruction Loss Base                                                  #
# ========================================================================= #


class ReconLossHandler(object):

    def __init__(self, reduction: str = 'mean'):
        self._reduction = reduction

    def activate(self, x):
        """
        The final activation of the model.
        - Never use this in a training loop.
        """
        raise NotImplementedError

    def activate_all(self, xs):
        return [self.activate(x) for x in xs]

    @final
    def compute_loss(self, x_partial_recon: torch.Tensor, x_targ: torch.Tensor) -> torch.Tensor:
        """
        Takes in an **unactivated** tensor from the model
        as well as an original target from the dataset.
        :return: The computed reduced loss
        """
        assert x_partial_recon.shape == x_targ.shape, f'x_partial_recon.shape={x_partial_recon.shape} x_targ.shape={x_targ.shape}'
        batch_loss = self._compute_unreduced_loss(x_partial_recon, x_targ)
        loss = loss_reduction(batch_loss, reduction=self._reduction)
        return loss

    @final
    def compute_ave_loss(self, xs_partial_recon: Sequence[torch.Tensor], xs_targ: Sequence[torch.Tensor]) -> torch.Tensor:
        return compute_ave_loss(self.compute_loss, xs_partial_recon, xs_targ)

    @final
    def compute_unreduced_loss(self, x_partial_recon: torch.Tensor, x_targ: torch.Tensor) -> torch.Tensor:
        return self._compute_unreduced_loss(x_partial_recon, x_targ)

    def _compute_unreduced_loss(self, x_partial_recon: torch.Tensor, x_targ: torch.Tensor) -> torch.Tensor:
        """
        Compute the loss without applying a reduction.
        - loss tensor should be the same shapes as the input tensors
        """
        raise NotImplementedError


# ========================================================================= #
# Reconstruction Losses                                                     #
# ========================================================================= #


class ReconLossHandlerMse(ReconLossHandler):
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

    def _compute_unreduced_loss(self, x_partial_recon, x_targ):
        # NOTE: x_targ is in the range [0, 1]... we scale this to be in the range [-1, 1]
        #       so that the MSE values are consistent. activating x_partial_recon instead
        #       changes the scale of the loss
        return F.mse_loss(x_partial_recon, (x_targ * 2) - 1, reduction='none')


class ReconLossHandlerBce(ReconLossHandler):
    """
    BCE loss should only be used with binary targets {0, 1}.
    - ignoring this and not using MSE is a prevalent error in VAE research.
    """

    def activate(self, x):
        # we allow the model output x to generally be in the range [-1, 1] and scale
        # it to the range [0, 1] here to match the targets.
        return torch.sigmoid(x)

    def _compute_unreduced_loss(self, x_partial_recon, x_targ):
        """
        Computes the Bernoulli loss for the sigmoid activation function

        REFERENCE:
            https://github.com/google-research/disentanglement_lib/blob/76f41e39cdeff8517f7fba9d57b09f35703efca9/disentanglement_lib/methods/shared/losses.py
            - the same when reduction=='mean_sum' for super().training_compute_loss()
        REFERENCE ALT:
            https://github.com/YannDubs/disentangling-vae/blob/master/disvae/models/losses.py
        """
        return F.binary_cross_entropy_with_logits(x_partial_recon, x_targ, reduction='none')


# ========================================================================= #
# Reconstruction Distributions                                              #
# ========================================================================= #


class ReconLossHandlerBernoulli(ReconLossHandlerBce):
    def _compute_unreduced_loss(self, x_partial_recon, x_targ):
        # This is exactly the same as the BCE version, but more 'correct'.
        return -torch.distributions.Bernoulli(logits=x_partial_recon).log_prob(x_targ)


class ReconLossHandlerContinuousBernoulli(ReconLossHandlerBce):
    """
    The continuous Bernoulli: fixing a pervasive error in variational autoencoders
    - Loaiza-Ganem G and Cunningham JP, NeurIPS 2019.
    - https://arxiv.org/abs/1907.06845
    """
    def _compute_unreduced_loss(self, x_partial_recon, x_targ):
        warnings.warn('Using continuous bernoulli distribution for reconstruction loss. This is not yet recommended!')
        # I think there is something wrong with this...
        # weird values...
        return -torch.distributions.ContinuousBernoulli(logits=x_partial_recon, lims=(0.49, 0.51)).log_prob(x_targ)


class ReconLossHandlerNormal(ReconLossHandlerMse):
    def _compute_unreduced_loss(self, x_partial_recon, x_targ):
        # this is almost the same as MSE, but scaled with a tiny offset
        # A value for scale should actually be passed...
        warnings.warn('Using normal distribution for reconstruction loss. This is not yet recommended!')
        # NOTE: x_targ is in the range [0, 1]... we scale this to be in the range [-1, 1]
        #       so that the MSE values are consistent. activating x_partial_recon instead
        #       changes the scale of the loss
        return -torch.distributions.Normal(x_partial_recon, 1.0).log_prob((x_targ * 2) - 1)


# ========================================================================= #
# Factory                                                                   #
# ========================================================================= #


def make_reconstruction_loss(name: str, reduction: str) -> ReconLossHandler:
    if name == 'mse':
        # from the normal distribution
        # binary values only in the set {0, 1}
        cls = ReconLossHandlerMse
    elif name == 'bce':
        # from the bernoulli distribution
        cls = ReconLossHandlerBce
    elif name == 'bernoulli':
        # reduces to bce
        # binary values only in the set {0, 1}
        cls = ReconLossHandlerBernoulli
    elif name == 'continuous_bernoulli':
        # bernoulli with a computed offset to handle values in the range [0, 1]
        cls = ReconLossHandlerContinuousBernoulli
    elif name == 'normal':
        # handle all real values
        cls = ReconLossHandlerNormal
    else:
        raise KeyError(f'Invalid vae reconstruction loss: {name}')
    # instantiate!
    return cls(reduction=reduction)


# ========================================================================= #
# END                                                                       #
# ========================================================================= #

