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

import re
import warnings
from typing import final
from typing import List
from typing import Sequence
from typing import Tuple
from typing import Union

import torch
import torch.nn.functional as F

from disent import registry
from disent.frameworks.helper.util import compute_ave_loss
from disent.nn.loss.reduction import batch_loss_reduction
from disent.nn.loss.reduction import loss_reduction
from disent.nn.modules import DisentModule
from disent.dataset.transform import FftKernel


# ========================================================================= #
# Reconstruction Loss Base                                                  #
# ========================================================================= #


class ReconLossHandler(DisentModule):

    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self._reduction = reduction

    def forward(self, *args, **kwargs):
        raise RuntimeError(f'Cannot call forward() on {self.__class__.__name__}')

    def activate(self, x_partial: torch.Tensor):
        """
        The final activation of the model.
        - Never use this in a training loop.
        """
        raise NotImplementedError

    def activate_all(self, xs_partial: Sequence[torch.Tensor]):
        return [self.activate(x_partial) for x_partial in xs_partial]

    @final
    def compute_loss(self, x_recon: torch.Tensor, x_targ: torch.Tensor) -> torch.Tensor:
        """
        Takes in activated tensors
        :return: The computed reduced loss
        """
        assert x_recon.shape == x_targ.shape, f'x_recon.shape={x_recon.shape} x_targ.shape={x_targ.shape}'
        batch_loss = self.compute_unreduced_loss(x_recon, x_targ)
        loss = loss_reduction(batch_loss, reduction=self._reduction)
        return loss

    @final
    def compute_loss_from_partial(self, x_partial_recon: torch.Tensor, x_targ: torch.Tensor) -> torch.Tensor:
        """
        Takes in an **unactivated** tensor from the model
        as well as an original target from the dataset.
        :return: The computed reduced loss
        """
        assert x_partial_recon.shape == x_targ.shape, f'x_partial_recon.shape={x_partial_recon.shape} x_targ.shape={x_targ.shape}'
        batch_loss = self.compute_unreduced_loss_from_partial(x_partial_recon, x_targ)
        loss = loss_reduction(batch_loss, reduction=self._reduction)
        return loss


    @final
    def compute_ave_loss(self, xs_recon: Sequence[torch.Tensor], xs_targ: Sequence[torch.Tensor]) -> torch.Tensor:
        """
        Compute the average over losses computed from corresponding tensor pairs in the sequence.
        """
        return compute_ave_loss(self.compute_loss, xs_recon, xs_targ)

    @final
    def compute_ave_loss_from_partial(self, xs_partial_recon: Sequence[torch.Tensor], xs_targ: Sequence[torch.Tensor]) -> torch.Tensor:
        """
        Compute the average over losses computed from corresponding tensor pairs in the sequence.
        """
        return compute_ave_loss(self.compute_loss_from_partial, xs_partial_recon, xs_targ)

    def compute_unreduced_loss(self, x_recon: torch.Tensor, x_targ: torch.Tensor) -> torch.Tensor:
        """
        Takes in activated tensors
        Compute the loss without applying a reduction, the loss
        tensor should be the same shapes as the input tensors
        :return: The computed unreduced loss
        """
        raise NotImplementedError

    def compute_unreduced_loss_from_partial(self, x_partial_recon: torch.Tensor, x_targ: torch.Tensor) -> torch.Tensor:
        """
        Takes in an **unactivated** tensor from the model
        Compute the loss without applying a reduction, the loss
        tensor should be the same shapes as the input tensors
        :return: The computed unreduced loss
        """
        raise NotImplementedError

    def _pairwise_reduce(self, unreduced_loss: torch.Tensor):
        assert self._reduction in ('mean', 'sum'), f'pairwise losses only support "mean" and "sum" reduction modes.'
        return batch_loss_reduction(unreduced_loss, reduction_dtype=None, reduction=self._reduction)

    def compute_pairwise_loss(self, x_recon: torch.Tensor, x_targ: torch.Tensor) -> torch.Tensor:
        return self._pairwise_reduce(self.compute_unreduced_loss(x_recon, x_targ))

    def compute_pairwise_loss_from_partial(self, x_partial_recon: torch.Tensor, x_targ: torch.Tensor) -> torch.Tensor:
        return self._pairwise_reduce(self.compute_unreduced_loss_from_partial(x_partial_recon, x_targ))


# ========================================================================= #
# Reconstruction Losses                                                     #
# ========================================================================= #


class ReconLossHandlerMse(ReconLossHandler):
    """
    MSE loss should be used with continuous targets between [0, 1].
    - using BCE for such targets is a prevalent error in VAE research.
    """

    def activate(self, x_partial: torch.Tensor) -> torch.Tensor:
        # mse requires no final activation
        return x_partial

    def compute_unreduced_loss(self, x_recon: torch.Tensor, x_targ: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(x_recon, x_targ, reduction='none')

    def compute_unreduced_loss_from_partial(self, x_partial_recon: torch.Tensor, x_targ: torch.Tensor) -> torch.Tensor:
        return self.compute_unreduced_loss(self.activate(x_partial_recon), x_targ)


class ReconLossHandlerMae(ReconLossHandlerMse):
    """
    MAE loss should be used with continuous targets between [0, 1].
    """
    def compute_unreduced_loss(self, x_recon, x_targ):
        return torch.abs(x_recon - x_targ)


class ReconLossHandlerBce(ReconLossHandler):
    """
    BCE loss should only be used with binary targets {0, 1}.
    - ignoring this and not using MSE is a prevalent error in VAE research.
    """

    def activate(self, x_partial: torch.Tensor):
        # we allow the model output x to generally be in the range [-1, 1] and scale
        # it to the range [0, 1] here to match the targets.
        return torch.sigmoid(x_partial)

    def compute_unreduced_loss(self, x_recon: torch.Tensor, x_targ: torch.Tensor) -> torch.Tensor:
        warnings.warn('binary cross entropy not computed over logits is inaccurate!')
        return F.binary_cross_entropy(x_recon, x_targ, reduction='none')

    def compute_unreduced_loss_from_partial(self, x_partial_recon, x_targ):
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

    def compute_unreduced_loss(self, x_recon, x_targ):
        # This is exactly the same as the BCE version, but more 'correct'.
        warnings.warn('bernoulli not computed over logits might be inaccurate!')
        return -torch.distributions.Bernoulli(probs=x_recon).log_prob(x_targ)

    def compute_unreduced_loss_from_partial(self, x_partial_recon, x_targ):
        # This is exactly the same as the BCE version, but more 'correct'.
        return -torch.distributions.Bernoulli(logits=x_partial_recon).log_prob(x_targ)


class ReconLossHandlerContinuousBernoulli(ReconLossHandlerBce):
    """
    The continuous Bernoulli: fixing a pervasive error in variational autoencoders
    - Loaiza-Ganem G and Cunningham JP, NeurIPS 2019.
    - https://arxiv.org/abs/1907.06845
    """

    def compute_unreduced_loss(self, x_recon, x_targ):
        warnings.warn('Using continuous bernoulli distribution for reconstruction loss. This is not yet recommended!')
        warnings.warn('continuous bernoulli not computed over logits might be inaccurate!')
        # I think there is something wrong with this...
        return -torch.distributions.ContinuousBernoulli(probs=x_recon, lims=(0.49, 0.51)).log_prob(x_targ)

    def compute_unreduced_loss_from_partial(self, x_partial_recon, x_targ):
        warnings.warn('Using continuous bernoulli distribution for reconstruction loss. This is not yet recommended!')
        # I think there is something wrong with this...
        return -torch.distributions.ContinuousBernoulli(logits=x_partial_recon, lims=(0.49, 0.51)).log_prob(x_targ)


class ReconLossHandlerNormal(ReconLossHandlerMse):

    def compute_unreduced_loss(self, x_recon, x_targ):
        # this is almost the same as MSE, but scaled with a tiny offset
        # A value for scale should actually be passed...
        warnings.warn('Using normal distribution for reconstruction loss. This is not yet recommended!')
        return -torch.distributions.Normal(x_recon, 1.0).log_prob(x_targ)


# ========================================================================= #
# Augmented Losses                                                          #
# ========================================================================= #


class AugmentedReconLossHandler(ReconLossHandler):

    def __init__(self, recon_loss_handler: ReconLossHandler, kernel: Union[str, torch.Tensor], wrap_weight=1.0, aug_weight=1.0):
        super().__init__(reduction=recon_loss_handler._reduction)
        # save variables
        self._recon_loss_handler = recon_loss_handler
        # must be a recon loss handler, but cannot nest augmented handlers
        assert isinstance(recon_loss_handler, ReconLossHandler)
        assert not isinstance(recon_loss_handler, AugmentedReconLossHandler)
        # load the kernel
        self._kernel = FftKernel(kernel=kernel, normalize=True)
        # kernel weighting
        assert 0 <= wrap_weight, f'loss_weight must be in the range [0, inf) but received: {repr(wrap_weight)}'
        assert 0 <= aug_weight, f'kern_weight must be in the range [0, inf) but received: {repr(aug_weight)}'
        self._wrap_weight = wrap_weight
        self._aug_weight = aug_weight
        # disable gradients
        for param in self.parameters():
            param.requires_grad = False

    def activate(self, x_partial: torch.Tensor):
        return self._recon_loss_handler.activate(x_partial)

    def compute_unreduced_loss(self, x_recon: torch.Tensor, x_targ: torch.Tensor) -> torch.Tensor:
        wrap_loss = self._recon_loss_handler.compute_unreduced_loss(x_recon, x_targ)
        aug_loss  = self._recon_loss_handler.compute_unreduced_loss(self._kernel(x_recon), self._kernel(x_targ))
        return (self._wrap_weight * wrap_loss) + (self._aug_weight * aug_loss)

    def compute_unreduced_loss_from_partial(self, x_partial_recon: torch.Tensor, x_targ: torch.Tensor) -> torch.Tensor:
        return self.compute_unreduced_loss(self.activate(x_partial_recon), x_targ)


# ========================================================================= #
# Registry & Factory                                                        #
# ========================================================================= #


# TODO: add ability to register parameterized reconstruction losses
_ARG_RECON_LOSSES: List[Tuple[re.Pattern, str, callable]] = [
    # (REGEX, EXAMPLE, FACTORY_FUNC)
    # - factory function takes at min one arg: fn(reduction) with one arg after that per regex capture group
    # - regex expressions are tested in order, expressions should be mutually exclusive or ordered such that more specialized versions occur first.
]


# NOTE: this function compliments make_kernel in transform/_augment.py
def make_reconstruction_loss(name: str, reduction: str) -> ReconLossHandler:
    if name in registry.RECON_LOSSES:
        # search normal losses!
        return registry.RECON_LOSSES[name](reduction)
    else:
        # regex search losses, and call with args!
        for r, _, fn in _ARG_RECON_LOSSES:
            result = r.search(name)
            if result is not None:
                return fn(reduction, *result.groups())
    # we couldn't find anything
    raise KeyError(f'Invalid vae reconstruction loss: {repr(name)} Valid losses include: {sorted(registry.RECON_LOSSES)}, examples of additional argument based losses include: {[example for _, example, _ in _ARG_RECON_LOSSES]}')


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
