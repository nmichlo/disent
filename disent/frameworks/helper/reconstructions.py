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
import os
import warnings
from typing import final
from typing import Sequence
from typing import Union
import re

import torch
import torch.nn.functional as F

import disent
from disent.frameworks.helper.reductions import batch_loss_reduction
from disent.frameworks.helper.reductions import loss_reduction
from disent.frameworks.helper.util import compute_ave_loss
from disent.util.math import torch_conv2d_channel_wise_fft
from disent.util import DisentModule
from disent.util.math import torch_box_kernel_2d

from deprecated import deprecated


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
        # we allow the model output x to generally be in the range [-1, 1] and scale
        # it to the range [0, 1] here to match the targets.
        # - this lets it learn more easily as the output is naturally centered on 1
        # - doing this here directly on the output is easier for visualisation etc.
        # - TODO: the better alternative is that we rather calculate the MEAN and STD over the dataset
        #         and normalise that.
        # - sigmoid is numerically not suitable with MSE
        return (x_partial + 1) / 2

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


@deprecated('Mse4 loss is being used during development to avoid a new hyper-parameter search')
class ReconLossHandlerMse4(ReconLossHandlerMse):
    def compute_unreduced_loss(self, x_recon: torch.Tensor, x_targ: torch.Tensor) -> torch.Tensor:
        return super().compute_unreduced_loss(x_recon, x_targ) * 4


@deprecated('Mae2 loss is being used during development to avoid a new hyper-parameter search')
class ReconLossHandlerMae2(ReconLossHandlerMae):
    def compute_unreduced_loss(self, x_recon: torch.Tensor, x_targ: torch.Tensor) -> torch.Tensor:
        return super().compute_unreduced_loss(x_recon, x_targ) * 2


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

    def __init__(self, recon_loss_handler: ReconLossHandler, kernel: Union[str, torch.Tensor], kernel_weight=1.0):
        super().__init__(reduction=recon_loss_handler._reduction)
        # save variables
        self._recon_loss_handler = recon_loss_handler
        # must be a recon loss handler, but cannot nest augmented handlers
        assert isinstance(recon_loss_handler, ReconLossHandler)
        assert not isinstance(recon_loss_handler, AugmentedReconLossHandler)
        # load the kernel
        if isinstance(kernel, str):
            kernel = torch.load(kernel)
        kernel = kernel.requires_grad_(False)
        # check stuffs
        assert isinstance(kernel, torch.Tensor)
        assert kernel.dtype == torch.float32
        assert kernel.ndim == 4, f'invalid number of kernel dims, required 4, given: {repr(kernel.ndim)}'  # B, C, H, W
        assert kernel.shape[0] == 1, f'invalid size of first kernel dim, required (1, ?, ?, ?), given: {repr(kernel.shape)}'  # B
        assert kernel.shape[0] in (1, 3), f'invalid size of second kernel dim, required (?, 1 or 3, ?, ?), given: {repr(kernel.shape)}'  # C
        # scale kernel -- just in case we didnt do this before
        kernel = kernel / kernel.sum()
        # save kernel
        self._kernel = torch.nn.Parameter(kernel, requires_grad=False)
        # kernel weighting
        assert 0 <= kernel_weight <= 1, f'kernel weight must be in the range [0, 1] but received: {repr(kernel_weight)}'
        self._kernel_weight = kernel_weight

    def activate(self, x_partial: torch.Tensor):
        return self._recon_loss_handler.activate(x_partial)

    def compute_unreduced_loss(self, x_recon: torch.Tensor, x_targ: torch.Tensor) -> torch.Tensor:
        aug_x_recon = torch_conv2d_channel_wise_fft(x_recon, self._kernel)
        aug_x_targ = torch_conv2d_channel_wise_fft(x_targ, self._kernel)
        aug_loss = self._recon_loss_handler.compute_unreduced_loss(aug_x_recon, aug_x_targ)
        loss = self._recon_loss_handler.compute_unreduced_loss(x_recon, x_targ)
        return (1. - self._kernel_weight) * loss + self._kernel_weight * aug_loss

    def compute_unreduced_loss_from_partial(self, x_partial_recon: torch.Tensor, x_targ: torch.Tensor) -> torch.Tensor:
        return self.compute_unreduced_loss(self.activate(x_partial_recon), x_targ)


# ========================================================================= #
# Factory                                                                   #
# ========================================================================= #


_RECON_LOSSES = {
    # ================================= #
    # from the normal distribution - real values in the range [0, 1]
    'mse': ReconLossHandlerMse,
    # mean absolute error
    'mae': ReconLossHandlerMae,
    # from the bernoulli distribution - binary values in the set {0, 1}
    'bce': ReconLossHandlerBce,
    # reduces to bce - binary values in the set {0, 1}
    'bernoulli': ReconLossHandlerBernoulli,
    # bernoulli with a computed offset to handle values in the range [0, 1]
    'continuous_bernoulli': ReconLossHandlerContinuousBernoulli,
    # handle all real values
    'normal': ReconLossHandlerNormal,
    # ================================= #
    # EXPERIMENTAL -- im just curious what would happen, haven't actually
    #                 done the maths or thought about this much.
    'mse4': ReconLossHandlerMse4,  # scaled as if computed over outputs of the range [-1, 1] instead of [0, 1]
    'mae2': ReconLossHandlerMae2,  # scaled as if computed over outputs of the range [-1, 1] instead of [0, 1]
}

_ARG_RECON_LOSSES = [
    # (REGEX, EXAMPLE, FACTORY_FUNC)
    # - factory function takes at min one arg: fn(reduction) with one arg after that per regex capture group
    # - regex expressions are tested in order, expressions should be mutually exclusive or ordered such that more specialized versions occur first.
    (re.compile(r'^([a-z\d]+)_(xy8)_r(47)_w(\d+\.\d+)$'),  'mse4_xy8_r47_w1.0', lambda reduction, loss, kern, radius, weight: AugmentedReconLossHandler(make_reconstruction_loss(loss, reduction=reduction), kernel=torch.load(os.path.abspath(os.path.join(disent.__file__, '../../data/adversarial_kernel', 'r47-1_s28800_adam_lr0.003_wd0.0_xy8x8.pt'))), kernel_weight=float(weight))),
    (re.compile(r'^([a-z\d]+)_(xy1)_r(47)_w(\d+\.\d+)$'),  'mse4_xy1_r47_w1.0', lambda reduction, loss, kern, radius, weight: AugmentedReconLossHandler(make_reconstruction_loss(loss, reduction=reduction), kernel=torch.load(os.path.abspath(os.path.join(disent.__file__, '../../data/adversarial_kernel', 'r47-1_s28800_adam_lr0.003_wd0.0_xy1x1.pt'))), kernel_weight=float(weight))),
    (re.compile(r'^([a-z\d]+)_(box)_r(\d+)_w(\d+\.\d+)$'), 'mse4_box_r31_w0.5', lambda reduction, loss, kern, radius, weight: AugmentedReconLossHandler(make_reconstruction_loss(loss, reduction=reduction), kernel=torch_box_kernel_2d(radius=int(radius))[None, ...], kernel_weight=float(weight))),
]


def make_reconstruction_loss(name: str, reduction: str) -> ReconLossHandler:
    if name in _RECON_LOSSES:
        # search normal losses!
        return _RECON_LOSSES[name](reduction)
    else:
        # regex search losses, and call with args!
        for r, _, fn in _ARG_RECON_LOSSES:
            result = r.search(name)
            if result is not None:
                return fn(reduction, *result.groups())
    # we couldn't find anything
    raise KeyError(f'Invalid vae reconstruction loss: {repr(name)} Valid losses include: {list(_RECON_LOSSES.keys())}, examples of additional argument based losses include: {[example for _, example, _ in _ARG_RECON_LOSSES]}')


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
