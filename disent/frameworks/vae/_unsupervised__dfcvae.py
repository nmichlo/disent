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

from dataclasses import dataclass
from numbers import Number
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

import torch
import torchvision
from torch import Tensor
from torch.nn import functional as F
from torchvision.models import vgg19_bn

from disent.frameworks.helper.util import compute_ave_loss
from disent.frameworks.vae._unsupervised__betavae import BetaVae
from disent.nn.loss.reduction import get_mean_loss_scale
from disent.dataset.transform.functional import check_tensor


# ========================================================================= #
# Dfc Vae                                                                   #
# ========================================================================= #


class DfcVae(BetaVae):
    """
    Deep Feature Consistent Variational Autoencoder.
    https://arxiv.org/abs/1610.00291
    - Uses features generated from a pretrained model as the loss.

    Reference implementation is from: https://github.com/AntixK/PyTorch-VAE

    Difference:
        1. MSE loss changed to BCE or MSE loss
        2. Mean taken over (batch for sum of pixels) not mean over (batch & pixels)
    """

    REQUIRED_OBS = 1

    @dataclass
    class cfg(BetaVae.cfg):
        feature_layers: Optional[List[Union[str, int]]] = None
        feature_inputs_mode: str = 'none'

    def __init__(self, model: 'AutoEncoder', cfg: cfg = None, batch_augment=None):
        super().__init__(model=model, cfg=cfg, batch_augment=batch_augment)
        # make dfc loss
        # TODO: this should be converted to a reconstruction loss handler that wraps another handler
        self._dfc_loss = DfcLossModule(feature_layers=self.cfg.feature_layers, input_mode=self.cfg.feature_inputs_mode)

    # --------------------------------------------------------------------- #
    # Overrides                                                             #
    # --------------------------------------------------------------------- #

    def compute_ave_recon_loss(self, xs_partial_recon: Sequence[torch.Tensor], xs_targ: Sequence[torch.Tensor]) -> Tuple[Union[torch.Tensor, Number], Dict[str, Any]]:
        # compute ave reconstruction loss
        pixel_loss = self.recon_handler.compute_ave_loss_from_partial(xs_partial_recon, xs_targ)  # (DIFFERENCE: 1)
        # compute ave deep features loss
        xs_recon = self.recon_handler.activate_all(xs_partial_recon)
        feature_loss = compute_ave_loss(self._dfc_loss.compute_loss, xs_recon, xs_targ, reduction=self.cfg.loss_reduction)
        # reconstruction error
        # TODO: not in reference implementation, but terms should be weighted
        # TODO: not in reference but feature loss is not scaled properly
        recon_loss = (pixel_loss + feature_loss) * 0.5
        # return logs
        return recon_loss, {
            'pixel_loss': pixel_loss,
            'feature_loss': feature_loss,
        }


# ========================================================================= #
# Helper Loss                                                               #
# ========================================================================= #


class DfcLossModule(torch.nn.Module):
    """
    Loss function for the Deep Feature Consistent Variational Autoencoder.
    https://arxiv.org/abs/1610.00291

    Reference implementation is from: https://github.com/AntixK/PyTorch-VAE

    Difference:
    - normalise data as torchvision.models require.

    # TODO: this should be converted to a reconstruction loss handler
    """

    def __init__(self, feature_layers: Optional[List[Union[str, int]]] = None, input_mode: str = 'none'):
        """
        :param feature_layers: List of string of IDs of feature layers in pretrained model
        """
        super().__init__()
        # feature layers to use
        self.feature_layers = set(['14', '24', '34', '43'] if (feature_layers is None) else [str(l) for l in feature_layers])
        # feature network
        self.feature_network = vgg19_bn(pretrained=True)
        # Freeze the pretrained feature network
        for param in self.feature_network.parameters():
            param.requires_grad = False
        # Evaluation Mode
        self.feature_network.eval()
        # input node
        assert input_mode in {'none', 'clamp', 'assert'}
        self.input_mode = input_mode

    def compute_loss(self, x_recon, x_targ, reduction='mean'):
        """
        x_recon and x_targ data should be an unnormalized RGB batch of
        data [B x C x H x W] in the range [0, 1].
        """
        features_recon = self._extract_features(x_recon)
        features_targ = self._extract_features(x_targ)
        # compute losses
        # TODO: not in reference implementation, but consider calculating mean feature loss rather than sum
        feature_loss = 0.0
        for (f_recon, f_targ) in zip(features_recon, features_targ):
            feature_loss += F.mse_loss(f_recon, f_targ, reduction='mean')
        # scale the loss accordingly
        # (DIFFERENCE: 2)
        return feature_loss * get_mean_loss_scale(x_targ, reduction=reduction)

    def _extract_features(self, inputs: Tensor) -> List[Tensor]:
        """
        Extracts the features from the pretrained model
        at the layers indicated by feature_layers.
        :param inputs: (Tensor) [B x C x H x W] unnormalised in the range [0, 1].
        :return: List of the extracted features
        """
        inputs = self._process_inputs(inputs)
        # normalise: https://pytorch.org/docs/stable/torchvision/models.html
        result = torchvision.transforms.functional.normalize(inputs, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # calculate all features
        features = []
        for (key, module) in self.feature_network.features._modules.items():
            result = module(result)
            if key in self.feature_layers:
                features.append(result)
        return features

    def _process_inputs(self, inputs: torch.Tensor) -> torch.Tensor:
        # check the input tensor
        if self.input_mode == 'assert':
            inputs = check_tensor(inputs, low=0, high=1, dtype=None)
        elif self.input_mode == 'clamp':
            inputs = torch.clamp(inputs, 0, 1)
        elif self.input_mode != 'none':
            raise KeyError(f'invalid input_mode={repr(self.input_mode)}')
        # repeat if missing dimensions, supports C in [1, 3]
        B, C, H, W = inputs.shape
        if C == 1:
            inputs = inputs.repeat(1, 3, 1, 1)
        assert (B, 3, H, W) == inputs.shape
        # done
        return inputs


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
