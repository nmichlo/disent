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

from typing import Any
from typing import Dict
from typing import Sequence
from typing import Tuple

import torch
from dataclasses import dataclass

from disent.frameworks.ae._unsupervised__ae import Ae
from disent.frameworks.vae._weaklysupervised__adavae import AdaVae


# ========================================================================= #
# Ada-GVAE                                                                  #
# ========================================================================= #


class AdaAe(Ae):
    """
    Custom implementation, removing Variational Auto-Encoder components of:
    Weakly Supervised Disentanglement Learning Without Compromises: https://arxiv.org/abs/2002.02886

    MODIFICATION:
    - L1 distance for deltas instead of KL divergence
    - adjustable threshold value
    """

    REQUIRED_OBS = 2

    @dataclass
    class cfg(Ae.cfg):
        ada_thresh_ratio: float = 0.5

    def hook_ae_intercept_zs(self, zs: Sequence[torch.Tensor]) -> Tuple[Sequence[torch.Tensor], Dict[str, Any]]:
        """
        Adaptive VAE Glue Method, putting the various components together
        1. find differences between deltas
        2. estimate a threshold for differences
        3. compute a shared mask from this threshold
        4. average together elements that should be considered shared

        TODO: the methods used in this function should probably be moved here
        TODO: this function could be turned into a torch.nn.Module!
        """
        z0, z1 = zs
        # shared elements that need to be averaged, computed per pair in the batch.
        share_mask = AdaVae.compute_shared_mask_from_zs(z0, z1, ratio=self.cfg.ada_thresh_ratio)
        # compute average posteriors
        new_zs = AdaVae.make_shared_zs(z0, z1, share_mask)
        # return new args & generate logs
        return new_zs, {
            'shared': share_mask.sum(dim=1).float().mean()
        }

# ========================================================================= #
# END                                                                       #
# ========================================================================= #
