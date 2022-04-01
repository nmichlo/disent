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

import logging
from dataclasses import dataclass
from typing import Any
from typing import Dict
from typing import Sequence
from typing import Tuple

import torch

from disent.frameworks.ae._supervised__adaneg_tae import AdaNegTripletAe
from disent.frameworks.vae._supervised__adaneg_tvae import AdaNegTripletVae
from disent.frameworks.vae._unsupervised__dotvae import DataOverlapMixin


log = logging.getLogger(__name__)


# ========================================================================= #
# Data Overlap Triplet AE                                                  #
# ========================================================================= #


class DataOverlapTripletAe(AdaNegTripletAe, DataOverlapMixin):
    """
    Michlo et al.
    https://github.com/nmichlo/msc-research

    This is the unsupervised version of the Adaptive Triplet Loss for Auto-Encoders not VAEs
    - Triplets are sampled from training minibatches and are ordered using some distance
      function taken directly over the datapoints.
    - Adaptive Triplet Loss is used to guide distance learning and encourage disentanglement
    """

    REQUIRED_OBS = 1

    @dataclass
    class cfg(AdaNegTripletAe.cfg, DataOverlapMixin.cfg):
        pass

    def __init__(self, model: 'AutoEncoder', cfg: cfg = None, batch_augment=None):
        super().__init__(model=model, cfg=cfg, batch_augment=batch_augment)
        # initialise mixin
        self.init_data_overlap_mixin()

    def hook_ae_compute_ave_aug_loss(self, zs: Sequence[torch.Tensor], xs_partial_recon: Sequence[torch.Tensor], xs_targ: Sequence[torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        [z], [x_targ_orig] = zs, xs_targ
        # 1. randomly generate and mine triplets using augmented versions of the inputs
        a_idxs, p_idxs, n_idxs = self.random_mined_triplets(x_targ_orig=x_targ_orig)
        # 2. compute triplet loss
        loss, loss_log = AdaNegTripletVae.estimate_ada_triplet_loss_from_zs(
            zs=[z[idxs] for idxs in (a_idxs, p_idxs, n_idxs)],
            cfg=self.cfg,
        )
        return loss, {
            **loss_log,
        }


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
