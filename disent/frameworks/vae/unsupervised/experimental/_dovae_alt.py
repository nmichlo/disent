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
from typing import Sequence

import torch
from torch.distributions import Normal

from disent.frameworks.vae.unsupervised.experimental._dovae import DataOverlapVae


# ========================================================================= #
# tvae                                                                      #
# ========================================================================= #


class DataOverlapAltVae(DataOverlapVae):

    REQUIRED_OBS = 1

    @dataclass
    class cfg(DataOverlapVae.cfg):
        overlap_num: int = 1024

    def hook_compute_ave_aug_loss(self, ds_posterior: Sequence[Normal], ds_prior: Sequence[Normal], zs_sampled: Sequence[torch.Tensor], xs_partial_recon: Sequence[torch.Tensor], xs_targ: Sequence[torch.Tensor]):
        # get values
        (d_loc,) = (d.loc for d in ds_posterior)
        (d_scale,) = (d.scale for d in ds_posterior)
        (x_targ,) = xs_targ

        # get random triples -- TODO: this does not generate unique pairs
        B, N = len(x_targ), self.cfg.overlap_num,
        a_idxs, p_idxs, n_idxs = torch.randint(B, size=(3, min(N, B**3)))

        # make new vars
        ds_posterior_NEW = tuple(Normal(d_loc[idxs], d_scale[idxs]) for idxs in (a_idxs, p_idxs, n_idxs))
        xs_targ_NEW = tuple(x_targ[idxs] for idxs in (a_idxs, p_idxs, n_idxs))

        # compute loss
        return super().hook_compute_ave_aug_loss(
            ds_posterior=ds_posterior_NEW,
            ds_prior=None,
            zs_sampled=None,
            xs_partial_recon=None,
            xs_targ=xs_targ_NEW,
        )

# ========================================================================= #
# END                                                                       #
# ========================================================================= #
