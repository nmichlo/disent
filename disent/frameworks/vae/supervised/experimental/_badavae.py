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
from typing import Any
from typing import Dict
from typing import Sequence
from typing import Tuple

import torch
from disent.frameworks.vae.weaklysupervised._adavae import AdaVae


# ========================================================================= #
# Guided Ada Vae                                                            #
# ========================================================================= #


class BoundedAdaVae(AdaVae):

    REQUIRED_OBS = 3

    @dataclass
    class cfg(AdaVae.cfg):
        pass

    def hook_intercept_zs(self, zs_params: Sequence['Params']) -> Tuple[Sequence['Params'], Dict[str, Any]]:
        a_z_params, p_z_params, n_z_params = zs_params

        # get distributions
        a_d_posterior, _ = self.params_to_dists(a_z_params)
        p_d_posterior, _ = self.params_to_dists(p_z_params)
        n_d_posterior, _ = self.params_to_dists(n_z_params)

        # get deltas
        a_p_deltas = AdaVae.compute_posterior_deltas(a_d_posterior, p_d_posterior, thresh_mode=self.cfg.ada_thresh_mode)
        a_n_deltas = AdaVae.compute_posterior_deltas(a_d_posterior, n_d_posterior, thresh_mode=self.cfg.ada_thresh_mode)

        # shared elements that need to be averaged, computed per pair in the batch.
        old_p_shared_mask = AdaVae.estimate_shared_mask(a_p_deltas, ratio=self.cfg.ada_thresh_ratio)
        old_n_shared_mask = AdaVae.estimate_shared_mask(a_n_deltas, ratio=self.cfg.ada_thresh_ratio)

        # modify threshold based on criterion and recompute if necessary
        # CORE of this approach!
        p_shared_mask, n_shared_mask = BoundedAdaVae.compute_constrained_masks(a_p_deltas, old_p_shared_mask, a_n_deltas, old_n_shared_mask)
        
        # make averaged variables
        new_args = AdaVae.make_averaged_params(a_z_params, p_z_params, p_shared_mask, average_mode=self.cfg.ada_average_mode)

        # TODO: n_z_params should not be here! this does not match the original version
        #       number of loss elements is not 2 like the original
        #       - recons gets 2 items, p & a only
        #       - reg gets 2 items, p & a only
        new_args = (*new_args, n_z_params)

        # return new args & generate logs
        # -- we only return 2 parameters a & p, not n
        return new_args, {
            'p_shared_before': old_p_shared_mask.sum(dim=1).float().mean(),
            'p_shared_after':      p_shared_mask.sum(dim=1).float().mean(),
            'n_shared_before': old_n_shared_mask.sum(dim=1).float().mean(),
            'n_shared_after':      n_shared_mask.sum(dim=1).float().mean(),
        }
    
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
    # HELPER                                                                #
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

    @staticmethod
    def compute_constrained_masks(p_kl_deltas, p_shared_mask, n_kl_deltas, n_shared_mask):
        # number of changed factors
        p_shared_num = torch.sum(p_shared_mask, dim=1, keepdim=True)
        n_shared_num = torch.sum(n_shared_mask, dim=1, keepdim=True)
    
        # POSITIVE SHARED MASK
        # order from smallest to largest
        p_sort_indices = torch.argsort(p_kl_deltas, dim=1)
        # p_shared should be at least n_shared
        new_p_shared_num = torch.max(p_shared_num, n_shared_num)
    
        # NEGATIVE SHARED MASK
        # order from smallest to largest
        n_sort_indices = torch.argsort(n_kl_deltas, dim=1)
        # n_shared should be at most p_shared
        new_n_shared_num = torch.min(p_shared_num, n_shared_num)
    
        # COMPUTE NEW MASKS
        new_p_shared_mask = torch.zeros_like(p_shared_mask)
        new_n_shared_mask = torch.zeros_like(n_shared_mask)
        for i, (new_shared_p, new_shared_n) in enumerate(zip(new_p_shared_num, new_n_shared_num)):
            new_p_shared_mask[i, p_sort_indices[i, :new_shared_p]] = True
            new_n_shared_mask[i, n_sort_indices[i, :new_shared_n]] = True
    
        # return masks
        return new_p_shared_mask, new_n_shared_mask


# ========================================================================= #
# END                                                                       #
# ========================================================================= #

