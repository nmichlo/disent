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
from typing import Sequence

from disent.frameworks.vae.supervised.experimental._adatvae import AdaTripletVae
from disent.frameworks.vae.supervised.experimental._adatvae import compute_ave_shared_params
from disent.frameworks.vae.supervised.experimental._adatvae import compute_triplet_shared_masks

log = logging.getLogger(__name__)


# ========================================================================= #
# Guided Ada Vae                                                            #
# ========================================================================= #


class AdaAveTripletVae(AdaTripletVae):
    """
    This was a more general attempt of the ada-tvae,
    that also averages representations passed to the decoder.
    - just averaging in this way without augmenting the loss with
      triplet, or ada_triplet is too weak of a supervision signal.
    """

    REQUIRED_OBS = 3

    @dataclass
    class cfg(AdaTripletVae.cfg):
        # adavae
        ada_thresh_mode: str = 'symmetric_kl'  # RESET OVERRIDEN VALUE
        # adaave_tvae
        adaave_augment_orig: bool = False  # triplet over original OR averaged embeddings
        adaave_decode_orig: bool = False  # decode & regularize original OR averaged embeddings

    def hook_intercept_zs(self, zs_params: Sequence['Params']):
        # triplet vae intercept -- in case detached
        zs_params, intercept_logs = super().hook_intercept_zs(zs_params)
        ds_posterior, _ = self.all_params_to_dists(zs_params)

        # compute shared masks, shared embeddings & averages over shared embeddings
        share_masks, share_logs = compute_triplet_shared_masks(ds_posterior, cfg=self.cfg)
        zs_params_shared, zs_params_shared_ave = compute_ave_shared_params(zs_params, share_masks, cfg=self.cfg)

        # get return values
        zs_params_augment = (zs_params if self.cfg.adaave_augment_orig else zs_params_shared_ave)
        zs_params_return = (zs_params if self.cfg.adaave_decode_orig else zs_params_shared_ave)

        # save params for aug_loss hook step
        self._curr_ada_loss_kwargs = dict(
            share_masks=share_masks,
            zs_params=zs_params,
            zs_params_shared=zs_params_shared,
            zs_params_shared_ave=zs_params_augment,  # USUALLY: zs_params_shared_ave
        )

        return zs_params_return, {
            **intercept_logs,
            **share_logs,
        }

    def hook_compute_ave_aug_loss(self, ds_posterior, ds_prior, zs_sampled, xs_partial_recon, xs_targ):
        """
        NOTE: we don't use input parameters here, this function will only work
              if called as part of training_step or do_training_step
        """
        # compute triplet loss
        result = AdaTripletVae.compute_ada_triplet_loss(**self._curr_ada_loss_kwargs, cfg=self.cfg)
        # cleanup temporary variables
        del self._curr_ada_loss_kwargs
        # we are done
        return result


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
