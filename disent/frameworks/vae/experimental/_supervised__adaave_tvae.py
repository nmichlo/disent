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
import warnings
from dataclasses import dataclass
from typing import Any
from typing import Dict
from typing import Sequence
from typing import Tuple

from deprecated import deprecated
from torch.distributions import Distribution
from torch.distributions import Normal

from disent.frameworks.vae.experimental._supervised__adatvae import AdaTripletVae
from disent.frameworks.vae.experimental._supervised__adatvae import compute_ave_shared_distributions
from disent.frameworks.vae.experimental._supervised__adatvae import compute_triplet_shared_masks

log = logging.getLogger(__name__)


# ========================================================================= #
# Guided Ada Vae                                                            #
# ========================================================================= #


@deprecated(reason='Rather use the AdaNegTripletVae')
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
        ada_thresh_mode: str = 'dist'  # RESET OVERRIDEN VALUE
        # adaave_tvae
        adaave_augment_orig: bool = True  # triplet over original OR averaged embeddings
        adaave_decode_orig: bool = True  # decode & regularize original OR averaged embeddings

    def __init__(self, make_optimizer_fn, make_model_fn, batch_augment=None, cfg=cfg):
        super().__init__(make_optimizer_fn=make_optimizer_fn, make_model_fn=make_model_fn, batch_augment=batch_augment, cfg=cfg)
        # checks
        if self.cfg.ada_thresh_mode != 'dist':
            warnings.warn(f'cfg.ada_thresh_mode == {repr(self.cfg.ada_thresh_mode)}. Modes other than "dist" do not work well!')
        val = (self.cfg.adat_triplet_loss != 'triplet_hard_ave_all')
        if self.cfg.adaave_augment_orig == val:
            warnings.warn(f'cfg.adaave_augment_orig == {repr(self.cfg.adaave_augment_orig)}. Modes other than {repr(val)} do not work well!')
        if self.cfg.adaave_decode_orig == False:
            warnings.warn(f'cfg.adaave_decode_orig == {repr(self.cfg.adaave_decode_orig)}. Modes other than True do not work well!')

    def hook_intercept_ds(self, ds_posterior: Sequence[Normal], ds_prior: Sequence[Normal]) -> Tuple[Sequence[Distribution], Sequence[Distribution], Dict[str, Any]]:
        # triplet vae intercept -- in case detached
        ds_posterior, ds_prior, intercept_logs = super().hook_intercept_ds(ds_posterior, ds_prior)

        # compute shared masks, shared embeddings & averages over shared embeddings
        share_masks, share_logs = compute_triplet_shared_masks(ds_posterior, cfg=self.cfg)
        ds_posterior_shared, ds_posterior_shared_ave = compute_ave_shared_distributions(ds_posterior, share_masks, cfg=self.cfg)

        # DIFFERENCE FROM ADAVAE | get return values
        # adavae: adaave_augment_orig == True, adaave_decode_orig == False
        ds_posterior_augment = (ds_posterior if self.cfg.adaave_augment_orig else ds_posterior_shared_ave)
        ds_posterior_return = (ds_posterior if self.cfg.adaave_decode_orig else ds_posterior_shared_ave)

        # save params for aug_loss hook step
        self._curr_ada_loss_kwargs = dict(
            share_masks=share_masks,
            zs=[d.mean for d in ds_posterior],
            zs_shared=[d.mean for d in ds_posterior_shared],
            zs_shared_ave=[d.mean for d in ds_posterior_augment],  # USUALLY: zs_params_shared_ave
        )

        return ds_posterior_return, ds_prior, {
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
