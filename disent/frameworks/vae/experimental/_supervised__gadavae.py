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

from torch.distributions import Distribution

from disent.frameworks.vae._weaklysupervised__adavae import AdaVae
from disent.frameworks.vae._weaklysupervised__adavae import compute_average_distribution
from disent.frameworks.vae.experimental._supervised__badavae import compute_constrained_masks


# ========================================================================= #
# Guided Ada Vae                                                            #
# ========================================================================= #


class GuidedAdaVae(AdaVae):

    REQUIRED_OBS = 3

    @dataclass
    class cfg(AdaVae.cfg):
        gada_anchor_ave_mode: str = 'average'

    def __init__(self, make_optimizer_fn, make_model_fn, batch_augment=None, cfg: cfg = None):
        super().__init__(make_optimizer_fn, make_model_fn, batch_augment=batch_augment, cfg=cfg)
        # how the anchor is averaged
        assert cfg.gada_anchor_ave_mode in {'thresh', 'average'}

    def hook_intercept_ds(self, ds_posterior: Sequence[Distribution], ds_prior: Sequence[Distribution]) -> Tuple[Sequence[Distribution], Sequence[Distribution], Dict[str, Any]]:
        """
        *NB* arguments must satisfy: d(l, l2) < d(l, l3) [positive dist < negative dist]
        - This function assumes that the distance between labels l, l2, l3
          corresponding to z, z2, z3 satisfy the criteria d(l, l2) < d(l, l3)
          ie. l2 is the positive sample, l3 is the negative sample
        """
        a_posterior, p_posterior, n_posterior = ds_posterior

        # get deltas
        a_p_deltas = AdaVae.compute_posterior_deltas(a_posterior, p_posterior, thresh_mode=self.cfg.ada_thresh_mode)
        a_n_deltas = AdaVae.compute_posterior_deltas(a_posterior, n_posterior, thresh_mode=self.cfg.ada_thresh_mode)

        # shared elements that need to be averaged, computed per pair in the batch.
        old_p_shared_mask = AdaVae.estimate_shared_mask(a_p_deltas, ratio=self.cfg.ada_thresh_ratio)
        old_n_shared_mask = AdaVae.estimate_shared_mask(a_n_deltas, ratio=self.cfg.ada_thresh_ratio)

        # modify threshold based on criterion and recompute if necessary
        # CORE of this approach!
        p_shared_mask, n_shared_mask = compute_constrained_masks(a_p_deltas, old_p_shared_mask, a_n_deltas, old_n_shared_mask)

        # make averaged variables
        # TODO: this can be merged with the gadavae/badavae
        ave_ap_a_posterior, ave_ap_p_posterior = AdaVae.make_averaged_distributions(a_posterior, p_posterior, p_shared_mask, average_mode=self.cfg.ada_average_mode)
        ave_an_a_posterior, ave_an_n_posterior = AdaVae.make_averaged_distributions(a_posterior, n_posterior, n_shared_mask, average_mode=self.cfg.ada_average_mode)
        ave_a_posterior = compute_average_distribution(ave_ap_a_posterior, ave_an_a_posterior, average_mode=self.cfg.ada_average_mode)

        # compute anchor average using the adaptive threshold | TODO: this doesn't really make sense
        anchor_ave_logs = {}
        if self.cfg.gada_anchor_ave_mode == 'thresh':
            ave_shared_mask = p_shared_mask * n_shared_mask
            ave_params, _ = AdaVae.make_averaged_distributions(a_posterior, ave_a_posterior, ave_shared_mask, average_mode=self.cfg.ada_average_mode)
            anchor_ave_logs['ave_shared'] = ave_shared_mask.sum(dim=1).float().mean()

        new_ds_posterior = ave_a_posterior, ave_ap_p_posterior, ave_an_n_posterior

        return new_ds_posterior, ds_prior, {
            'p_shared_before': old_p_shared_mask.sum(dim=1).float().mean(),
            'p_shared_after':      p_shared_mask.sum(dim=1).float().mean(),
            'n_shared_before': old_n_shared_mask.sum(dim=1).float().mean(),
            'n_shared_after':      n_shared_mask.sum(dim=1).float().mean(),
            **anchor_ave_logs,
        }


# ========================================================================= #
# END                                                                       #
# ========================================================================= #

