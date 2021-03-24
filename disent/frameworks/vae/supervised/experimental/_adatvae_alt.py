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
import warnings
from dataclasses import dataclass
from dataclasses import field
from typing import Mapping
from typing import Sequence

import torch
import numpy as np
from torch.distributions import Normal

from disent.frameworks.helper.triplet_loss import configured_triplet, configured_dist_triplet
from disent.frameworks.vae.unsupervised import Vae
from disent.frameworks.vae.supervised import TripletVae
from disent.frameworks.vae.weaklysupervised import AdaVae
import logging

from disent.frameworks.vae.weaklysupervised._adavae import compute_average
from disent.schedule import CyclicSchedule
from disent.schedule import Schedule
from experiment.util.hydra_utils import instantiate_recursive
from experiment.util.hydra_utils import make_target_dict


log = logging.getLogger(__name__)


# ========================================================================= #
# Guided Ada Vae                                                            #
# ========================================================================= #


class AltAdaTripletVae(TripletVae):

    REQUIRED_OBS = 3

    @dataclass
    class cfg(TripletVae.cfg, AdaVae.cfg):
        pass

    def hook_intercept_zs(self, zs_params: Sequence['Params']):
        # triplet vae intercept -- in case detached
        zs_params, intercept_logs = super().hook_intercept_zs(zs_params)
        # ================================= #
        # get components
        # ================================= #
        a_z_params, p_z_params, n_z_params = zs_params
        a_d_posterior, _ = self.params_to_dists(a_z_params)
        p_d_posterior, _ = self.params_to_dists(p_z_params)
        n_d_posterior, _ = self.params_to_dists(n_z_params)
        # ================================= #
        # compute averaged triplet
        # ================================= #
        # compute the deltas
        ap_z_deltas = AdaVae.compute_deltas(a_d_posterior, p_d_posterior, thresh_mode=self.cfg.thresh_mode)
        an_z_deltas = AdaVae.compute_deltas(a_d_posterior, n_d_posterior, thresh_mode=self.cfg.thresh_mode)
        pn_z_deltas = AdaVae.compute_deltas(p_d_posterior, n_d_posterior, thresh_mode=self.cfg.thresh_mode)
        # shared elements that need to be averaged, computed per pair in the batch.
        ap_share_mask = AdaVae.estimate_shared_mask(ap_z_deltas, ratio=self.cfg.thresh_ratio)
        an_share_mask = AdaVae.estimate_shared_mask(an_z_deltas, ratio=self.cfg.thresh_ratio)
        pn_share_mask = AdaVae.estimate_shared_mask(pn_z_deltas, ratio=self.cfg.thresh_ratio)
        # compute all averages
        ave_ap_z_params, ave_pa_z_params = AdaVae.make_averaged(a_z_params, p_z_params, ap_share_mask, average_mode=self.cfg.average_mode)
        ave_an_z_params, ave_na_z_params = AdaVae.make_averaged(a_z_params, n_z_params, an_share_mask, average_mode=self.cfg.average_mode)
        ave_pn_z_params, ave_np_z_params = AdaVae.make_averaged(p_z_params, n_z_params, pn_share_mask, average_mode=self.cfg.average_mode)
        # compute averages
        ave_a_mean, ave_a_logvar = compute_average(ave_ap_z_params.mean, ave_ap_z_params.logvar, ave_an_z_params.mean, ave_an_z_params.logvar, average_mode=self.cfg.average_mode)
        ave_p_mean, ave_p_logvar = compute_average(ave_pa_z_params.mean, ave_pa_z_params.logvar, ave_pn_z_params.mean, ave_pn_z_params.logvar, average_mode=self.cfg.average_mode)
        ave_n_mean, ave_n_logvar = compute_average(ave_na_z_params.mean, ave_na_z_params.logvar, ave_np_z_params.mean, ave_np_z_params.logvar, average_mode=self.cfg.average_mode)
        # create new z_params
        new_z_params = (
            a_z_params.__class__(ave_a_mean, ave_a_logvar),
            a_z_params.__class__(ave_p_mean, ave_p_logvar),
            a_z_params.__class__(ave_n_mean, ave_n_logvar),
        )
        # ================================= #
        # return values
        # ================================= #
        return new_z_params, {
            **intercept_logs,
            'ap_shared': ap_share_mask.sum(dim=1).float().mean(),
            'an_shared': an_share_mask.sum(dim=1).float().mean(),
            'pn_shared': pn_share_mask.sum(dim=1).float().mean(),
        }


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
