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

#  ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~
#  MIT License
#
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#
from dataclasses import dataclass

import torch

from disent.frameworks.helper.triplet_loss import configured_triplet
from disent.frameworks.vae.supervised.experimental._adatvae import AdaTripletVae


# ========================================================================= #
# tvae                                                                      #
# ========================================================================= #


class ReconTripletVae(AdaTripletVae):

    @dataclass
    class cfg(AdaTripletVae.cfg):
        # OVERRIDE - triplet vae configs
        detach: bool = False
        detach_decoder: bool = False
        detach_no_kl: bool = False
        detach_logvar: float = 0  # std = 0.5, logvar = ln(std**2) ~= -2,77
        # OVERRIDE - triplet loss configs
        triplet_scale: float = 0
        # TRIPLET_MODE
        recon_triplet_mode: str = 'triplet'
        # OVERRIDE ADAVAE
        # adatvae: what version of triplet to use
        triplet_mode: str = 'ada_p_orig_lerp'
        # adatvae: annealing
        lerp_step_start: int = 3600
        lerp_step_end: int = 14400
        lerp_goal: float = 1.0

    def augment_loss(self, ds_posterior, xs_targ):
        # get values
        a_z_mean, p_z_mean_OLD, n_z_mean_OLD = ds_posterior[0].mean, ds_posterior[1].mean, ds_posterior[2].mean
        a_x_targ, p_x_targ, n_x_targ = xs_targ

        # CORE OF THIS APPROACH
        # ++++++++++++++++++++++++++++++++++++++++++ #
        # calculate which are wrong!
        a_p_losses = self._recons.training_compute_loss(a_x_targ, p_x_targ, reduction='none').mean(dim=(-3, -2, -1))
        a_n_losses = self._recons.training_compute_loss(a_x_targ, n_x_targ, reduction='none').mean(dim=(-3, -2, -1))
        # swap if wrong!
        swap_mask = (a_p_losses > a_n_losses)[:, None].repeat(1, a_z_mean.shape[-1])
        p_z_mean = torch.where(swap_mask, n_z_mean_OLD, p_z_mean_OLD)
        n_z_mean = torch.where(swap_mask, p_z_mean_OLD, n_z_mean_OLD)
        # ++++++++++++++++++++++++++++++++++++++++++ #

        if self.cfg.recon_triplet_mode == 'triplet':
            triplet_loss = configured_triplet(a_z_mean, p_z_mean, n_z_mean, cfg=self.cfg)
            logs_triplet = {}
        elif self.cfg.recon_triplet_mode == 'ada_triplet':
            self.steps += 1
            triplet_loss, logs_triplet = self.ada_triplet_loss(
                zs_mean=(ds_posterior[0].mean, ds_posterior[1].mean, ds_posterior[2].mean),
                step=self.steps,
                cfg=self.cfg,
            )
        else:
            raise KeyError

        return triplet_loss, {
            'recon_triplet_loss': triplet_loss,
            **self.measure_differences(ds_posterior, xs_targ),
            **logs_triplet,
        }

    def measure_differences(self, ds_posterior, xs_targ):
        d0_posterior, d1_posterior, d2_posterior = ds_posterior
        x0_targ, x1_targ, x2_targ = xs_targ

        # TESTS
        # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #
        # EFFECTIVELY SUMMED MSE WITH TERM: (u_1 - u_0)**2
        kl_0_1 = self._distributions.compute_kl_loss(d0_posterior, d1_posterior, reduction='none')
        kl_0_2 = self._distributions.compute_kl_loss(d0_posterior, d2_posterior, reduction='none')
        mu_0_1 = (d0_posterior.mean - d1_posterior.mean) ** 2
        mu_0_2 = (d0_posterior.mean - d2_posterior.mean) ** 2
        # z differences
        kl_mu_differences_all = torch.mean(((kl_0_1 < kl_0_2) ^ (mu_0_1 < mu_0_2)).to(torch.float32))
        kl_mu_differences_ave = torch.mean(((kl_0_1.mean(dim=-1) < kl_0_2.mean(dim=-1)) ^ (mu_0_1.mean(dim=-1) < mu_0_2.mean(dim=-1))).to(torch.float32))
        # obs differences
        assert self.cfg.recon_loss == 'mse', 'only mse loss is supported'
        xs_0_1 = self._recons.training_compute_loss(x0_targ, x1_targ, reduction='none').mean(dim=(-3, -2, -1))
        xs_0_2 = self._recons.training_compute_loss(x0_targ, x2_targ, reduction='none').mean(dim=(-3, -2, -1))
        # get differences
        kl_xs_differences_all = torch.mean(((kl_0_1 < kl_0_2) ^ (xs_0_1 < xs_0_2)[..., None]).to(torch.float32))
        mu_xs_differences_all = torch.mean(((mu_0_1 < mu_0_2) ^ (xs_0_1 < xs_0_2)[..., None]).to(torch.float32))
        kl_xs_differences_ave = torch.mean(((kl_0_1.mean(dim=-1) < kl_0_2.mean(dim=-1)) ^ (xs_0_1 < xs_0_2)).to(torch.float32))
        mu_xs_differences_ave = torch.mean(((mu_0_1.mean(dim=-1) < mu_0_2.mean(dim=-1)) ^ (xs_0_1 < xs_0_2)).to(torch.float32))
        # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #
        return {
            'kl_mu_differences_all': kl_mu_differences_all,
            'kl_mu_differences_ave': kl_mu_differences_ave,
            'kl_xs_differences_all': kl_xs_differences_all,
            'mu_xs_differences_all': mu_xs_differences_all,
            'kl_xs_differences_ave': kl_xs_differences_ave,
            'mu_xs_differences_ave': mu_xs_differences_ave,
        }


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
