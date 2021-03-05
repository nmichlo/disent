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

import torch

from disent.frameworks.vae.unsupervised import BetaVae


# ========================================================================= #
# tvae                                                                      #
# ========================================================================= #


class TripleVae(BetaVae):

    def compute_training_loss(self, batch, batch_idx):
        (x0, x1, x2), (x0_targ, x1_targ, x2_targ) = batch['x'], batch['x_targ']

        # FORWARD
        # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #
        # latent distribution parametrisation
        z0_params = self.training_encode_params(x0)
        z1_params = self.training_encode_params(x1)
        z2_params = self.training_encode_params(x2)
        # sample from latent distribution
        (d0_posterior, d0_prior), z0_sampled = self.training_params_to_distributions_and_sample(z0_params)
        (d1_posterior, d1_prior), z1_sampled = self.training_params_to_distributions_and_sample(z1_params)
        (d2_posterior, d2_prior), z2_sampled = self.training_params_to_distributions_and_sample(z2_params)
        # reconstruct without the final activation
        x0_partial_recon = self.training_decode_partial(z0_sampled)
        x1_partial_recon = self.training_decode_partial(z1_sampled)
        x2_partial_recon = self.training_decode_partial(z2_sampled)
        # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #

        # LOSS
        # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #
        # reconstruction error
        recon_loss_0 = self.training_recon_loss(x0_partial_recon, x0_targ)  # E[log p(x|z)]
        recon_loss_1 = self.training_recon_loss(x1_partial_recon, x1_targ)  # E[log p(x|z)]
        recon_loss_2 = self.training_recon_loss(x2_partial_recon, x2_targ)  # E[log p(x|z)]
        ave_recon_loss = (recon_loss_0 + recon_loss_1 + recon_loss_2) / 3
        # KL divergence
        kl_loss_0 = self.training_kl_loss(d0_posterior, d0_prior)  # D_kl(q(z|x) || p(z|x))
        kl_loss_1 = self.training_kl_loss(d1_posterior, d1_prior)  # D_kl(q(z|x) || p(z|x))
        kl_loss_2 = self.training_kl_loss(d2_posterior, d2_prior)  # D_kl(q(z|x) || p(z|x))
        ave_kl_loss = (kl_loss_0 + kl_loss_1 + kl_loss_2) / 3
        # compute kl regularisation
        ave_kl_reg_loss = self.training_regularize_kl(ave_kl_loss)
        # augment loss
        aug_loss, logs_aug = self.augment_loss(
            ds_posterior=(d0_posterior, d1_posterior, d2_posterior),
            xs_targ=(x0_targ, x1_targ, x2_targ),
        )
        # compute combined loss - must be same as the BetaVAE
        loss = ave_recon_loss + ave_kl_reg_loss + aug_loss
        # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #

        return {
            'train_loss': loss,
            'recon_loss': ave_recon_loss,
            'kl_reg_loss': ave_kl_reg_loss,
            'kl_loss': ave_kl_loss,
            'elbo': -(ave_recon_loss + ave_kl_loss),
            # differences
            **self.measure_differences(
                ds_posterior=(d0_posterior, d1_posterior, d2_posterior),
                xs_targ=(x0_targ, x1_targ, x2_targ),
            ),
            # extra loss
            **logs_aug,
        }

    def augment_loss(self, ds_posterior, xs_targ):
        return 0, {}

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
