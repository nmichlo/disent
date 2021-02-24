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
from dataclasses import dataclass

from torch.distributions import Distribution

from disent.frameworks.vae.unsupervised import BetaVae


# ========================================================================= #
# Ada-GVAE                                                                  #
# ========================================================================= #


class AdaVae(BetaVae):

    """
    Weakly Supervised Disentanglement Learning Without Compromises: https://arxiv.org/abs/2002.02886
    - pretty much a beta-vae with averaging between decoder outputs to form weak supervision signal.
    - GAdaVAE:   Averaging from https://arxiv.org/abs/1809.02383
    - ML-AdaVAE: Averaging from https://arxiv.org/abs/1705.08841

    MODIFICATION:
    - Symmetric KL Calculation used by default, described in: https://openreview.net/pdf?id=8VXvj1QNRl1
    """

    @dataclass
    class cfg(BetaVae.cfg):
        average_mode: str = 'gvae'
        symmetric_kl: bool = True

    def __init__(self, make_optimizer_fn, make_model_fn, batch_augment=None, cfg: cfg = None):
        super().__init__(make_optimizer_fn, make_model_fn, batch_augment=batch_augment, cfg=cfg)
        # averaging modes
        self._compute_average_fn = {
            'gvae': compute_average_gvae,
            'ml-vae': compute_average_ml_vae
        }[self.cfg.average_mode]

    def compute_training_loss(self, batch, batch_idx):
        """
        (✓) Visual inspection against reference implementation:
            https://github.com/google-research/disentanglement_lib (GroupVAEBase & MLVae)
            - only difference for GroupVAEBase & MLVae how the mean parameterisations are calculated
        """
        (x0, x1), (x0_targ, x1_targ) = batch['x'], batch['x_targ']

        # FORWARD
        # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #
        # latent distribution parameterizations
        z0_params = self.training_encode_params(x0)
        z1_params = self.training_encode_params(x1)
        # intercept and mutate z [SPECIFIC TO ADAVAE]
        (z0_params, z1_params), intercept_logs = self.intercept_z(all_params=(z0_params, z1_params))
        # sample from latent distribution
        (d0_posterior, d0_prior), z0_sampled = self.training_params_to_distributions_and_sample(z0_params)
        (d1_posterior, d1_prior), z1_sampled = self.training_params_to_distributions_and_sample(z1_params)
        # reconstruct without the final activation
        x0_partial_recon = self.training_decode_partial(z0_sampled)
        x1_partial_recon = self.training_decode_partial(z1_sampled)
        # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #

        # LOSS
        # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #
        # reconstruction error
        recon0_loss = self.training_recon_loss(x0_partial_recon, x0_targ)  # E[log p(x|z)]
        recon1_loss = self.training_recon_loss(x1_partial_recon, x1_targ)  # E[log p(x|z)]
        ave_recon_loss = (recon0_loss + recon1_loss) / 2
        # KL divergence
        kl0_loss = self.training_kl_loss(d0_posterior, d0_prior)  # D_kl(q(z|x) || p(z|x), d0_prior)
        kl1_loss = self.training_kl_loss(d1_posterior, d1_prior)  # D_kl(q(z|x) || p(z|x), d1_prior)
        ave_kl_loss = (kl0_loss + kl1_loss) / 2
        # compute kl regularisation
        ave_kl_reg_loss = self.training_regularize_kl(ave_kl_loss)
        # compute combined loss - must be same as the BetaVAE
        loss = ave_recon_loss + ave_kl_reg_loss
        # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #

        return {
            'train_loss': loss,
            'recon_loss': ave_recon_loss,
            'kl_reg_loss': ave_kl_reg_loss,
            'kl_loss': ave_kl_loss,
            'elbo': -(ave_recon_loss + ave_kl_loss),
            **intercept_logs,
        }

    def intercept_z(self, all_params):
        """
        Adaptive VAE Glue Method, putting the various components together
        1. find differences between deltas
        2. estimate a threshold for differences
        3. compute a shared mask from this threshold
        4. average together elements that should be considered shared

        (✓) Visual inspection against reference implementation:
            https://github.com/google-research/disentanglement_lib (aggregate_argmax)
        """
        z0_params, z1_params = all_params
        # compute the deltas
        d0_posterior, _ = self.training_params_to_distributions(z0_params)  # numerical accuracy errors
        d1_posterior, _ = self.training_params_to_distributions(z1_params)  # numerical accuracy errors
        z_deltas = self.compute_kl_deltas(d0_posterior, d1_posterior, symmetric_kl=self.cfg.symmetric_kl)
        # shared elements that need to be averaged, computed per pair in the batch.
        share_mask = self.compute_shared_mask(z_deltas)
        # compute average posteriors
        new_args = self.compute_averaged(z0_params, z1_params, share_mask, compute_average_fn=self._compute_average_fn)
        # return new args & generate logs
        return new_args, {'shared': share_mask.sum(dim=1).float().mean()}

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
    # HELPER                                                                #
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

    @classmethod
    def compute_kl_deltas(cls, d0_posterior: Distribution, d1_posterior: Distribution, symmetric_kl: bool):
        """
        (✓) Visual inspection against reference implementation
        https://github.com/google-research/disentanglement_lib (compute_kl)
        - difference is that they don't multiply by 0.5 to get true kl, but that's not needed

        TODO: this might be numerically unstable with f32 passed to distributions
        """
        # shared elements that need to be averaged, computed per pair in the batch.
        # [𝛿_i ...]
        if symmetric_kl:
            # FROM: https://openreview.net/pdf?id=8VXvj1QNRl1
            kl_deltas_d1_d0 = torch.distributions.kl_divergence(d1_posterior, d0_posterior)
            kl_deltas_d0_d1 = torch.distributions.kl_divergence(d0_posterior, d1_posterior)
            kl_deltas = (0.5 * kl_deltas_d1_d0) + (0.5 * kl_deltas_d0_d1)
        else:
            kl_deltas = torch.distributions.kl_divergence(d1_posterior, d0_posterior)
        # return values
        return kl_deltas

    @classmethod
    def compute_shared_mask(cls, z_deltas):
        """
        Core of the adaptive VAE algorithm, estimating which factors
        have changed (or in this case which are shared and should remained unchanged
        by being be averaged) between pairs of observations.

        (✓) Visual inspection against reference implementation:
            https://github.com/google-research/disentanglement_lib (aggregate_argmax)
            - Implementation conversion is non-trivial, items are histogram binned.
              If we are in the second histogram bin, ie. 1, then kl_deltas <= kl_threshs
            - TODO: (aggregate_labels) An alternative mode exists where you can bind the
                    latent variables to any individual label, by one-hot encoding which
                    latent variable should not be shared: "enforce that each dimension
                    of the latent code learns one factor (dimension 1 learns factor 1)
                    and enforce that each factor of variation is encoded in a single
                    dimension."
        """
        # threshold τ
        z_threshs = cls.estimate_threshold(z_deltas)
        # true if 'unchanged' and should be average
        shared_mask = z_deltas < z_threshs
        # return
        return shared_mask


    @classmethod
    def estimate_threshold(cls, kl_deltas, keepdim=True):
        """
        Compute the threshold for each image pair in a batch of kl divergences of all elements of the latent distributions.
        It should be noted that for a perfectly trained model, this threshold is always correct.

        (✓) Visual inspection against reference implementation:
            https://github.com/google-research/disentanglement_lib (aggregate_argmax)
        """
        maximums = kl_deltas.max(axis=1, keepdim=keepdim).values
        minimums = kl_deltas.min(axis=1, keepdim=keepdim).values
        return (0.5 * minimums) + (0.5 * maximums)

    @classmethod
    def compute_averaged(cls, z0_params, z1_params, share_mask, compute_average_fn: callable):
        # compute average posteriors
        ave_mean, ave_logvar = compute_average_fn(
            z0_params.mean, z0_params.logvar,
            z1_params.mean, z1_params.logvar,
        )
        # select averages
        ave_z0_mean = torch.where(share_mask, ave_mean, z0_params.mean)
        ave_z1_mean = torch.where(share_mask, ave_mean, z1_params.mean)
        ave_z0_logvar = torch.where(share_mask, ave_logvar, z0_params.logvar)
        ave_z1_logvar = torch.where(share_mask, ave_logvar, z1_params.logvar)
        # return values
        return z0_params.__class__(ave_z0_mean, ave_z0_logvar), z1_params.__class__(ave_z1_mean, ave_z1_logvar)


# ========================================================================= #
# Averaging Functions                                                       #
# ========================================================================= #


def compute_average_gvae(z0_mean, z0_logvar, z1_mean, z1_logvar):
    """
    Compute the arithmetic mean of the encoder distributions.
    - Ada-GVAE Averaging function

    (✓) Visual inspection against reference implementation:
        https://github.com/google-research/disentanglement_lib (GroupVAEBase.model_fn)
    """
    # helper
    z0_var, z1_var = z0_logvar.exp(), z1_logvar.exp()
    # averages
    ave_var = (z0_var + z1_var) * 0.5
    ave_mean = (z0_mean + z1_mean) * 0.5
    # mean, logvar
    return ave_mean, ave_var.log()  # natural log

def compute_average_ml_vae(z0_mean, z0_logvar, z1_mean, z1_logvar):
    """
    Compute the product of the encoder distributions.
    - Ada-ML-VAE Averaging function

    (✓) Visual inspection against reference implementation:
        https://github.com/google-research/disentanglement_lib (MLVae.model_fn)

    # TODO: recheck
    """
    # helper
    z0_var, z1_var = z0_logvar.exp(), z1_logvar.exp()
    # Diagonal matrix inverse: E^-1 = 1 / E
    # https://proofwiki.org/wiki/Inverse_of_Diagonal_Matrix
    z0_invvar, z1_invvar = z0_var.reciprocal(), z1_var.reciprocal()
    # average var: E^-1 = E1^-1 + E2^-1
    # disentanglement_lib: ave_var = 2 * z0_var * z1_var / (z0_var + z1_var)
    ave_var = 2 * (z0_invvar + z1_invvar).reciprocal()
    # average mean: u^T = (u1^T E1^-1 + u2^T E2^-1) E
    # disentanglement_lib: ave_mean = (z0_mean/z0_var + z1_mean/z1_var) * ave_var * 0.5
    ave_mean = (z0_mean*z0_invvar + z1_mean*z1_invvar) * ave_var * 0.5
    # mean, logvar
    return ave_mean, ave_var.log()  # natural log


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
