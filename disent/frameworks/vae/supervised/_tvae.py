from dataclasses import dataclass

import torch
from disent.frameworks.vae.unsupervised import BetaVae
from disent.frameworks.helper.triplet_loss import configured_triplet, TripletLossConfig, TripletConfigTypeHint


# ========================================================================= #
# tvae                                                                      #
# ========================================================================= #


class TripletVae(BetaVae):

    @dataclass
    class cfg(BetaVae.cfg, TripletLossConfig):
        # tvae: no loss from decoder -> encoder
        detach: bool = False
        detach_decoder: bool = True
        detach_no_kl: bool = False
        detach_logvar: float = -2  # std = 0.5, logvar = ln(std**2) ~= -2,77

    def compute_training_loss(self, batch, batch_idx):
        (a_x, p_x, n_x), (a_x_targ, p_x_targ, n_x_targ) = batch['x'], batch['x_targ']

        # FORWARD
        # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #
        # latent distribution parametrisation
        a_z_params = self.training_encode_params(a_x)
        p_z_params = self.training_encode_params(p_x)
        n_z_params = self.training_encode_params(n_x)
        # get zeros
        if self.cfg.detach and (self.cfg.detach_logvar is not None):
            a_z_params.logvar = torch.full_like(a_z_params.logvar, self.cfg.detach_logvar)
            p_z_params.logvar = torch.full_like(p_z_params.logvar, self.cfg.detach_logvar)
            n_z_params.logvar = torch.full_like(n_z_params.logvar, self.cfg.detach_logvar)
        # sample from latent distribution
        (a_d_posterior, a_d_prior), a_z_sampled = self.training_params_to_distributions_and_sample(a_z_params)
        (p_d_posterior, p_d_prior), p_z_sampled = self.training_params_to_distributions_and_sample(p_z_params)
        (n_d_posterior, n_d_prior), n_z_sampled = self.training_params_to_distributions_and_sample(n_z_params)
        # detach samples so no gradient flows through them
        if self.cfg.detach and self.cfg.detach_decoder:
            a_z_sampled = a_z_sampled.detach()
            p_z_sampled = p_z_sampled.detach()
            n_z_sampled = n_z_sampled.detach()
        # reconstruct without the final activation
        a_x_partial_recon = self.training_decode_partial(a_z_sampled)
        p_x_partial_recon = self.training_decode_partial(p_z_sampled)
        n_x_partial_recon = self.training_decode_partial(n_z_sampled)
        # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #

        # LOSS
        # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #
        # reconstruction error
        a_recon_loss = self.training_recon_loss(a_x_partial_recon, a_x_targ)  # E[log p(x|z)]
        p_recon_loss = self.training_recon_loss(p_x_partial_recon, p_x_targ)  # E[log p(x|z)]
        n_recon_loss = self.training_recon_loss(n_x_partial_recon, n_x_targ)  # E[log p(x|z)]
        ave_recon_loss = (a_recon_loss + p_recon_loss + n_recon_loss) / 3
        # KL divergence
        if self.cfg.detach and self.cfg.detach_no_kl:
            ave_kl_loss = 0
        else:
            a_kl_loss = self.training_kl_loss(a_d_posterior, a_d_prior)  # D_kl(q(z|x) || p(z|x))
            p_kl_loss = self.training_kl_loss(p_d_posterior, p_d_prior)  # D_kl(q(z|x) || p(z|x))
            n_kl_loss = self.training_kl_loss(n_d_posterior, n_d_prior)  # D_kl(q(z|x) || p(z|x))
            ave_kl_loss = (a_kl_loss + p_kl_loss + n_kl_loss) / 3
        # compute kl regularisation
        ave_kl_reg_loss = self.training_regularize_kl(ave_kl_loss)
        # augment loss
        augment_loss, augment_loss_logs = self.augment_loss(z_means=(a_z_params.mean, p_z_params.mean, n_z_params.mean))
        # compute combined loss - must be same as the BetaVAE
        loss = ave_recon_loss + ave_kl_reg_loss + augment_loss
        # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #

        return {
            'train_loss': loss,
            'recon_loss': ave_recon_loss,
            'kl_reg_loss': ave_kl_reg_loss,
            'kl_loss': ave_kl_loss,
            'elbo': -(ave_recon_loss + ave_kl_loss),
            **augment_loss_logs,
        }

    def augment_loss(self, z_means):
        return self.augment_loss_triplet(z_means, self.cfg)

    @staticmethod
    def augment_loss_triplet(z_means, cfg: TripletConfigTypeHint):
        anc, pos, neg = z_means
        # loss is scaled and everything
        loss = configured_triplet(anc, pos, neg, cfg=cfg)
        # return loss & log
        return loss, {
            f'{cfg.triplet_loss}_L{cfg.triplet_p}': loss
        }


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
