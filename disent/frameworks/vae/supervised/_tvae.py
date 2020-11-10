from dataclasses import dataclass

import torch
from disent.frameworks.vae.loss import bce_loss_with_logits, kl_normal_loss
from disent.frameworks.vae.unsupervised import BetaVae
from disent.loss.triplet import triplet_loss


# ========================================================================= #
# tvae                                                                      #
# ========================================================================= #


class TripletVae(BetaVae):

    @dataclass
    class cfg(BetaVae.cfg):
        # tvae: triplet stuffs
        triplet_loss: str = 'triplet',
        triplet_margin_min: float = 0.1,
        triplet_margin: float = 10,
        triplet_scale: float = 100,
        triplet_p: int = 2,
        # tvae: no loss from decoder -> encoder
        detach: bool = False,
        detach_decoder: bool = True,
        detach_no_kl: bool = False,
        detach_logvar: float = -2,  # std = 0.5, logvar = ln(std**2) ~= -2,77

    def compute_training_loss(self, batch, batch_idx):
        (a_x, p_x, n_x), (a_x_targ, p_x_targ, n_x_targ) = batch['x'], batch['x_targ']

        # FORWARD
        # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #
        # latent distribution parametrisation
        a_z_mean, a_z_logvar = self.encode_gaussian(a_x)
        p_z_mean, p_z_logvar = self.encode_gaussian(p_x)
        n_z_mean, n_z_logvar = self.encode_gaussian(n_x)
        # get zeros
        if self.cfg.detach and (self.cfg.detach_logvar is not None):
            a_z_logvar = torch.full_like(a_z_logvar, self.cfg.detach_logvar)
            p_z_logvar = torch.full_like(p_z_logvar, self.cfg.detach_logvar)
            n_z_logvar = torch.full_like(n_z_logvar, self.cfg.detach_logvar)
        # sample from latent distribution
        a_z_sampled = self.reparameterize(a_z_mean, a_z_logvar)
        p_z_sampled = self.reparameterize(p_z_mean, p_z_logvar)
        n_z_sampled = self.reparameterize(n_z_mean, n_z_logvar)
        # detach samples so no gradient flows through them
        if self.cfg.detach and self.cfg.detach_decoder:
            a_z_sampled = a_z_sampled.detach()
            p_z_sampled = p_z_sampled.detach()
            n_z_sampled = n_z_sampled.detach()
        # reconstruct without the final activation
        a_x_recon = self.decode_partial(a_z_sampled)
        p_x_recon = self.decode_partial(p_z_sampled)
        n_x_recon = self.decode_partial(n_z_sampled)
        # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #

        # LOSS
        # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #
        # reconstruction error
        a_recon_loss = bce_loss_with_logits(a_x_recon, a_x_targ)  # E[log p(x|z)]
        p_recon_loss = bce_loss_with_logits(p_x_recon, p_x_targ)  # E[log p(x|z)]
        n_recon_loss = bce_loss_with_logits(n_x_recon, n_x_targ)  # E[log p(x|z)]
        ave_recon_loss = (a_recon_loss + p_recon_loss + n_recon_loss) / 3
        # KL divergence
        if self.cfg.detach and self.cfg.detach_no_kl:
            ave_kl_loss = 0
            ave_kl_reg_loss = 0
        else:
            a_kl_loss = kl_normal_loss(a_z_mean, a_z_logvar)  # D_kl(q(z|x) || p(z|x))
            p_kl_loss = kl_normal_loss(p_z_mean, p_z_logvar)  # D_kl(q(z|x) || p(z|x))
            n_kl_loss = kl_normal_loss(n_z_mean, n_z_logvar)  # D_kl(q(z|x) || p(z|x))
            ave_kl_loss = (a_kl_loss + p_kl_loss + n_kl_loss) / 3
            # compute kl regularisation
            ave_kl_reg_loss = self.kl_regularization(ave_kl_loss)
        # augment loss (0 for this)
        augment_loss, augment_loss_logs = self.augment_loss(z_means=(a_z_mean, p_z_mean, n_z_mean), z_logvars=(a_z_logvar, p_z_logvar, n_z_logvar), z_samples=(a_z_sampled, p_z_sampled, n_z_sampled))
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

    def augment_loss(self, z_means, z_logvars, z_samples):
        a_z_mean, p_z_mean, n_z_mean = z_means
        return augment_loss_triplet(a_z_mean, p_z_mean, n_z_mean, scale=self.cfg.triplet_scale, margin=self.cfg.triplet_margin, p=self.cfg.triplet_p)


# ========================================================================= #
# Default Triplet                                                           #
# ========================================================================= #


def augment_loss_triplet(a_z_mean, p_z_mean, n_z_mean, scale=1., margin=10., p=2):
    augmented_loss = scale * triplet_loss(a_z_mean, p_z_mean, n_z_mean, margin_min=None, margin_max=margin, p=p)
    return augmented_loss, {
        f'triplet_L{p}': augmented_loss,
    }


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
