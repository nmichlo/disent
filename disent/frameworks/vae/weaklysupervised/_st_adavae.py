import numpy as np
from disent.frameworks.vae.loss import bce_loss_with_logits, kl_normal_loss
from disent.frameworks.vae.weaklysupervised import AdaVae


# ========================================================================= #
# Swapped Target AdaVae                                                     #
# ========================================================================= #


class SwappedTargetAdaVae(AdaVae):

    def __init__(
            self,
            make_optimizer_fn,
            make_model_fn,
            make_augment_fn=None,
            beta=4,
            average_mode='gvae',
            swap_chance=0.1
    ):
        super().__init__(make_optimizer_fn, make_model_fn, make_augment_fn=make_augment_fn, beta=beta, average_mode=average_mode)
        assert swap_chance >= 0
        self.swap_chance = swap_chance

    def compute_training_loss(self, batch, batch_idx):
        (x0, x1), (x0_targ, x1_targ) = batch['x'], batch['x_targ']

        # random change for the target not to be equal to the input
        if np.random.random() < self.swap_chance:
            x0_targ, x1_targ = x1_targ, x0_targ

        # FORWARD
        # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #
        # latent distribution parametrisation
        z0_mean, z0_logvar = self.encode_gaussian(x0)
        z1_mean, z1_logvar = self.encode_gaussian(x1)
        # intercept and mutate z [SPECIFIC TO ADAVAE]
        (z0_mean, z0_logvar, z1_mean, z1_logvar), intercept_logs = self.intercept_z(z0_mean, z0_logvar, z1_mean, z1_logvar)
        # sample from latent distribution
        z0_sampled = self.reparameterize(z0_mean, z0_logvar)
        z1_sampled = self.reparameterize(z1_mean, z1_logvar)
        # reconstruct without the final activation
        x0_recon = self.decode_partial(z0_sampled)
        x1_recon = self.decode_partial(z1_sampled)
        # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #

        # LOSS
        # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #
        # reconstruction error
        recon0_loss = bce_loss_with_logits(x0_recon, x0_targ)  # E[log p(x|z)]
        recon1_loss = bce_loss_with_logits(x1_recon, x1_targ)  # E[log p(x|z)]
        ave_recon_loss = (recon0_loss + recon1_loss) / 2
        # KL divergence
        kl0_loss = kl_normal_loss(z0_mean, z0_logvar)     # D_kl(q(z|x) || p(z|x))
        kl1_loss = kl_normal_loss(z1_mean, z1_logvar)     # D_kl(q(z|x) || p(z|x))
        ave_kl_loss = (kl0_loss + kl1_loss) / 2
        # compute kl regularisation
        ave_kl_reg_loss = self.kl_regularization(ave_kl_loss)
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


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
