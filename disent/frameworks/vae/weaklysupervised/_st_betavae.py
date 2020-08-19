import numpy as np
from disent.frameworks.vae.unsupervised import BetaVae
from disent.frameworks.vae.loss import bce_loss_with_logits, kl_normal_loss


# ========================================================================= #
# Swapped Target BetaVAE                                                    #
# ========================================================================= #


class SwappedTargetBetaVae(BetaVae):

    def __init__(self, make_optimizer_fn, make_model_fn, beta=4, swap_chance=0.1):
        super().__init__(make_optimizer_fn, make_model_fn, beta=beta)
        assert swap_chance >= 0
        self.swap_chance = swap_chance

    def compute_loss(self, batch, batch_idx):
        x0, x1 = batch

        # random change for the target not to be equal to the input
        x_input, x_target = x0, x0
        if np.random.random() < self.swap_chance:
            x_target = x1

        # FORWARD
        # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #
        # latent distribution parametrisation
        z_mean, z_logvar = self.encode_gaussian(x_input)
        # sample from latent distribution
        z = self.reparameterize(z_mean, z_logvar)
        # reconstruct without the final activation
        x_recon = self.decode_partial(z)
        # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #

        # LOSS
        # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #
        # reconstruction error
        recon_loss = bce_loss_with_logits(x_recon, x_target)  # E[log p(x|z)]
        # KL divergence
        kl_loss = kl_normal_loss(z_mean, z_logvar)     # D_kl(q(z|x) || p(z|x))
        # compute kl regularisation
        kl_reg_loss = self.kl_regularization(kl_loss)
        # compute combined loss
        loss = recon_loss + kl_reg_loss
        # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #

        return {
            'train_loss': loss,
            'recon_loss': recon_loss,
            'kl_reg_loss': kl_reg_loss,
            'kl_loss': kl_loss,
            'elbo': -(recon_loss + kl_loss),
        }


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
