import torch
from disent.frameworks.vae.weaklysupervised import AdaVae
from disent.frameworks.vae.loss import bce_loss_with_logits, kl_normal_loss


# ========================================================================= #
# Guided Ada Vae                                                            #
# ========================================================================= #


class GuidedAdaVae(AdaVae):
    
    def __init__(self, make_optimizer_fn, make_model_fn, beta=4, average_mode='gvae', anchor_ave_mode='average'):
        super().__init__(make_optimizer_fn, make_model_fn, beta=beta, average_mode=average_mode)
        # how the anchor is averaged
        assert anchor_ave_mode in {'thresh', 'average'}
        self.anchor_ave_mode = anchor_ave_mode

    def compute_loss(self, batch, batch_idx):
        a_x, p_x, n_x = batch
        # FORWARD
        # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #
        # latent distribution parametrisation
        a_z_mean, a_z_logvar = self.encode_gaussian(a_x)
        p_z_mean, p_z_logvar = self.encode_gaussian(p_x)
        n_z_mean, n_z_logvar = self.encode_gaussian(n_x)
        # intercept and mutate z [SPECIFIC TO ADAVAE]
        (a_z_mean, a_z_logvar, p_z_mean, p_z_logvar, n_z_mean, n_z_logvar), intercept_logs = self.intercept_z(a_z_mean, a_z_logvar, p_z_mean, p_z_logvar, n_z_mean, n_z_logvar)
        # sample from latent distribution
        a_z_sampled = self.reparameterize(a_z_mean, a_z_logvar)
        p_z_sampled = self.reparameterize(p_z_mean, p_z_logvar)
        n_z_sampled = self.reparameterize(n_z_mean, n_z_logvar)
        # reconstruct without the final activation
        a_x_recon = self.decode_partial(a_z_sampled)
        p_x_recon = self.decode_partial(p_z_sampled)
        n_x_recon = self.decode_partial(n_z_sampled)
        # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #

        # LOSS
        # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #
        # reconstruction error
        a_recon_loss = bce_loss_with_logits(a_x, a_x_recon)  # E[log p(x|z)]
        p_recon_loss = bce_loss_with_logits(p_x, p_x_recon)  # E[log p(x|z)]
        n_recon_loss = bce_loss_with_logits(n_x, n_x_recon)  # E[log p(x|z)]
        ave_recon_loss = (a_recon_loss + p_recon_loss + n_recon_loss) / 3
        # KL divergence
        a_kl_loss = kl_normal_loss(a_z_mean, a_z_logvar)     # D_kl(q(z|x) || p(z|x))
        p_kl_loss = kl_normal_loss(p_z_mean, p_z_logvar)     # D_kl(q(z|x) || p(z|x))
        n_kl_loss = kl_normal_loss(n_z_mean, n_z_logvar)     # D_kl(q(z|x) || p(z|x))
        ave_kl_loss = (a_kl_loss + p_kl_loss + n_kl_loss) / 3
        # compute kl regularisation
        ave_kl_reg_loss = self.kl_regularization(ave_kl_loss)
        # augment loss (0 for this)
        augment_loss, augment_loss_logs = self.augment_loss(a_z_mean, a_z_logvar, p_z_mean, p_z_logvar, n_z_mean, n_z_logvar)
        # compute combined loss - must be same as the BetaVAE
        loss = ave_recon_loss + ave_kl_reg_loss + augment_loss
        # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #

        return {
            'train_loss': loss,
            'recon_loss': ave_recon_loss,
            'kl_reg_loss': ave_kl_reg_loss,
            'kl_loss': ave_kl_loss,
            'elbo': -(ave_recon_loss + ave_kl_loss),
            **intercept_logs,
            **augment_loss_logs,
        }

    def intercept_z(self, a_z_mean, a_z_logvar, p_z_mean, p_z_logvar, n_z_mean, n_z_logvar):
        """
        *NB* arguments must satisfy: d(l, l2) < d(l, l3) [positive dist < negative dist]
        - This function assumes that the distance between labels l, l2, l3
          corresponding to z, z2, z3 satisfy the criteria d(l, l2) < d(l, l3)
          ie. l2 is the positive sample, l3 is the negative sample
        """
        # shared elements that need to be averaged, computed per pair in the batch.
        p_kl_deltas, p_kl_threshs, old_p_shared_mask = AdaVae.estimate_shared(a_z_mean, a_z_logvar, p_z_mean, p_z_logvar)
        n_kl_deltas, n_kl_threshs, old_n_shared_mask = AdaVae.estimate_shared(a_z_mean, a_z_logvar, n_z_mean, n_z_logvar)

        # modify threshold based on criterion and recompute if necessary
        # CORE of this approach!
        p_shared_mask, n_shared_mask = compute_constrained_masks(p_kl_deltas, old_p_shared_mask, n_kl_deltas, old_n_shared_mask)
        
        # make averaged variables
        pa_z_mean, pa_z_logvar, p_z_mean, p_z_logvar = self.make_averaged(a_z_mean, a_z_logvar, p_z_mean, p_z_logvar, p_shared_mask)
        na_z_mean, na_z_logvar, n_z_mean, n_z_logvar = self.make_averaged(a_z_mean, a_z_logvar, n_z_mean, n_z_logvar, n_shared_mask)
        ave_mean, ave_logvar = self.compute_average(pa_z_mean, pa_z_logvar, na_z_mean, na_z_logvar)

        anchor_ave_logs = {}
        if self.anchor_ave_mode == 'thresh':
            # compute anchor average using the adaptive threshold
            ave_shared_mask = p_shared_mask * n_shared_mask
            ave_mean, ave_logvar, _, _ = self.make_averaged(a_z_mean, a_z_logvar, ave_mean, ave_logvar, ave_shared_mask)
            anchor_ave_logs['ave_shared'] = ave_shared_mask.sum(dim=1).float().mean()

        new_args = ave_mean, ave_logvar, p_z_mean, p_z_logvar, n_z_mean, n_z_logvar
        return new_args, {
            'p_shared_before': old_p_shared_mask.sum(dim=1).float().mean(),
            'p_shared_after':      p_shared_mask.sum(dim=1).float().mean(),
            'n_shared_before': old_n_shared_mask.sum(dim=1).float().mean(),
            'n_shared_after':      n_shared_mask.sum(dim=1).float().mean(),
            **anchor_ave_logs,
        }
    
    def augment_loss(self, a_z_mean, a_z_logvar, p_z_mean, p_z_logvar, n_z_mean, n_z_logvar):
        return 0, {}


# ========================================================================= #
# HELPER                                                                    #
# ========================================================================= #


def compute_constrained_masks(p_kl_deltas, p_shared_mask, n_kl_deltas, n_shared_mask):
    # number of changed factors
    p_shared_num = torch.sum(p_shared_mask, dim=1, keepdim=True)
    n_shared_num = torch.sum(n_shared_mask, dim=1, keepdim=True)

    # POSITIVE SHARED MASK
    # order from smallest to largest
    p_sort_indices = torch.argsort(p_kl_deltas, dim=1)
    # p_shared should be at least n_shared
    new_p_shared_num = torch.max(p_shared_num, n_shared_num)

    # NEGATIVE SHARED MASK
    # order from smallest to largest
    n_sort_indices = torch.argsort(n_kl_deltas, dim=1)
    # n_shared should be at most p_shared
    new_n_shared_num = torch.min(p_shared_num, n_shared_num)

    # COMPUTE NEW MASKS
    new_p_shared_mask = torch.zeros_like(p_shared_mask)
    new_n_shared_mask = torch.zeros_like(n_shared_mask)
    for i, (new_shared_p, new_shared_n) in enumerate(zip(new_p_shared_num, new_n_shared_num)):
        new_p_shared_mask[i, p_sort_indices[i, :new_shared_p]] = True
        new_n_shared_mask[i, n_sort_indices[i, :new_shared_n]] = True

    # return masks
    return new_p_shared_mask, new_n_shared_mask


# ========================================================================= #
# END                                                                       #
# ========================================================================= #

