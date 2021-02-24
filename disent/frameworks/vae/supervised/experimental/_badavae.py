from dataclasses import dataclass

import torch
from disent.frameworks.vae.weaklysupervised import AdaVae


# ========================================================================= #
# Guided Ada Vae                                                            #
# ========================================================================= #


class BoundedAdaVae(AdaVae):

    @dataclass
    class cfg(AdaVae.cfg):
        pass

    def compute_training_loss(self, batch, batch_idx):
        (a_x, p_x, n_x), (a_x_targ, p_x_targ, n_x_targ) = batch['x'], batch['x_targ']

        # FORWARD
        # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #
        # latent distribution parametrisation
        a_z_params = self.training_encode_params(a_x)
        p_z_params = self.training_encode_params(p_x)
        n_z_params = self.training_encode_params(n_x)
        # intercept and mutate z [SPECIFIC TO ADAVAE]
        (a_z_params, p_z_params), intercept_logs = self.intercept_z(all_params=(a_z_params, p_z_params, n_z_params))
        # sample from latent distribution
        (d0_posterior, d0_prior), a_z_sampled = self.training_params_to_distributions_and_sample(a_z_params)
        (d1_posterior, d1_prior), p_z_sampled = self.training_params_to_distributions_and_sample(p_z_params)
        # reconstruct without the final activation
        a_x_partial_recon = self.training_decode_partial(a_z_sampled)
        p_x_partial_recon = self.training_decode_partial(p_z_sampled)
        # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #

        # LOSS
        # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #
        # reconstruction error
        a_recon_loss = self.training_recon_loss(a_x_partial_recon, a_x_targ)  # E[log p(x|z)]
        p_recon_loss = self.training_recon_loss(p_x_partial_recon, p_x_targ)  # E[log p(x|z)]
        ave_recon_loss = (a_recon_loss + p_recon_loss) / 2
        # KL divergence
        a_kl_loss = self.training_kl_loss(d0_posterior, d0_prior)     # D_kl(q(z|x) || p(z|x))
        p_kl_loss = self.training_kl_loss(d1_posterior, d1_prior)     # D_kl(q(z|x) || p(z|x))
        ave_kl_loss = (a_kl_loss + p_kl_loss) / 2
        # compute kl regularisation
        ave_kl_reg_loss = self.training_regularize_kl(ave_kl_loss)
        # augment loss (0 for this)
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
            **intercept_logs,
            **augment_loss_logs,
        }

    def intercept_z(self, all_params):
        a_z_params, p_z_params, n_z_params = all_params

        # get distributions
        a_d_posterior, _ = self.training_params_to_distributions(a_z_params)
        p_d_posterior, _ = self.training_params_to_distributions(p_z_params)
        n_d_posterior, _ = self.training_params_to_distributions(n_z_params)

        # get deltas
        a_p_deltas = AdaVae.compute_kl_deltas(a_d_posterior, p_d_posterior, symmetric_kl=self.cfg.symmetric_kl)
        a_n_deltas = AdaVae.compute_kl_deltas(a_d_posterior, n_d_posterior, symmetric_kl=self.cfg.symmetric_kl)

        # shared elements that need to be averaged, computed per pair in the batch.
        old_p_shared_mask = AdaVae.compute_shared_mask(a_p_deltas)
        old_n_shared_mask = AdaVae.compute_shared_mask(a_n_deltas)

        # modify threshold based on criterion and recompute if necessary
        # CORE of this approach!
        p_shared_mask, n_shared_mask = BoundedAdaVae.compute_constrained_masks(a_p_deltas, old_p_shared_mask, a_n_deltas, old_n_shared_mask)
        
        # make averaged variables
        new_args = AdaVae.compute_averaged(a_z_params, p_z_params, p_shared_mask, self._compute_average_fn)

        # return new args & generate logs
        return new_args, {
            'p_shared_before': old_p_shared_mask.sum(dim=1).float().mean(),
            'p_shared_after':      p_shared_mask.sum(dim=1).float().mean(),
            'n_shared_before': old_n_shared_mask.sum(dim=1).float().mean(),
            'n_shared_after':      n_shared_mask.sum(dim=1).float().mean(),
        }
    
    def augment_loss(self, z_means):
        return 0, {}

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
    # HELPER                                                                #
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

    @staticmethod
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

