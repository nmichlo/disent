# import logging
#
# import torch
#
# from disent.frameworks.other.msp import MatrixSubspaceProjection
# from disent.frameworks.supervised.gadavae import GuidedAdaVae, triplet_loss
# from disent.frameworks.unsupervised.vae import TrainingData, bce_loss_with_logits, kl_normal_loss
#
#
# log = logging.getLogger(__name__)
#
#
# # ========================================================================= #
# # Guided Ada Vae                                                            #
# # ========================================================================= #
#
#
# class MspGuidedAdaVae(GuidedAdaVae):
#
#     def __init__(self, msp: MatrixSubspaceProjection, beta=4, triplet_margin=0.3, triplet_scale=1, msp_scale=1):
#         assert triplet_scale > 0, f'{triplet_scale=} must be > 0'
#         assert msp_scale > 0, f'{msp_scale=} must be > 0'
#         super().__init__(beta, triplet_margin=triplet_margin, triplet_scale=triplet_scale)
#         # matrix subspace projection
#         self.msp_scale = msp_scale
#         self.msp = msp
#
#     def intercept_z(self, a_z_mean, a_z_logvar, p_z_mean, p_z_logvar, n_z_mean, n_z_logvar):
#         # DO NOTHING
#         new_args = a_z_mean, a_z_logvar, p_z_mean, p_z_logvar, n_z_mean, n_z_logvar
#         return new_args, {}
#
#     def compute_loss(self, a_data: TrainingData, p_data: TrainingData, n_data: TrainingData):
#         # COMPUTE LOSS FOR TRIPLE:
#         (a_x, a_x_recon, a_z_mean, a_z_logvar, a_z_sampled) = a_data
#         (p_x, p_x_recon, p_z_mean, p_z_logvar, p_z_sampled) = p_data
#         (n_x, n_x_recon, n_z_mean, n_z_logvar, n_z_sampled) = n_data
#
#         # reconstruction error
#         a_recon_loss = bce_loss_with_logits(a_x, a_x_recon)  # E[log p(x|z)]
#         p_recon_loss = bce_loss_with_logits(p_x, p_x_recon)  # E[log p(x|z)]
#         n_recon_loss = bce_loss_with_logits(n_x, n_x_recon)  # E[log p(x|z)]
#         ave_recon_loss = (a_recon_loss + p_recon_loss + n_recon_loss) / 3
#
#         # KL divergence
#         a_kl_loss = kl_normal_loss(a_z_mean, a_z_logvar)  # D_kl(q(z|x) || p(z|x))
#         p_kl_loss = kl_normal_loss(p_z_mean, p_z_logvar)  # D_kl(q(z|x) || p(z|x))
#         n_kl_loss = kl_normal_loss(n_z_mean, n_z_logvar)  # D_kl(q(z|x) || p(z|x))
#         ave_kl_loss = (a_kl_loss + p_kl_loss + n_kl_loss) / 3
#
#         # MSP - Average In Label Space
#         a_y = self.msp.latent_to_labels(a_z_mean)
#         p_y = self.msp.latent_to_labels(p_z_mean)
#         n_y = self.msp.latent_to_labels(n_z_mean)
#
#         # TODO: this is not necessary - PERFORM AVERAGING LIKE USUAL AND USE MSP WITH TRIPLET LOSS
#         with torch.no_grad():
#             # MSP - thresholds in label space (without KL)
#             p_y_delta = torch.abs(a_y - p_y)
#             n_y_delta = torch.abs(a_y - n_y)
#             p_y_ave_thresh = 0.5 * (p_y_delta.min(dim=1, keepdim=True).values + p_y_delta.max(dim=1, keepdim=True).values)
#             n_y_ave_thresh = 0.5 * (n_y_delta.min(dim=1, keepdim=True).values + n_y_delta.max(dim=1, keepdim=True).values)
#
#             # MSP - averages in label space
#             p_y_mask = (p_y_delta < p_y_ave_thresh)
#             n_y_mask = (n_y_delta < n_y_ave_thresh)
#             masked_ap_y_ave = p_y_mask*(a_y + p_y)*0.5
#             masked_an_y_ave = n_y_mask*(a_y + n_y)*0.5
#             a_ap_y_ave = (~p_y_mask)*a_y + masked_ap_y_ave
#             p_ap_y_ave = (~p_y_mask)*p_y + masked_ap_y_ave
#             a_an_y_ave = (~n_y_mask)*a_y + masked_an_y_ave
#             n_an_y_ave = (~n_y_mask)*n_y + masked_an_y_ave
#
#         # MSP - loss
#         loss_y_msp_ap_a = self.msp.loss_batch(a_z_mean, a_ap_y_ave)
#         loss_y_msp_ap_p = self.msp.loss_batch(p_z_mean, p_ap_y_ave)
#         loss_y_msp_an_a = self.msp.loss_batch(a_z_mean, a_an_y_ave)
#         loss_y_msp_an_n = self.msp.loss_batch(n_z_mean, n_an_y_ave)
#         loss_y_msp_ave = (loss_y_msp_ap_a + loss_y_msp_ap_p + loss_y_msp_an_a + loss_y_msp_an_n) / 4
#
#         # MSP - triplet on labels
#         # loss_y_triplet = torch.norm(a_y-p_y) - torch.norm(a_y-n_y)
#         loss_y_triplet = triplet_loss(a_y, p_y, n_y, alpha=self.triplet_margin)
#
#         # regularisation loss
#         reg_loss = self.beta * ave_kl_loss
#         t_loss = self.triplet_scale * loss_y_triplet
#         msp_loss = self.msp_scale * loss_y_msp_ave
#
#         # compute combined loss
#         loss = ave_recon_loss + reg_loss + (t_loss + msp_loss)
#
#         loss_dict = {
#             'train_loss': loss,
#             'reconstruction_loss': ave_recon_loss,
#             'regularize_loss': reg_loss,
#             'kl_loss': ave_kl_loss,
#             'elbo': -(ave_recon_loss + ave_kl_loss),
#             'triplet_loss': t_loss,
#             'msp_loss': msp_loss,
#         }
#
#         return loss_dict
#
#
# # ========================================================================= #
# # END                                                                       #
# # ========================================================================= #
#
