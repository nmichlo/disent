from disent.frameworks.supervised.gadavae import GuidedAdaVae
import torch

# ========================================================================= #
# tgadavae                                                                  #
# ========================================================================= #


class TripletGuidedAdaVae(GuidedAdaVae):
    
    def __init__(
            self,
            make_optimizer_fn,
            make_model_fn,
            beta=4,
            average_mode='gvae',
            anchor_ave_mode='average',
            triplet_alpha=0.1,
            triplet_scale=1,
    ):
        super().__init__(make_optimizer_fn, make_model_fn, beta=beta, average_mode=average_mode, anchor_ave_mode=anchor_ave_mode)
        self.triplet_alpha = triplet_alpha
        self.triplet_scale = triplet_scale

    def augment_loss(self, a_z_mean, a_z_logvar, p_z_mean, p_z_logvar, n_z_mean, n_z_logvar):
        loss_triplet = triplet_loss(a_z_mean, p_z_mean, n_z_mean, alpha=self.triplet_alpha)
        augmented_loss = self.triplet_scale * loss_triplet
        return augmented_loss, {'triplet_loss': loss_triplet}


# ========================================================================= #
# TRIPLET LOSS                                                              #
# ========================================================================= #


def triplet_loss(anchor, positive, negative, alpha=0.3):
    # import tensorflow as tf
    # positive_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), 1)
    # negative_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), 1)
    # loss_1 = tf.add(tf.subtract(positive_dist, negative_dist), alpha)
    # loss = tf.reduce_sum(tf.maximum(loss_1, 0.0), 0)

    positive_dist = torch.sum((anchor - positive)**2, dim=1)
    negative_dist = torch.sum((anchor - negative)**2, dim=1)
    clamped = torch.clamp_min((positive_dist - negative_dist) + alpha, 0)
    loss = torch.mean(clamped, dim=0)  # TODO: this was sum
    return loss


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
