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
            batch_augment=None,
            beta=4,
            average_mode='gvae',
            symmetric_kl=True,
            swap_chance=0.1
    ):
        super().__init__(make_optimizer_fn, make_model_fn, batch_augment=batch_augment, beta=beta, average_mode=average_mode, symmetric_kl=symmetric_kl)
        assert swap_chance >= 0
        self.swap_chance = swap_chance

    def compute_training_loss(self, batch, batch_idx):
        (x0, x1), (x0_targ, x1_targ) = batch['x'], batch['x_targ']

        # random change for the target not to be equal to the input
        if np.random.random() < self.swap_chance:
            x0_targ, x1_targ = x1_targ, x0_targ

        return super(SwappedTargetAdaVae, self).compute_training_loss({
            'x': (x0, x1),
            'x_targ': (x0_targ, x1_targ),
        }, batch_idx)


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
