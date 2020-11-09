import numpy as np
from disent.frameworks.vae.unsupervised import BetaVae
from disent.frameworks.vae.loss import bce_loss_with_logits, kl_normal_loss


# ========================================================================= #
# Swapped Target BetaVAE                                                    #
# ========================================================================= #


class SwappedTargetBetaVae(BetaVae):

    def __init__(
            self,
            make_optimizer_fn,
            make_model_fn,
            batch_augment=None,
            beta=4,
            # swapped target
            swap_chance=0.1
    ):
        super().__init__(make_optimizer_fn, make_model_fn, batch_augment=batch_augment, beta=beta)
        assert swap_chance >= 0
        self.swap_chance = swap_chance

    def compute_training_loss(self, batch, batch_idx):
        (x0, x1), (x0_targ, x1_targ) = batch['x'], batch['x_targ']

        # random change for the target not to be equal to the input
        if np.random.random() < self.swap_chance:
            x0_targ, x1_targ = x1_targ, x0_targ

        return super(SwappedTargetBetaVae, self).compute_training_loss({
            'x': (x0,),
            'x_targ': (x0_targ,),
        }, batch_idx)


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
