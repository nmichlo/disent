from dataclasses import dataclass

import numpy as np
from disent.frameworks.vae.unsupervised import BetaVae


# ========================================================================= #
# Swapped Target BetaVAE                                                    #
# ========================================================================= #


class SwappedTargetBetaVae(BetaVae):

    @dataclass
    class cfg(BetaVae.cfg):
        swap_chance: float = 0.1

    def __init__(self, make_optimizer_fn, make_model_fn, batch_augment=None, cfg: cfg = None):
        super().__init__(make_optimizer_fn, make_model_fn, batch_augment=batch_augment, cfg=cfg)
        assert cfg.swap_chance >= 0

    def compute_training_loss(self, batch, batch_idx):
        (x0, x1), (x0_targ, x1_targ) = batch['x'], batch['x_targ']

        # random change for the target not to be equal to the input
        if np.random.random() < self.cfg.swap_chance:
            x0_targ, x1_targ = x1_targ, x0_targ

        return super(SwappedTargetBetaVae, self).compute_training_loss({
            'x': (x0,),
            'x_targ': (x0_targ,),
        }, batch_idx)


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
