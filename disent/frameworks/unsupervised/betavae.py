from disent.frameworks.unsupervised.vae import VaeLoss


# ========================================================================= #
# Beta-VAE Loss                                                             #
# ========================================================================= #


class BetaVaeLoss(VaeLoss):
    def __init__(self, beta=4):
        self.beta = beta

    def regularizer(self, kl_loss, z_mean, z_logvar, z_sampled):
        return self.beta * kl_loss


# ========================================================================= #
# Beta-VAE-H Loss                                                           #
# ========================================================================= #


class BetaVaeHLoss(BetaVaeLoss):
    """
    Compute the Beta-VAE loss as in [1]

    [1] Higgins, Irina, et al. "beta-vae: Learning basic visual concepts with
    a constrained variational framework." (2016).

    (NOTE: BetaVAEB is from understanding disentanglement in Beta VAEs)
    """

    def __init__(self, anneal_end_steps, beta=4):
        super().__init__(beta)
        self.n_train_steps = 0
        self.anneal_end_steps = anneal_end_steps
        raise NotImplementedError('n_train_steps is not yet implemented for BetaVaeHLoss, it will not yet work')

    def regularizer(self, kl_loss, z_mean, z_logvar, z_sampled):
        print('WARNING: training steps not updated')
        anneal_reg = lerp_step(0, 1, self.n_train_steps, self.anneal_end_steps)  # if is_train else 1
        return (anneal_reg * self.beta) * kl_loss


# ========================================================================= #
# HELPER                                                                    #
# ========================================================================= #


def lerp(a, b, t):
    """Linear interpolation between parameters, respects bounds when t is out of bounds [0, 1]"""
    assert a < b
    t = max(0, min(t, 1))
    # precise method, guarantees v==b when t==1 | simplifies to: a + t*(b-a)
    return (1 - t) * a + t * b


def lerp_step(a, b, step, max_steps):
    """Linear interpolation based on a step count."""
    if max_steps <= 0:
        return b
    return lerp(a, b, step / max_steps)


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
