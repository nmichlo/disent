from disent.frameworks.vae.unsupervised import Vae


# ========================================================================= #
# Beta-VAE Loss                                                             #
# ========================================================================= #


class BetaVae(Vae):

    def __init__(self, make_optimizer_fn, make_model_fn, beta=4):
        super().__init__(make_optimizer_fn, make_model_fn)
        assert beta >= 0
        self.beta = beta

    def kl_regularization(self, kl_loss):
        return self.beta * kl_loss


# ========================================================================= #
# Beta-VAE-H Loss                                                           #
# ========================================================================= #


class BetaVaeH(BetaVae):
    """
    Compute the Beta-VAE loss as in [1]

    [1] Higgins, Irina, et al. "beta-vae: Learning basic visual concepts with
    a constrained variational framework." (2016).

    (NOTE: BetaVAEB is from understanding disentanglement in Beta VAEs)
    """

    def __init__(self, make_optimizer_fn, make_model_fn, anneal_end_steps=0, beta=4):
        super().__init__(make_optimizer_fn, make_model_fn, beta=beta)
        self.anneal_end_steps = anneal_end_steps

    def kl_regularization(self, kl_loss):
        # anneal
        anneal_reg = lerp_step(0, 1, self.trainer.global_step, self.anneal_end_steps)
        # compute kl regularisation
        return anneal_reg * self.beta * kl_loss


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
