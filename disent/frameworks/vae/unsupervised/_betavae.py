from dataclasses import dataclass

import torch

from disent.frameworks.vae.unsupervised import Vae


# ========================================================================= #
# Beta-VAE Loss                                                             #
# ========================================================================= #


class BetaVae(Vae):

    @dataclass
    class cfg(Vae.cfg):
        beta: float = 4

    def __init__(self, make_optimizer_fn, make_model_fn, batch_augment=None, cfg: cfg = None):
        super().__init__(make_optimizer_fn, make_model_fn, batch_augment=batch_augment, cfg=cfg)
        assert self.cfg.beta >= 0, 'beta must be >= 0'

    def training_regularize_kl(self, kl_loss):
        if self.cfg.beta == 0:
            # numerical stability
            return torch.zeros_like(kl_loss)
        else:
            return self.cfg.beta * kl_loss


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

    @dataclass
    class cfg(BetaVae.cfg):
        anneal_end_steps: int = 0

    def training_regularize_kl(self, kl_loss):
        # anneal
        anneal_reg = lerp_step(0, 1, self.trainer.global_step, self.cfg.anneal_end_steps)
        # compute kl regularisation
        return anneal_reg * self.cfg.beta * kl_loss


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
