#  ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~
#  MIT License
#
#  Copyright (c) 2021 Nathan Juraj Michlo
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in
#  all copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.
#  ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~

import torch
from torch.distributions import Distribution
from torch.distributions import Distribution
from torch.distributions import Distribution
from torch.distributions import Distribution
from torch.distributions import Distribution
from torch.distributions import Distribution
from torch.distributions import Distribution
from torch.distributions import Distribution
from torch.distributions import Distribution
from torch.distributions import Distribution
from torch.distributions import Distribution
from torch.distributions import Distribution
from torch.distributions import Distribution
from torch.distributions import Distribution


# ========================================================================= #
# Kl Losses                                                                 #
# ========================================================================= #


def kl_loss_direct_reverse(posterior: Distribution, prior: Distribution, z_sampled: torch.Tensor = None):
    # This is how the original VAE/BetaVAE papers do it.
    # - we compute the reverse kl divergence directly instead of approximating it
    # - kl(post|prior)
    # FORWARD vs. REVERSE kl (https://www.tuananhle.co.uk/notes/reverse-forward-kl.html)
    # - If we minimize the kl(post|prior) or the reverse/exclusive KL, the zero-forcing/mode-seeking behavior arises.
    # - If we minimize the kl(prior|post) or the forward/inclusive KL, the mass-covering/mean-seeking behavior arises.
    return torch.distributions.kl_divergence(posterior, prior)


def kl_loss_approx_reverse(posterior: Distribution, prior: Distribution, z_sampled: torch.Tensor = None):
    # This is how pytorch-lightning-bolts does it:
    # - kl(post|prior)
    # See issue: https://github.com/PyTorchLightning/pytorch-lightning-bolts/issues/565
    # - we approximate the reverse kl divergence instead of computing it analytically
    assert z_sampled is not None, 'to compute the approximate kl loss, z_sampled needs to be defined (cfg.kl_mode="approx")'
    return posterior.log_prob(z_sampled) - prior.log_prob(z_sampled)


def kl_loss_direct_forward(posterior: Distribution, prior: Distribution, z_sampled: torch.Tensor = None):
    # compute the forward kl
    # - kl(prior|post)
    return torch.distributions.kl_divergence(prior, posterior)


def kl_loss_approx_forward(posterior: Distribution, prior: Distribution, z_sampled: torch.Tensor = None):
    # compute the approximate forward kl
    # - kl(prior|post)
    assert z_sampled is not None, 'to compute the approximate kl loss, z_sampled needs to be defined (cfg.kl_mode="approx")'
    return prior.log_prob(z_sampled) - posterior.log_prob(z_sampled)


def kl_loss_direct_symmetric(posterior: Distribution, prior: Distribution, z_sampled: torch.Tensor = None):
    # compute the (scaled) symmetric kl
    # - 0.5 * kl(prior|post) + 0.5 * kl(prior|post)
    return 0.5 * kl_loss_direct_reverse(posterior, prior, z_sampled) + 0.5 * kl_loss_direct_forward(posterior, prior, z_sampled)


def kl_loss_approx_symmetric(posterior: Distribution, prior: Distribution, z_sampled: torch.Tensor = None):
    # compute the approximate (scaled) symmetric kl
    # - 0.5 * kl(prior|post) + 0.5 * kl(prior|post)
    return 0.5 * kl_loss_approx_reverse(posterior, prior, z_sampled) + 0.5 * kl_loss_approx_forward(posterior, prior, z_sampled)


_KL_LOSS_MODES = {
    # reverse kl -- how it should be done for VAEs
    'direct':         kl_loss_direct_reverse,  # alias for reverse modes
    'approx':         kl_loss_approx_reverse,  # alias for reverse modes
    'direct_reverse': kl_loss_direct_reverse,
    'approx_reverse': kl_loss_approx_reverse,
    # forward kl
    'direct_forward': kl_loss_direct_forward,
    'approx_forward': kl_loss_approx_forward,
    # symmetric kl
    'direct_symmetric': kl_loss_direct_symmetric,
    'approx_symmetric': kl_loss_approx_symmetric,
}


def kl_loss(posterior: Distribution, prior: Distribution, z_sampled: torch.Tensor = None, mode='direct'):
    return _KL_LOSS_MODES[mode](posterior, prior, z_sampled)


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
