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

from dataclasses import dataclass

import numpy as np

from disent.frameworks.vae._weaklysupervised__adavae import AdaVae

# ========================================================================= #
# Swapped Target AdaVae                                                     #
# ========================================================================= #


class SwappedInputAdaVae(AdaVae):
    """
    This is a naive example of a framework that randomly swaps the inputs to
    the VAE, effectively creating a type of denoising auto-encoder.
    """

    REQUIRED_OBS = 2

    @dataclass
    class cfg(AdaVae.cfg):
        swap_chance: float = 0.1

    def __init__(self, model: "AutoEncoder", cfg: cfg = None, batch_augment=None):
        super().__init__(model=model, cfg=cfg, batch_augment=batch_augment)
        assert cfg.swap_chance >= 0

    def do_training_step(self, batch, batch_idx):
        (x0, x1), (x0_targ, x1_targ) = self._get_xs_and_targs(batch, batch_idx)

        # random chance to swap the inputs
        if np.random.random() < self.cfg.swap_chance:
            x0, x1 = x1, x0

        return super(SwappedInputAdaVae, self).do_training_step(
            {
                "x": (x0, x1),
                "x_targ": (x0_targ, x1_targ),
            },
            batch_idx,
        )


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
