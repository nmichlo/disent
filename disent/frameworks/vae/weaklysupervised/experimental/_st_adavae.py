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
from disent.frameworks.vae.weaklysupervised._adavae import AdaVae


# ========================================================================= #
# Swapped Target AdaVae                                                     #
# ========================================================================= #


class SwappedTargetAdaVae(AdaVae):

    REQUIRED_OBS = 2

    @dataclass
    class cfg(AdaVae.cfg):
        swap_chance: float = 0.1

    def __init__(self, make_optimizer_fn, make_model_fn, batch_augment=None, cfg: cfg = None):
        super().__init__(make_optimizer_fn, make_model_fn, batch_augment=batch_augment, cfg=cfg)
        assert cfg.swap_chance >= 0

    def do_training_step(self, batch, batch_idx):
        (x0, x1), (x0_targ, x1_targ) = batch['x'], batch['x_targ']

        # random change for the target not to be equal to the input
        if np.random.random() < self.cfg.swap_chance:
            x0_targ, x1_targ = x1_targ, x0_targ

        return super(SwappedTargetAdaVae, self).do_training_step({
            'x': (x0, x1),
            'x_targ': (x0_targ, x1_targ),
        }, batch_idx)


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
