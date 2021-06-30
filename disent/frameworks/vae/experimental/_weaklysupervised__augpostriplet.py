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

import logging
from dataclasses import dataclass
from typing import Union

import torch

from disent.frameworks.vae._supervised__tvae import TripletVae
from disent.util.config import instantiate_object_if_needed


log = logging.getLogger(__name__)


# ========================================================================= #
# Guided Ada Vae                                                            #
# ========================================================================= #


class AugPosTripletVae(TripletVae):

    REQUIRED_OBS = 2  # third obs is generated from augmentations

    @dataclass
    class cfg(TripletVae.cfg):
        overlap_augment: Union[dict, callable] = None

    def __init__(self, make_optimizer_fn, make_model_fn, batch_augment=None, cfg: cfg = None):
        super().__init__(make_optimizer_fn, make_model_fn, batch_augment=batch_augment, cfg=cfg)
        self._augment = None
        # initialise & check augment
        self._augment = instantiate_object_if_needed(self.cfg.overlap_augment)
        assert callable(self._augment), f'augment is not callable: {repr(self._augment)}'

    def do_training_step(self, batch, batch_idx):
        (a_x, n_x), (a_x_targ, n_x_targ) = self._get_xs_and_targs(batch, batch_idx)

        # generate augmented items
        with torch.no_grad():
            p_x_targ = a_x_targ
            p_x = self._augment(a_x)
            # a_x = self._aug(a_x)
            # n_x = self._aug(n_x)

        batch['x'], batch['x_targ'] = (a_x, p_x, n_x), (a_x_targ, p_x_targ, n_x_targ)
        # compute!
        return super().do_training_step(batch, batch_idx)


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
