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
from typing import Optional

import torch
from torch import nn
from disent.util.strings import colors as c


log = logging.getLogger(__name__)


# ========================================================================= #
# Basic Weight Initialisation                                               #
# ========================================================================= #


_WEIGHT_INIT_FNS = {
    'xavier_uniform':     lambda weight: nn.init.xavier_uniform_(weight, gain=1.0),  # gain=1
    'xavier_normal':      lambda weight: nn.init.xavier_normal_(weight, gain=1.0),   # gain=1
    'xavier_normal__0.1': lambda weight: nn.init.xavier_normal_(weight, gain=0.1),   # gain=0.1
    # kaiming -- also known as "He initialisation"
    'kaiming_uniform':            lambda weight: nn.init.kaiming_uniform_(weight, a=0, mode='fan_in', nonlinearity='relu'),  # fan_in, relu
    'kaiming_normal':             lambda weight: nn.init.kaiming_normal_(weight, a=0, mode='fan_in', nonlinearity='relu'),   # fan_in, relu
    'kaiming_normal__fan_out':    lambda weight: nn.init.kaiming_normal_(weight, a=0, mode='fan_out', nonlinearity='relu'),  # fan_in, relu
    # other
    'orthogonal':    lambda weight: nn.init.orthogonal_(weight, gain=1),  # gain=1
    'normal':        lambda weight: nn.init.normal_(weight, mean=0., std=1.),      # gain=1
    'normal__0.1':   lambda weight: nn.init.normal_(weight, mean=0., std=0.1),     # gain=0.1
    'normal__0.01':  lambda weight: nn.init.normal_(weight, mean=0., std=0.01),    # gain=0.01
    'normal__0.001': lambda weight: nn.init.normal_(weight, mean=0., std=0.001),   # gain=0.01
}


# TODO: clean this up! this is terrible...
def init_model_weights(model: nn.Module, mode: Optional[str] = 'xavier_normal', log_level=logging.INFO) -> nn.Module:
    count = 0

    # get default mode
    if mode is None:
        mode = 'default'

    def _apply_init_weights(m):
        nonlocal count
        init, count = False, count + 1

        # actually initialise!
        if mode == 'default':
            pass
        elif mode in _WEIGHT_INIT_FNS:
            if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
                _WEIGHT_INIT_FNS[mode](m.weight)
                nn.init.zeros_(m.bias)
                init = True
        else:
            raise KeyError(f'Unknown init mode: {repr(mode)}, valid modes are: {["default"] + sorted(_WEIGHT_INIT_FNS)}')

        # print messages
        if init:
            log.log(log_level, f'| {count:03d} {c.lGRN}INIT{c.RST}: {m.__class__.__name__}')
        else:
            log.log(log_level, f'| {count:03d} {c.lRED}SKIP{c.RST}: {m.__class__.__name__}')

    log.log(log_level, f'Initialising Model Layers: {mode}')
    model.apply(_apply_init_weights)

    return model


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
