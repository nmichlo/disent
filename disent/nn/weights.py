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

from disent.nn.activations import swish
from disent.util.strings import colors as c


log = logging.getLogger(__name__)


# ========================================================================= #
# Basic Weight Initialisation                                               #
# ========================================================================= #


_ACTIVATIONS = {
    'relu': torch.relu,
    'sigmoid': torch.sigmoid,
    'tanh': torch.tanh,
    'swish': swish,
}


def _activation_mean_std(activation_fn, samples: int = 1024*10):
    out = activation_fn(torch.randn(samples, dtype=torch.float64))
    return out.mean().item(), out.std().item()


def init_model_weights(model: nn.Module, mode: Optional[str] = 'xavier_normal', log_level=logging.INFO) -> nn.Module:
    """
    This whole function is very naive and most likely quite wrong.
    -- be careful using it!
    """
    count = 0

    # get default mode
    if mode is None:
        mode = 'default'

    # get scaling
    mean, std, activation = None, None, None
    if len(mode.split('__scale_')) == 2:
        mode, activation = mode.split('__scale_')
        mean, std = _activation_mean_std(_ACTIVATIONS[activation], samples=1024*10)

    def init_normal(m):
        nonlocal count
        init, count = False, count + 1

        # actually initialise!
        if mode == 'xavier_normal':
            if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
                init = True
        elif mode == 'normal':
            if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.normal_(m.weight)
                nn.init.zeros_(m.bias)
                init = True
        elif mode == 'default':
            pass
        else:
            raise KeyError(f'Unknown init mode: {repr(mode)}, valid modes are: {["xavier_normal", "default"]}')

        # scale values
        # -- this is very naive and most likely wrong!
        if std is not None:
            if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
                with torch.no_grad():
                    m.weight /= std
                    init = True

        # print messages
        if init:
            log.log(log_level, f'| {count:03d} {c.lGRN}INIT{c.RST}: {m.__class__.__name__}')
        else:
            log.log(log_level, f'| {count:03d} {c.lRED}SKIP{c.RST}: {m.__class__.__name__}')

    log.log(log_level, f'Initialising Model Layers: {mode}{f"__scale_{activation} (mean={mean}, std={std})" if activation else ""}')
    model.apply(init_normal)

    return model


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
