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

import pytorch_lightning as pl
import torch


# ========================================================================= #
# Base Modules                                                              #
# ========================================================================= #


class DisentModule(torch.nn.Module):

    def _forward_unimplemented(self, *args):
        # Annoying fix applied by torch for Module.forward:
        # https://github.com/python/mypy/issues/8795
        raise RuntimeError('This should never run!')

    def forward(self, *args, **kwargs):
        raise NotImplementedError


class DisentLightningModule(pl.LightningModule):

    def _forward_unimplemented(self, *args):
        # Annoying fix applied by torch for Module.forward:
        # https://github.com/python/mypy/issues/8795
        raise RuntimeError('This should never run!')


# ========================================================================= #
# Utility Layers                                                            #
# ========================================================================= #


class BatchView(DisentModule):
    def __init__(self, size):
        super().__init__()
        self._size = (-1, *size)

    def forward(self, x):
        return x.view(*self._size)


class Unsqueeze3D(DisentModule):
    def forward(self, x):
        assert x.ndim == 2
        return x.view(*x.shape, 1, 1)  # (B, N) -> (B, N, 1, 1)


class Flatten3D(DisentModule):
    def forward(self, x):
        assert x.ndim == 4
        return x.view(x.shape[0], -1)  # (B, C, H, W) -> (B, C*H*W)


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
