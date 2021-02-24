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

from disent.util import DisentModule


log = logging.getLogger(__name__)


# ========================================================================= #
# Utility Layers                                                            #
# ========================================================================= #


class Print(DisentModule):
    """From: https://github.com/1Konny/Beta-VAE/blob/master/model.py"""
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, tensor):
        log.debug(self.layer, '|', tensor.shape, '->')
        output = self.layer.forward(tensor)
        log.debug(output.shape)
        return output


class BatchView(DisentModule):
    """From: https://github.com/1Konny/Beta-VAE/blob/master/model.py"""
    def __init__(self, size):
        super().__init__()
        self.size = (-1, *size)

    def forward(self, tensor):
        return tensor.view(*self.size)


class Unsqueeze3D(DisentModule):
    """From: https://github.com/amir-abdi/disentanglement-pytorch"""
    def forward(self, x):
        x = x.unsqueeze(-1)
        x = x.unsqueeze(-1)
        return x


class Flatten3D(DisentModule):
    """From: https://github.com/amir-abdi/disentanglement-pytorch"""
    def forward(self, x):
        x = x.view(x.size()[0], -1)
        return x

# ========================================================================= #
# END                                                                       #
# ========================================================================= #
