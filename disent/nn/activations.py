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
from disent.nn.modules import DisentModule


# ========================================================================= #
# Swish                                                                     #
# ========================================================================= #


class _SwishFunction(torch.autograd.Function):
    """
    Modified from:
    https://github.com/ceshine/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/utils.py
    """

    # This uses less memory than the standard implementation,
    # by re-computing the gradient on the backward pass

    @staticmethod
    def forward(ctx, x: torch.Tensor):
        y = x * torch.sigmoid(x)
        ctx.save_for_backward(x)
        return y

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x = ctx.saved_variables[0]
        sigmoid_x = torch.sigmoid(x)
        return grad_output * (sigmoid_x * (1 + x * (1 - sigmoid_x)))


def swish(tensor: torch.Tensor):
    return _SwishFunction.apply(tensor)


class Swish(DisentModule):
    def forward(self, tensor: torch.Tensor):
        return _SwishFunction.apply(tensor)


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
