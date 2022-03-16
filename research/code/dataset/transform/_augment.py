#  ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~
#  MIT License
#
#  Copyright (c) 2022 Nathan Juraj Michlo
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


import os

import torch
import research
from disent.dataset.transform._augment import _scale_kernel
from disent.util.deprecate import deprecated


# ========================================================================= #
# Kernel Registry                                                           #
# -- import this file to register the functions!                            #
# ========================================================================= #


@torch.no_grad()
def _load_xy8_r47():
    return torch.load(os.path.abspath(os.path.join(research.__file__, '../part03_learnt_overlap/e01_learn_to_disentangle/data', 'r47-1_s28800_adam_lr0.003_wd0.0_xy8x8.pt'))).detach()


@torch.no_grad()
def _load_xy1_r47():
    return torch.load(os.path.abspath(os.path.join(research.__file__, '../part03_learnt_overlap/e01_learn_to_disentangle/data', 'r47-1_s28800_adam_lr0.003_wd0.0_xy1x1.pt'))).detach()


@deprecated('kernel `xy8_r47` has been deprecated! It is not correctly scaled, please use `xy8s_r47` instead!')
def _make_xy8_r47(kern: str = None, radius: str = None):
    return _load_xy8_r47()


@deprecated('kernel `xy1_r47` has been deprecated! It is not correctly scaled, please use `xy1s_r47` instead!')
def _make_xy1_r47(kern: str = None, radius: str = None):
    return _load_xy1_r47()


def _make_xy8s_r47(kern: str = None, radius: str = None):
    return _scale_kernel(_load_xy8_r47(), 'sum')


def _make_xy1s_r47(kern: str = None, radius: str = None):
    return _scale_kernel(_load_xy1_r47(), 'sum')


def _make_xy8m_r47(kern: str = None, radius: str = None):
    return _scale_kernel(_load_xy8_r47(), 'maxsum')


def _make_xy1m_r47(kern: str = None, radius: str = None):
    return _scale_kernel(_load_xy1_r47(), 'maxsum')


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
