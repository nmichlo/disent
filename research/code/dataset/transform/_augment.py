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

import logging
import os
from typing import Callable

import torch
import research
from disent.dataset.transform._augment import _scale_kernel
from disent.util.deprecate import deprecated


log = logging.getLogger(__name__)


# ========================================================================= #
# Helper                                                                    #
# -- import this file to register the functions!                            #
# ========================================================================= #


_KERNEL_DIR = os.path.abspath(os.path.join(research.__file__, '..', 'part03_learnt_overlap/e01_learn_to_disentangle/data'))


def _get_make_kernel_fn(file_name: str, *, normalize_mode: str, root_dir: str = _KERNEL_DIR) -> Callable[[], torch.Tensor]:
    def _make_kernel(kern: str = None, radius: str = None) -> torch.Tensor:
        path = os.path.join(root_dir, file_name)
        log.debug(f'Loading kernel from: {path}')
        with torch.no_grad():
            kernel = torch.load(path)
            return _scale_kernel(kernel, mode=normalize_mode)
    return _make_kernel


# ========================================================================= #
# Kernel Registry                                                           #
# -- import this file to register the functions!                            #
# ========================================================================= #


# old kernels, with negative values -- these should be removed
_make_xy1_r47  = deprecated(fn=_get_make_kernel_fn('OLD_r47-1_s28800_adam_lr0.003_wd0.0_xy1x1.pt', normalize_mode='none'),   msg='kernel `xy1_r47` has been deprecated! It is not correctly scaled, please use `xy1_abs63` instead!')
_make_xy8_r47  = deprecated(fn=_get_make_kernel_fn('OLD_r47-1_s28800_adam_lr0.003_wd0.0_xy8x8.pt', normalize_mode='none'),   msg='kernel `xy8_r47` has been deprecated! It is not correctly scaled, please use `xy8_abs63` instead!')

# kernels learnt with `kernel = abs(params)` parameterization
# - no negative values
_make_xy1_abs63  = _get_make_kernel_fn('MSC_abs_r63-1_s28800_b512_adam_lr0.001_wd0.0_xysquares_1x1.pt',  normalize_mode='none')
_make_xy2_abs63  = _get_make_kernel_fn('MSC_abs_r63-1_s28800_b512_adam_lr0.001_wd0.0_xysquares_2x2.pt',  normalize_mode='none')
_make_xy4_abs63  = _get_make_kernel_fn('MSC_abs_r63-1_s28800_b512_adam_lr0.001_wd0.0_xysquares_4x4.pt',  normalize_mode='none')
_make_xy8_abs63  = _get_make_kernel_fn('MSC_abs_r63-1_s28800_b512_adam_lr0.001_wd0.0_xysquares_8x8.pt',  normalize_mode='none')

# kernels learnt with `kernel = params` parameterization
# - has negative values
_make_xy1_none63 = _get_make_kernel_fn('MSC_none_r63-1_s28800_b512_adam_lr0.001_wd0.0_xysquares_1x1.pt', normalize_mode='none')
_make_xy2_none63 = _get_make_kernel_fn('MSC_none_r63-1_s28800_b512_adam_lr0.001_wd0.0_xysquares_2x2.pt', normalize_mode='none')
_make_xy4_none63 = _get_make_kernel_fn('MSC_none_r63-1_s28800_b512_adam_lr0.001_wd0.0_xysquares_4x4.pt', normalize_mode='none')
_make_xy8_none63 = _get_make_kernel_fn('MSC_none_r63-1_s28800_b512_adam_lr0.001_wd0.0_xysquares_8x8.pt', normalize_mode='none')


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
