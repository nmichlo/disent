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

import numpy as np
from wandb.wandb_torch import torch


# ========================================================================= #
# DEBUG                                                                     #
# ========================================================================= #


def replace_arrays_with_shapes(obj):
    """
    recursively replace all arrays of an object
    with their shapes to make debugging easier!
    """
    if isinstance(obj, (torch.Tensor, np.ndarray)):
        return obj.shape
    elif isinstance(obj, dict):
        return {replace_arrays_with_shapes(k): replace_arrays_with_shapes(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return list(replace_arrays_with_shapes(v) for v in obj)
    elif isinstance(obj, tuple):
        return tuple(replace_arrays_with_shapes(v) for v in obj)
    elif isinstance(obj, set):
        return {replace_arrays_with_shapes(k) for k in obj}
    else:
        return obj


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
