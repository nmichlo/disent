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
import os

import numpy as np
import torch


"""
helpful functions that do not fit nicely into any other file.
"""

log = logging.getLogger(__name__)

# ========================================================================= #
# seeds                                                                     #
# ========================================================================= #


def is_test_run():
    """
    This is used internally to test some scripts. There is no need
    to use this function in your own scripts.
    """
    return bool(os.environ.get('DISENT_TEST_RUN', False))


def _set_test_run():
    os.environ['DISENT_TEST_RUN'] = 'True'


# ========================================================================= #
# Conversion                                                                #
# ========================================================================= #


def to_numpy(array) -> np.ndarray:
    """
    Handles converting any array like object to a numpy array.
    specifically with support for a tensor
    """
    # TODO: replace... maybe with kornia
    if torch.is_tensor(array):
        return array.cpu().detach().numpy()
    # recursive conversion
    # not super efficient but allows handling of PIL.Image and other nested data.
    elif isinstance(array, (list, tuple)):
        return np.stack([to_numpy(elem) for elem in array], axis=0)
    else:
        return np.array(array)


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
