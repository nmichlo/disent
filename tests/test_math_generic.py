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
import pytest
import torch

from disent.nn.functional._util_generic import generic_as_int32
from disent.nn.functional._util_generic import generic_max
from disent.nn.functional._util_generic import generic_min
from disent.nn.functional._util_generic import generic_ndim
from disent.nn.functional._util_generic import generic_shape


# ========================================================================= #
# Helper                                                                    #
# ========================================================================= #


def _assert_type_and_value(input, target):
    # check types are the same
    assert type(input) == type(target)
    # specific checks
    if isinstance(target, (int, float)):
        assert input == target
    elif isinstance(target, (np.ndarray, np.int32, np.int64, np.float32, np.float64)):
        assert input.dtype == target.dtype
        assert np.all(input == target)
    elif isinstance(target, torch.Tensor):
        assert input.dtype == target.dtype
        assert input.device == target.device
        assert torch.all(input == target)
    else:  # pragma: no cover
        raise NotImplementedError('This should never happen in tests!')


# ========================================================================= #
# TESTS                                                                     #
# ========================================================================= #


def test_generic_as_int32():
    # scalars
    _assert_type_and_value(input=generic_as_int32(-1),  target=-1)
    _assert_type_and_value(input=generic_as_int32(-1.), target=-1)
    _assert_type_and_value(input=generic_as_int32(1.5), target=1)
    # torch
    _assert_type_and_value(input=generic_as_int32(torch.as_tensor([-1.5, 0, 1.0])), target=torch.as_tensor([-1, 0, 1], dtype=torch.int32))
    # numpy
    _assert_type_and_value(input=generic_as_int32(np.array([-1.5, 0, 1.0])), target=np.array([-1, 0, 1], dtype=np.int32))
    # unsupported
    with pytest.raises(TypeError, match='invalid type'):
        generic_as_int32(None)


def test_generic_max():
    # scalars
    _assert_type_and_value(input=generic_max(-1),  target=-1)
    _assert_type_and_value(input=generic_max(1.5), target=1.5)
    # torch
    _assert_type_and_value(input=generic_max(torch.as_tensor([-1, 0, 1])), target=torch.as_tensor(1, dtype=torch.int64))
    _assert_type_and_value(input=generic_max(torch.as_tensor([-1.0, 0.0, 1.0])), target=torch.as_tensor(1.0, dtype=torch.float32))
    # numpy
    _assert_type_and_value(input=generic_max(np.array([-1, 0, 1])), target=np.int64(1))
    _assert_type_and_value(input=generic_max(np.array([-1.0, 0.0, 1.0])), target=np.float64(1.0))
    # unsupported
    with pytest.raises(TypeError, match='invalid type'):
        generic_max(None)


def test_generic_min():
    # scalars
    _assert_type_and_value(input=generic_min(-1),  target=-1)
    _assert_type_and_value(input=generic_min(1.5), target=1.5)
    # torch
    _assert_type_and_value(input=generic_min(torch.as_tensor([-1, 0, 1])), target=torch.as_tensor(-1, dtype=torch.int64))
    _assert_type_and_value(input=generic_min(torch.as_tensor([-1.0, 0.0, 1.0])), target=torch.as_tensor(-1.0, dtype=torch.float32))
    # numpy
    _assert_type_and_value(input=generic_min(np.array([-1, 0, 1])), target=np.int64(-1))
    _assert_type_and_value(input=generic_min(np.array([-1.0, 0.0, 1.0])), target=np.float64(-1.0))
    # unsupported
    with pytest.raises(TypeError, match='invalid type'):
        generic_min(None)


def test_generic_shape():
    # scalars
    assert generic_shape(1.) == ()
    assert generic_shape(-1) == ()
    # torch
    assert generic_shape(torch.as_tensor([-1, 0, 1])) == (3,)
    assert generic_shape(torch.as_tensor([-1.0, 0.0, 1.0])) == (3,)
    # numpy
    assert generic_shape(np.array([-1, 0, 1])) == (3,)
    assert generic_shape(np.array([-1.0, 0.0, 1.0])) == (3,)
    # unsupported
    with pytest.raises(TypeError, match='invalid type'):
        generic_shape(None)


def test_generic_ndim():
    # scalars
    assert generic_ndim(1.) == 0
    assert generic_ndim(-1) == 0
    # torch
    assert generic_ndim(torch.as_tensor([-1, 0, 1])) == 1
    assert generic_ndim(torch.as_tensor([-1.0, 0.0, 1.0])) == 1
    # numpy
    assert generic_ndim(np.array([-1, 0, 1])) == 1
    assert generic_ndim(np.array([-1.0, 0.0, 1.0])) == 1
    # unsupported
    with pytest.raises(TypeError, match='invalid type'):
        generic_ndim(None)


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
