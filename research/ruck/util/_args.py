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

from argparse import Namespace
from typing import Optional
from typing import Sequence


class HParamsMixin(object):

    __hparams = None

    def save_hyperparameters(self, ignore: Optional[Sequence[str]] = None, include: Optional[Sequence[str]] = None):
        import inspect
        import warnings
        # get ignored values
        ignored = set() if (ignore is None) else set(ignore)
        included = set() if (include is None) else set(include)
        assert all(str.isidentifier(k) for k in ignored)
        assert all(str.isidentifier(k) for k in included)
        # get function params & signature
        locals = inspect.currentframe().f_back.f_locals
        params = inspect.signature(self.__class__.__init__)
        # get values
        (self_param, *params) = params.parameters.items()
        # check that self is correct & skip it
        assert self_param[0] == 'self'
        assert locals[self_param[0]] is self
        # get other values
        values = {}
        for k, v in params:
            if k in ignored: continue
            if v.kind == v.VAR_KEYWORD: warnings.warn('variable keywords argument saved, consider converting to explicit arguments.')
            if v.kind == v.VAR_POSITIONAL: warnings.warn('variable positional argument saved, consider converting to explicit named arguments.')
            values[k] = locals[k]
        # get extra values
        for k in included:
            assert k != 'self'
            assert k not in values, 'k has already been included'
            values[k] = locals[k]
        # done!
        self.__hparams = Namespace(**values)

    @property
    def hparams(self):
        return self.__hparams
