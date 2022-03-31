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

import contextlib
import os
import sys
from contextlib import contextmanager
from typing import Any
from typing import Dict


# ========================================================================= #
# TEST UTILS                                                                #
# ========================================================================= #


@contextmanager
def no_stdout():
    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    yield
    sys.stdout = old_stdout


@contextmanager
def no_stderr():
    old_stderr = sys.stderr
    sys.stderr = open(os.devnull, 'w')
    yield
    sys.stderr = old_stderr


@contextlib.contextmanager
def temp_wd(new_wd):
    old_wd = os.getcwd()
    os.chdir(new_wd)
    yield
    os.chdir(old_wd)


@contextlib.contextmanager
def temp_sys_args(new_argv):
    # TODO: should this copy values?
    old_argv = sys.argv
    sys.argv = new_argv
    yield
    sys.argv = old_argv


@contextmanager
def temp_environ(environment: Dict[str, Any]):
    # TODO: should this copy values? -- could use unittest.mock.patch.dict(...)
    # save the old environment
    existing_env = {}
    for k in environment:
        if k in os.environ:
            existing_env[k] = os.environ[k]
    # update the environment
    os.environ.update(environment)
    # run the context
    try:
        yield
    finally:
        # restore the original environment
        for k in environment:
            if k in existing_env:
                os.environ[k] = existing_env[k]
            else:
                del os.environ[k]


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
