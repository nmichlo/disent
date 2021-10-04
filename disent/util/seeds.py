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
import logging
import random
import numpy as np


log = logging.getLogger(__name__)


# ========================================================================= #
# seeds                                                                     #
# ========================================================================= #


def seed(long=777):
    """
    https://pytorch.org/docs/stable/notes/randomness.html
    """
    if long is None:
        log.warning(f'[SEEDING]: no seed was specified. Seeding skipped!')
        return
    # seed python
    random.seed(long)
    # seed numpy
    np.random.seed(long)
    # seed torch - it can be slow to import
    try:
        import torch
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(long)  # also calls: torch.cuda.manual_seed_all
    except ImportError:
        log.warning(f'[SEEDING]: torch is not installed. Skipped seeding torch methods!')
    # done!
    log.info(f'[SEEDED]: {long}')


class TempNumpySeed(contextlib.ContextDecorator):
    def __init__(self, seed=None, offset=0):
        # check and normalize seed
        if seed is not None:
            try:
                seed = int(seed)
            except:
                raise ValueError(f'{seed=} is not int-like!')
        # offset seed
        if seed is not None:
            seed += offset
        # save values
        self._seed = seed
        self._state = None

    def __enter__(self):
        if self._seed is not None:
            self._state = np.random.get_state()
            np.random.seed(self._seed)

    def __exit__(self, *args, **kwargs):
        if self._seed is not None:
            np.random.set_state(self._state)
            self._state = None

    def _recreate_cm(self):
        # TODO: do we need to override this?
        return self

# ========================================================================= #
# END                                                                       #
# ========================================================================= #
