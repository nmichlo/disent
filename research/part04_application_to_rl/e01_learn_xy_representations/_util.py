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
import sys
from typing import Optional

import psutil


log = logging.getLogger(__name__)


# ========================================================================= #
# Helper                                                                    #
# ========================================================================= #


def get_num_workers(
    num_workers: Optional[int] = None,
    default_max: int = 16,
) -> int:
    if sys.platform == 'darwin':
        auto_workers = 0
        if num_workers is None:
            log.warning(f'MacOS detected, setting num_workers to {auto_workers} to avoid Dataloader bug.')
            num_workers = auto_workers
        elif num_workers > auto_workers:
            log.warning(f'MacOS detected, but manually set num_workers is greater than zero at {num_workers}, might result in a Dataloader bug!')
    else:
        auto_workers = min(psutil.cpu_count(logical=False), default_max)
        if num_workers is None:
            num_workers = auto_workers
            log.warning(f'Automatically set num_workers to {num_workers}, cpu_count is {psutil.cpu_count(logical=False)}, max auto workers is {default_max}')
        elif num_workers > auto_workers:
            log.warning(f'manually set num_workers at {num_workers} might be too high, cpu_count is {psutil.cpu_count(logical=False)}, max auto workers is {default_max}')
    return num_workers


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
