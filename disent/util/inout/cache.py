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
from functools import wraps
from typing import Callable
from typing import Dict
from typing import NoReturn
from typing import Optional
from typing import Union

from disent.util.inout.hashing import normalise_hash
from disent.util.inout.hashing import hash_file
from disent.util.inout.hashing import validate_file_hash


log = logging.getLogger(__name__)


# ========================================================================= #
# Function Caching                                                          #
# ========================================================================= #


class stalefile(object):
    """
    decorator that only runs the wrapped function if a
    file does not exist, or its hash does not match.
    """

    def __init__(
        self,
        file: str,
        hash: Optional[Union[str, Dict[str, str]]],
        hash_type: str = 'md5',
        hash_mode: str = 'fast',
    ):
        self.file = file
        self.hash = normalise_hash(hash=hash, hash_mode=hash_mode)
        self.hash_type = hash_type
        self.hash_mode = hash_mode

    def __call__(self, func: Callable[[str], NoReturn]) -> Callable[[], str]:
        @wraps(func)
        def wrapper() -> str:
            if self.is_stale():
                log.debug(f'calling wrapped function: {func} because the file is stale: {repr(self.file)}')
                func(self.file)
                validate_file_hash(self.file, hash=self.hash, hash_type=self.hash_type, hash_mode=self.hash_mode)
            else:
                log.debug(f'skipped wrapped function: {func} because the file is fresh: {repr(self.file)}')
            return self.file
        return wrapper

    def is_stale(self):
        fhash = hash_file(file=self.file, hash_type=self.hash_type, hash_mode=self.hash_mode, missing_ok=True)
        if not fhash:
            log.info(f'file is stale because it does not exist: {repr(self.file)}')
            return True
        if fhash != self.hash:
            log.warning(f'file is stale because the computed {self.hash_mode} {self.hash_type} hash: {fhash} does not match the target hash: {self.hash} for file: {repr(self.file)}')
            return True
        log.debug(f'file is fresh: {repr(self.file)}')
        return False

    def __bool__(self):
        return self.is_stale()


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
