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
from abc import ABCMeta
from typing import Callable
from typing import NoReturn

from disent.data.util.in_out import hash_file


log = logging.getLogger(__name__)


# ========================================================================= #
# Base Job                                                                  #
# ========================================================================= #


class CachedJob(object):
    """
    Base class for cached jobs. A job is some arbitrary directed chains
    of computations where child jobs depend on parent jobs, and jobs
    can be skipped if it has already been run and the cache is valid.

    Jobs are always deterministic, and if run and cached should never go out of date.

    NOTE: if needed it would be easy to add support directed acyclic graphs, and sub-graphs
    NOTE: this is probably overkill, but it makes the code to write a new dataset nice and clean...
    """

    def __init__(self, job_fn: Callable[[], NoReturn], is_cached_fn: Callable[[], bool]):
        self._parent = None
        self._child = None
        self._job_fn = job_fn
        self._is_cached_fn = is_cached_fn

    def __repr__(self):
        return f'{self.__class__.__name__}'

    def set_parent(self, parent: 'CachedJob'):
        if not isinstance(parent, CachedJob):
            raise TypeError(f'{self}: parent job was not an instance of: {CachedJob.__class__}')
        if self._parent is not None:
            raise RuntimeError(f'{self}: parent has already been set')
        if parent._child is not None:
            raise RuntimeError(f'{parent}: child has already been set')
        self._parent = parent
        parent._child = self
        return parent

    def set_child(self, child: 'CachedJob'):
        child.set_parent(self)
        return child

    def run(self, force=False, recursive=False) -> 'CachedJob':
        # visit parents always
        if recursive:
            if self._parent is not None:
                self._parent.run(force=force, recursive=recursive)
        # skip if fresh
        if not force:
            if not self._is_cached_fn():
                log.debug(f'{self}: skipped non-stale job')
                return self
        # don't visit parents if fresh
        if not recursive:
            if self._parent is not None:
                self._parent.run(force=force, recursive=recursive)
        # run nodes
        log.debug(f'{self}: run stale job')
        self._job_fn()
        return self


# ========================================================================= #
# Base File Job                                                             #
# ========================================================================= #


class CachedJobFile(CachedJob):

    """
    An abstract cached job that only runs if a file does not exist,
    or the files hash sum does not match the given value.
    """

    def __init__(
        self,
        make_file_fn: Callable[[str], NoReturn],
        path: str,
        hash: str,
        hash_type: str = 'md5',
        hash_mode: str = 'full',
    ):
        # set attributes
        self.path = path
        self.hash = hash
        self.hash_type = hash_type
        self.hash_mode = hash_mode
        # generate
        self._make_file_fn = make_file_fn
        # check hash
        super().__init__(job_fn=self.__job_fn, is_cached_fn=self.__is_cached_fn)

    def __compute_hash(self) -> str:
        return hash_file(self.path, hash_type=self.hash_type, hash_mode=self.hash_mode)

    def __is_cached_fn(self) -> bool:
        # stale if the file does not exist
        if not os.path.exists(self.path):
            log.warning(f'{self}: stale because file does not exist: {repr(self.path)}')
            return True
        # stale if the hash does not match
        fhash = self.__compute_hash()
        if self.hash != fhash:
            log.warning(f'{self}: stale because computed {self.hash_mode} {self.hash_type} hash: {repr(fhash)} does not match expected hash: {repr(self.hash)} for: {repr(self.path)}')
            return True
        # not stale, we don't need to do anything!
        return False

    def __job_fn(self):
        self._make_file_fn(self.path)
        # check the hash
        fhash = self.__compute_hash()
        if self.hash != fhash:
            raise RuntimeError(f'{self}: computed {self.hash_mode} {self.hash_type} hash: {repr(fhash)} for newly generated file {repr(self.path)} does not match expected hash: {repr(self.hash)}')
        else:
            log.debug(f'{self}: successfully generated file: {repr(self.path)} with correct {self.hash_mode} {self.hash_type} hash: {fhash}')


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
