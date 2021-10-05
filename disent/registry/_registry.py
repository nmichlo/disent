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

from typing import Any
from typing import Callable
from typing import Dict
from typing import NoReturn
from typing import Optional
from typing import Sequence
from typing import Set
from typing import Tuple


# ========================================================================= #
# Basic Cached Item                                                        #
# ========================================================================= #


class CachedValue(object):

    def __init__(self):
        # save the path
        self._is_cached = False
        self._cache_value = None
        # handled by the registry
        self._is_registered = False
        self._assert_valid_value = None

    @property
    def value(self) -> Any:
        # cache the imported value
        if not self._is_cached:
            if not self._is_registered:
                raise RuntimeError('registry item must be linked to a registry before the value can be computed!')
            self._is_cached = True
            # generate the value
            value = self._generate()
            self._assert_valid_value(value)  # never None if _is_registered
            self._cache_value = value
        # do the import!
        return self._cache_value

    def link_to_registry(self, registry: 'Registry') -> 'CachedValue':
        # check we have only ever been linked once!
        if self._is_registered:
            raise RuntimeError('registry item has been registered more than once!')
        if not isinstance(registry, Registry):
            raise TypeError(f'expected registry to be of type: {Registry.__name__}, got: {type(registry)}')
        # initialize link
        self._is_registered = True
        self._assert_valid_value = registry.assert_valid_value
        return self

    def __repr__(self):
        return f'{self.__class__.__name__}()'

    def _generate(self) -> Any:
        raise NotImplementedError


# ========================================================================= #
# Registry                                                                  #
# ========================================================================= #


class Registry(object):

    def __init__(
        self,
        assert_valid_key,
        assert_valid_value,
    ):
        self._keys_to_values: Dict[str, Any] = {}
        self._unique_values: Set[Any] = set()
        self._unique_keys: Set[Any] = set()
        # checks!
        assert (assert_valid_key is None) or callable(assert_valid_key), f'assert_valid_key must be None or callable'
        assert (assert_valid_value is None) or callable(assert_valid_value), f'assert_valid_value must be None or callable'
        self._assert_valid_key = assert_valid_key
        self._assert_valid_value = assert_valid_value

    def __call__(
        self,
        aliases: Sequence[str]
    ):
        def _decorator(fn):
            # register the function
            self.register(value=fn, aliases=aliases)
            # return the original function
            return fn
        return _decorator

    def register(
        self,
        value: Any,
        aliases: Sequence[str],
    ) -> 'Registry':
        # check keys
        if len(aliases) < 1:
            raise ValueError(f'aliases must be specified, got an empty sequence')
        for k in aliases:
            if not str.isidentifier(k):
                raise ValueError(f'alias is not a valid identifier: {repr(k)}')
            if k in self._keys_to_values:
                raise RuntimeError(f'registry already contains key: {repr(k)}')
            self.assert_valid_key(k)
        # check value
        if value in self._unique_values:
            raise RuntimeError(f'registry already contains value: {value}')
        # handle caching
        if isinstance(value, CachedValue):
            self.assert_valid_value(value)
        else:
            value.link_to_registry(self)
        # register value & keys
        self._unique_values.add(value)
        self._unique_keys.add(aliases[0])
        for k in aliases:
            self._keys_to_values[k] = value
        # done!
        return self

    def __contains__(self, key: str):
        return key in self._keys_to_values

    def __getitem__(self, key: str):
        if key not in self._keys_to_values:
            raise KeyError(f'registry does not contain the key: {repr(key)}, valid keys include: {sorted(self._keys_to_values.keys())}')
        # handle caching
        value = self._keys_to_values[key]
        if isinstance(value, CachedValue):
            value = value.value
        return value

    def __iter__(self):
        return self.iter_unique_keys()

    def __len__(self):
        return len(self._unique_values)

    def iter_unique_keys(self):
        yield from self._unique_keys

    def iter_all_keys(self):
        yield from self._keys_to_values.keys()

    def assert_valid_value(self, value: Any) -> NoReturn:
        if self._assert_valid_value is not None:
            self._assert_valid_value(value)

    def assert_valid_key(self, key: str) -> NoReturn:
        if self._assert_valid_key is not None:
            self._assert_valid_key(key)


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
