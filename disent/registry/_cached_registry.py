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
from typing import Set
from typing import Tuple


# ========================================================================= #
# Basic Cached Item                                                        #
# ========================================================================= #


class CachedRegistryItem(object):

    def __init__(self, keys: Tuple[str, ...]):
        # aliases
        assert isinstance(keys, tuple), f'Sequence of keys must be a tuple, got: {repr(keys)}'
        assert all(map(str.isidentifier, keys)), f'All keys must be valid python identifiers, got: {sorted(keys)}'
        self._keys = keys
        # save the path
        self._is_cached = False
        self._cache_value = None
        # handled by the registry
        self._is_registered = False
        self._assert_valid_value = None

    @property
    def keys(self) -> Tuple[str, ...]:
        return self._keys

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

    def purge(self) -> 'CachedRegistryItem':
        self._is_cached = False
        self._cache_value = None
        return self

    def link_to_registry(self, registry: 'CachedRegistry') -> 'CachedRegistryItem':
        # check we have only ever been linked once!
        if self._is_registered:
            raise RuntimeError('registry item has been registered more than once!')
        if not isinstance(registry, CachedRegistry):
            raise TypeError(f'expected registry to be of type: {CachedRegistry.__name__}, got: {type(registry)}')
        # link!
        self._is_registered = True
        self._assert_valid_value = registry.assert_valid_value
        return self

    def __repr__(self):
        return f'{self.__class__.__name__}({repr(self._keys)})'

    def _generate(self) -> Any:
        raise NotImplementedError


# ========================================================================= #
# Basic Cached Registry                                                     #
# ========================================================================= #


class CachedRegistry(object):

    def __init__(
        self,
        assert_valid_key: Callable[[Any], NoReturn] = None,
        assert_valid_value: Callable[[Any], NoReturn] = None,
    ):
        self._keys_to_items: Dict[str, CachedRegistryItem] = {}
        self._unique_items: Set[CachedRegistryItem] = set()
        # checks!
        assert (assert_valid_key is None) or callable(assert_valid_key), f'assert_valid_key must be None or callable!'
        assert (assert_valid_value is None) or callable(assert_valid_value), f'assert_valid_value must be None or callable!'
        self._assert_valid_key = assert_valid_key
        self._assert_valid_value = assert_valid_value

    def register(self, registry_item: CachedRegistryItem):
        # check the item
        if not isinstance(registry_item, CachedRegistryItem):
            raise TypeError(f'expected registry_item to be an instance of {CachedRegistryItem.__name__}')
        if registry_item in self._unique_items:
            raise RuntimeError(f'registry already contains registry_item: {registry_item}')
        # add to this registry
        self._unique_items.add(registry_item)
        registry_item.link_to_registry(self)
        # register keys
        for k in registry_item.keys:
            # check key
            if self._assert_valid_key is not None:
                self._assert_valid_key(k)
            if k in self._keys_to_items:
                raise RuntimeError(f'registry already contains key: {repr(k)}')
            # register key
            self._keys_to_items[k] = registry_item

    def __contains__(self, key: str):
        return key in self._keys_to_items

    def __getitem__(self, key: str):
        # get the registry item
        registry_item: Optional[CachedRegistryItem] = self._keys_to_items.get(key, None)
        # check that we have the key
        if registry_item is None:
            raise KeyError(f'registry does not contain the key: {repr(key)}, valid keys include: {sorted(self._keys_to_items.keys())}')
        # obtain or generate the cache value
        return registry_item.value

    def __iter__(self):
        return self.unique_keys()

    def unique_keys(self):
        for registry_item in self._unique_items:
            yield registry_item.keys[0]

    def all_keys(self):
        yield from self._keys_to_items.keys()

    def purge(self) -> 'CachedRegistry':
        for registry_item in self._unique_items:
            registry_item.purge()
        return self

    def assert_valid_value(self, value: Any) -> NoReturn:
        if self._assert_valid_value is not None:
            self._assert_valid_value(value)

    def assert_valid_key(self, key: str) -> NoReturn:
        if self._assert_valid_key is not None:
            self._assert_valid_key(key)


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
