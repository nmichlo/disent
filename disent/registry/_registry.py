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

import inspect
import re
from abc import abstractmethod
from functools import lru_cache
from typing import Any
from typing import Callable
from typing import Dict
from typing import Generic
from typing import Iterable
from typing import Iterator
from typing import List
from typing import MutableMapping
from typing import MutableSequence
from typing import NoReturn
from typing import Optional
from typing import overload
from typing import Protocol
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import TypeVar
from typing import Union

from disent.util.function import wrapped_partial
from disent.util.imports import import_obj_partial
from disent.util.imports import _check_and_split_path


# ========================================================================= #
# Type Hints                                                                #
# ========================================================================= #


K = TypeVar('K')
V = TypeVar('V')
AliasesHint = Union[str, Tuple[str, ...]]


# ========================================================================= #
# Provided Values                                                           #
# ========================================================================= #


class ProvidedValue(Generic[V]):
    """
    Base class for providing immutable values using the `get` method.
    - Subclasses should override this
    """

    def get(self, name: str) -> V:
        raise NotImplementedError

    def matches(self, name: str) -> bool:
        raise NotImplementedError

    def __repr__(self):
        return f'{self.__class__.__name__}()'


class StaticValue(ProvidedValue[V]):
    """
    Provide static values. Simply a see-through wrapper
    around already generated / constant values.
    """

    def __init__(self, value: V):
        self._value = value

    def get(self, name: str) -> V:
        return self._value

    def matches(self, name: str) -> bool:
        return True

    def __repr__(self):
        return f'{self.__class__.__name__}({repr(self._value)})'


class StaticImport(StaticValue[V]):
    def __init__(self, fn: V, *partial_args, **partial_kwargs):
        super().__init__(wrapped_partial(fn, *partial_args, **partial_kwargs))


class LazyValue(ProvidedValue[V]):
    """
    Use a function to provide a value by generating and caching
    the result only when this value is first needed.
    """

    def __init__(self, generate_fn: Callable[[], V]):
        assert callable(generate_fn)
        self._generate_fn = generate_fn
        self._is_generated = False
        self._value = None

    def get(self, name: str) -> V:
        # cache the value
        if not self._is_generated:
            self._is_generated = True
            self._value = self._generate_fn()
        return self._value

    def matches(self, name: str) -> bool:
        return True

    def clear(self):
        self._is_generated = False
        self._value = None

    def __repr__(self):
        return f'{self.__class__.__name__}({repr(self._generate_fn)})'


class LazyImport(LazyValue[V]):
    """
    Like lazy value, but instead takes in the import path to a callable object.
    Any remaining args and kwargs are used to partially parameterize the object.
    - The partial object is returned, and not called. This should be
      the same as importing the value directly when `get` is called!
    """

    def __init__(self, import_path: str, *partial_args, **partial_kwargs):
        # function imports the object when called
        def generate_fn():
            return import_obj_partial(import_path, *partial_args, **partial_kwargs)
        # initialise the lazy value
        super().__init__(generate_fn=generate_fn)


class RegexValue(ProvidedValue):

    def __init__(
        self,
        pattern: Union[str, re.Pattern],
        example: str,
        factory_fn: Callable[[...], V],
    ):
        self._pattern = pattern
        self._example = example
        self._factory_fn = factory_fn

        # check the regex type & convert
        if isinstance(self._pattern, str):
            self._pattern = re.compile(self._pattern)
        if not isinstance(self._pattern, re.Pattern):
            raise TypeError(f'regex pattern must be a regex `str` or `re.Pattern`, got: {repr(self._pattern)}')
        if self._pattern.groups < 1:
            raise ValueError(f'regex pattern must contain at least one group, got: {repr(self._pattern)}')

        # check the factory function
        if not callable(self._factory_fn):
            raise TypeError(f'generator function must be callable, got: {self._factory_fn}')
        signature = inspect.signature(self._factory_fn)
        if len(signature.parameters) != self._pattern.groups:
            raise ValueError(f'signature has incorrect number of parameters: {repr(signature)} compared to the number of groups in the regex pattern: {repr(self._pattern)}')

        # check the example
        if not isinstance(self._example, str):
            raise TypeError(f'example must be a `str`, got: {type(self._example)}')
        if not self._example:
            raise ValueError(f'example must not be empty, got: {type(self._example)}')
        # check that the regex matches the example!
        result = self._pattern.search(self._example)
        if result is None:
            raise ValueError(f'could not match example: {repr(self._example)} to regex: {repr(self._pattern)}')

    @property
    def pattern(self) -> re.Pattern:
        return self._pattern

    @property
    def example(self) -> str:
        return self._example

    def get(self, name: str) -> V:
        result = self._pattern.search(name)
        if result is None:
            raise KeyError(f'pattern: {self.pattern} does not match given name: {repr(name)}. The following example would be valid: {repr(self.example)}')
        return self._factory_fn(*result.groups())

    def matches(self, name: str) -> bool:
        return self._pattern.search(name) is not None


# ========================================================================= #
# Provided Dict                                                             #
# ========================================================================= #


class _ItemProto(Protocol[K, V]):
    def _setitem(self, k: K, v: V) -> NoReturn: ...
    def _getitem(self, k: K) -> V: ...


class DictProviders(MutableMapping[K, V]):
    """
    A dictionary that only allows instances of
    provided values to be added to it.
    - The returned values are obtained directly from the providers
    """

    def __init__(self):
        self._providers: Dict[K, ProvidedValue[V]] = {}

    def __getitem__(self, k: K) -> V:
        return self._getitem(k)

    def __contains__(self, k: K):
        return k in self._providers

    def __setitem__(self, k: K, v: ProvidedValue[V]) -> NoReturn:
        self._setitem(k, v)

    def __delitem__(self, k: K) -> NoReturn:
        del self._providers[k]

    def __len__(self) -> int:
        return len(self._providers)

    def __iter__(self) -> Iterator[K]:
        yield from self._providers

    # allow easier overriding in subclasses without calling super() which can get confusing

    def _getitem(self, k: K) -> V:
        provider = self._providers[k]
        return provider.get(k)

    def _setitem(self, k: K, v: ProvidedValue[V]) -> NoReturn:
        if not isinstance(v, ProvidedValue):
            raise TypeError(f'Values stored in {self.__class__.__name__} must be instances of: {ProvidedValue.__name__}, got: {repr(v)}')
        self._providers[k] = v


class DictUnwrapProviders(DictProviders[K, V]):
    """
    Unlike ProvidedDict above, the returned values
    are only obtained from the providers once.
    - Providers are deleted and replaced with the values they return
    - Provided values should not be nested
    """

    # change the type hint
    __setitem__: Callable[[K, Union[V, ProvidedValue[V]]], NoReturn]

    def _getitem(self, k: K) -> V:
        item = self._providers[k]
        # unwrap and delete the provider, storing the value instead
        if isinstance(item, ProvidedValue):
            item = item.get(k)
            assert not isinstance(item, ProvidedValue), 'provided values cannot be nested.'
            self._providers[k] = item
        # get the items
        return item

    def _setitem(self, k: K, v: V) -> NoReturn:
        self._providers[k] = v


# ========================================================================= #
# Registry - Mixin                                                          #
# ========================================================================= #


class _RegistryMixin(Generic[V]):

    def __init__(self, name: str):
        if not str.isidentifier(name):
            raise ValueError(f'Registry names must be valid identifiers, got: {repr(name)}')
        # initialise
        self._name = name
        super().__init__()

    @property
    def name(self) -> str:
        return self._name

    def __repr__(self):
        return f'{self.__class__.__name__}({self._name})'

    # --- CORE --- #

    def __setitem__(self, aliases: AliasesHint, v: Any) -> None:
        self._setitems(aliases, v)

    def __getitem__(self: Union[_ItemProto, '_RegistryMixin'], k: str) -> V:
        value = self._getitem(k)
        self._check_provided_value(value)
        return value

    def __delitem__(self, k: str) -> None:
        raise RuntimeError(f'Registry: {repr(self.name)} does not support item deletion. Tried to remove key: {repr(k)}')

    # --- HELPER --- #

    def _setitems(self: Union[_ItemProto, '_RegistryMixin'], aliases: AliasesHint, v: Union[V, ProvidedValue[V]]) -> None:
        aliases = self._normalise_aliases(aliases)
        # check all the aliases
        for k in aliases:
            if not str.isidentifier(k):
                raise ValueError(f'Keys stored in registry: {repr(self.name)} must be valid identifiers, got: {repr(k)}')
            if k in self:
                raise RuntimeError(f'Tried to overwrite existing key: {repr(k)} in registry: {repr(self.name)}')
            self._check_key(k)
        # set all the aliases
        for k in aliases:
            self._setitem(k, self._check_and_normalise_value(v))

    def _normalise_aliases(self, aliases: AliasesHint, check_nonempty: bool = True) -> Tuple[str]:
        if isinstance(aliases, str):
            aliases = (aliases,)
        if not isinstance(aliases, tuple):
            raise TypeError(f'Multiple aliases must be provided to registry: {repr(self.name)} as a Tuple[str], got: {repr(aliases)}')
        if check_nonempty:
            if len(aliases) < 1:
                raise ValueError(f'At least one alias must be provided to registry: {repr(self.name)}, got: {repr(aliases)}')
        return aliases

    # --- OVERRIDABLE --- #

    def _check_and_normalise_value(self, v: ProvidedValue[V]) -> ProvidedValue[V]:
        return v

    def _check_provided_value(self, v: V) -> NoReturn:
        pass

    def _check_key(self, k: str) -> NoReturn:
        pass

    # --- MISSING VALUES --- #

    def setmissing(self, alias: AliasesHint, value: V) -> NoReturn:
        # find missing keys
        aliases = self._normalise_aliases(alias)
        missing = tuple(alias for alias in aliases if (alias not in self))
        # register missing keys
        if missing:
            self._setitems(missing, value)

    @property
    def setm(self) -> '_RegistrySetMissing':
        # instead of checking values manually, at the cost of some efficiency,
        # this allows us to register values multiple times with hardly modified notation!
        # -- only modifies unset values
        # set once:    `REGISTRY['key'] = val`
        # set default: `REGISTRY.setm['key'] = val`
        return self._RegistrySetMissing(self)

    class _RegistrySetMissing(object):
        def __init__(self, registry: '_RegistryMixin'):
            self._registry: _RegistryMixin = registry

        def __setitem__(self, aliases: str, v: ProvidedValue[V]) -> NoReturn:
            self._registry.setmissing(aliases, v)


# ========================================================================= #
# Registry - Mixin                                                          #
# ========================================================================= #


class RegistryProviders(_RegistryMixin[V], DictProviders[str, V]):
    """
    A registry is an immutable `DictProviders` that can also take in
    tuples as keys to set aliases to the same value for all of those keys!
    """

    # change the type hint
    __setitem__: Callable[[AliasesHint, ProvidedValue[V]], NoReturn]


class RegistryUnwrapProviders(_RegistryMixin[V], DictUnwrapProviders[str, V]):
    """
    A registry is an immutable `DictUnwrapProviders` that can also take in
    tuples as keys to set aliases to the same value for all of those keys!
    """

    # change the type hint
    __setitem__: Callable[[AliasesHint, Union[V, ProvidedValue[V]]], NoReturn]


# ========================================================================= #
# Import Registry                                                           #
# ========================================================================= #


class RegistryImports(RegistryProviders):
    """
    A registry for arbitrary imports.
    -- supports decorating functions and classes
    """

    def register(
        self,
        aliases: Optional[AliasesHint] = None,
        auto_alias: bool = True,
        partial_args: Tuple[Any, ...] = None,
        partial_kwargs: Dict[str, Any] = None,
    ) -> Callable[[V], V]:
        """
        Register a function or object to this registry.
        - can be used as a decorator @register(...)
        - automatically chooses an alias based on the function name
        - specify defaults for the function with the args and kwargs
        """
        # default values
        if aliases is None: aliases = ()
        if partial_args is None: partial_args = ()
        if partial_kwargs is None: partial_kwargs = {}
        aliases = self._normalise_aliases(aliases, check_nonempty=False)

        # add the function name as an alias if it does not already exist,
        # then register the partially parameterised function as a static value
        def _decorator(orig_fn):
            keys = self._append_auto_alias(self._get_fn_alias(orig_fn), aliases=aliases, auto_alias=auto_alias)
            self[keys] = StaticImport(orig_fn, *partial_args, **partial_kwargs)
            return orig_fn
        return _decorator

    def register_import(
        self,
        import_path: str,
        aliases: Optional[AliasesHint] = None,
        auto_alias: bool = True,
        *partial_args,
        **partial_kwargs,
    ) -> NoReturn:
        """
        Register an import path and automatically obtain an alias from it.
        - This is the same as: registry[(import_name, *aliases)] = LazyImport(import_path, *partial_args, **partial_kwargs)
        """
        # normalise aliases
        if aliases is None: aliases = ()
        aliases = self._normalise_aliases(aliases, check_nonempty=False)
        # add object alias
        (*_, alias) = _check_and_split_path(import_path)
        aliases = self._append_auto_alias(alias, aliases=aliases, auto_alias=auto_alias)
        # register the lazy import
        self[aliases] = LazyImport(import_path=import_path, *partial_args, **partial_kwargs)

    # --- ALIAS HELPER --- #

    def _append_auto_alias(self, alias: Optional[str], aliases: Tuple[str, ...], auto_alias: bool):
        if auto_alias:
            if alias is not None:
                if alias not in self:
                    aliases = (alias, *aliases)
                elif not aliases:
                    raise RuntimeError(f'automatic alias: {repr(alias)} already exists for registry: {repr(self.name)} and no alternative aliases were specified.')
            elif not aliases:
                raise RuntimeError(f'Cannot add value to registry: {repr(self.name)}, no automatic alias was found!')
        elif not aliases:
            raise RuntimeError(f'Cannot add value to registry: {repr(self.name)}, no manual aliases were specified and automatic aliasing is disabled!')
        return aliases

    @staticmethod
    def _get_fn_alias(fn) -> Optional[str]:
        if hasattr(fn, '__name__'):
            if str.isidentifier(fn.__name__):
                return fn.__name__
        return None

    # --- OVERRIDDEN --- #

    def _check_and_normalise_value(self, v: ProvidedValue[V]) -> ProvidedValue[V]:
        if not isinstance(v, (LazyImport, StaticImport)):
            raise TypeError(f'Values stored in registry: {repr(self.name)} must be instances of: {(LazyImport.__name__, StaticImport.__name__)}, got: {repr(v)}')
        return v

    # --- OVERRIDABLE --- #

    def _check_provided_value(self, v: V) -> NoReturn:
        pass

    def _check_key(self, k: str) -> NoReturn:
        pass


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
