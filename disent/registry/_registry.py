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
from abc import ABC
from typing import Any
from typing import Callable
from typing import Dict
from typing import Generic
from typing import Iterator
from typing import List
from typing import MutableMapping
from typing import NoReturn
from typing import Optional
from typing import Protocol
from typing import Set
from typing import Tuple
from typing import TypeVar
from typing import Union

from disent.util.function import wrapped_partial
from disent.util.imports import _check_and_split_path
from disent.util.imports import import_obj
from disent.util.imports import import_obj_partial


# ========================================================================= #
# Type Hints                                                                #
# ========================================================================= #


K = TypeVar('K')
V = TypeVar('V')
T = TypeVar('T')
AliasesHint = Union[str, Tuple[str, ...]]


class _FactoryFn(Protocol[V]):
    def __call__(self, *args) -> V: ...




# ========================================================================= #
# Provided Values                                                           #
# ========================================================================= #


class ProvidedValue(Generic[V], ABC):
    """
    Base class for providing immutable values using the `get` method.
    - Subclasses should override this
    """

    def get(self) -> V:
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

    def get(self) -> V:
        return self._value

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

    def get(self) -> V:
        # cache the value
        if not self._is_generated:
            self._is_generated = True
            self._value = self._generate_fn()
        return self._value

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


# ========================================================================= #
# Provided Dict                                                             #
# ========================================================================= #


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
        return provider.get()

    def _setitem(self, k: K, v: ProvidedValue[V]) -> NoReturn:
        if not isinstance(v, ProvidedValue):
            raise TypeError(f'Values stored in {self.__class__.__name__} must be instances of: {ProvidedValue.__name__}, got: {repr(v)}')
        self._providers[k] = v


# ========================================================================= #
# Registry - Mixin                                                          #
# ========================================================================= #


class Registry(DictProviders[str, V]):

    def __init__(self, name: str):
        if not str.isidentifier(name):
            raise ValueError(f'Registry names must be valid identifiers, got: {repr(name)}')
        # initialise
        self._name = name
        super().__init__()

    @property
    def static_examples(self) -> List[str]:
        return list(self._providers.keys())

    @property
    def examples(self) -> List[str]:
        return self.static_examples

    @property
    def name(self) -> str:
        return self._name

    def __repr__(self):
        return f'{self.__class__.__name__}({self._name})'

    # --- CORE --- #

    def __setitem__(self, aliases: AliasesHint, v: ProvidedValue[V]) -> NoReturn:
        self._setitems(aliases, v)

    def __getitem__(self, k: str) -> V:
        value = self._getitem(k)
        self._check_provided_value(value)
        return value

    def __delitem__(self, k: str) -> None:
        raise RuntimeError(f'Registry: {repr(self.name)} does not support item deletion. Tried to remove key: {repr(k)}')

    # --- HELPER --- #

    def _setitems(self, aliases: AliasesHint, v: Union[V, ProvidedValue[V]]) -> None:
        aliases = self._normalise_aliases(aliases)
        # check all the aliases
        for k in aliases:
            if not str.isidentifier(k):
                raise ValueError(f'Keys stored in registry: {repr(self.name)} must be valid identifiers, got: {repr(k)}')
            if k in self:
                raise RuntimeError(f'Tried to overwrite existing key: {repr(k)} in registry: {repr(self.name)}')
            self._check_key(k)
        # check the value
        v = self._check_and_normalise_value(v)
        # set all the aliases
        for k in aliases:
            self._setitem(k, v)

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
        def __init__(self, registry: 'Registry'):
            self._registry = registry

        def __setitem__(self, aliases: str, v: ProvidedValue[V]) -> NoReturn:
            self._registry.setmissing(aliases, v)


# ========================================================================= #
# Import Registry                                                           #
# ========================================================================= #


# TODO: merge this with the dynamic registry below?
class RegistryImports(Registry[V]):
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
    ) -> Callable[[T], T]:
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
# Dynamic Registry                                                          #
# ========================================================================= #

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
# Constructor                                                               #
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #


class RegexConstructor(object):

    def __init__(
        self,
        pattern: Union[str, re.Pattern],
        example: str,
        factory_fn: Union[_FactoryFn[V], str],
    ):
        self._pattern = self._check_pattern(pattern)
        self._example = self._check_example(example, self._pattern)
        # we can delay loading of the function if it is a string!
        self._factory_fn = factory_fn if isinstance(factory_fn, str) else self._check_factory_fn(factory_fn, self._pattern)

    @classmethod
    def _check_pattern(cls, pattern: Union[str, re.Pattern]):
        # check the regex type & convert
        if isinstance(pattern, str):
            pattern = re.compile(pattern)
        if not isinstance(pattern, re.Pattern):
            raise TypeError(f'regex pattern must be a regex `str` or `re.Pattern`, got: {repr(pattern)}')
        if pattern.groups < 1:
            raise ValueError(f'regex pattern must contain at least one group, got: {repr(pattern)}')
        return pattern

    @classmethod
    def _check_factory_fn(cls, factory_fn: _FactoryFn[V], pattern: re.Pattern) -> _FactoryFn[V]:
        # we have an actual function, we can check it!
        if not callable(factory_fn):
            raise TypeError(f'generator function must be callable, got: {factory_fn}')
        signature = inspect.signature(factory_fn)
        if len(signature.parameters) != pattern.groups:
            raise ValueError(f'signature has incorrect number of parameters: {repr(signature)} compared to the number of groups in the regex pattern: {repr(pattern)}')
        return factory_fn

    @classmethod
    def _check_example(cls, example: str, pattern: re.Pattern) -> str:
        # check the example
        if not isinstance(example, str):
            raise TypeError(f'example must be a `str`, got: {type(example)}')
        if not example:
            raise ValueError(f'example must not be empty, got: {type(example)}')
        # check that the regex matches the example!
        if pattern.search(example) is None:
            raise ValueError(f'could not match example: {repr(example)} to regex: {repr(pattern)}')
        return example

    @property
    def pattern(self) -> re.Pattern:
        return self._pattern

    @property
    def example(self) -> str:
        return self._example

    def construct(self, name: str) -> V:
        # get the results
        result = self._pattern.search(name)
        if result is None:
            raise KeyError(f'pattern: {self.pattern} does not match given name: {repr(name)}. The following example would be valid: {repr(self.example)}')
        # get the function -- load via the path
        if isinstance(self._factory_fn, str):
            fn = import_obj(self._factory_fn)
            self._factory_fn = self._check_factory_fn(fn, self._pattern)
        # construct
        return self._factory_fn(*result.groups())

    def can_construct(self, name: str) -> bool:
        return self._pattern.search(name) is not None


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
# Cached Linear Search                                                      #
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #


class RegexProvidersSearch(object):

    def __init__(self):
        self._patterns: Set[re.Pattern] = set()
        self._constructors: List[RegexConstructor] = []
        # caching
        self._cache = {}
        self._cache_dirty = False

    @property
    def regex_constructors(self) -> List[RegexConstructor]:
        return list(self._constructors)

    def construct(self, arg_str: str):
        provider = self.get_constructor(arg_str)
        # build the object
        if provider is not None:
            return provider.construct(arg_str)
        # no result was found!
        raise KeyError(f'could not construct an item from the given argument string: {repr(arg_str)}, valid patterns include: {[p.pattern for p in self._constructors]}')

    def can_construct(self, arg_str: str) -> bool:
        return self.get_constructor(arg_str) is not None

    def get_constructor(self, arg_str: str) -> Optional[RegexConstructor]:
        # TODO: clean up this cache!
        # check cache -- remove None entries if dirty
        if self._cache_dirty:
            self._cache = {k: v for k, v in self._cache.items() if v is not None}
            self._cache_dirty = False
        if arg_str in self._cache:
            return self._cache[arg_str]
        # check the input string
        if not isinstance(arg_str, str):
            raise TypeError(f'regex factory can only construct from `str`, got: {repr(arg_str)}')
        if not arg_str:
            raise ValueError(f'regex factory can only construct from non-empty `str`, got: {repr(arg_str)}')
        # match the values
        constructor = None
        for c in self:
            if c.can_construct(arg_str):
                constructor = c
                break
        # cache the value
        self._cache[arg_str] = constructor
        if len(self._cache) > 128:
            self._cache.popitem()
        return constructor

    def has_pattern(self, pattern: Union[str, re.Pattern]) -> bool:
        if isinstance(pattern, str):
            pattern = re.compile(pattern)
        return pattern in self._patterns

    def __len__(self) -> int:
        return len(self._constructors)

    def __iter__(self) -> Iterator[RegexConstructor]:
        yield from self._constructors

    def append(self, constructor: RegexConstructor):
        if not isinstance(constructor, RegexConstructor):
            raise TypeError(f'regex factory only accepts {RegexConstructor.__name__} providers.')
        if constructor.pattern in self._patterns:
            raise RuntimeError(f'regex factory already contains the regex pattern: {repr(constructor.pattern)}')
        # append value!
        self._patterns.add(constructor.pattern)
        self._constructors.append(constructor)
        self._cache_dirty = True


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
# Dynamic Registry                                                          #
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #


class RegexRegistry(Registry[V]):

    """
    Registry that allows registering of regex expressions that can be used to
    construct values if there is no static value found!
    - Regular expressions are checked in the order they are registered.
    - `name in registry` checks if any of the expression matches, it does not check for an existing regex
    - `len(registry)` returns the number of examples available, each item & regex factory
    - `for example in registry` returns the examples available, each item & regex factory should be called if we use these to construct values `registry[example]`

    To check for an already added regex expression, use:
    - `has_regex(expr)`
    """

    def __init__(self, name: str):
        self._regex_providers = RegexProvidersSearch()
        super().__init__(name)

    # --- CORE ... UPDATED WITH LINEAR SEARCH --- #

    @property
    def regex_constructors(self) -> List[RegexConstructor]:
        return self._regex_providers.regex_constructors

    @property
    def regex_examples(self) -> List[str]:
        return [constructor.example for constructor in self._regex_providers.regex_constructors]

    @property
    def examples(self) -> List[str]:
        return [*self.static_examples, *self.regex_examples]

    def __getitem__(self, k: str) -> V:
        assert isinstance(k, str), f'invalid key: {repr(k)}, must be a `str`'
        # the regex provider is cached so this should be efficient for the same value calls
        # -- we do not cache the actual provided value!
        if k in self._providers:
            return self._getitem(k)
        elif self._regex_providers.can_construct(k):
            return self._regex_providers.construct(k)
        raise KeyError(f'dynamic registry: {repr(self.name)} cannot construct item with key: {repr(k)}. Valid static values: {sorted(self._providers.keys())}. Valid dynamic examples: {[p.example for p in self._regex_providers]}')

    def __setitem__(self, aliases: AliasesHint, v: ProvidedValue[V]) -> NoReturn:
        if isinstance(aliases, re.Pattern) or isinstance(v, RegexConstructor):
            raise RuntimeError(f'register dynamic values to the dynamic registry: {repr(self.name)} with the `register_regex` or `register_constructor` methods.')
        super().__setitem__(aliases, v)

    def __contains__(self, k: K):
        if k in self._providers:
            return True
        if self._regex_providers.can_construct(k):
            return True
        return False

    def __len__(self) -> int:
        return len(self._providers) + len(self._regex_providers)

    def __iter__(self) -> Iterator[K]:
        yield from self._providers
        yield from (p.example for p in self._regex_providers)

    # --- OVERRIDABLE --- #

    def _check_regex_constructor(self, constructor: RegexConstructor):
        pass

    # --- DYNAMIC VALUES --- #

    def has_regex(self, pattern: Union[str, re.Pattern]) -> bool:
        return self._regex_providers.has_pattern(pattern)

    def register_constructor(self, constructor: RegexConstructor) -> 'RegexRegistry':
        """
        Register a regex constructor
        """
        if not isinstance(constructor, RegexConstructor):
            raise TypeError(f'dynamic registry: {repr(self.name)} only accepts dynamic {RegexConstructor.__name__}, got: {repr(constructor)}')
        self._check_regex_constructor(constructor)
        self._regex_providers.append(constructor)
        return self

    def register_regex(self, pattern: Union[str, re.Pattern], example: str, factory_fn: Optional[Union[_FactoryFn[V], str]] = None):
        """
        Register and create a regex constructor
        """
        def _register_wrapper(fn: T) -> T:
            self.register_constructor(RegexConstructor(pattern=pattern, example=example, factory_fn=fn))
            return fn
        return _register_wrapper if (factory_fn is None) else _register_wrapper(factory_fn)

    def register_missing_constructor(self, constructor: RegexConstructor):
        """
        Only register a regex constructor if the pattern does not already exist!
        """
        if not self.has_regex(constructor.pattern):
            return self.register_constructor(constructor)

    def register_missing_regex(self, pattern: Union[str, re.Pattern], example: str, factory_fn: Optional[Union[_FactoryFn[V], str]] = None):
        """
        Only register and create a regex constructor if the pattern does not already exist!
        """
        if not self.has_regex(pattern):
            return self.register_regex(pattern=pattern, example=example, factory_fn=factory_fn)
        elif factory_fn is None:
            return lambda fn: fn  # dummy wrapper

    # --- MISSING VALUES --- #

    # override from the parent class!
    class _RegistrySetMissing(Registry._RegistrySetMissing):

        _registry: 'RegexRegistry'

        def register_constructor(self, constructor: RegexConstructor):
            """
            Only register a regex constructor if the pattern does not already exist!
            """
            return self._registry.register_missing_constructor(constructor=constructor)

        def register_regex(self, pattern: Union[str, re.Pattern], example: str, factory_fn: Optional[Union[_FactoryFn[V], str]] = None):
            """
            Only register and create a regex constructor if the pattern does not already exist!
            """
            return self._registry.register_missing_regex(pattern=pattern, example=example, factory_fn=factory_fn, )


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
