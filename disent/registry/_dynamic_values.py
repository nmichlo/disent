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


import re
from typing import Any
from typing import Callable
from typing import Dict
from typing import NamedTuple
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union


# ========================================================================= #
# Value Registry                                                            #
# ========================================================================= #


class _ValueFactory(NamedTuple):
    name: str
    regex: re.Pattern
    example: str
    factory_fn: callable


class DynamicRegistry(object):

    # --- INIT --- #

    def __init__(
        self,
        name: str,
        parent_static_values: dict = None,
        make_static_value: callable = None,
        make_dynamic_value: callable = None,
        allowed_value_types: Sequence[type] = None,
        is_valid_value: Optional[Callable[[Any], Union[bool, Any]]] = None
    ):
        self._name = name
        self._registered_static: Dict[str, Any] = {} if (parent_static_values is None) else parent_static_values
        self._registered_dynamic: Dict[str, _ValueFactory] = {}
        self._allowed_value_types: Tuple[Any] = None if (allowed_value_types is None) else tuple(allowed_value_types)
        self._is_valid_value = is_valid_value
        # defaults
        self._make_static_value = make_static_value if (make_static_value is not None) else (lambda value, **kwargs: value)
        self._make_dynamic_value = make_dynamic_value if (make_dynamic_value is not None) else (lambda factory_fn, *args, **kwargs: factory_fn(*args, **kwargs))
        # checks
        if not callable(self._make_static_value):
            raise TypeError(f'`make_static_value` must be callable, eg: `lambda value, **kwargs: value`, got: {repr(self._make_static_value)}')
        if not callable(self._make_dynamic_value):
            raise TypeError(f'`make_dynamic_value` must be callable, eg: `lambda factory_fn, *args, **kwargs: factory_fn(*args)`, got: {repr(self._make_dynamic_value)}')
        if self._is_valid_value is not None:
            if not callable(self._is_valid_value):
                raise TypeError(f'`is_valid_value` must be callable, eg: `lambda value: True`, got: {repr(self._is_valid_value)}')

    @property
    def name(self) -> str:
        return self._name

    def __repr__(self):
        return f'{self.__class__.__name__}(name={repr(self.name)}, ...)'

    # --- REGISTRY --- #

    def register_static(self, name: str, value: Any):
        name = self._check_name(name)
        value = self._check_value(value)
        # register the static value
        self._registered_static[name] = value

    def register_dynamic(self, name: str, regex: Union[str, re.Pattern], example: str, factory_fn: callable):
        name = self._check_name(name)
        regex = self._check_regex(regex)
        example = self._check_example(example)
        factory_fn = self._check_factory_fn(factory_fn, regex=regex, example=example)
        # register the value
        self._registered_dynamic[name] = _ValueFactory(name=name, regex=regex, example=example, factory_fn=factory_fn)

    def make_value(self, name: str, **kwargs):
        # search static values
        if name in self._registered_static:
            static_value = self._make_static_value(self._registered_static[name], **kwargs)
            return self._check_value(static_value)
        # search dynamic values -- this is slow, cache this!
        for dyn in self._registered_dynamic.values():
            result = dyn.regex.search(name)
            if result is not None:
                dynamic_value = self._make_dynamic_value(dyn.factory_fn, *result.groups(), **kwargs)
                return self._check_value(dynamic_value)
        # we couldn't find anything
        raise KeyError(f'Invalid dynamic registry name: {repr(name)}. Valid static names include: {sorted(self._registered_static.keys())} and examples of dynamic names include: {[d.example for d in self._registered_dynamic.values()]}')

    # --- CHECKS --- #

    def _check_name(self, name: str) -> str:
        # check types
        if not isinstance(name, str):
            raise TypeError(f'register name must be a `str`, got: {type(name)}')
        if not name:
            raise ValueError(f'register name must not be empty, got: {type(name)}')
        # check the name does not exist
        if name in self._registered_static:
            raise ValueError(f'registered static value already exists, cannot register: {repr(name)}')
        if name in self._registered_dynamic:
            raise ValueError(f'registered dynamic value already exists, cannot register: {repr(name)}')
        # done!
        return name

    def _check_regex(self, regex: Union[str, re.Pattern]) -> re.Pattern:
        # check the regex type & convert
        if isinstance(regex, str):
            regex = re.compile(regex)
        if not isinstance(regex, re.Pattern):
            raise TypeError(f'regex must be a regex `str` or `re.Pattern`, got: {type(regex)}')
        # make sure there are groups
        if regex.groups < 1:
            raise ValueError(f'regex must contain at least one group, got: {repr(regex)}')
        # done!
        return regex

    def _check_example(self, example: str) -> str:
        if not isinstance(example, str):
            raise TypeError(f'example must be a `str`, got: {type(example)}')
        if not example:
            raise ValueError(f'example must not be empty, got: {type(example)}')
        return example

    def _check_factory_fn(self, factory_fn: callable, regex: re.Pattern, example: str):
        if not callable(factory_fn):
            raise TypeError(f'`factory_fn` must be callable, got: {repr(factory_fn)}')
        # check that the regex matches the example!
        result = regex.search(example)
        if result is None:
            raise ValueError(f'could not match example: {repr(example)} to regex: {repr(regex)}')
        # check that we can create the object!
        # if check_factory:
        #     dynamic_value = self._make_dynamic_value(factory_fn, *result.groups())
        #     self._check_value(dynamic_value)

    def _check_value(self, value: Any) -> Any:
        if self._allowed_value_types is not None:
            if not isinstance(value, self._allowed_value_types):
                raise TypeError(f'registry value has an incorrect type, got: {value}, must be of types: {self._allowed_value_types}')
        if self._is_valid_value is not None:
            is_valid = self._is_valid_value(value)
            if isinstance(is_valid, bool):
                if not is_valid:
                    raise ValueError(f'registry value is invalid, got: {value}')
        return value

# ========================================================================= #
# END                                                                       #
# ========================================================================= #
