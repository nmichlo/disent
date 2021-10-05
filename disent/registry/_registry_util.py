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
from typing import Dict
from typing import ForwardRef
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import TypeVar

from disent.registry._cached_registry import CachedRegistry
from disent.registry._cached_registry import CachedRegistryItem


# ========================================================================= #
# Import Helper                                                             #
# ========================================================================= #


def _check_and_split_path(import_path: str) -> Tuple[str, ...]:
    segments = import_path.split('.')
    # make sure each segment is a valid python identifier
    if not all(map(str.isidentifier, segments)):
        raise ValueError(f'import path is invalid: {repr(import_path)}')
    # return the segments!
    return tuple(segments)


def _import(import_path: str):
    # checks
    segments = _check_and_split_path(import_path)
    # split path
    module_path, attr_name = '.'.join(segments[:-1]), segments[-1]
    # import the module
    import importlib
    try:
        module = importlib.import_module(module_path)
    except Exception as e:
        raise ImportError(f'failed to import module: {repr(module_path)}') from e
    # import the attrs
    try:
        attr = getattr(module, attr_name)
    except Exception as e:
        raise ImportError(f'failed to get attribute: {repr(attr_name)} on module: {repr(module_path)}') from e
    # done
    return attr


# ========================================================================= #
# Implement Registry Item                                                   #
# ========================================================================= #


class CachedImportItem(CachedRegistryItem):

    def __init__(self, keys: Tuple[str, ...], import_path: str):
        super().__init__(keys)
        # make sure the path is valid!
        _check_and_split_path(import_path)
        self._import_path = import_path

    def _generate(self) -> Any:
        return _import(self._import_path)


# ========================================================================= #
# Import Info                                                               #
# ========================================================================= #


class _ImportInfo(object):

    def __init__(
        self,
        aliases: Sequence[str] = None,
    ):
        self._aliases = tuple(aliases) if (aliases is not None) else ()
        # check values
        if not all(map(str.isidentifier, self._aliases)):
            raise ValueError(f'aliases should all be identifiers: {repr(self._aliases)}')

    @property
    def aliases(self) -> Tuple[str]:
        return self._aliases


# hack to trick pycharm
T = TypeVar('T')


def import_info(
    aliases: Sequence[str] = None,
) -> T:
    return _ImportInfo(aliases=aliases)


# ========================================================================= #
# Import Registry Meta-Class                                                #
# ========================================================================= #


class ImportRegistryMeta(object):
    """
    Check for class properties that are annotated with Type[str] and their values are CachedImportItems.
    - Extracts these values and constructs a registry from them!


    >>> if False:
    >>>    import disent.dataset.data
    >>>
    >>> class DataRegistry(metaclass=ImportRegistryMeta):
    >>>    XYObject: Type['disent.dataset.data.XYObjectData'] = import_info()
    """

    def __init__(cls, name, bases, dct):
        cls.__registry = CachedRegistry(
            assert_valid_key=getattr(cls, 'assert_valid_key', None),
            assert_valid_value=getattr(cls, 'assert_valid_value', None),
        )
        # sort
        annotated = dct['__annotations__']
        assigned = {k: v for k, v in dct.items() if not k.startswith('_')}
        # check corresponding
        assert not (annotated.keys() - assigned.keys()), f'Some annotated fields do not have a corresponding assignment: {sorted(annotated.keys() - assigned.keys())}'
        assert not (assigned.keys() - annotated.keys()), f'Some assigned fields do not have a corresponding annotation: {sorted(assigned.keys() - annotated.keys())}'
        # check types
        t = Type[object].__class__
        incorrect_annotations: Dict[str, t] = {k: v for k, v in annotated.items() if not (type(v) == t and hasattr(v, '__args__') and isinstance(v.__args__, tuple) and len(v.__args__) == 1 and isinstance(v.__args__[0], ForwardRef))}
        incorrect_assigned: Dict[str, _ImportInfo] = {k: v for k, v in assigned.items() if not isinstance(v, _ImportInfo)}
        assert not incorrect_annotations, f'Annotations must be Type[str], incorrect annotations include: {sorted(incorrect_annotations.keys())}'
        assert not incorrect_assigned, f'Assignments must be {_ImportInfo.__name__} instances, incorrect assignments include: {sorted(incorrect_assigned.keys())}'
        # get values
        for key, typ, val in ((k, annotated[k], assigned[k]) for k in annotated.keys()):
            # extract reference
            [ref] = typ.__args__
            import_path = ref.__forward_arg__
            assert isinstance(import_path, str)
            # get values
            cls.__registry.register(CachedImportItem(
                keys=(key, *val.aliases),
                import_path=import_path,
            ))

    def __contains__(cls, key): return key in cls.__registry
    def __getitem__(cls, key): return cls.__registry[key]
    def __iter__(cls): return cls.__registry.__iter__()

    def __getattr__(cls, key):
        if key not in cls.__registry:
            raise AttributeError(f'invalid attribute: {repr(key)}, must be one of: {sorted(cls.__registry.all_keys())}')
        return cls.__registry[key]

    def assert_valid_key(self, key: str):
        """overridable"""

    def assert_valid_value(self, value: Any):
        """overridable"""


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
