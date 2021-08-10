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


# ========================================================================= #
# Registry Helper                                                           #
# ========================================================================= #


class PathBuilder(object):
    """
    Path builder stores the path taken down attributes
      - This is used to trick pycharm type hinting. In the example
        below, `Cars3dData` will be an instance of `_PathBuilder`,
        but will type hint to `disent.dataset.data.Cars3dData`
        ```
        disent = _PathBuilder()
        if False:
            import disent.dataset.data
        Cars3dData = disent.dataset.data._groundtruth__cars3d.Cars3dData
        ```
    """

    def __init__(self, *segments):
        self.__segments = tuple(segments)

    def __getattr__(self, item: str):
        return PathBuilder(*self.__segments, item)

    def _do_import_(self):
        import importlib
        import_module, import_name = '.'.join(self.__segments[:-1]), self.__segments[-1]
        try:
            module = importlib.import_module(import_module)
        except Exception as e:
            raise ImportError(f'failed to import module: {repr(import_module)} ({".".join(self.__segments)})') from e
        try:
            obj = getattr(module, import_name)
        except Exception as e:
            raise ImportError(f'failed to get attribute on module: {repr(import_name)} ({".".join(self.__segments)})') from e
        return obj


def LazyImportMeta(to_lowercase: bool = True):
    """
    Lazy import paths metaclass checks for stored instances of `_PathBuilder` on a class and returns the
    imported version of the attribute instead of the `_PathBuilder` itself.
      - Used to perform lazy importing of classes and objects inside a module
    """

    if to_lowercase:
        def transform(item):
            if isinstance(item, str):
                return item.lower()
            return item
    else:
        def transform(item):
            return item

    class _LazyImportMeta:
        def __init__(cls, name, bases, dct):
            cls.__unimported = {}  # Dict[str, _PathBuilder]
            cls.__imported = {}    # Dict[str, Any]
            # check annotations
            for key, value in dct.items():
                if isinstance(value, PathBuilder):
                    assert str.isidentifier(key), f'registry key is not an identifier: {repr(key)}'
                    key = transform(key)
                    cls.__unimported[key] = value

        def __contains__(cls, item):
            item = transform(item)
            return (item in cls.__unimported)

        def __getitem__(cls, item):
            item = transform(item)
            if item not in cls.__imported:
                if item not in cls.__unimported:
                    raise KeyError(f'invalid key: {repr(item)}, must be one of: {sorted(cls.__unimported.keys())}')
                cls.__imported[item] = cls.__unimported[item]._do_import_()
            return cls.__imported[item]

        def __getattr__(cls, item):
            item = transform(item)
            if item not in cls.__unimported:
                raise AttributeError(f'invalid attribute: {repr(item)}, must be one of: {sorted(cls.__unimported.keys())}')
            return cls[item]

        def __iter__(self):
            yield from (transform(item) for item in self.__unimported.keys())

    return _LazyImportMeta


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
