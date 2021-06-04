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

import dataclasses
import inspect
from functools import lru_cache
from typing import Any
from typing import Dict
from typing import Set
from typing import Tuple
from typing import Union


# ========================================================================= #
# Task Builders                                                             #
# - I know this is overkill... but i was having fun...                      #
# ========================================================================= #


IN = object()
TASK = object()


@dataclasses.dataclass
class Task(object):
    fn: callable
    name: str
    # parameters
    params: Tuple[str, ...]
    params_inputs: Tuple[str, ...]
    params_parents: Tuple[str, ...]
    params_optional: Tuple[str, ...]


@lru_cache()
def _task_handler_get_task(fn) -> Task:
    # get name
    name = fn.__name__
    if name.startswith('task__'):
        name = name[len('task__'):]
    elif name.startswith('_task__'):
        name = name[len('_task__'):]
    if not name:
        raise ValueError(f'task function has empty name: {repr(fn.__name__)}')
    # get parameters
    inputs, parents, optional, params = [], [], [], []
    for arg_name, param in inspect.signature(fn).parameters.items():
        if param.default is param.empty:
            raise RuntimeError(f'task {repr(name)} has non-keyword argument: {repr(arg_name)}')
        elif param.default is TASK:
            parents.append(arg_name)
        elif param.default is IN:
            inputs.append(arg_name)
        else:
            optional.append(arg_name)
        params.append(arg_name)
    # return task
    return Task(fn=fn, name=name, params=tuple(params), params_inputs=tuple(inputs), params_parents=tuple(parents), params_optional=tuple(optional))


@lru_cache()
def _task_handler_get_parents(
    task_names: Tuple[str, ...],
    task_fns: Tuple[callable, ...],
) -> Tuple[callable, ...]:
    if not task_fns:
        raise ValueError(f'No task functions were given: {task_fns}')
    # get functions that this depends on
    task_map = {task.name: task for task in (_task_handler_get_task(fn) for fn in task_fns)}
    # get all dependencies
    unprocessed, compute = set(task_names), set()
    # check they are valid
    if unprocessed - task_map.keys():
        raise KeyError(f'Specified task names do not exist: {sorted(unprocessed)}, valid task names are: {sorted(task_map.keys())}')
    # add all parents
    while unprocessed:
        name = unprocessed.pop()
        compute.add(name)
        unprocessed.update(task_map[name].params_parents)
    # done!
    task_fns_minimal = tuple(fn for fn in task_fns if _task_handler_get_task(fn).name in compute)
    compute = tuple(task_map[name].fn for name in compute)
    return compute, task_fns_minimal


@lru_cache()
def _task_handler_check_arguments(
    compute: Tuple[callable, ...],
    task_fns: Tuple[callable, ...],
    input_symbol_names: Tuple[str, ...],
    strict: bool = True,
    disable_options: bool = True,
):
    # strict mode checks all parameters even if not needed
    if strict:
        compute = task_fns
    # get parameters from functions & parents of tasks marked to compute
    inputs = {name for fn in compute for name in _task_handler_get_task(fn).params_inputs}
    parents = {name for fn in compute for name in _task_handler_get_task(fn).params_parents}
    optional = {name for fn in compute for name in _task_handler_get_task(fn).params_optional}
    # check that we have no options if they are disabled
    if disable_options:
        if optional:
            raise RuntimeError(f'Optional symbols have been disabled: {sorted(optional)}, set `disable_options=False` to skip this error.')
    # check dependencies between tasks
    if inputs   & parents:  raise RuntimeError(f'An input symbol has the same name as a parent symbol: {sorted(inputs & parents)}')
    if optional & parents:  raise RuntimeError(f'An optional symbol has the same name as a parent symbol: {sorted(optional & parents)}')
    if inputs   & optional: raise RuntimeError(f'An input symbol has the same name as an optional symbol: {sorted(inputs & optional)}')
    # check against kwargs
    input_symbol_names = set(input_symbol_names)
    if input_symbol_names & parents: raise RuntimeError(f'A given argument has the same name as a parent symbol: {sorted(input_symbol_names & parents)}')
    if inputs - input_symbol_names: raise RuntimeError(f'All the required inputs have not been passed as arguments: {sorted(inputs - input_symbol_names)}')
    if input_symbol_names - (inputs | optional): raise RuntimeError(f'Invalid arguments were found that are not input or optional symbols: {sorted(input_symbol_names - (inputs | optional))}')
    # done checks!


class TaskHandler(object):

    def __init__(
        self,
        task_names: Union[str, Tuple[str, ...]],
        task_fns: Tuple[callable, ...],
        symbols: Dict[str, Any] = None,
        strict: bool = True,
        disable_options: bool = True
    ):
        self._task_names_orig                = task_names
        self._task_names: Tuple[str, ...]    = (task_names,) if isinstance(task_names, str) else tuple(task_names)
        self._task_fns: Tuple[callable, ...] = task_fns
        # get defaults
        symbols = {} if (symbols is None) else dict(symbols)
        # get compute graph
        compute, self._task_fns_min = _task_handler_get_parents(task_names=self._task_names, task_fns=task_fns)
        _task_handler_check_arguments(compute=compute, task_fns=task_fns, input_symbol_names=tuple(sorted(symbols.keys())), strict=strict, disable_options=disable_options)
        # dispatch variables
        self._compute: Set[callable]     = set(compute)
        self._compute_all: Set[callable] = set(task_fns)
        self._symbols: Dict[str, Any]    = symbols

    def dispatch_all(self):
        for task in self._task_fns_min:
            self.dispatch(task)
        return self

    def dispatch(self, fn):
        if fn in self._compute:
            task = _task_handler_get_task(fn)
            kwargs = {name: self._symbols[name] for name in task.params if name in self._symbols}
            result = task.fn(**kwargs)
            self._symbols[task.name] = result
        elif fn not in self._compute_all:
            raise KeyError(f'tried to dispatch function that has not been registered: {_task_handler_get_task(fn).name}')
        return self

    def result(self):
        if isinstance(self._task_names_orig, str):
            return self._symbols[self._task_names_orig]
        else:
            return tuple(self._symbols[name] for name in self._task_names)

    @staticmethod
    def compute(task_names: Union[str, Tuple[str, ...]], task_fns: Tuple[callable, ...], symbols: Dict[str, Any] = None, strict: bool = True, disable_options=True) -> Tuple[Any, ...]:
        # create the compute graph
        handler = TaskHandler(
            task_names=task_names,
            task_fns=task_fns,
            symbols=symbols,
            strict=strict,
            disable_options=disable_options,
        )
        # these may or may not be evaluated!
        handler.dispatch_all()
        # return the results
        return handler.result()


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
