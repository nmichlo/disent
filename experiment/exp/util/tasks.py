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
import warnings
from collections import namedtuple
from typing import Any
from typing import Dict
from typing import Optional
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import Union


# ========================================================================= #
# Task Builders                                                             #
# ========================================================================= #

_INPUT_ = object()
_COMPUTED_ = object()


Task = namedtuple('Task', ('name', 'names_input', 'names_required', 'names_optional', 'fn'))

_NamesType = Union[str, Sequence[str]]
_InputsType = Optional[Dict[str, Any]]


# I know this is overkill... and hides functionality where used...
# but I was having fun... and so I don't regret it!
class Tasks(object):
    """
    Graph that can dynamically compute what is required or not based on function signatures, and function names.

    Functions are Tasks. Tasks can have:
        - Input Values (specified on invocation,    <param>=_INPUT_)
        - computed values (computed by other tasks, <param>=_COMPUTED_)
        - optional values (default parameters,      <param>=<value>)
    """

    def __init__(self, task_fns: Sequence[callable], debug=False):
        self._tasks, (self._deps_inputs, self._deps_required, self._deps_optional) = self._build_tasks(task_fns)
        # typed deps
        self._dep_types = {
            **{k: 'input' for k in self._deps_inputs},
            **{k: 'required' for k in self._deps_required},
            **{k: 'optional' for k in self._deps_optional},
        }
        self._debug = debug

    @classmethod
    def _fn_to_task(cls, fn) -> Task:
        # get name
        name = fn.__name__
        if name.startswith('_task__'):
            name = name[len('_task__'):]
        if not name:
            raise ValueError(f'task function has empty name: {repr(fn.__name__)}')
        # get parameters
        inputs, required, optional = [], [], []
        for arg_name, param in inspect.signature(fn).parameters.items():
            if param.default is param.empty:
                warnings.warn(f'non-explicit notation used for RESULT, convert to keyword argument: "{arg_name}=RESULT"')
                required.append(arg_name)
            elif param.default is _COMPUTED_:
                required.append(arg_name)
            elif param.default is _INPUT_:
                inputs.append(arg_name)
            else:
                optional.append(arg_name)
        # return task
        return Task(name=name, names_input=inputs, names_required=required, names_optional=optional, fn=fn)

    @classmethod
    def _build_tasks(cls, task_fns: Sequence[callable]) -> Tuple[Dict[str, Task], Tuple[Set[str], Set[str], Set[str]]]:
        # create all tasks
        tasks = {}
        for fn in task_fns:
            task = cls._fn_to_task(fn)
            if task.name in tasks:
                raise RuntimeError(f'task with name: {repr(task.name)} has already been added!')
            tasks[task.name] = task
        # collect deps
        deps_inputs   = {dep for task in tasks.values() for dep in task.names_input}
        deps_required = {dep for task in tasks.values() for dep in task.names_required}
        deps_optional = {opt for task in tasks.values() for opt in task.names_optional}
        # check deps
        if deps_inputs   & deps_optional: raise RuntimeError(f'An input dependency has the same name as an optional dependency: {sorted(deps_inputs & deps_optional)}')
        if deps_inputs   & deps_required: raise RuntimeError(f'An input dependency has the same name as a required dependency: {sorted(deps_inputs & deps_required)}')
        if deps_optional & deps_required: raise RuntimeError(f'An optional dependency has the same name as a required dependency: {sorted(deps_optional & deps_required)}')
        # return everything
        return tasks, (deps_inputs, deps_required, deps_optional)

    def compute(self, tasks: _NamesType, inputs: _InputsType = None, options: _InputsType = None, result_overrides: _InputsType = None):
        # normalise
        _names = self._normalise_task_names(tasks)
        _inputs = self._normalise_inputs(inputs)
        _options = self._normalise_options(options)
        results = self._normalise_result_overrides(result_overrides)

        # compute results
        def _do_task(name: str):
            if name not in results:
                task = self._tasks[name]
                results[name] = task.fn(
                    **{input_name:  _inputs[input_name]   for input_name  in task.names_input},
                    **{parent_name: _do_task(parent_name) for parent_name in task.names_required},
                    **{option_name: _options[option_name] for option_name in task.names_optional if option_name in _options},
                )
                if self._debug:
                    print(f'computed: {name}')
            return results[name]

        # return values
        if isinstance(tasks, str):
            return _do_task(tasks)
        else:
            return tuple(_do_task(name) for name in _names)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
    # Check & Normalise Inputs                                              #
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

    def _normalise_task_names(self, names: _NamesType) -> Sequence[str]:
        # normalise input tasks
        names = [names] if isinstance(names, str) else list(names)
        # check input tasks
        if not names:
            raise KeyError(f'No task names specified: {names}, valid task names are: {sorted(self._tasks.keys())}')
        if set(names) - self._tasks.keys():
            raise KeyError(f'Specified task names are invalid: {sorted(set(names) - self._tasks.keys())}, valid task names are: {sorted(self._tasks.keys())}')
        # return everything
        return names

    def _normalise_inputs(self, inputs: _InputsType):
        # check missing
        if inputs is None:
            inputs = {}
        if self._deps_inputs - inputs.keys():
            raise KeyError(f'Input dependencies were not specified in the input list: {sorted(self._deps_inputs - inputs.keys())}')
        if inputs.keys() - self._deps_inputs:
            raise KeyError(f'Specified inputs: {sorted(inputs.keys() - self._deps_inputs)} are not valid input dependencies: {sorted(self._deps_inputs)}')
        # return everything
        return inputs

    def _normalise_options(self, options: _InputsType):
        if options is None:
            options = {}
        if options.keys() - self._deps_optional:
            raise KeyError(f'Specified overridden optional inputs are invalid: {sorted(options.keys() - self._deps_optional)}, valid optional input names are: {sorted(self._deps_optional)}')
        return options

    def _normalise_result_overrides(self, result_overrides: _InputsType):
        if result_overrides is None:
            result_overrides = {}
        if result_overrides.keys() - self._tasks.keys():
            raise KeyError(f'Specified overridden task names are invalid: {sorted(result_overrides.keys() - self._tasks.keys())}, valid task names are: {sorted(self._tasks.keys())}')
        return result_overrides


# ========================================================================= #
# END                                                                       #
# ========================================================================= #

