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
import heapq
from typing import Any
from typing import Callable
from typing import Dict
from typing import List


# ========================================================================= #
# Type Hints                                                                #
# ========================================================================= #


ValueFnHint     = Callable[[Any], Any]
StatFnHint      = Callable[[Any], Any]


# ========================================================================= #
# Logbook                                                                   #
# ========================================================================= #


class StatsGroup(object):

    def __init__(self, value_fn: ValueFnHint = None, **stats_fns: StatFnHint):
        assert all(str.isidentifier(key) for key in stats_fns.keys())
        assert stats_fns
        self._value_fn = value_fn
        self._stats_fns = stats_fns

    @property
    def keys(self) -> List[str]:
        return list(self._stats_fns.keys())

    def compute(self, value: Any) -> Dict[str, Any]:
        if self._value_fn is not None:
            value = self._value_fn(value)
        return {
            key: stat_fn(value)
            for key, stat_fn in self._stats_fns.items()
        }


class Logbook(object):

    def __init__(self, *external_keys: str, **stats_groups: StatsGroup):
        self._all_ordered_keys = []
        self._external_keys = []
        self._stats_groups = {}
        self._history = []
        # register values
        for k in external_keys:
            self.register_external_stat(k)
        for k, v in stats_groups.items():
            self.register_stats_group(k, v)

    def _assert_key_valid(self, name: str):
        if not str.isidentifier(name):
            raise ValueError(f'stat name is not a valid identifier: {repr(name)}')
        return name

    def _assert_key_available(self, name: str):
        if name in self._external_keys:
            raise ValueError(f'external stat already named: {repr(name)}')
        if name in self._stats_groups:
            raise ValueError(f'stat group already named: {repr(name)}')
        return name

    def register_external_stat(self, name: str):
        self._assert_key_available(self._assert_key_available(name))
        # add stat
        self._external_keys.append(name)
        self._all_ordered_keys.append(name)
        return self

    def register_stats_group(self, name: str, stats_group: StatsGroup):
        self._assert_key_available(self._assert_key_available(name))
        assert isinstance(stats_group, StatsGroup)
        assert stats_group not in self._stats_groups.values()
        # add stat group
        self._stats_groups[name] = stats_group
        self._all_ordered_keys.extend(f'{name}:{key}' for key in stats_group.keys)
        return self

    def record(self, population: 'PopulationHint', **external_values):
        # extra stats
        if set(external_values.keys()) != set(self._external_keys):
            raise KeyError(f'required external_values: {sorted(self._external_keys)}, got: {sorted(external_values.keys())}')
        # external values
        stats = dict(external_values)
        # generate stats
        for name, stat_group in self._stats_groups.items():
            for key, value in stat_group.compute(population).items():
                stats[f'{name}:{key}'] = value
        # order stats
        assert set(stats.keys()) == set(self._all_ordered_keys)
        record = {k: stats[k] for k in self._all_ordered_keys}
        # record and return stats
        self._history.append(record)
        return dict(record)

    @property
    def history(self) -> List[Dict[str, Any]]:
        return list(self._history)


# ========================================================================= #
# HallOfFame                                                                #
# ========================================================================= #


@dataclasses.dataclass(order=True)
class HallOfFameItem:
    fitness: float
    member: Any = dataclasses.field(compare=False)


class HallOfFame(object):

    def __init__(self, n_best: int = 5, maximize: bool = True):
        self._maximize = maximize
        self._n_best = n_best
        self._heap = []

    def update(self, population: 'PopulationHint'):
        for member in population:
            item = HallOfFameItem(fitness=member.fitness, member=member)
            if len(self._heap) < self._n_best:
                heapq.heappush(self._heap, item)
            else:
                heapq.heappushpop(self._heap, item)


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
