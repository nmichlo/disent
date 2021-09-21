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
from typing import Any
from typing import Any
from typing import Any
from typing import Dict
from typing import List
from typing import List

from research.ruck._history import StatsGroup
from research.ruck.util._args import HParamsMixin


# ========================================================================= #
# Members                                                                   #
# ========================================================================= #


class MemberIsNotEvaluatedError(Exception):
    pass


class MemberAlreadyEvaluatedError(Exception):
    pass


class Member(object):

    def __init__(self, value: Any):
        self._value = value
        self._fitness = None

    @property
    def value(self) -> Any:
        return self._value

    @property
    def fitness(self):
        if not self.is_evaluated:
            raise MemberIsNotEvaluatedError('The member has not been evaluated, the fitness has not yet been set.')
        return self._fitness

    @fitness.setter
    def fitness(self, value):
        if self.is_evaluated:
            raise MemberAlreadyEvaluatedError('The member has already been evaluated, the fitness can only ever be set once. Create a new member instead!')
        self._fitness = value

    @property
    def is_evaluated(self) -> bool:
        return (self._fitness is not None)


# ========================================================================= #
# Population                                                                #
# ========================================================================= #


PopulationHint  = List[Member]


# ========================================================================= #
# Module                                                                    #
# ========================================================================= #


class EaModule(HParamsMixin):

    # HELPER

    def get_stating_population(self) -> PopulationHint:
        start_values = self.get_starting_population_values()
        assert len(start_values) > 0
        return [Member(m) for m in start_values]

    # OVERRIDE

    def get_stats_groups(self) -> Dict[str, StatsGroup]:
        return {}

    @property
    def num_generations(self) -> int:
        raise NotImplementedError

    def get_starting_population_values(self) -> List[Any]:
        raise NotImplementedError

    def generate_offspring(self, population: PopulationHint) -> PopulationHint:
        raise NotImplementedError

    def select_population(self, population: PopulationHint, offspring: PopulationHint) -> PopulationHint:
        raise NotImplementedError

    def evaluate_member(self, value: Any) -> float:
        raise NotImplementedError


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
