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


import logging
import os
from argparse import Namespace
from typing import Any
from typing import Dict
from typing import List
from typing import NoReturn
from typing import Optional
from typing import Sequence
from typing import Union

import numpy as np
from tqdm import tqdm

from disent.util.iters import chunked
from research.ruck._history import HallOfFame
from research.ruck._history import Logbook
from research.ruck._history import StatsGroup


log = logging.getLogger(__name__)


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
# Problem                                                                   #
# ========================================================================= #


class EaProblem(object):

    def __init__(self):
        self.__hparams = None

    def save_hyperparameters(self, ignore: Optional[Union[str, Sequence[str]]] = None):
        import inspect
        import warnings
        # get ignored values
        if ignore is None:            ignored = set()
        elif isinstance(ignore, str): ignored = {ignore}
        else:                         ignored = set(ignore)
        assert all(str.isidentifier(k) for k in ignored)
        # get function params & signature
        locals = inspect.currentframe().f_back.f_locals
        params = inspect.signature(self.__class__.__init__)
        # get values
        (self_param, *params) = params.parameters.items()
        # check that self is correct & skip it
        assert self_param[0] == 'self'
        assert locals[self_param[0]] is self
        # get other values
        values = {}
        for k, v in params:
            if k in ignored: continue
            if v.kind == v.VAR_KEYWORD: warnings.warn('variable keywords argument saved, consider converting to explicit arguments.')
            if v.kind == v.VAR_POSITIONAL: warnings.warn('variable positional argument saved, consider converting to explicit named arguments.')
            values[k] = locals[k]
        # done!
        self.__hparams = Namespace(**values)

    @property
    def hparams(self):
        return self.__hparams

    def get_stats_groups(self) -> Optional[Dict[str, StatsGroup]]:
        return None

    def get_starting_population_values(self) -> PopulationHint:
        raise NotImplementedError

    def generate_offspring(self, population: PopulationHint) -> PopulationHint:
        raise NotImplementedError

    def select_population(self, population: PopulationHint, offspring: PopulationHint) -> PopulationHint:
        raise NotImplementedError

    def evaluate_member(self, value: Any) -> float:
        raise NotImplementedError


# ========================================================================= #
# Run                                                                       #
# ========================================================================= #


def _check_population(population: PopulationHint, required_size: int) -> NoReturn:
    assert len(population) > 0, 'population must not be empty'
    assert len(population) == required_size, 'population size is invalid'
    assert all(isinstance(member, Member) for member in population), 'items in population are not members'


def _create_default_logbook(problem: EaProblem) -> Logbook:
    logbook = Logbook('gen', 'evals')
    logbook.register_stats_group('fit', StatsGroup(lambda pop: [m.fitness for m in pop], min=np.min, max=np.max, mean=np.mean))
    # register problem stats
    stat_groups = problem.get_stats_groups()
    for k, v in (stat_groups if (stat_groups is not None) else {}).items():
        logbook.register_stats_group(k, v)
    # done!
    return logbook


def _evaluate_invalid(population: PopulationHint, eval_fn) -> int:
    unevaluated = chunked([member for member in population if not member.is_evaluated], chunk_size=16)
    fitnesses = [_evaluate_batch(batch, eval_fn) for batch in unevaluated]
    # update fitness values
    evaluations = 0
    for fitness_batch, member_batch in zip(fitnesses, unevaluated):
        for fitness, member in zip(fitness_batch, member_batch):
            member.fitness = fitness
            evaluations += 1
    # update hall of fame
    return evaluations


def _evaluate_batch(batch: PopulationHint, eval_fn) -> List[float]:
    return [eval_fn(member.value) for member in batch]


def run_ea(
    problem: EaProblem,
    generations: int = 100,
    num_workers: int = min(os.cpu_count(), 16),
    progress: bool = True,
):
    # ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~ #
    # INIT                              #
    # ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~ #

    assert isinstance(problem, EaProblem)
    assert generations > 0
    assert num_workers > 0

    # instantiate helpers
    halloffame = HallOfFame(n_best=5, maximize=True)
    logbook = _create_default_logbook(problem=problem)

    # get & check population
    population = [Member(v) for v in problem.get_starting_population_values()]
    population_size = len(population)
    _check_population(population, required_size=population_size)

    # ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~ #
    # RUN                               #
    # ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~ #

    # evaluate population
    evaluations = _evaluate_invalid(population, eval_fn=problem.evaluate_member)
    halloffame.update(population)
    # update statistics with new population
    logbook.record(population, gen=0, evals=evaluations)

    # run
    with tqdm(range(1, generations+1), desc='generation', disable=not progress) as p:
        for i in p:
            offspring = problem.generate_offspring(population)
            # evaluate population
            evaluations = _evaluate_invalid(offspring, eval_fn=problem.evaluate_member)
            halloffame.update(offspring)
            # get population
            population = problem.select_population(population, offspring)
            _check_population(population, required_size=population_size)
            # update statistics with new population
            logbook.record(population, gen=i, evals=evaluations)

    # done!
    return population, logbook, halloffame


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
