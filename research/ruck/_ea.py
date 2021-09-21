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
from typing import Any
from typing import Dict
from typing import List

import numpy as np
import ray
from tqdm import tqdm

from disent.util.iters import chunked
from research.ruck._history import HallOfFame
from research.ruck._history import Logbook
from research.ruck._history import StatsGroup
from research.ruck.util._args import HParamsMixin


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
# Utils Trainer                                                             #
# ========================================================================= #


def _check_population(population: PopulationHint, required_size: int) -> PopulationHint:
    assert len(population) > 0, 'population must not be empty'
    assert len(population) == required_size, 'population size is invalid'
    assert all(isinstance(member, Member) for member in population), 'items in population are not members'
    return population


def _get_batch_size(total: int) -> int:
    resources = ray.available_resources()
    if 'CPU' not in resources:
        return total
    else:
        cpus = int(resources['CPU'])
        batch_size = (total + cpus - 1) // cpus
        return batch_size


# ========================================================================= #
# Functional Trainer                                                        #
# ========================================================================= #


def _evaluate_invalid(population: PopulationHint, eval_fn) -> int:
    unevaluated = [member for member in population if not member.is_evaluated]
    unevaluated = chunked(unevaluated, chunk_size=_get_batch_size(len(unevaluated)))
    # fetch values!
    fitnesses = ray.get([_evaluate_batch.remote(batch, eval_fn) for batch in unevaluated])
    # update fitness values
    evaluations = 0
    for fitness_batch, member_batch in zip(fitnesses, unevaluated):
        for fitness, member in zip(fitness_batch, member_batch):
            member.fitness = fitness
            evaluations += 1
    # update hall of fame
    return evaluations

@ray.remote
def _evaluate_batch(batch: PopulationHint, eval_fn) -> List[float]:
    return [eval_fn(member.value) for member in batch]


def yield_population_steps(module: EaModule):
    # 1. create population
    population = module.get_stating_population()
    population_size = len(population)
    population = _check_population(population, required_size=population_size)
    # 2. evaluate population
    evaluations = _evaluate_invalid(population, eval_fn=module.evaluate_member)

    # yield initial population
    yield 0, population, evaluations, population

    # training loop
    for i in range(1, module.num_generations+1):
        # 1. generate offspring
        offspring = module.generate_offspring(population)
        # 2. evaluate
        evaluations = _evaluate_invalid(offspring, eval_fn=module.evaluate_member)
        # 3. select
        population = module.select_population(population, offspring)
        population = _check_population(population, required_size=population_size)

        # yield steps
        yield i, offspring, evaluations, population


# ========================================================================= #
# Class Trainer                                                             #
# ========================================================================= #


class Trainer(object):

    def __init__(
        self,
        num_workers: int = min(os.cpu_count(), 16),
        progress: bool = True,
        history_n_best: int = 5,
    ):
        self._num_workers = num_workers
        self._progress = progress
        self._history_n_best = history_n_best
        assert self._num_workers > 0
        assert self._history_n_best > 0

    def fit(self, module: EaModule):
        assert isinstance(module, EaModule)
        # history trackers
        logbook, halloffame = self._create_default_trackers(module)
        # progress bar and training loop
        with tqdm(total=module.num_generations+1, desc='generation', disable=not self._progress, ncols=120) as p:
            for gen, offspring, evals, population in yield_population_steps(module):
                # update statistics with new population
                halloffame.update(offspring)
                stats = logbook.record(population, gen=gen, evals=evals)
                # update progress bar
                p.update()
                p.set_postfix(dict(evals=evals, fit_max=stats['fit:max']))
        # done
        return population, logbook, halloffame

    def _create_default_trackers(self, module: EaModule):
        halloffame = HallOfFame(
            n_best=self._history_n_best,
            maximize=True,
        )
        logbook = Logbook(
            'gen', 'evals',
            fit=StatsGroup(lambda pop: [m.fitness for m in pop], min=np.min, max=np.max, mean=np.mean),
            **module.get_stats_groups()
        )
        return logbook, halloffame


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
