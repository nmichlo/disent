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
from typing import List

import numpy as np
# import ray
import ray
from tqdm import tqdm

from disent.util.iters import chunked
from disent.util.iters import iter_chunks
from research.ruck import EaModule
from research.ruck import Member
from research.ruck import PopulationHint
from research.ruck._history import HallOfFame
from research.ruck._history import Logbook
from research.ruck._history import StatsGroup


log = logging.getLogger(__name__)


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
# Evaluate Helper                                                           #
# ========================================================================= #


def _eval_sequential(population: PopulationHint, eval_fn):
    return [eval_fn(member.value) for member in population]


_evaluate_ray = ray.remote(_eval_sequential)


def _eval_multiproc(population: PopulationHint, eval_fn):
    member_batches = iter_chunks(population, chunk_size=_get_batch_size(len(population)))
    score_batches = ray.get([_evaluate_ray.remote(members, eval_fn=eval_fn) for members in member_batches])
    return [score for score_batch in score_batches for score in score_batch]


# ========================================================================= #
# Evaluate Invalid                                                          #
# ========================================================================= #


def _evaluate_invalid(population: PopulationHint, eval_fn, multiproc: bool = False):
    unevaluated = [member for member in population if not member.is_evaluated]
    # get scores
    if multiproc:
        scores = _eval_multiproc(unevaluated, eval_fn)
    else:
        scores = _eval_sequential(unevaluated, eval_fn)
    # set values
    for member, score in zip(unevaluated, scores):
        member.fitness = score
    # return number of evaluations
    return len(unevaluated)


# ========================================================================= #
# Functional Trainer                                                        #
# ========================================================================= #

def yield_population_steps(module: EaModule, multiproc: bool = True):
    # TODO: this should be using ray.put and ray.get instead of transferring data around! About 3x slower because of this
    # TODO: this should be using ray.put and ray.get instead of transferring data around! About 3x slower because of this
    # TODO: this should be using ray.put and ray.get instead of transferring data around! About 3x slower because of this

    # 1. create population
    population = module.gen_starting_population()
    population_size = len(population)
    population = _check_population(population, required_size=population_size)

    # 2. evaluate population
    evaluations = _evaluate_invalid(population, eval_fn=module.evaluate_member, multiproc=multiproc)

    # yield initial population
    yield 0, population, evaluations, population

    # training loop
    for i in range(1, module.num_generations+1):
        # 1. generate offspring
        offspring = module.generate_offspring(population)
        # 2. evaluate
        evaluations = _evaluate_invalid(offspring, eval_fn=module.evaluate_member, multiproc=multiproc)
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
        multiproc: bool = True,
        progress: bool = True,
        history_n_best: int = 5,
    ):
        self._multiproc = multiproc
        self._progress = progress
        self._history_n_best = history_n_best
        assert self._history_n_best > 0

    def fit(self, module: EaModule):
        assert isinstance(module, EaModule)
        # history trackers
        logbook, halloffame = self._create_default_trackers(module)
        # progress bar and training loop
        with tqdm(total=module.num_generations+1, desc='generation', disable=not self._progress, ncols=120) as p:
            for gen, offspring, evals, population in yield_population_steps(module, multiproc=self._multiproc):
                # update statistics with new population
                halloffame.update(offspring)
                stats = logbook.record(population, gen=gen, evals=evals)
                # update progress bar
                p.update()
                p.set_postfix({k: stats[k] for k in module.get_progress_stats()})
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
