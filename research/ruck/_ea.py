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
import logging
import random
import time
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import scipy
from scipy.stats import mode
from tqdm import tqdm

from disent.util.iters import chunked
from disent.util.math.random import random_choice_prng


log = logging.getLogger(__name__)


# ========================================================================= #
# Types                                                                     #
# ========================================================================= #


PopulationHint  = List['Member']
EvalFnHint      = Callable[[Any], float]
OffspringFnHint = Callable[[PopulationHint], PopulationHint]
NextPopFnHint   = Callable[[PopulationHint, PopulationHint], PopulationHint]

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

    def record(self, population: PopulationHint, **external_values):
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

    def update(self, population: PopulationHint):
        for member in population:
            item = HallOfFameItem(fitness=member.fitness, member=member)
            if len(self._heap) < self._n_best:
                heapq.heappush(self._heap, item)
            else:
                heapq.heappushpop(self._heap, item)

# ========================================================================= #
# Members                                                                   #
# ========================================================================= #


class BaseEA(object):

    def __init__(
        self,
        evaluate_member: EvalFnHint,
        generate_offspring: OffspringFnHint,
        select_population: NextPopFnHint,
        stats_groups: Optional[Dict[str, StatsGroup]] = None
    ):
        super().__init__()
        self._evaluate_member = evaluate_member
        self._generate_offspring = generate_offspring
        self._select_population = select_population
        self._stats_groups = dict(stats_groups) if (stats_groups is not None) else {}

    def _check_population(self, population: PopulationHint, required_size: int):
        assert len(population) == required_size, 'population size is invalid'
        assert all(isinstance(member, Member) for member in population), 'items in population are not members'

    def _create_default_logbook(self) -> Logbook:
        return Logbook(
            'gen', 'evals',
            fit=StatsGroup(value_fn=lambda pop: [m.fitness for m in pop], min=np.min, max=np.max, mean=np.mean),
            **self._stats_groups,
        )

    def fit(
        self,
        population_values: List[Any],
        generations: int = 100,
        progress: bool = True,
    ) -> (PopulationHint, Logbook, HallOfFame):
        # initialize
        logbook = self._create_default_logbook()
        halloffame = HallOfFame(n_best=5, maximize=True)
        # get population
        population_size = len(population_values)
        population = [Member(value) for value in population_values]
        self._check_population(population, population_size)
        # evaluate population
        evaluations = self.evaluate_invalid(population)
        halloffame.update(population)
        # update statistics
        logbook.record(population, gen=0, evals=evaluations)
        # run
        with tqdm(range(1, generations+1), desc='generation', disable=not progress) as progress:
            for i in progress:
                offspring = self._generate_offspring(population)
                # evaluate population
                evaluations = self.evaluate_invalid(offspring)
                halloffame.update(offspring)
                # get population
                population = self._select_population(population, offspring)
                self._check_population(population, population_size)
                # update statistics with new population
                logbook.record(population, gen=i, evals=evaluations)
        # done!
        return population, logbook, halloffame

    def evaluate_invalid(self, population: PopulationHint) -> int:
        unevaluated = chunked([member for member in population if not member.is_evaluated], chunk_size=16)
        fitnesses = [self._evaluate_batch(batch) for batch in unevaluated]
        # update fitness values
        evaluations = 0
        for fitness_batch, member_batch in zip(fitnesses, unevaluated):
            for fitness, member in zip(fitness_batch, member_batch):
                member.fitness = fitness
                evaluations += 1
        # update hall of fame
        return evaluations

    def _evaluate_batch(self, batch: PopulationHint) -> List[float]:
        return [self._evaluate_member(member.value) for member in batch]


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

    # def copy(self):
    #     member = Member(deepcopy(self._value))
    #     member._fitness = deepcopy(self._fitness)
    #     return member

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


MateFnHint = Callable[[Any, Any], Tuple[Any, Any]]
MutateFnHint = Callable[[Any], Any]


def crossover_and_mutate(
    population: PopulationHint,
    p_mate: float,
    mate: MateFnHint,
    p_mutate: float,
    mutate: MutateFnHint,
) -> PopulationHint:
    """
    Apply crossover AND mutation.
    Modified individuals are independent of the population,
    requiring their fitness to be re-evaluated.

    NB: Mate & Mutate should return copies of the received values.

    ** Modified from DEAP **
    """
    offspring = list(population)

    # EXTRA
    random.shuffle(offspring)

    # Apply crossover
    for i in range(1, len(offspring), 2):
        if random.random() < p_mate:
            value0, value1 = mate(offspring[i - 1].value, offspring[i].value)
            offspring[i - 1], offspring[i] = Member(value0), Member(value1)

    # Apply Mutation
    for i in range(len(offspring)):
        if random.random() < p_mutate:
            value = mutate(offspring[i].value)
            offspring[i] = Member(value)

    return offspring


def crossover_or_mutate_or_reproduce(
    population: PopulationHint,
    num_offspring: int,  # lambda_
    p_mate: float,
    mate: MateFnHint,
    p_mutate: float,
    mutate: MutateFnHint,
) -> PopulationHint:
    """
    Apply crossover OR mutation OR reproduction
    Modified individuals are independent of the population,
    requiring their fitness to be re-evaluated.

    NB: Mate & Mutate should return copies of the received values.

    ** Modified from DEAP **
    """
    assert (p_mate + p_mutate) <= 1.0, 'The sum of the crossover and mutation probabilities must be smaller or equal to 1.0.'

    offspring = []
    for _ in range(num_offspring):
        op_choice = random.random()
        if op_choice < p_mate:
            # Apply crossover
            ind1, ind2 = random.sample(population, 2)
            value, _ = mate(ind1.value, ind2.value)
            offspring.append(Member(value))
        elif op_choice < p_mate + p_mutate:
            # Apply mutation
            ind = random.choice(population)
            value = mutate(ind.value)
            offspring.append(Member(value))
        else:
            # Apply reproduction
            offspring.append(random.choice(population))

    return offspring


# ========================================================================= #
# Mate                                                                      #
# ========================================================================= #


def mate_crossover(a, b):
    assert a.ndim == 1
    assert a.shape == b.shape
    i, j = np.random.randint(0, len(a), size=2)
    i, j = min(i, j), max(i, j)
    new_a = np.concatenate([a[:i], b[i:j], a[j:]], axis=0)
    new_b = np.concatenate([b[:i], a[i:j], b[j:]], axis=0)
    return new_a, new_b


# ========================================================================= #
# Mutate                                                                    #
# ========================================================================= #


def mutate_flip_bits(a, p=0.05):
    return a ^ (np.random.random(a.shape) < p)


def mutate_flip_bit_types(a, p=0.05):
    if np.random.random() < 0.5:
        # flip set bits
        return a ^ ((np.random.random(a.shape) < p) & a)
    else:
        # flip unset bits
        return a ^ ((np.random.random(a.shape) < p) & ~a)



# ========================================================================= #
# Selection                                                                 #
# ========================================================================= #


def select_best(population: PopulationHint, num: int):
    return sorted(population, key=lambda m: m.fitness, reverse=True)[:num]


def select_worst(population: PopulationHint, num: int):
    return sorted(population, key=lambda m: m.fitness, reverse=False)[:num]


def select_random(population: PopulationHint, num: int):
    return random_choice_prng(population, size=num, replace=False)


def select_tournament(population: PopulationHint, num: int, k: int = 3):
    key = lambda m: m.fitness
    return [
        max(random.sample(population, k=k), key=key)
        for _ in range(num)
    ]


# ========================================================================= #
# Test EA                                                                   #
# ========================================================================= #


def _get_gen_and_select_fns(
    mate_fn,
    mutate_fn,
    select_fn,
    mode: str = 'simple',
    p_mate: float = 0.5,
    p_mutate: float = 0.5,
    offspring_num: int = 128,
    population_num: int = 128,
):
    if mode == 'simple':
        def _generate(population):          return crossover_and_mutate(population=select_fn(population, len(population)), p_mate=p_mate, mate=mate_fn, p_mutate=p_mutate, mutate=mutate_fn)
        def _select(population, offspring): return offspring
    elif mode == 'mu_plus_lambda':
        def _generate(population):          return crossover_or_mutate_or_reproduce(population, num_offspring=offspring_num, p_mate=p_mate, mate=mate_fn, p_mutate=p_mutate, mutate=mutate_fn)
        def _select(population, offspring): return select_fn(population + offspring, population_num)
    elif mode == 'mu_comma_lambda':
        def _generate(population):          return crossover_or_mutate_or_reproduce(population, num_offspring=offspring_num, p_mate=p_mate, mate=mate_fn, p_mutate=p_mutate, mutate=mutate_fn)
        def _select(population, offspring): return select_fn(offspring, population_num)
    else:
        raise KeyError(f'invalid mode: {repr(mode)}')
    return _generate, _select


def onemax():
    # mate   cxTwoPointNumpy
    # mutate tools.mutFlipBit
    # select tools.selNSGA2  # deap.tools.selTournament

    POPULATION_SIZE = 128

    _gen, _sel = _get_gen_and_select_fns(
        mate_fn=mate_crossover,
        mutate_fn=lambda a: mutate_flip_bit_types(a, p=0.05),
        select_fn=select_tournament,
        mode='mu_plus_lambda',
        offspring_num=POPULATION_SIZE,
        population_num=POPULATION_SIZE,
    )

    ea = BaseEA(
        evaluate_member=lambda value: value.mean(),
        generate_offspring=_gen,
        select_population=_sel,
    )

    population = [(np.random.random(10_000) < 0.5) for i in range(POPULATION_SIZE)]

    population, logbook, halloffame = ea.fit(population, generations=40, progress=True)

    for entry in logbook.history:
        print(entry)


if __name__ == '__main__':
    # about 10x faster than the onemax
    # numpy version given for deap
    # -- https://github.com/DEAP/deap/blob/master/examples/ga/onemax_numpy.py

    t = time.time()
    onemax()
    print(time.time() - t)



# ========================================================================= #
# Apply Helpers                                                             #
# ========================================================================= #


# def get_vary_or(mate_fn, mutate_fn, p_mate=0.5, p_mutate=0.2):
#     return ApplyOne(
#         VaryMate(mate_fn=mate_fn),
#         VaryMutate(mutate_fn=mutate_fn),
#         VaryRandom(),
#         probabilities=[p_mate, p_mutate, 1.0-(p_mate+p_mutate)]
#     )
#
#
# def get_vary_and(mate_fn, mutate_fn, p_mate=0.5, p_mutate=0.2):
#     return ApplyAny(
#         VaryMate(mate_fn=mate_fn),
#         VaryMutate(mutate_fn=mutate_fn),
#         probabilities=[p_mate, p_mutate]
#     )
#
#
# class Apply(object):
#     def apply(self, population):
#         raise NotImplementedError
#
#
# class ApplyOne(Apply):
#     def __init__(self, *applicators: Apply, probabilities=Sequence[float]):
#         assert len(applicators) == len(probabilities)
#         assert all(p >= 0 for p in probabilities)
#         assert sum(probabilities) == 1.0
#         self._applicators = applicators
#         self._probabilities = probabilities
#
#     def apply(self, population) -> Member:
#         applicator = np.random.choice(self._applicators, p=self._probabilities)
#         return applicator.apply(population)
#
#
# class ApplyAny(Apply):
#     def __init__(self, *applicators: Apply, probabilities=Sequence[float]):
#         assert len(applicators) == len(probabilities)
#         assert all(p >= 0 for p in probabilities)
#         self._applicators = applicators
#         self._probabilities = probabilities
#
#     def apply(self, *args, **kwargs):
#         do_applies = np.random.random(len(self._probabilities)) < self._probabilities
#         for applicator, do_apply in zip(self._applicators, do_applies):
#             if do_apply:
#                 applicator(*args, **kwargs)
#
#
#
# class VaryRandom(Vary):
#     def vary(self, population: PopulationHint, idx: int) -> Member:
#         return np.random.choice(population)
#
#
# class VaryMate(Vary):
#     def __init__(self, mate_fn: Callable[[Any, Any], Any]):
#         self._mate_fn = mate_fn
#
#     def vary(self, population: PopulationHint, idx: int) -> Member:
#         member0, member1 = random_choice_prng(population, 2, replace=False)
#         value, _ = self._mate_fn(member0.value, member1.value)
#         return Member(value=value)
#
#
# class VaryMutate(Vary):
#     def __init__(self, mutate_fn: Callable[[Any], Any]):
#         self._mutate_fn = mutate_fn
#
#     def vary(self, population: PopulationHint, idx: int) -> Member:
#         member = np.random.choice(population)
#         value = deepcopy(member.value)
#         return Member(value=self._mutate_fn(value))


# ========================================================================= #
# Population                                                                #
# ========================================================================= #


# class Population(object):
#
#     def __init__(self, members: Iterable[Member]):
#         self._members = [
#             (member.copy() if isinstance(member, Member) else Member(member))
#             for member in members
#         ]
#
#     def __iter__(self):
#         yield from self._members
#
#     def copy(self):
#         return Population(member.copy() for member in self._members)
#
#     def __len__(self):
#         return len(self._members)
#
#     def __getitem__(self, idx: int):
#         return self._members[idx]


# ========================================================================= #
# Basic Genetic Algorithm                                                   #
# ========================================================================= #


# class DiversifySequence(object):
#
#     def __init__(self, mutators: Sequence[Callable[[Population], None]]):
#         self._mutators = list(mutators)
#
#     def diversify(self, population: Population):
#         offspring = population.copy()
#         for mutator in self._mutators:
#             mutator(offspring)
#         return offspring


# class Mutator(object):
#
#     def __init__(self, p=0.5):
#         self.p = p
#
#     def __call__(self, offspring: Population):
#         for member in offspring:
#             if random.random() < self.p:
#                 self._in_place_mutate(member)
#
#     def _in_place_mutate(self, member):
#         raise NotImplementedError()


# class MutatorMutate(Mutator):
#
#     def _in_place_mutate(self, member):
#         member.value ^=
