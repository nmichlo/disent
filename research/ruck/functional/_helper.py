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

import random

from research.ruck._ea import Member
from research.ruck._ea import PopulationHint
from research.ruck.functional import MateFnHint
from research.ruck.functional import MutateFnHint


# ========================================================================= #
# Crossover & Mutate Helpers                                                #
# ========================================================================= #


def apply_mate_and_mutate(
    population: PopulationHint,
    mate: MateFnHint,
    mutate: MutateFnHint,
    p_mate: float,
    p_mutate: float,
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


def apply_mate_or_mutate_or_reproduce(
    population: PopulationHint,
    mate: MateFnHint,
    mutate: MutateFnHint,
    p_mate: float,
    p_mutate: float,
    num_offspring: int,  # lambda_
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
# Gen & Select                                                              #
# ========================================================================= #


def factory_ea_alg(
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
        def _generate(population):          return apply_mate_and_mutate(population=select_fn(population, len(population)), p_mate=p_mate, mate=mate_fn, p_mutate=p_mutate, mutate=mutate_fn)
        def _select(population, offspring): return offspring
    elif mode == 'mu_plus_lambda':
        def _generate(population):          return apply_mate_or_mutate_or_reproduce(population, num_offspring=offspring_num, p_mate=p_mate, mate=mate_fn, p_mutate=p_mutate, mutate=mutate_fn)
        def _select(population, offspring): return select_fn(population + offspring, population_num)
    elif mode == 'mu_comma_lambda':
        def _generate(population):          return apply_mate_or_mutate_or_reproduce(population, num_offspring=offspring_num, p_mate=p_mate, mate=mate_fn, p_mutate=p_mutate, mutate=mutate_fn)
        def _select(population, offspring): return select_fn(offspring, population_num)
    else:
        raise KeyError(f'invalid mode: {repr(mode)}')
    return _generate, _select


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
