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

import heapq
from random import random


class HallOfFame():

    def __init__(self, n: int = 1):
        self._n = n
        self._heap = []

    def update(self, population: 'Population'):
        best = sorted(population, reverse=True, key=lambda member: member.fitness)
        for member in best:
            heapq.heappush(self._heap, member)
            if len(self._heap) > self._n:
                heapq.heappop(self._heap)

def ea_simple(
    population,
    toolbox,
    cxpb,
    mutpb,
    ngen,
    stats=None,
    halloffame=None,
):

    def update(individuals):
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in individuals if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(individuals)

    update(population)

    # Begin the generational process
    for gen in range(1, ngen + 1):
        # Select the next generation individuals
        offspring = toolbox.select(population, len(population))
        # Vary the pool of individuals
        offspring = varAnd(offspring, toolbox, cxpb, mutpb)
        # recompute fitness & add to hall of fame
        update(offspring)
        # Replace the current population by the offspring
        population[:] = offspring

    return population #, logbook


class MemberIsDirtyError(Exception):
    pass

class MemberIsNotDirtyError(Exception):
    pass


class Member(object):

    def __init__(self, value, fitness=None):
        self.value = value
        self._fitness = fitness

    def copy(self):
        return Member(deepcopy(self.value))

    @property
    def fitness(self):
        if self.is_dirty:
            raise MemberIsDirtyError('The member is dirty, the fitness has not been computed.')
        return self._fitness

    def set_fitness(self, value):
        assert value is not None
        if not self.is_dirty:
            raise MemberIsDirtyError('The members fitness value cannot be set, it has not been marked as dirty!')
        self._fitness = value

    @property
    def is_dirty(self) -> bool:
        return (self._fitness is None)

    def set_dirty(self):
        if self.is_dirty:
            raise MemberIsDirtyError('The member is already marked as dirty!')
        self._fitness = None

    def __lt__(self, other: 'Member'):
        # make this work with heapq
        return self.fitness < other.fitness





class Population(object):

    def __init__(self, members: Iterable[Member]):
        self._members = list(members)

    def __iter__(self):
        yield from self._members

    def copy(self):
        return Population(member.copy() for member in self._members)

    def __len__(self):
        return len(self._members)

    def __getitem__(self, idx: int):
        return self._members[idx]


class DiversifySequence(object):

    def __init__(self, mutators: Sequence[Callable[[Population], None]]):
        self._mutators = list(mutators)

    def diversify(self, population: Population):
        offspring = population.copy()
        for mutator in self._mutators:
            mutator(offspring)
        return offspring


class Mutator(object):

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, offspring: Population):
        for member in offspring:
            if random.random() < self.p:
                self._in_place_mutate(member)

    def _in_place_mutate(self, member):
        raise NotImplementedError()


# class MutatorMutate(Mutator):
#
#     def _in_place_mutate(self, member):
#         member.value ^=




def default_vary(population: Population, cxpb, mutpb):
        # copy all the individuals
        offspring = population.copy()

        # Apply crossover and mutation on the offspring
        for i in range(1, len(offspring), 2):
            if random.random() < cxpb:
                offspring[i - 1], offspring[i] = toolbox.mate(offspring[i - 1], offspring[i])
                del offspring[i - 1].fitness.values, offspring[i].fitness.values

        for i in range(len(offspring)):
            if random.random() < mutpb:
                offspring[i], = toolbox.mutate(offspring[i])
                del offspring[i].fitness.values

        return offspring
