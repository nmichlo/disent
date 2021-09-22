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
from typing import Callable

from disent.util.math.random import random_choice_prng
from research.ruck import PopulationHint


# ========================================================================= #
# Select                                                                    #
# ========================================================================= #


SelectFnHint = Callable[[PopulationHint, int], PopulationHint]


def select_best(population: PopulationHint, num: int) -> PopulationHint:
    return sorted(population, key=lambda m: m.fitness, reverse=True)[:num]


def select_worst(population: PopulationHint, num: int) -> PopulationHint:
    return sorted(population, key=lambda m: m.fitness, reverse=False)[:num]


def select_random(population: PopulationHint, num: int) -> PopulationHint:
    return random_choice_prng(population, size=num, replace=False)


def select_tournament(population: PopulationHint, num: int, k: int = 3) -> PopulationHint:
    key = lambda m: m.fitness
    return [
        max(random.sample(population, k=k), key=key)
        for _ in range(num)
    ]


# ========================================================================= #
# Selection                                                                 #
# ========================================================================= #
