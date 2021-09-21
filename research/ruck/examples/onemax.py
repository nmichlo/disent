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
import time
import warnings
from argparse import Namespace
from copy import deepcopy
from typing import Optional
from typing import Sequence
from typing import Union

import numpy as np
from pytorch_lightning import LightningModule

from research.ruck._ea import EaProblem
from research.ruck._ea import PopulationHint
from research.ruck._ea import run_ea
from research.ruck.functional import mate_crossover_1d
from research.ruck.functional import mutate_flip_bit_types
from research.ruck.functional import select_tournament
from research.ruck.functional._helper import crossover_and_mutate


class OneMaxProblem(EaProblem):

    def __init__(
        self,
        population_size: int = 128,
        member_size: int = 10_000,
        p_mate: float = 0.5,
        p_mutate: float = 0.5,
    ):
        super().__init__()
        self.save_hyperparameters()

    def get_starting_population_values(self) -> PopulationHint:
        return [
            np.random.random(10_000) < 0.5
            for _ in range(self.hparams.population_size)
        ]

    def generate_offspring(self, population: PopulationHint) -> PopulationHint:
        return crossover_and_mutate(
            population=select_tournament(population, len(population)),  # tools.selNSGA2
            mate=mate_crossover_1d,
            mutate=mutate_flip_bit_types,
            p_mate=self.hparams.p_mate,
            p_mutate=self.hparams.p_mutate,
        )

    def select_population(self, population: PopulationHint, offspring: PopulationHint) -> PopulationHint:
        return offspring

    def evaluate_member(self, value: np.ndarray) -> float:
        return value.mean()


if __name__ == '__main__':
    # about 10x faster than the onemax
    # numpy version given for deap
    # -- https://github.com/DEAP/deap/blob/master/examples/ga/onemax_numpy.py

    t = time.time()
    problem = OneMaxProblem(population_size=128, member_size=10_000)
    run_ea(problem, generations=40)
    print(time.time() - t)
