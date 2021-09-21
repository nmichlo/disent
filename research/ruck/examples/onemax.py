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
import numpy as np
import ray

from disent.util.profiling import Timer
from research.ruck import *


class OneMaxProblem(EaProblem):

    def __init__(
        self,
        generations: int = 40,
        population_size: int = 128,
        member_size: int = 10_000,
        p_mate: float = 0.5,
        p_mutate: float = 0.5,
    ):
        super().__init__()
        self.save_hyperparameters()

    def get_num_generations(self) -> int:
        return self.hparams.generations

    def get_starting_population_values(self) -> PopulationHint:
        return [
            np.random.random(self.hparams.member_size) < 0.5
            for _ in range(self.hparams.population_size)
        ]

    def generate_offspring(self, population: PopulationHint) -> PopulationHint:
        # SEE: R.factory_ea_alg -- TODO: make it easier to swap!
        return R.apply_mate_and_mutate(
            population=R.select_tournament(population, len(population)),  # tools.selNSGA2
            mate=R.mate_crossover_1d,
            mutate=R.mutate_flip_bit_types,
            p_mate=self.hparams.p_mate,
            p_mutate=self.hparams.p_mutate,
        )

    def select_population(self, population: PopulationHint, offspring: PopulationHint) -> PopulationHint:
        return offspring

    def evaluate_member(self, value: np.ndarray) -> float:
        return value.mean()


if __name__ == '__main__':
    # about 10x faster than the onemax (0.25s vs 2.6s)
    # numpy version given for deap
    # -- https://github.com/DEAP/deap/blob/master/examples/ga/onemax_numpy.py

    logging.basicConfig(level=logging.INFO)
    ray.init(num_cpus=16)

    with Timer('ruck:onemax'):
        problem = OneMaxProblem(generations=4000, population_size=128, member_size=10_000)
        run_ea(problem)
