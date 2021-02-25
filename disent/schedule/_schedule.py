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

import numpy as np

from disent.schedule.lerp import cyclical_anneal
from disent.schedule.lerp import lerp_step


# ========================================================================= #
# Schedules                                                                 #
# ========================================================================= #


class Schedule(object):

    def __call__(self, step: int, value):
        return self.compute_value(step=step, value=value)

    def compute_value(self, step: int, value):
        raise NotImplementedError


# ========================================================================= #
# Value Schedules                                                           #
# ========================================================================= #


class LinearSchedule(Schedule):
    """
    A simple lerp schedule based on some start and end ratio.

    Multiples the value based on the step by some
    computed value that is in the range [0, 1]
    """

    def __init__(self, max_step: int, r_start: float = 0.0, r_end: float = 1.0):
        assert max_step > 0
        self.max_step = max_step
        self.r_start = r_start
        self.r_end = r_end

    def compute_value(self, step: int, value):
        ratio = lerp_step(
            step=step,
            max_step=self.max_step,
            a=self.r_start,
            b=self.r_end,
        )
        return value * ratio


class CyclicSchedule(Schedule):
    """
    Cyclical schedule based on:
    https://arxiv.org/abs/1903.10145

    Multiples the value based on the step by some
    computed value that is in the range [0, 1]
    """

    # TODO: maybe move this api into cyclical_anneal
    def __init__(self, period: int, repeats: int = None, r_start=0.0, r_end=1.0, end_value='end', mode='linear'):
        self.period = period
        self.repeats = repeats
        self.end_value = {'start': 'low', 'end': 'high'}[end_value]
        self.mode = mode
        # scale values
        self.r_start = r_start
        self.r_end = r_end

    def compute_value(self, step: int, value):
        # outputs value in range [0, 1]
        ratio = cyclical_anneal(
            step=step,
            period=self.period,
            low_ratio=0.0,
            high_ratio=0.0,
            repeats=self.repeats,
            start_low=True,
            end_value=self.end_value,
            mode=self.mode
        )
        # scale value between [r_min, r_max]
        sratio = self.r_start + ratio * (self.r_end - self.r_start)
        result = value * sratio
        # compute value!
        return result


class SingleSchedule(CyclicSchedule):
    """
    A single repeat version of CyclicSchedule that automatically
    chooses if its going from high to low or low to high based on the start_value.

    Multiples the value based on the step by some
    computed value that is in the range [0, 1]
    """

    def __init__(self, max_step, r_start=0.0, r_end=1.0, mode='linear'):
        super().__init__(
            period=max_step,
            repeats=1,
            r_start=r_start,
            r_end=r_end,
            end_value='end',
            mode=mode,
        )


# ========================================================================= #
# Clip Schedules                                                            #
# ========================================================================= #


class ClipSchedule(Schedule):
    """
    This schedule shifts the step, or clips the value
    """

    def __init__(self, schedule: Schedule, min_step=None, max_step=None, min_value=None, max_value=None):
        assert isinstance(schedule, Schedule)
        self.schedule = schedule
        self.min_step = min_step
        self.max_step = max_step
        self.min_value = min_value
        self.max_value = max_value

    def compute_value(self, step: int, value):
        if self.max_step is not None: step = np.minimum(self.max_step, step)
        if self.min_step is not None: step = np.maximum(self.min_step, step) - self.min_step
        result = self.schedule(step, value)
        if self.max_value is not None: result = np.minimum(self.max_value, result)
        if self.min_value is not None: result = np.maximum(self.min_value, result)
        return result


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
