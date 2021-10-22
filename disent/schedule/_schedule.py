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

from typing import Union
from typing import Optional

import numpy as np

from disent.schedule.lerp import cyclical_anneal
from disent.schedule.lerp import lerp
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


class NoopSchedule(Schedule):

    def compute_value(self, step: int, value):
        # does absolutely nothing!
        return value


# ========================================================================= #
# Value Schedules                                                           #
# ========================================================================= #


def _common_lerp_value(ratio, value, r_start: float, r_end: float):
    # scale the value such that it (which should be in the range [0, 1]) between [r_min, r_max]
    return lerp(
        ratio,
        start_val=value * r_start,
        end_val=value * r_end,
    )


def _completion_ratio(step: int, start_step: int, end_step: int):
    ratio = lerp_step(
        step=(step - start_step),
        max_step=(end_step - start_step),
        start_val=0.0,
        end_val=1.0,
    )
    return ratio


class LinearSchedule(Schedule):
    """
    A simple lerp schedule based on some start and end ratio.

    Multiples the value based on the step by some
    computed value that is in the range [0, 1]
    """

    def __init__(
        self,
        start_step: int,
        end_step: int,
        r_start: float = 0.0,
        r_end: float = 1.0,
    ):
        """
        :param min_step: The step at which the schedule starts and the value unfreezes
        :param max_step: The step at which the schedule finishes and the value freezes
        :param r_start: The ratio of the original value that the schedule will start with
        :param r_end: The ratio of the original value that the schedule will end with
        """
        assert start_step >= 0
        assert end_step > 0
        assert start_step < end_step
        self.start_step = start_step
        self.end_step = end_step
        self.r_start = r_start
        self.r_end = r_end

    def compute_value(self, step: int, value):
        # completion ratio in range [0, 1]. If step < start_step return 0, if step > end_step return 1
        ratio = _completion_ratio(step=step, start_step=self.start_step, end_step=self.end_step)
        # lerp the value into the range [r_start * value, r_end * value] according to the ratio
        return _common_lerp_value(ratio, value=value, r_start=self.r_start, r_end=self.r_end)


class CyclicSchedule(Schedule):
    """
    Cyclical schedule based on:
    https://arxiv.org/abs/1903.10145

    Multiples the value based on the step by some
    computed value that is in the range [0, 1]
    """

    # TODO: maybe move this api into cyclical_anneal
    def __init__(
        self,
        period: int,
        start_step: Optional[int] = None,
        repeats: Optional[int] = None,
        r_start: float = 0.0,
        r_end: float = 1.0,
        end_mode: str = 'end',
        mode: str = 'linear',
        p_low: float = 0.0,
        p_high: float = 0.0,
    ):
        """
        :param period: The number of steps it takes for the schedule to repeat
        :param start_step: The step when the schedule will start, if this is None
                           then no modification to the step is performed. Equivalent to
                           `start_step=0` if no negative step values are passed.
        :param repeats: The number of repeats of this schedule. The end_step of the schedule will
                        be `start_step + repeats*period`. If `repeats is None` or `repeats < 0` then the
                        schedule never ends.
        :param r_start: The ratio of the original value that the schedule will start with
        :param r_end: The ratio of the original value that the schedule will end with
        :param end_mode: what of value the schedule should take after finishing [start, end]
        :param mode: The kind of function use to interpolate between the start and finish [linear, sigmoid, cosine]
        :param p_low: The portion of the period at the start that is spent at the minimum value
        :param p_high: The portion of the period that at the end is spent at the maximum value
        """
        # checks
        if (repeats is not None) and (repeats < 0):
            repeats = None
        # set values
        self.period = period
        self.repeats = repeats
        self.start_step = start_step
        self.end_value = {'start': 'low', 'end': 'high'}[end_mode]
        self.mode = mode
        # scale values
        self.r_start = r_start
        self.r_end = r_end
        # portions of low and high -- low + high <= 1.0 -- low + slope + high == 1.0
        self.p_low = p_low
        self.p_high = p_high
        # checks
        assert (start_step is None) or (start_step >= 0)

    def compute_value(self, step: int, value):
        # shift the start
        if self.start_step is not None:
            step = max(0, step - self.start_step)
        # outputs value in range [0, 1]
        ratio = cyclical_anneal(
            step=step,
            period=self.period,
            low_ratio=self.p_low,
            high_ratio=self.p_high,
            repeats=self.repeats,
            start_low=True,
            end_value=self.end_value,
            mode=self.mode
        )
        return _common_lerp_value(ratio, value=value, r_start=self.r_start, r_end=self.r_end)


class SingleSchedule(CyclicSchedule):
    """
    A single repeat version of CyclicSchedule that automatically
    chooses if its going from high to low or low to high based on the start_value.

    Multiples the value based on the step by some
    computed value that is in the range [0, 1]
    """

    def __init__(
        self,
        start_step: int,
        end_step: int,
        r_start: float = 0.0,
        r_end: float = 1.0,
        mode: str = 'linear',
    ):
        """
        :param start_step: The step when the schedule will start
        :param end_step: The step when the schedule will finish
        :param r_start: The ratio of the original value that the schedule will start with
        :param r_end: The ratio of the original value that the schedule will end with
        :param mode: The kind of function use to interpolate between the start and finish [linear, sigmoid, cosine]
        """
        super().__init__(
            period=(end_step - start_step),
            start_step=start_step,
            repeats=1,
            r_start=r_start,
            r_end=r_end,
            end_mode='end',
            mode=mode,
            p_low=0.0,  # adjust the start and end steps instead
            p_high=0.0, # adjust the start and end steps instead
        )


class CosineWaveSchedule(Schedule):
    """
    A simple cosine wave schedule based on some start and end ratio.
    -- note this starts at zero by default

    Multiples the value based on the step by some
    computed value that is in the range [0, 1]
    """

    # TODO: add r_start
    # TODO: add start_step
    # TODO: add repeats
    def __init__(
        self,
        period: int,
        r_start: float = 0.0,
        r_end: float = 1.0,
    ):
        """
        :param period: The number of steps it takes for the schedule to repeat
        :param r_start: The ratio of the original value that the schedule will start with
        :param r_end: The ratio of the original value that the schedule will end with
        """
        assert period > 0
        self.period = period
        self.r_start = r_start
        self.r_end = r_end

    def compute_value(self, step: int, value):
        cosine_ratio = 0.5 * (1 + np.cos(step * (2 * np.pi / self.period) + np.pi))
        # lerp the value into the range [r_start * value, r_end * value] according to the ratio
        return _common_lerp_value(cosine_ratio, value=value, r_start=self.r_start, r_end=self.r_end)


# ========================================================================= #
# Clip Schedules                                                            #
# ========================================================================= #


class ClipSchedule(Schedule):
    """
    This schedule shifts the step, or clips the value
    """

    def __init__(
        self,
        schedule: Schedule,
        min_step: Optional[int] = None,
        max_step: Optional[int] = None,
        shift_step: Union[bool, int] = True,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
    ):
        """
        :param schedule:
        :param min_step: The minimum step passed to the sub-schedule
        :param max_step: The maximum step passed to the sub-schedule
        :param shift_step: (if bool) Shift all the step values passed to the sub-schedule,
                           at or before min_step the sub-schedule will get `0`, at or after
                           max_step the sub-schedule will get `max_step-shift_step`
                           (if int) Add the given value to the step passed to the sub-schedule
        :param min_value: The minimum value returned from the sub-schedule
        :param max_value: The maximum value returned from the sub-schedule
        """
        assert isinstance(schedule, Schedule)
        self.schedule = schedule
        # step settings
        self.min_step = min_step
        self.max_step = max_step
        # shift step
        self.shift_step = shift_step
        if isinstance(shift_step, bool):
            if self.min_step is not None:
                self.shift_step = -self.min_step
        # value settings
        self.min_value = min_value
        self.max_value = max_value

    def compute_value(self, step: int, value):
        if self.max_step is not None: step = np.minimum(self.max_step, step)
        if self.min_step is not None: step = np.maximum(self.min_step, step)
        if self.shift_step is not None: step += self.shift_step
        result = self.schedule(step, value)
        if self.max_value is not None: result = np.minimum(self.max_value, result)
        if self.min_value is not None: result = np.maximum(self.min_value, result)
        return result


# ========================================================================= #
# END                                                                       #
# ========================================================================= #


if __name__ == '__main__':

    def plot_schedules(*schedules: Schedule, total: int = 1000, value=1):
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, len(schedules), figsize=(3*len(schedules), 3))
        for ax, s in zip(axs, schedules):
            xs = list(range(total))
            ys = [s(x, value) for x in xs]
            ax.set_xlim([-0.05 * total, total + 0.05 * total])
            ax.set_ylim([-0.05 * value, value + 0.05 * value])
            ax.plot(xs, ys)
            ax.set_title(f'{s.__class__.__name__}')
        fig.tight_layout()
        plt.show()

    def main():

        # these should be equivalent
        plot_schedules(
            LinearSchedule(start_step=100, end_step=900, r_start=0.1, r_end=0.8),
            LinearSchedule(start_step=200, end_step=800, r_start=0.9, r_end=0.2),
            SingleSchedule(start_step=100, end_step=900, r_start=0.1, r_end=0.8),
            SingleSchedule(start_step=200, end_step=800, r_start=0.9, r_end=0.2),
            # LinearSchedule(min_step=900, max_step=100, r_start=0.1, r_end=0.8), # INVALID
            # LinearSchedule(min_step=900, max_step=100, r_start=0.8, r_end=0.1), # INVALID
        )

        plot_schedules(
            CyclicSchedule(period=300, start_step=0,   repeats=None, r_start=0.1, r_end=0.8, end_mode='end',   mode='linear',  p_low=0.00, p_high=0.00),
            CyclicSchedule(period=300, start_step=0,   repeats=2,    r_start=0.9, r_end=0.2, end_mode='start', mode='linear',  p_low=0.25, p_high=0.00),
            CyclicSchedule(period=300, start_step=200, repeats=2,    r_start=0.9, r_end=0.2, end_mode='end', mode='linear',    p_low=0.00, p_high=0.25),
            CyclicSchedule(period=300, start_step=0,   repeats=2,    r_start=0.1, r_end=0.8, end_mode='end',   mode='cosine',  p_low=0.25, p_high=0.25),
            CyclicSchedule(period=300, start_step=250, repeats=None, r_start=0.1, r_end=0.8, end_mode='end',   mode='sigmoid', p_low=0.00, p_high=0.00),
        )

        plot_schedules(
            SingleSchedule(start_step=0,   end_step=800, r_start=0.1, r_end=0.8, mode='linear'),
            SingleSchedule(start_step=100, end_step=800, r_start=0.8, r_end=0.1, mode='linear'),
            SingleSchedule(start_step=100, end_step=800, r_start=0.8, r_end=0.1, mode='linear'),
        )

    main()
