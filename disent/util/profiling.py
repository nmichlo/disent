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
import time
from math import log10
from contextlib import ContextDecorator


log = logging.getLogger(__name__)


# ========================================================================= #
# Memory Usage                                                              #
# ========================================================================= #


def get_memory_usage(pretty: bool = False):
    import os
    import psutil
    process = psutil.Process(os.getpid())
    num_bytes = process.memory_info().rss  # in bytes
    # format the bytes
    if pretty:
        from disent.util.strings.fmt import bytes_to_human
        return bytes_to_human(num_bytes)
    else:
        return num_bytes


# ========================================================================= #
# Context Manager Timer                                                     #
# ========================================================================= #


class Timer(ContextDecorator):

    """
    Timer class, can be used with a with statement to
    measure the execution time of a block of code!

    Examples:

        1. get the runtime
        ```
        with Timer() as t:
            time.sleep(1)
        print(t.pretty)
        ```

        2. automatic print
        ```
        with Timer(name="example") as t:
            time.sleep(1)
        ```

        3. reuse timer to measure multiple instances
        ```
        t = Timer()
        for i in range(100):
            with t:
                time.sleep(0.95)
            if t.elapsed > 3:
                break
        print(t)
        ```
    """

    def __init__(self, name: str = None, log_level: int = logging.INFO):
        self._start_time: int = None
        self._end_time: int = None
        self._total_time = 0
        self.name = name
        self._log_level = log_level

    def __enter__(self):
        self._start_time = time.time_ns()
        return self

    def __exit__(self, *args, **kwargs):
        self._end_time = time.time_ns()
        # add elapsed time to total time, and reset the timer!
        self._total_time += (self._end_time - self._start_time)
        self._start_time = None
        self._end_time = None
        # print results
        if self.name:
            if self._log_level is None:
                print(f'{self.name}: {self.pretty}')
            else:
                log.log(self._log_level, f'{self.name}: {self.pretty}')

    def restart(self):
        assert self._start_time is not None, 'timer must have been started before we can restart it'
        assert self._end_time is None, 'timer cannot be restarted if it is finished'
        self._start_time = time.time_ns()

    @property
    def elapsed_ns(self) -> int:
        if self._start_time is not None:
            # running
            return self._total_time + (time.time_ns() - self._start_time)
        # finished running
        return self._total_time

    @property
    def elapsed_ms(self) -> float:
        return self.elapsed_ns / 1_000_000

    @property
    def elapsed(self) -> float:
        return self.elapsed_ns / 1_000_000_000

    @property
    def pretty(self) -> str:
        return Timer.prettify_time(self.elapsed_ns)

    def __int__(self): return self.elapsed_ns
    def __float__(self): return self.elapsed
    def __str__(self): return self.pretty
    def __repr__(self): return self.pretty

    @staticmethod
    def prettify_time(ns: int) -> str:
        if ns == 0:
            return 'N/A'
        elif ns < 0:
            return 'NaN'
        # get power of 1000
        pow = min(3, int(log10(ns) // 3))
        time = ns / 1000**pow
        # get pretty string!
        if pow < 3 or time < 60:
            # less than 1 minute
            name = ['ns', 'µs', 'ms', 's'][pow]
            return f'{time:.3f}{name}'
        else:
            # 1 or more minutes
            s = int(time)
            d, s = divmod(s, 86400)
            h, s = divmod(s, 3600)
            m, s = divmod(s, 60)
            if d > 0:   return f'{d}d:{h}h:{m}m'
            elif h > 0: return f'{h}h:{m}m:{s}s'
            else:       return f'{m}m:{s}s'


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
