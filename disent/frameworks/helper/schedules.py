
import math
import numpy as np
import pytest


# ========================================================================= #
# Original Cyclical Annealing Schedules                                     #
# - https://arxiv.org/abs/1903.10145                                        #
# - https://github.com/haofuml/cyclical_annealing                           #
# !!! we keep these around as tests to bootstrap our newer functions        #
# ========================================================================= #


def _ORIG_frange_cycle_linear(v_min, v_max, total_steps, repeats=4, ratio=0.5):
    L = np.ones(total_steps)
    period = total_steps / repeats
    v_delta = (v_max - v_min) / (period * ratio)
    # linear schedule
    for c in range(repeats):
        v, i = v_min, 0
        while v <= v_max and (int(i + c * period) < total_steps):
            L[int(i + c * period)] = v
            v += v_delta
            i += 1
    return L


def _ORIG_frange_cycle_sigmoid(v_min, v_max, total_steps, repeats=4, ratio=0.5):
    L = np.ones(total_steps)
    period = total_steps / repeats
    v_delta = (v_max - v_min) / (period * ratio)  # step is in [0,1]
    # transform into [-6, 6] for plots: v*12.-6.
    for c in range(repeats):
        v, i = v_min, 0
        while v <= v_max:
            L[int(i + c * period)] = 1.0 / (1.0 + np.exp(- (v * 12. - 6.)))
            v += v_delta
            i += 1
    return L


#  function  = 1 − cos(a), where a scans from 0 to pi/2
def _ORIG_frange_cycle_cosine(v_min, v_max, total_steps, repeats=4, ratio=0.5):
    L = np.ones(total_steps)
    period = total_steps / repeats
    v_delta = (v_max - v_min) / (period * ratio)  # step is in [0,1]
    # transform into [0, pi] for plots:
    for c in range(repeats):
        v, i = v_min, 0
        while v <= v_max:
            L[int(i + c * period)] = 0.5 - .5 * math.cos(v * math.pi)
            v += v_delta
            i += 1
    return L


def _ORIG_frange(v_min, v_max, v_delta, total_steps):
    L = np.ones(total_steps)
    v, i = v_min, 0
    while v <= v_max:
        L[i] = v
        v += v_delta
        i += 1
    return L


# ========================================================================= #
# Cyclical Annealing Schedules - Activations                                #
# - same api as the original versions but cleaned up!                       #
# !!! we keep these around to keep the same API as the original functions   #
# ========================================================================= #


def activate_linear(v):
    return v


def activate_sigmoid(v):
    return 1.0 / (1.0 + np.exp(- (v * 12. - 6.)))


def activate_cosine(v):
    return 0.5 - .5 * np.cos(v * np.pi)


# ========================================================================= #
# Cleaned Cyclical Annealing Schedules                                      #
# - same api as the original versions but cleaned up!                       #
# !!! we keep these around to keep the same API as the original functions   #
# ========================================================================= #


def _CLEANED_frange_cycle(v_min, v_max, total_steps, repeats=4, ratio=0.5, activation=activate_linear, v_delta=None):
    L = np.ones(total_steps)
    period = total_steps / repeats
    # check one or the other
    if ratio is not None:
        assert v_delta is None
    if v_delta is not None:
        assert ratio is None
    # handle last case
    if v_delta is None:
        v_delta = (v_max - v_min) / (period * ratio)
    # linear schedule
    for c in range(repeats):
        v, i = v_min, 0
        while v <= v_max and (int(i + c * period) < total_steps):
            L[int(i + c * period)] = activation(v)
            v += v_delta
            i += 1
    return L


def _CLEANED_frange_cycle_linear(v_min, v_max, total_steps, repeats=4, ratio=0.5):
    return _CLEANED_frange_cycle(v_min, v_max, total_steps, repeats=repeats, ratio=ratio, activation=activate_linear)


def _CLEANED_frange_cycle_sigmoid(v_min, v_max, total_steps, repeats=4, ratio=0.5):
    return _CLEANED_frange_cycle(v_min, v_max, total_steps, repeats=repeats, ratio=ratio, activation=activate_sigmoid)


def _CLEANED_frange_cycle_cosine(v_min, v_max, total_steps, repeats=4, ratio=0.5):
    return _CLEANED_frange_cycle(v_min, v_max, total_steps, repeats=repeats, ratio=ratio, activation=activate_cosine)


def _CLEANED_frange(v_min, v_max, v_delta, total_steps):
    return _CLEANED_frange_cycle(v_min, v_max, total_steps, repeats=1, ratio=None, activation=activate_linear, v_delta=v_delta)


@pytest.mark.parametrize(['cleaned_fn', 'orig_fn'], [
    (_CLEANED_frange_cycle_linear,  _ORIG_frange_cycle_linear),
    (_CLEANED_frange_cycle_sigmoid, _ORIG_frange_cycle_sigmoid),
    (_CLEANED_frange_cycle_cosine,  _ORIG_frange_cycle_cosine),
])
@pytest.mark.parametrize(('v_min', 'v_max', 'total_steps', 'repeats', 'ratio'), [
    (0.2,  0.5, 99,  4, 0.6),
    (0.2,  0.5, 13,  3, 0.5),
    (0.0,  0.5, 101, 4, 0.5),
    (0.0,  1.0, 103, 4, 0.1),
    (0.1,  1.0, 77,  4, 0.5),
    (-0.1, 1.0, 77,  4, 0.8),
    (0.1,  1.1, 77,  4, 0.5),
])
def test_cleaned_cycle_equal(cleaned_fn, orig_fn, v_min, v_max, total_steps, repeats, ratio):
    orig = orig_fn(v_min, v_max, total_steps, repeats, ratio)
    cleaned = cleaned_fn(v_min, v_max, total_steps, repeats, ratio)
    assert np.allclose(orig, cleaned)



@pytest.mark.parametrize(['cleaned_fn', 'orig_fn'], [
    (_CLEANED_frange,  _ORIG_frange),
])
@pytest.mark.parametrize(('v_min', 'v_max', 'v_delta', 'total_steps'), [
    (0.2,  0.5, 0.6, 99),
    (0.2,  0.5, 0.5, 13),
    (0.0,  0.5, 0.5, 101),
    (0.0,  1.0, 0.1, 103),
    (0.1,  1.0, 0.5, 77),
    (-0.1, 1.0, 0.8, 77),
    (0.1,  1.1, 0.5, 77),
])
def test_cleaned_cycle_equal(orig_fn, cleaned_fn, v_min, v_max, v_delta, total_steps):
    orig = orig_fn(v_min, v_max, v_delta, total_steps)
    cleaned = cleaned_fn(v_min, v_max, v_delta, total_steps)
    assert np.allclose(orig, cleaned)


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
