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
import warnings
from timeit import timeit

import matplotlib.pyplot as plt
import numba
import numpy as np

from disent.util.seeds import seed


@numba.njit
def _dist_change(dist_matrix, i: int, j: int, k: int, l: int):
    return dist_matrix[i, k] + dist_matrix[j, l] - dist_matrix[i, j] - dist_matrix[k, l]


@numba.njit
def two_opt_fast(route, dist_matrix):
    """
    This is not actually the correct two opt
    algorithm, but it generates decent results.
    """
    route = np.copy(route)
    # repeat until there is no improvement
    improved = True
    while improved:
        improved = False
        for i in range(1, len(route) - 2):
            for j in range(i + 2, len(route)):
                if _dist_change(dist_matrix, route[i-1], route[i], route[j-1], route[j]) < 0:
                    route[i:j] = route[i:j][::-1]
                    improved = True
    return route


@numba.njit
def two_opt(route, dist_matrix):
    route = np.copy(route)
    # repeat until there is no improvement
    improved = True
    while improved:
        improved = False
        for i in range(1, len(route) - 2):
            for j in range(i + 2, len(route)):
                if _dist_change(dist_matrix, route[i-1], route[i], route[j-1], route[j]) < 0:
                    route[i:j] = route[i:j][::-1]
                    improved = True
                    break
            if improved:
                break
    return route


def traveling_salesman(points, route=None, two_opt_fn=two_opt):
    points = np.array(points)
    assert points.ndim == 2
    # check unique
    unique_points = np.unique(points, axis=0)
    if len(unique_points) < len(points):
        warnings.warn('duplicate points found, adding small random offsets to points!')
        points = points + np.random.randn(*points.shape) * 1e-10 * np.max(points)
    # get distances
    dist_mat = np.linalg.norm(points[:, None, :] - points[None, :, :], ord=2, axis=-1)
    # get route
    if route is None:
        route = np.arange(len(points))
    # done
    return two_opt_fn(route, dist_mat)


def path_length(points, route=None, closed=False):
    # get defaults
    if route is None:
        route = np.arange(len(points))
    # checks
    assert points.ndim == 2
    assert route.ndim == 1
    # handle case
    if closed:
        idxs_from, idxs_to = route, np.roll(route, -1)
    else:
        idxs_from, idxs_to = route[:-1], route[1:]
    # get dists
    return np.sum(np.linalg.norm(points[idxs_to] - points[idxs_from], ord=2, axis=-1))


def random_route(length):
    route = np.arange(length)
    np.random.shuffle(route)
    return route


if __name__ == '__main__':

    # generate random points
    seed(42)
    points = np.random.rand(128, 2)

    dist_matrix = np.linalg.norm(points[:, None, :] - points[None, :, :], ord=2, axis=-1)
    rand_route = random_route(len(points))

    # warmup
    route_fast = two_opt_fast(rand_route, dist_matrix)
    route_slow = two_opt(rand_route, dist_matrix)

    # time tests
    print('TIMES')
    print(timeit(lambda: two_opt_fast(rand_route, dist_matrix), number=25) / 25)
    print(timeit(lambda:      two_opt(rand_route, dist_matrix), number=25) / 25)

    print('\nPATH LENGTHS')
    print(path_length(points, route_fast))
    print(path_length(points, route_slow))

    # plot route
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(5, 10))
    ax1.set_title('fast')
    ax0.scatter(*points.T)
    ax0.plot(*points[route_fast].T)
    ax1.set_title('slow')
    ax1.scatter(*points.T)
    ax1.plot(*points[route_slow].T)
    # display
    fig.tight_layout()
    plt.show()
