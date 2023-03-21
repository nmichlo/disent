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

from disent.dataset.data import XYSquaresData


class XYSquaresSampler(XYSquaresData):
    def sample_1d_boxes(self, size=None):
        size = (2,) if (size is None) else ((size, 2) if isinstance(size, int) else (*size, 2))
        # sample x0, y0
        s0 = self._offset + self._spacing * np.random.randint(0, self._placements, size=size)
        # sample x1, y1
        s1 = s0 + self._square_size
        # return (x0, y0), (x1, y1)
        return s0, s1

    def sample_1d_overlap(self, size=None):
        s0, s1 = self.sample_1d_boxes(size=size)
        # compute overlap
        return np.maximum(np.min(s1, axis=-1) - np.max(s0, axis=-1), 0)

    def sample_1d_delta(self, size=None):
        s0, s1 = self.sample_1d_boxes(size=size)
        # compute differences
        l_delta = np.max(s0, axis=-1) - np.min(s0, axis=-1)
        r_delta = np.max(s1, axis=-1) - np.min(s1, axis=-1)
        # return delta
        return np.minimum(l_delta + r_delta, self._square_size * 2)


if __name__ == "__main__":
    print("\nDecreasing Spacing & Increasing Size")
    for ss, gs in [(8, 8), (9, 7), (17, 6), (25, 5), (33, 4), (41, 3), (49, 2), (57, 1)][::-1]:
        d = XYSquaresSampler(square_size=ss, grid_spacing=gs, grid_size=8, no_warnings=True)
        print(
            "ss={:2d} gs={:1d} overlap={:7.4f} delta={:7.4f}".format(
                ss, gs, d.sample_1d_overlap(size=1_000_000).mean(), d.sample_1d_delta(size=1_000_000).mean()
            )
        )

    print("\nDecreasing Spacing")
    for i in range(8):
        ss, gs = 8, 8 - i
        d = XYSquaresSampler(square_size=ss, grid_spacing=gs, grid_size=8, no_warnings=True)
        print(
            "ss={:2d} gs={:1d} overlap={:7.4f} delta={:7.4f}".format(
                ss, gs, d.sample_1d_overlap(size=1_000_000).mean(), d.sample_1d_delta(size=1_000_000).mean()
            )
        )

    print("\nDecreasing Spacing & Keeping Dimension Size Constant")
    for i in range(8):
        ss, gs = 8, 8 - i
        d = XYSquaresSampler(square_size=ss, grid_spacing=gs, grid_size=None, no_warnings=True)
        print(
            "ss={:2d} gs={:1d} overlap={:7.4f} delta={:7.4f}".format(
                ss, gs, d.sample_1d_overlap(size=1_000_000).mean(), d.sample_1d_delta(size=1_000_000).mean()
            )
        )
