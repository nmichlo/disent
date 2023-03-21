#  ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~
#  MIT License
#
#  Copyright (c) 2022 Nathan Juraj Michlo
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

from disent.dataset.data import GroundTruthData


class RandomData(GroundTruthData):
    """
    Dataset that returns deterministic pseudorandom observations.
    """

    factor_sizes = (8, 16, 32, 64)
    factor_names = ("r1", "r2", "r3", "r4")
    img_shape = (64, 64, 1)

    def _get_observation(self, idx: int):
        rng = np.random.default_rng(idx + 1)
        return rng.integers(0, 256, size=self.img_shape, dtype="uint8")


if __name__ == "__main__":

    def main():
        """
        Compute the dataset statistics!

        results:
        >>> vis_mean: [0.4999966931838419]
        >>> vis_std: [0.2897895504502549]
        """
        from disent.dataset.transform import ToImgTensorF32
        from disent.dataset.util.stats import compute_data_mean_std

        data = RandomData(transform=ToImgTensorF32(size=64))
        mean, std = compute_data_mean_std(data, progress=True, num_workers=0, batch_size=256)

        print(f"vis_mean: {mean.tolist()}")
        print(f"vis_std: {std.tolist()}")

    main()
