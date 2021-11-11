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


from disent.registry import REGISTRIES


# ========================================================================= #
# TESTS                                                                     #
# ========================================================================= #


COUNTS = {
    'DATASETS': 6,
    'SAMPLERS': 8,
    'FRAMEWORKS': 10,
    'RECON_LOSSES': 6,
    'LATENT_DISTS': 2,
    'OPTIMIZERS': 30,
    'METRICS': 5,
    'SCHEDULES': 5,
    'MODELS': 8,
}




def test_registry_loading():
    # load everything and check the counts
    total = 0
    for registry in REGISTRIES:
        count = 0
        for name in REGISTRIES[registry]:
            loaded = REGISTRIES[registry][name]
            count += 1
            total += 1
        assert COUNTS[registry] == count, f'invalid count for: {registry}'
    assert total == sum(COUNTS.values()), f'invalid total'


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
