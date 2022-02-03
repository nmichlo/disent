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

COUNTS_RESEARCH = {     # pragma: delete-on-release
    'DATASETS': 10,     # pragma: delete-on-release
    'SAMPLERS': 8,      # pragma: delete-on-release
    'FRAMEWORKS': 25,   # pragma: delete-on-release
    'RECON_LOSSES': 6,  # pragma: delete-on-release
    'LATENT_DISTS': 2,  # pragma: delete-on-release
    'OPTIMIZERS': 30,   # pragma: delete-on-release
    'METRICS': 9,       # pragma: delete-on-release
    'SCHEDULES': 5,     # pragma: delete-on-release
    'MODELS': 8,        # pragma: delete-on-release
}                       # pragma: delete-on-release


def _check_all_registries(counts):
    # load everything and check the counts
    total = 0
    for registry in REGISTRIES:
        count = 0
        for name in REGISTRIES[registry]:
            loaded = REGISTRIES[registry][name]
            count += 1
            total += 1
        assert count == counts[registry], f'invalid count for: {registry}'
    assert total == sum(counts.values()), f'invalid total'


def test_registry_loading():
    _check_all_registries(COUNTS)
    # check the research components               # pragma: delete-on-release
    from research.code import register_to_disent  # pragma: delete-on-release
    register_to_disent()                          # pragma: delete-on-release
    _check_all_registries(COUNTS_RESEARCH)        # pragma: delete-on-release


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
