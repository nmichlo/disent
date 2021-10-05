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


from disent.registry import REGISTRY


# ========================================================================= #
# TESTS                                                                     #
# ========================================================================= #


COUNTS = {
    'DATASET': 6,
    'SAMPLER': 8,
    'FRAMEWORK': 10,
    'RECON_LOSS': 6,
    'LATENT_DIST': 2,
    'OPTIMIZER': 30,
    'METRIC': 5,
    'SCHEDULE': 5,
    'MODEL': 8,
}


COUNTS = {             # pragma: delete-on-release
    'DATASET': 9,      # pragma: delete-on-release
    'SAMPLER': 8,      # pragma: delete-on-release
    'FRAMEWORK': 25,   # pragma: delete-on-release
    'RECON_LOSS': 8,   # pragma: delete-on-release
    'LATENT_DIST': 2,  # pragma: delete-on-release
    'OPTIMIZER': 30,   # pragma: delete-on-release
    'METRIC': 7,       # pragma: delete-on-release
    'SCHEDULE': 5,     # pragma: delete-on-release
    'MODEL': 8,        # pragma: delete-on-release
}                      # pragma: delete-on-release


def test_registry_loading():
    # load everything and check the counts
    total = 0
    for registry in REGISTRY:
        count = 0
        for name in REGISTRY[registry]:
            loaded = REGISTRY[registry][name]
            count += 1
            total += 1
        assert COUNTS[registry] == count, f'invalid count for: {registry}'
    assert total == sum(COUNTS.values()), f'invalid total'


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
