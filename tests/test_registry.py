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

import pytest
import disent.registry as R


# ========================================================================= #
# TESTS                                                                     #
# ========================================================================= #


COUNTS = {
    'DATASETS': 15,
    'SAMPLERS': 8,
    'FRAMEWORKS': 15,
    'RECON_LOSSES': 9,
    'LATENT_HANDLERS': 2,
    'OPTIMIZERS': 30,
    'METRICS': 9,
    'SCHEDULES': 5,
    'MODELS': 8,
    'KERNELS': 2,
}

COUNTS = {
    'DATASETS': 17,
    'SAMPLERS': 8,
    'FRAMEWORKS': 17,
    'RECON_LOSSES': 9,
    'LATENT_HANDLERS': 2,
    'OPTIMIZERS': 30,
    'METRICS': 9,
    'SCHEDULES': 5,
    'MODELS': 8,
    'KERNELS': 2,
}


@pytest.mark.parametrize('registry_key', COUNTS.keys())
def test_registry_loading(registry_key):
    from docs.examples.extend_experiment.code import register_to_disent
    register_to_disent()
    register_to_disent()  # must be able to call more than once!
    # load everything and check the counts
    count = 0
    for example in R.REGISTRIES[registry_key]:
        loaded = R.REGISTRIES[registry_key][example]
        count += 1
    assert count == COUNTS[registry_key], f'invalid count for: {registry_key}'


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
