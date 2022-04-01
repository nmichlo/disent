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
import os
import os.path

import pytest

import experiment.run as experiment_run
from experiment.util.hydra_main import hydra_main
from tests.util import temp_environ
from tests.util import temp_sys_args


# ========================================================================= #
# TESTS                                                                     #
# ========================================================================= #


EXAMPLE_CFG_DIR = os.path.abspath(os.path.join(__file__, '../..', 'docs/examples/extend_experiment/config'))


@pytest.mark.parametrize(('env', 'args'), [
    # test the standard configs
    (dict(), ['run_action=skip']),
    (dict(), ['run_action=prepare_data']),
    (dict(), ['run_action=train']),
    # test the configs with the research components
    # -- we need to modify the search path
    # -- we need to register all the components
    (dict(DISENT_CONFIGS_PREPEND=EXAMPLE_CFG_DIR), ['run_action=train', 'dataset=E--xyblocks',                    'metrics=test', 'framework=E--si-betavae', 'schedule=adanegtvae_up_all']),
    (dict(DISENT_CONFIGS_PREPEND=EXAMPLE_CFG_DIR), ['run_action=train', 'dataset=E--mask-dthr-pseudorandom.yaml', 'metrics=none', 'framework=adavae_os',     'schedule=beta_cyclic']),
])
def test_experiment_run(env, args):
    # show full errors in hydra
    os.environ['HYDRA_FULL_ERROR'] = '1'

    # temporarily set the environment and the arguments
    with temp_environ(env), temp_sys_args([experiment_run.__file__, *args]):
        # run the hydra experiment
        # 1. sets the default search path to `experiment/config`
        # 2. add to the search path with the `DISENT_CONFIGS_PREPEND` and `DISENT_CONFIGS_APPEND` environment variables
        # 3. enable the ${exit:<msg>} and various other resolvers for omegaconf/hydra
        hydra_main(
            callback=experiment_run.run_action,
            config_name='config_test',
            log_level=logging.DEBUG,
        )



# ========================================================================= #
# END                                                                       #
# ========================================================================= #
