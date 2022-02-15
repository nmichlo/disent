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

import hydra
import pytest

import experiment.run as experiment_run
from tests.util import temp_environ
from tests.util import temp_sys_args


# ========================================================================= #
# TESTS                                                                     #
# ========================================================================= #
                                                                                                                 # pragma: delete-on-release
                                                                                                                 # pragma: delete-on-release
RESEARCH_CFG_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'research/config'))  # pragma: delete-on-release


@pytest.mark.parametrize(('env', 'args'), [
    # test the standard configs
    (dict(), ['run_action=skip']),
    (dict(), ['run_action=prepare_data']),
    (dict(), ['run_action=train']),
    # test the configs with the research components  # pragma: delete-on-release
    # -- we need to modify the search path           # pragma: delete-on-release
    # -- we need to register all the components      # pragma: delete-on-release
    (dict(DISENT_CONFIGS_PREPEND=RESEARCH_CFG_DIR), ['run_action=train', 'dataset=X--xysquares', 'metrics=test', 'framework=X--adaae_os', 'schedule=adavae_down_all']),  # pragma: delete-on-release
])
def test_experiment_run(env, args):
    # show full errors in hydra
    os.environ['HYDRA_FULL_ERROR'] = '1'

    logging.basicConfig(level=logging.DEBUG)

    # Patch Hydra and OmegaConf:
    # 1. sets the default search path to `experiment/config`
    # 2. add to the search path with the `DISENT_CONFIGS_PREPEND` and `DISENT_CONFIGS_APPEND` environment variables
    #    NOTE: --config-dir has lower priority than all these, --config-path has higher priority.
    # 3. enable the ${exit:<msg>} resolver for omegaconf/hydra
    experiment_run.patch_hydra()

    # temporarily set the environment and the arguments
    with temp_environ(env), temp_sys_args([experiment_run.__file__, *args]):
        # TODO: why does this not work when config_path is absolute?
        #      ie. config_path=os.path.join(os.path.dirname(experiment_run.__file__), 'config')
        hydra_main = hydra.main(config_path=None, config_name='config_test')(experiment_run.run_action)
        hydra_main()


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
