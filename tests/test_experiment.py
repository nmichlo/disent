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

import os
import os.path

import hydra
import pytest

import experiment.run as experiment_run
from tests.util import temp_sys_args


# ========================================================================= #
# TESTS                                                                     #
# ========================================================================= #


@pytest.mark.parametrize('args', [
    ['run_action=skip'],
    ['run_action=prepare_data'],
    ['run_action=train'],
])
def test_experiment_run(args):
    os.environ['HYDRA_FULL_ERROR'] = '1'

    # TODO: why does this not work when config_path is absolute?
    #      ie. config_path=os.path.join(os.path.dirname(experiment_run.__file__), 'config')
    with temp_sys_args([experiment_run.__file__, *args]):
        hydra_main = hydra.main(config_path='config', config_name='config_test')(experiment_run.run_action)
        hydra_main()


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
