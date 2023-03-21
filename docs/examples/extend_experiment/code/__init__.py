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

import logging

import disent.registry as R
from docs.examples.extend_experiment.code.random_data import RandomData

log = logging.getLogger(__name__)


def register_to_disent():
    log.info("Registering example with disent!")

    # DATASETS.setm[...] is an alias for DATASETS[...] that only sets the value if it does not already exist.
    # -- register_to_disent should be able to be called multiple times in the same run!

    # register: datasets
    R.DATASETS.setm["pseudorandom"] = R.LazyImport("docs.examples.extend_experiment.code.random_data.RandomData")
    R.DATASETS.setm["xyblocks"] = R.LazyImport(
        "docs.examples.extend_experiment.code.groundtruth__xyblocks.XYBlocksData"
    )

    # register: VAEs
    R.FRAMEWORKS.setm["si_ada_vae"] = R.LazyImport(
        "docs.examples.extend_experiment.code.weaklysupervised__si_adavae.SwappedInputAdaVae"
    )
    R.FRAMEWORKS.setm["si_beta_vae"] = R.LazyImport(
        "docs.examples.extend_experiment.code.weaklysupervised__si_betavae.SwappedInputBetaVae"
    )
