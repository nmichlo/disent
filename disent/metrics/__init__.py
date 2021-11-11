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

# Expose
from ._dci import metric_dci
from ._factor_vae import metric_factor_vae
from ._mig import metric_mig
from ._sap import metric_sap
from ._unsupervised import metric_unsupervised


# ========================================================================= #
# Fast Metric Settings                                                      #
# ========================================================================= #


# helper imports
from disent.util.function import wrapped_partial as _wrapped_partial


FAST_METRICS = {
    'dci':                 _wrapped_partial(metric_dci,                 num_train=1000, num_test=500, boost_mode='sklearn'),
    'factor_vae':          _wrapped_partial(metric_factor_vae,          num_train=700,  num_eval=350, num_variance_estimate=1000),  # may not be accurate, but it just takes waay too long otherwise 20+ seconds
    'mig':                 _wrapped_partial(metric_mig,                 num_train=2000),
    'sap':                 _wrapped_partial(metric_sap,                 num_train=2000, num_test=1000),
    'unsupervised':        _wrapped_partial(metric_unsupervised,        num_train=2000),
}

DEFAULT_METRICS = {
    'dci':                 metric_dci,
    'factor_vae':          metric_factor_vae,
    'mig':                 metric_mig,
    'sap':                 metric_sap,
    'unsupervised':        metric_unsupervised,
}
