
# Expose
from ._dci import metric_dci
from ._factor_vae import metric_factor_vae
from ._mig import metric_mig
from ._sap import metric_sap
from ._unsupervised import metric_unsupervised

# Nathan Michlo et. al
from ._flatness import metric_flatness


# ========================================================================= #
# Fast Metric Settings                                                      #
# ========================================================================= #


from disent.util import wrapped_partial as _wrapped_partial


FAST_METRICS = {
    'dci':          _wrapped_partial(metric_dci,          num_train=1000, num_test=500, boost_mode='sklearn'),  # takes
    'factor_vae':   _wrapped_partial(metric_factor_vae,   num_train=700, num_eval=350, num_variance_estimate=1000),  # may not be accurate, but it just takes waay too long otherwise 20+ seconds
    'flatness':     _wrapped_partial(metric_flatness,     factor_repeats=128),
    'mig':          _wrapped_partial(metric_mig,          num_train=2000),
    'sap':          _wrapped_partial(metric_sap,          num_train=2000, num_test=1000),
    'unsupervised': _wrapped_partial(metric_unsupervised, num_train=2000),
}

DEFAULT_METRICS = {
    'dci':          metric_dci,
    'factor_vae':   metric_factor_vae,
    'flatness':     metric_flatness,
    'mig':          metric_mig,
    'sap':          metric_sap,
    'unsupervised': metric_unsupervised,
}
