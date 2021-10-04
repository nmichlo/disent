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

"""
The disent registry only contains implemented versions of
classes and functions, no interfaces are included.
  - for interfaces and creating your own data types, extend
    the base classes from the various locations.

*NB* All modules and classes are lazily imported!

# TODO: this needs to be more flexible
        - support custom registration
        - support aliases
        - support validation of objects
        - add factory methods
"""


# TODO: this file needs to be cleaned up!!!
# TODO: this file needs to be cleaned up!!!
# TODO: this file needs to be cleaned up!!!
# TODO: this file needs to be cleaned up!!!
# TODO: this file needs to be cleaned up!!!
# TODO: this file needs to be cleaned up!!!
# TODO: this file needs to be cleaned up!!!
# TODO: this file needs to be cleaned up!!!
# TODO: this file needs to be cleaned up!!!
# TODO: this file needs to be cleaned up!!!
# TODO: this file needs to be cleaned up!!!
# TODO: this file needs to be cleaned up!!!


# ========================================================================= #
# Fake Imports                                                              #
# ========================================================================= #


from disent.registry import _registry_util as _R


# this is to trick the PyCharm type hinting system, used in
# conjunction with the `if False: import ...` statements which
# should never actually be run!
_disent          = _R.PathBuilder('disent')
_torch           = _R.PathBuilder('torch')
_torch_optimizer = _R.PathBuilder('torch_optimizer')


if False:
    import disent           as _disent
    import torch            as _torch
    import torch_optimizer  as _torch_optimizer
    # force pycharm to load hints
    import disent.dataset.data                           as _
    import disent.dataset.sampling                       as _
    import disent.frameworks.ae                          as _
    import disent.frameworks.vae                         as _
    import disent.frameworks.helper.reconstructions      as _
    import disent.frameworks.helper.latent_distributions as _
    import disent.metrics                                as _
    import disent.schedule                               as _
    import disent.model.ae                               as _
    import torch.optim                                   as _


# ========================================================================= #
# Registries                                                                #
# TODO: registries should support multiple aliases
# ========================================================================= #


# changes here should also update `disent/dataset/data/__init__.py`
class DATASET(metaclass=_R.LazyImportMeta()):
    # [groundtruth -- impl]
    Cars3d            = _disent.dataset.data._groundtruth__cars3d.Cars3dData
    DSprites          = _disent.dataset.data._groundtruth__dsprites.DSpritesData
    Mpi3d             = _disent.dataset.data._groundtruth__mpi3d.Mpi3dData
    SmallNorb         = _disent.dataset.data._groundtruth__norb.SmallNorbData
    Shapes3d          = _disent.dataset.data._groundtruth__shapes3d.Shapes3dData
    XYObject          = _disent.dataset.data._groundtruth__xyobject.XYObjectData


# changes here should also update `disent/dataset/sampling/__init__.py`
class SAMPLER(metaclass=_R.LazyImportMeta()):
    # [ground truth samplers]
    GT_Dist        = _disent.dataset.sampling._groundtruth__dist.GroundTruthDistSampler
    GT_Pair        = _disent.dataset.sampling._groundtruth__pair.GroundTruthPairSampler
    GT_PairOrig   = _disent.dataset.sampling._groundtruth__pair_orig.GroundTruthPairOrigSampler
    GT_Single      = _disent.dataset.sampling._groundtruth__single.GroundTruthSingleSampler
    GT_Triple      = _disent.dataset.sampling._groundtruth__triplet.GroundTruthTripleSampler
    # [any dataset samplers]
    Single         = _disent.dataset.sampling._single.SingleSampler
    Random         = _disent.dataset.sampling._random__any.RandomSampler
    # [episode samplers]
    RandomEpisode = _disent.dataset.sampling._random__episodes.RandomEpisodeSampler


# changes here should also update `disent/frameworks/ae/__init__.py` & `disent/frameworks/vae/__init__.py`
class FRAMEWORK(metaclass=_R.LazyImportMeta()):
    # [AE]
    TripletAe              = _disent.frameworks.ae._supervised__tae.TripletAe
    Ae                     = _disent.frameworks.ae._unsupervised__ae.Ae
    # [VAE]
    TripletVae             = _disent.frameworks.vae._supervised__tvae.TripletVae
    BetaTcVae              = _disent.frameworks.vae._unsupervised__betatcvae.BetaTcVae
    BetaVae                = _disent.frameworks.vae._unsupervised__betavae.BetaVae
    DfcVae                 = _disent.frameworks.vae._unsupervised__dfcvae.DfcVae
    DipVae                 = _disent.frameworks.vae._unsupervised__dipvae.DipVae
    InfoVae                = _disent.frameworks.vae._unsupervised__infovae.InfoVae
    Vae                    = _disent.frameworks.vae._unsupervised__vae.Vae
    AdaVae                 = _disent.frameworks.vae._weaklysupervised__adavae.AdaVae


# changes here should also update `disent/frameworks/helper/reconstructions.py`
class RECON_LOSS(metaclass=_R.LazyImportMeta(to_lowercase=True)):
    # [STANDARD LOSSES]
    Mse                 = _disent.frameworks.helper.reconstructions.ReconLossHandlerMse  # from the normal distribution - real values in the range [0, 1]
    Mae                 = _disent.frameworks.helper.reconstructions.ReconLossHandlerMae  # mean absolute error
    # [STANDARD DISTRIBUTIONS]
    Bce                 = _disent.frameworks.helper.reconstructions.ReconLossHandlerBce                  # from the bernoulli distribution - binary values in the set {0, 1}
    Bernoulli           = _disent.frameworks.helper.reconstructions.ReconLossHandlerBernoulli            # reduces to bce - binary values in the set {0, 1}
    ContinuousBernoulli = _disent.frameworks.helper.reconstructions.ReconLossHandlerContinuousBernoulli  # bernoulli with a computed offset to handle values in the range [0, 1]
    Normal              = _disent.frameworks.helper.reconstructions.ReconLossHandlerNormal               # handle all real values


# changes here should also update `disent/frameworks/helper/latent_distributions.py`
class LATENT_DIST(metaclass=_R.LazyImportMeta()):
    Normal = _disent.frameworks.helper.latent_distributions.LatentDistsHandlerNormal
    Laplace = _disent.frameworks.helper.latent_distributions.LatentDistsHandlerLaplace


# non-disent classes
class OPTIMIZER(metaclass=_R.LazyImportMeta()):
    # [torch]
    Adadelta  = _torch.optim.adadelta.Adadelta
    Adagrad   = _torch.optim.adagrad.Adagrad
    Adam      = _torch.optim.adam.Adam
    Adamax    = _torch.optim.adamax.Adamax
    AdamW     = _torch.optim.adamw.AdamW
    ASGD      = _torch.optim.asgd.ASGD
    LBFGS     = _torch.optim.lbfgs.LBFGS
    RMSprop   = _torch.optim.rmsprop.RMSprop
    Rprop     = _torch.optim.rprop.Rprop
    SGD       = _torch.optim.sgd.SGD
    SparseAdam= _torch.optim.sparse_adam.SparseAdam
    # [torch_optimizer] - non-optimizers: Lookahead
    A2GradExp = _torch_optimizer.A2GradExp
    A2GradInc = _torch_optimizer.A2GradInc
    A2GradUni = _torch_optimizer.A2GradUni
    AccSGD    = _torch_optimizer.AccSGD
    AdaBelief = _torch_optimizer.AdaBelief
    AdaBound  = _torch_optimizer.AdaBound
    AdaMod    = _torch_optimizer.AdaMod
    Adafactor = _torch_optimizer.Adafactor
    Adahessian= _torch_optimizer.Adahessian
    AdamP     = _torch_optimizer.AdamP
    AggMo     = _torch_optimizer.AggMo
    Apollo    = _torch_optimizer.Apollo
    DiffGrad  = _torch_optimizer.DiffGrad
    Lamb      = _torch_optimizer.Lamb
    NovoGrad  = _torch_optimizer.NovoGrad
    PID       = _torch_optimizer.PID
    QHAdam    = _torch_optimizer.QHAdam
    QHM       = _torch_optimizer.QHM
    RAdam     = _torch_optimizer.RAdam
    Ranger    = _torch_optimizer.Ranger
    RangerQH  = _torch_optimizer.RangerQH
    RangerVA  = _torch_optimizer.RangerVA
    SGDP      = _torch_optimizer.SGDP
    SGDW      = _torch_optimizer.SGDW
    SWATS     = _torch_optimizer.SWATS
    Shampoo   = _torch_optimizer.Shampoo
    Yogi      = _torch_optimizer.Yogi


# changes here should also update `disent/metrics/__init__.py`
class METRIC(metaclass=_R.LazyImportMeta()):
    dci                 = _disent.metrics._dci.metric_dci
    factor_vae          = _disent.metrics._factor_vae.metric_factor_vae
    mig                 = _disent.metrics._mig.metric_mig
    sap                 = _disent.metrics._sap.metric_sap
    unsupervised        = _disent.metrics._unsupervised.metric_unsupervised


# changes here should also update `disent/schedule/__init__.py`
class SCHEDULE(metaclass=_R.LazyImportMeta()):
    Clip       = _disent.schedule._schedule.ClipSchedule
    CosineWave = _disent.schedule._schedule.CosineWaveSchedule
    Cyclic     = _disent.schedule._schedule.CyclicSchedule
    Linear     = _disent.schedule._schedule.LinearSchedule
    Noop       = _disent.schedule._schedule.NoopSchedule


# changes here should also update `disent/model/ae/__init__.py`
class MODEL(metaclass=_R.LazyImportMeta()):
    # [DECODER]
    EncoderConv64     = _disent.model.ae._vae_conv64.EncoderConv64
    EncoderConv64Norm = _disent.model.ae._norm_conv64.EncoderConv64Norm
    EncoderFC         = _disent.model.ae._vae_fc.EncoderFC
    EncoderTest       = _disent.model.ae._test.EncoderTest
    # [ENCODER]
    DecoderConv64     = _disent.model.ae._vae_conv64.DecoderConv64
    DecoderConv64Norm = _disent.model.ae._norm_conv64.DecoderConv64Norm
    DecoderFC         = _disent.model.ae._vae_fc.DecoderFC
    DecoderTest       = _disent.model.ae._test.DecoderTest


# ========================================================================= #
# Registry of all Registries                                                #
# ========================================================================= #


# self-reference -- for testing purposes
class REGISTRY(metaclass=_R.LazyImportMeta()):
    DATASET       = _disent.registry.DATASET
    SAMPLER       = _disent.registry.SAMPLER
    FRAMEWORK     = _disent.registry.FRAMEWORK
    RECON_LOSS    = _disent.registry.RECON_LOSS
    LATENT_DIST   = _disent.registry.LATENT_DIST
    OPTIMIZER     = _disent.registry.OPTIMIZER
    METRIC        = _disent.registry.METRIC
    SCHEDULE      = _disent.registry.SCHEDULE
    MODEL         = _disent.registry.MODEL


# ========================================================================= #
# cleanup                                                                   #
# ========================================================================= #


del _disent
del _torch
del _torch_optimizer
