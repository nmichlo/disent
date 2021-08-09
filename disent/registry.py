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
"""


# ========================================================================= #
# Registry Helper                                                           #
# ========================================================================= #


class _PathBuilder(object):
    """
    Path builder stores the path taken down attributes
      - This is used to trick pycharm type hinting. In the example
        below, `Cars3dData` will be an instance of `_PathBuilder`,
        but will type hint to `disent.dataset.data.Cars3dData`
        ```
        disent = _PathBuilder()
        if False:
            import disent.dataset.data
        Cars3dData = disent.dataset.data._groundtruth__cars3d.Cars3dData
        ```
    """

    def __init__(self, *segments):
        self.__segments = tuple(segments)

    def __getattr__(self, item: str):
        return _PathBuilder(*self.__segments, item)

    def _do_import_(self):
        import importlib
        import_module, import_name = '.'.join(self.__segments[:-1]), self.__segments[-1]
        try:
            module = importlib.import_module(import_module)
        except Exception as e:
            raise ImportError(f'failed to import module: {repr(import_module)} ({".".join(self.__segments)})') from e
        try:
            obj = getattr(module, import_name)
        except Exception as e:
            raise ImportError(f'failed to get attribute on module: {repr(import_name)} ({".".join(self.__segments)})') from e
        return obj


class _LazyImportPathsMeta:
    """
    Lazy import paths metaclass checks for stored instances of `_PathBuilder` on a class and returns the
    imported version of the attribute instead of the `_PathBuilder` itself.
      - Used to perform lazy importing of classes and objects inside a module
    """

    def __init__(cls, name, bases, dct):
        cls.__unimported = {}  # Dict[str, _PathBuilder]
        cls.__imported = {}    # Dict[str, Any]
        # check annotations
        for key, value in dct.items():
            if isinstance(value, _PathBuilder):
                assert str.islower(key) and str.isidentifier(key), f'registry key is not a lowercase identifier: {repr(key)}'
                cls.__unimported[key] = value

    def __contains__(cls, item):
        return (item in cls.__unimported)

    def __getitem__(cls, item):
        if item not in cls.__unimported:
            raise KeyError(f'invalid key: {repr(item)}, must be one of: {sorted(cls.__unimported.keys())}')
        if item not in cls.__imported:
            cls.__imported[item] = cls.__unimported[item]._do_import_()
        return cls.__imported[item]

    def __getattr__(cls, item):
        if item not in cls.__unimported:
            raise AttributeError(f'invalid attribute: {repr(item)}, must be one of: {sorted(cls.__unimported.keys())}')
        return cls[item]

    def __iter__(self):
        yield from self.__unimported.keys()


# ========================================================================= #
# Fake Imports                                                              #
# ========================================================================= #


# this is to trick the PyCharm type hinting system, used in
# conjunction with the `if False: import ...` statements which
# should never actually be run!
_disent          = _PathBuilder('disent')
_torch           = _PathBuilder('torch')
_torch_optimizer = _PathBuilder('torch_optimizer')

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
# ========================================================================= #


# changes here should also update `disent/dataset/data/__init__.py`
class DATASET(metaclass=_LazyImportPathsMeta):
    # groundtruth -- impl
    cars3d            = _disent.dataset.data._groundtruth__cars3d.Cars3dData
    dsprites          = _disent.dataset.data._groundtruth__dsprites.DSpritesData
    mpi3d             = _disent.dataset.data._groundtruth__mpi3d.Mpi3dData
    smallnorb         = _disent.dataset.data._groundtruth__norb.SmallNorbData
    shapes3d          = _disent.dataset.data._groundtruth__shapes3d.Shapes3dData
    xyblocks          = _disent.dataset.data._groundtruth__xyblocks.XYBlocksData
    xyobject          = _disent.dataset.data._groundtruth__xyobject.XYObjectData
    xysquares         = _disent.dataset.data._groundtruth__xysquares.XYSquaresData
    xysquares_minimal = _disent.dataset.data._groundtruth__xysquares.XYSquaresMinimalData


# changes here should also update `disent/dataset/sampling/__init__.py`
class SAMPLER(metaclass=_LazyImportPathsMeta):
    # single sampler
    single         = _disent.dataset.sampling._single.SingleSampler
    # ground truth samplers
    gt_dist        = _disent.dataset.sampling._groundtruth__dist.GroundTruthDistSampler
    gt_pair        = _disent.dataset.sampling._groundtruth__pair.GroundTruthPairSampler
    gt_pair_orig   = _disent.dataset.sampling._groundtruth__pair_orig.GroundTruthPairOrigSampler
    gt_single      = _disent.dataset.sampling._groundtruth__single.GroundTruthSingleSampler
    gt_triple      = _disent.dataset.sampling._groundtruth__triplet.GroundTruthTripleSampler
    # any dataset samplers
    random         = _disent.dataset.sampling._random__any.RandomSampler
    # episode samplers
    random_episode = _disent.dataset.sampling._random__episodes.RandomEpisodeSampler


# changes here should also update `disent/frameworks/ae/__init__.py` & `disent/frameworks/vae/__init__.py`
class FRAMEWORK(metaclass=_LazyImportPathsMeta):
    # ~=~=~=~=~=~=~=~=~=~=~=~=~ #
    # [AE] - supervised frameworks
    tae              = _disent.frameworks.ae._supervised__tae.TripletAe
    # [AE] - unsupervised frameworks
    ae               = _disent.frameworks.ae._unsupervised__ae.Ae
    # [AE] - weakly supervised frameworks
    # ~=~=~=~=~=~=~=~=~=~=~=~=~ #
    # [VAE] - supervised frameworks
    tvae             = _disent.frameworks.vae._supervised__tvae.TripletVae
    # [VAE] - unsupervised frameworks
    betatcvae        = _disent.frameworks.vae._unsupervised__betatcvae.BetaTcVae
    betavae          = _disent.frameworks.vae._unsupervised__betavae.BetaVae
    dfcvae           = _disent.frameworks.vae._unsupervised__dfcvae.DfcVae
    dipvae           = _disent.frameworks.vae._unsupervised__dipvae.DipVae
    infovae          = _disent.frameworks.vae._unsupervised__infovae.InfoVae
    vae              = _disent.frameworks.vae._unsupervised__vae.Vae
    # [VAE] - weakly supervised frameworks
    adavae           = _disent.frameworks.vae._weaklysupervised__adavae.AdaVae
    # ~=~=~=~=~=~=~=~=~=~=~=~=~ #
    # [AE - EXPERIMENTAL] - supervised frameworks
    exp__adanegtae   = _disent.frameworks.ae.experimental._supervised__adaneg_tae.AdaNegTripletAe
    # [AE - EXPERIMENTAL] - unsupervised frameworks
    exp__dotae       = _disent.frameworks.ae.experimental._unsupervised__dotae.DataOverlapTripletAe
    # [AE - EXPERIMENTAL] - weakly supervised frameworks
    exp__adaae       = _disent.frameworks.ae.experimental._weaklysupervised__adaae.AdaAe
    # ~=~=~=~=~=~=~=~=~=~=~=~=~ #
    # [VAE - EXPERIMENTAL] - supervised frameworks
    exp__adaavetvae  = _disent.frameworks.vae.experimental._supervised__adaave_tvae.AdaAveTripletVae
    exp__adanegtvae  = _disent.frameworks.vae.experimental._supervised__adaneg_tvae.AdaNegTripletVae
    exp__adatvae     = _disent.frameworks.vae.experimental._supervised__adatvae.AdaTripletVae
    exp__badavae     = _disent.frameworks.vae.experimental._supervised__badavae.BoundedAdaVae
    exp__gadavae     = _disent.frameworks.vae.experimental._supervised__gadavae.GuidedAdaVae
    exp__badatvae    = _disent.frameworks.vae.experimental._supervised__tbadavae.TripletBoundedAdaVae
    exp__gadatvae    = _disent.frameworks.vae.experimental._supervised__tgadavae.TripletGuidedAdaVae
    # [VAE - EXPERIMENTAL] - unsupervised frameworks
    exp__dorvae      = _disent.frameworks.vae.experimental._unsupervised__dorvae.DataOverlapRankVae
    exp__dotvae      = _disent.frameworks.vae.experimental._unsupervised__dotvae.DataOverlapTripletVae
    # [VAE - EXPERIMENTAL] - weakly supervised frameworks
    exp__augpos_tvae = _disent.frameworks.vae.experimental._weaklysupervised__augpostriplet.AugPosTripletVae
    exp__st_adavae   = _disent.frameworks.vae.experimental._weaklysupervised__st_adavae.SwappedTargetAdaVae
    exp__st_betavae  = _disent.frameworks.vae.experimental._weaklysupervised__st_betavae.SwappedTargetBetaVae
    # ~=~=~=~=~=~=~=~=~=~=~=~=~ #


# changes here should also update `disent/frameworks/helper/reconstructions.py`
class RECON_LOSS(metaclass=_LazyImportPathsMeta):
    # ~=~=~=~=~=~=~=~=~=~=~=~=~ #
    # [STANDARD LOSSES]
    # from the normal distribution - real values in the range [0, 1]
    mse                  = _disent.frameworks.helper.reconstructions.ReconLossHandlerMse
    # mean absolute error
    mae                  = _disent.frameworks.helper.reconstructions.ReconLossHandlerMae
    # ~=~=~=~=~=~=~=~=~=~=~=~=~ #
    # [STANDARD DISTRIBUTIONS]
    # from the bernoulli distribution - binary values in the set {0, 1}
    bce                  = _disent.frameworks.helper.reconstructions.ReconLossHandlerBce
    # reduces to bce - binary values in the set {0, 1}
    bernoulli            = _disent.frameworks.helper.reconstructions.ReconLossHandlerBernoulli
    # bernoulli with a computed offset to handle values in the range [0, 1]
    continuous_bernoulli = _disent.frameworks.helper.reconstructions.ReconLossHandlerContinuousBernoulli
    # handle all real values
    normal               = _disent.frameworks.helper.reconstructions.ReconLossHandlerNormal
    # ~=~=~=~=~=~=~=~=~=~=~=~=~ #
    # [EXPERIMENTAL LOSSES]
    mse4                 = _disent.frameworks.helper.reconstructions.ReconLossHandlerMse4  # scaled as if computed over outputs of the range [-1, 1] instead of [0, 1]
    mae2                 = _disent.frameworks.helper.reconstructions.ReconLossHandlerMae2  # scaled as if computed over outputs of the range [-1, 1] instead of [0, 1]
    # ~=~=~=~=~=~=~=~=~=~=~=~=~ #


# changes here should also update `disent/frameworks/helper/latent_distributions.py`
class LATENT_DIST(metaclass=_LazyImportPathsMeta):
    normal = _disent.frameworks.helper.latent_distributions.LatentDistsHandlerNormal
    laplace = _disent.frameworks.helper.latent_distributions.LatentDistsHandlerLaplace


# non-disent classes
class OPTIMIZER(metaclass=_LazyImportPathsMeta):
    # torch
    adadelta    = _torch.optim.adadelta.Adadelta
    adagrad     = _torch.optim.adagrad.Adagrad
    adam        = _torch.optim.adam.Adam
    adamax      = _torch.optim.adamax.Adamax
    adam_w      = _torch.optim.adamw.AdamW
    asgd        = _torch.optim.asgd.ASGD
    lbfgs       = _torch.optim.lbfgs.LBFGS
    rmsprop     = _torch.optim.rmsprop.RMSprop
    rprop       = _torch.optim.rprop.Rprop
    sgd         = _torch.optim.sgd.SGD
    sparse_adam = _torch.optim.sparse_adam.SparseAdam
    # torch_optimizer  [non-optimizers: Lookahead]
    a2grad_exp  = _torch_optimizer.A2GradExp
    a2grad_inc  = _torch_optimizer.A2GradInc
    a2grad_uni  = _torch_optimizer.A2GradUni
    acc_sgd     = _torch_optimizer.AccSGD
    adabelief   = _torch_optimizer.AdaBelief
    adabound    = _torch_optimizer.AdaBound
    adamod      = _torch_optimizer.AdaMod
    adafactor   = _torch_optimizer.Adafactor
    adahessian  = _torch_optimizer.Adahessian
    adam_p      = _torch_optimizer.AdamP
    aggmo       = _torch_optimizer.AggMo
    apollo      = _torch_optimizer.Apollo
    diffgrad    = _torch_optimizer.DiffGrad
    lamb        = _torch_optimizer.Lamb
    novograd    = _torch_optimizer.NovoGrad
    pid         = _torch_optimizer.PID
    qhadam      = _torch_optimizer.QHAdam
    qhm         = _torch_optimizer.QHM
    radam       = _torch_optimizer.RAdam
    ranger      = _torch_optimizer.Ranger
    ranger_qh   = _torch_optimizer.RangerQH
    ranger_va   = _torch_optimizer.RangerVA
    sgd_p       = _torch_optimizer.SGDP
    sgd_w       = _torch_optimizer.SGDW
    swats       = _torch_optimizer.SWATS
    shampoo     = _torch_optimizer.Shampoo
    yogi        = _torch_optimizer.Yogi


# changes here should also update `disent/metrics/__init__.py`
class METRIC(metaclass=_LazyImportPathsMeta):
    dci                 = _disent.metrics._dci.metric_dci
    factor_vae          = _disent.metrics._factor_vae.metric_factor_vae
    flatness            = _disent.metrics._flatness.metric_flatness
    flatness_components = _disent.metrics._flatness_components.metric_flatness_components
    mig                 = _disent.metrics._mig.metric_mig
    sap                 = _disent.metrics._sap.metric_sap
    unsupervised        = _disent.metrics._unsupervised.metric_unsupervised


# changes here should also update `disent/schedule/__init__.py`
class SCHEDULE(metaclass=_LazyImportPathsMeta):
    clip   = _disent.schedule._schedule.ClipSchedule
    cosine = _disent.schedule._schedule.CosineWaveSchedule
    cyclic = _disent.schedule._schedule.CyclicSchedule
    linear = _disent.schedule._schedule.LinearSchedule
    noop   = _disent.schedule._schedule.NoopSchedule


# changes here should also update `disent/model/ae/__init__.py`
class MODEL_ENCODER(metaclass=_LazyImportPathsMeta):
    conv64     = _disent.model.ae._vae_conv64.EncoderConv64
    conv64norm = _disent.model.ae._norm_conv64.EncoderConv64Norm
    fc         = _disent.model.ae._vae_fc.EncoderFC
    test       = _disent.model.ae._test.EncoderTest


# changes here should also update `disent/model/ae/__init__.py`
class MODEL_DECODER(metaclass=_LazyImportPathsMeta):
    conv64     = _disent.model.ae._vae_conv64.DecoderConv64
    conv64norm = _disent.model.ae._norm_conv64.DecoderConv64Norm
    fc         = _disent.model.ae._vae_fc.DecoderFC
    test       = _disent.model.ae._test.DecoderTest


# ========================================================================= #
# Registry of all Registries                                                #
# ========================================================================= #


# self-reference -- for testing purposes
class REGISTRY(metaclass=_LazyImportPathsMeta):
    dataset       = _disent.registry.DATASET
    sampler       = _disent.registry.SAMPLER
    framework     = _disent.registry.FRAMEWORK
    recon_loss    = _disent.registry.RECON_LOSS
    latent_dist   = _disent.registry.LATENT_DIST
    optimizer     = _disent.registry.OPTIMIZER
    metric        = _disent.registry.METRIC
    schedule      = _disent.registry.SCHEDULE
    model_encoder = _disent.registry.MODEL_ENCODER
    model_decoder = _disent.registry.MODEL_DECODER


# ========================================================================= #
# cleanup                                                                   #
# ========================================================================= #


del _disent
del _torch
del _torch_optimizer
