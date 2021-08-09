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
disent = _PathBuilder('disent')
torch = _PathBuilder('torch')
torch_optimizer = _PathBuilder('torch_optimizer')

if False:
    # disent
    import disent.dataset.data
    import disent.dataset.sampling
    import disent.frameworks.ae
    import disent.frameworks.vae
    import disent.frameworks.helper.reconstructions
    import disent.frameworks.helper.latent_distributions
    import disent.metrics
    import disent.schedule
    import disent.model.ae
    # torch
    import torch.optim
    # torch_optimizer
    import torch_optimizer


# ========================================================================= #
# Registries                                                                #
# ========================================================================= #


# changes here should also update `disent/dataset/data/__init__.py`
class DATASET(metaclass=_LazyImportPathsMeta):
    # groundtruth -- impl
    cars3d            = disent.dataset.data._groundtruth__cars3d.Cars3dData
    dsprites          = disent.dataset.data._groundtruth__dsprites.DSpritesData
    mpi3d             = disent.dataset.data._groundtruth__mpi3d.Mpi3dData
    smallnorb         = disent.dataset.data._groundtruth__norb.SmallNorbData
    shapes3d          = disent.dataset.data._groundtruth__shapes3d.Shapes3dData
    xyblocks          = disent.dataset.data._groundtruth__xyblocks.XYBlocksData
    xyobject          = disent.dataset.data._groundtruth__xyobject.XYObjectData
    xysquares         = disent.dataset.data._groundtruth__xysquares.XYSquaresData
    xysquares_minimal = disent.dataset.data._groundtruth__xysquares.XYSquaresMinimalData


# changes here should also update `disent/dataset/sampling/__init__.py`
class SAMPLER(metaclass=_LazyImportPathsMeta):
    # single sampler
    single         = disent.dataset.sampling._single.SingleSampler
    # ground truth samplers
    gt_dist        = disent.dataset.sampling._groundtruth__dist.GroundTruthDistSampler
    gt_pair        = disent.dataset.sampling._groundtruth__pair.GroundTruthPairSampler
    gt_pair_orig   = disent.dataset.sampling._groundtruth__pair_orig.GroundTruthPairOrigSampler
    gt_single      = disent.dataset.sampling._groundtruth__single.GroundTruthSingleSampler
    gt_triple      = disent.dataset.sampling._groundtruth__triplet.GroundTruthTripleSampler
    # any dataset samplers
    random         = disent.dataset.sampling._random__any.RandomSampler
    # episode samplers
    random_episode = disent.dataset.sampling._random__episodes.RandomEpisodeSampler


# changes here should also update `disent/frameworks/ae/__init__.py` & `disent/frameworks/vae/__init__.py`
class FRAMEWORK(metaclass=_LazyImportPathsMeta):
    # [AE]
    # - supervised frameworks
    triplet_ae  = disent.frameworks.ae._supervised__tae.TripletAe
    # - unsupervised frameworks
    ae          = disent.frameworks.ae._unsupervised__ae.Ae
    # - weakly supervised frameworks
    # <ADD>

    # [VAE]
    # - supervised frameworks
    triplet_vae  = disent.frameworks.vae._supervised__tvae.TripletVae
    # - unsupervised frameworks
    beta_tc_vae  = disent.frameworks.vae._unsupervised__betatcvae.BetaTcVae
    beta_vae     = disent.frameworks.vae._unsupervised__betavae.BetaVae
    dfc_vae      = disent.frameworks.vae._unsupervised__dfcvae.DfcVae
    dip_vae      = disent.frameworks.vae._unsupervised__dipvae.DipVae
    info_vae     = disent.frameworks.vae._unsupervised__infovae.InfoVae
    vae          = disent.frameworks.vae._unsupervised__vae.Vae
    # - weakly supervised frameworks
    ada_vae      = disent.frameworks.vae._weaklysupervised__adavae.AdaVae

    # [AE - EXPERIMENTAL]



    # [VAE - EXPERIMENTAL]


# changes here should also update `disent/frameworks/helper/reconstructions.py`
class RECON_LOSS(metaclass=_LazyImportPathsMeta):
    # STANDARD LOSSES:
    # from the normal distribution - real values in the range [0, 1]
    mse = disent.frameworks.helper.reconstructions.ReconLossHandlerMse
    # mean absolute error
    mae = disent.frameworks.helper.reconstructions.ReconLossHandlerMae

    # STANDARD DISTRIBUTIONS:
    # from the bernoulli distribution - binary values in the set {0, 1}
    bce = disent.frameworks.helper.reconstructions.ReconLossHandlerBce
    # reduces to bce - binary values in the set {0, 1}
    bernoulli = disent.frameworks.helper.reconstructions.ReconLossHandlerBernoulli
    # bernoulli with a computed offset to handle values in the range [0, 1]
    continuous_bernoulli = disent.frameworks.helper.reconstructions.ReconLossHandlerContinuousBernoulli
    # handle all real values
    normal = disent.frameworks.helper.reconstructions.ReconLossHandlerNormal

    # EXPERIMENTAL LOSSES -- im just curious what would happen, haven't actually done the maths or thought about this much.
    mse4 = disent.frameworks.helper.reconstructions.ReconLossHandlerMse4  # scaled as if computed over outputs of the range [-1, 1] instead of [0, 1]
    mae2 = disent.frameworks.helper.reconstructions.ReconLossHandlerMae2  # scaled as if computed over outputs of the range [-1, 1] instead of [0, 1]


# changes here should also update `disent/frameworks/helper/latent_distributions.py`
class LATENT_DIST(metaclass=_LazyImportPathsMeta):
    normal = disent.frameworks.helper.latent_distributions.LatentDistsHandlerNormal
    laplace = disent.frameworks.helper.latent_distributions.LatentDistsHandlerLaplace


# non-disent classes
class OPTIMIZER(metaclass=_LazyImportPathsMeta):
    # torch
    adadelta    = torch.optim.adadelta.Adadelta
    adagrad     = torch.optim.adagrad.Adagrad
    adam        = torch.optim.adam.Adam
    adamax      = torch.optim.adamax.Adamax
    adam_w       = torch.optim.adamw.AdamW
    asgd        = torch.optim.asgd.ASGD
    lbfgs       = torch.optim.lbfgs.LBFGS
    rmsprop     = torch.optim.rmsprop.RMSprop
    rprop       = torch.optim.rprop.Rprop
    sgd         = torch.optim.sgd.SGD
    sparse_adam = torch.optim.sparse_adam.SparseAdam
    # torch_optimizer
    a2grad_exp  = torch_optimizer.A2GradExp
    a2grad_inc  = torch_optimizer.A2GradInc
    a2grad_uni  = torch_optimizer.A2GradUni
    acc_sgd     = torch_optimizer.AccSGD
    adabelief  = torch_optimizer.AdaBelief
    adabound   = torch_optimizer.AdaBound
    adamod     = torch_optimizer.AdaMod
    adafactor  = torch_optimizer.Adafactor
    adahessian = torch_optimizer.Adahessian
    adam_p     = torch_optimizer.AdamP
    aggmo      = torch_optimizer.AggMo
    apollo     = torch_optimizer.Apollo
    diffgrad   = torch_optimizer.DiffGrad
    lamb       = torch_optimizer.Lamb
    # lookahead  = torch_optimizer.Lookahead
    novograd   = torch_optimizer.NovoGrad
    pid        = torch_optimizer.PID
    qhadam     = torch_optimizer.QHAdam
    qhm        = torch_optimizer.QHM
    radam      = torch_optimizer.RAdam
    ranger     = torch_optimizer.Ranger
    ranger_qh  = torch_optimizer.RangerQH
    ranger_va  = torch_optimizer.RangerVA
    sgd_p      = torch_optimizer.SGDP
    sgd_w      = torch_optimizer.SGDW
    swats      = torch_optimizer.SWATS
    shampoo    = torch_optimizer.Shampoo
    yogi       = torch_optimizer.Yogi


# changes here should also update `disent/metrics/__init__.py`
class METRIC(metaclass=_LazyImportPathsMeta):
    dci                 = disent.metrics._dci.metric_dci
    factor_vae          = disent.metrics._factor_vae.metric_factor_vae
    flatness            = disent.metrics._flatness.metric_flatness
    flatness_components = disent.metrics._flatness_components.metric_flatness_components
    mig                 = disent.metrics._mig.metric_mig
    sap                 = disent.metrics._sap.metric_sap
    unsupervised        = disent.metrics._unsupervised.metric_unsupervised


# changes here should also update `disent/schedule/__init__.py`
class SCHEDULE(metaclass=_LazyImportPathsMeta):
    clip   = disent.schedule._schedule.ClipSchedule
    cosine = disent.schedule._schedule.CosineWaveSchedule
    cyclic = disent.schedule._schedule.CyclicSchedule
    linear = disent.schedule._schedule.LinearSchedule
    noop   = disent.schedule._schedule.NoopSchedule


# changes here should also update `disent/model/ae/__init__.py`
class MODEL_ENCODER(metaclass=_LazyImportPathsMeta):
    conv64     = disent.model.ae._vae_conv64.EncoderConv64
    conv64norm = disent.model.ae._norm_conv64.EncoderConv64Norm
    fc         = disent.model.ae._vae_fc.EncoderFC
    test       = disent.model.ae._test.EncoderTest


# changes here should also update `disent/model/ae/__init__.py`
class MODEL_DECODER(metaclass=_LazyImportPathsMeta):
    conv64     = disent.model.ae._vae_conv64.DecoderConv64
    conv64norm = disent.model.ae._norm_conv64.DecoderConv64Norm
    fc         = disent.model.ae._vae_fc.DecoderFC
    test       = disent.model.ae._test.DecoderTest


# ========================================================================= #
# Registry of all Registries                                                #
# ========================================================================= #


# self-reference -- for testing purposes
class REGISTRY(metaclass=_LazyImportPathsMeta):
    dataset       = disent.registry.DATASET
    sampler       = disent.registry.SAMPLER
    framework     = disent.registry.FRAMEWORK
    recon_loss    = disent.registry.RECON_LOSS
    latent_dist   = disent.registry.LATENT_DIST
    optimizer     = disent.registry.OPTIMIZER
    metric        = disent.registry.METRIC
    schedule      = disent.registry.SCHEDULE
    model_encoder = disent.registry.MODEL_ENCODER
    model_decoder = disent.registry.MODEL_DECODER


# ========================================================================= #
# Registry - Framework                                                      #
# ========================================================================= #
