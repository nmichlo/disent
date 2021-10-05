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
        - support validation of objects
        - add factory methods
"""


# ========================================================================= #
# Fake Imports                                                              #
# ========================================================================= #

from typing import Type as _T
from disent.registry._registry_util import ImportRegistryMeta as _ImportRegistryMeta
from disent.registry._registry_util import import_info as _import_info


# ========================================================================= #
# Registries                                                                #
# ========================================================================= #


if False:
    import disent.dataset.data


# changes here should also update `disent/dataset/data/__init__.py`
class DATASET(metaclass=_ImportRegistryMeta):
    # [groundtruth -- impl]
    Cars3d:            _T['disent.dataset.data._groundtruth__cars3d.Cars3dData']              = _import_info()
    DSprites:          _T['disent.dataset.data._groundtruth__dsprites.DSpritesData']          = _import_info()
    Mpi3d:             _T['disent.dataset.data._groundtruth__mpi3d.Mpi3dData']                = _import_info()
    SmallNorb:         _T['disent.dataset.data._groundtruth__norb.SmallNorbData']             = _import_info()
    Shapes3d:          _T['disent.dataset.data._groundtruth__shapes3d.Shapes3dData']          = _import_info()
    XYBlocks:          _T['disent.dataset.data._groundtruth__xyblocks.XYBlocksData']          = _import_info()  # pragma: delete-on-release
    XYObject:          _T['disent.dataset.data._groundtruth__xyobject.XYObjectData']          = _import_info()
    XYSquares:         _T['disent.dataset.data._groundtruth__xysquares.XYSquaresData']        = _import_info()  # pragma: delete-on-release
    XYSquares_Minimal: _T['disent.dataset.data._groundtruth__xysquares.XYSquaresMinimalData'] = _import_info()  # pragma: delete-on-release


if False:
    import disent.dataset.sampling


# changes here should also update `disent/dataset/sampling/__init__.py`
class SAMPLER(metaclass=_ImportRegistryMeta):
    # [ground truth samplers]
    GT_Dist:        _T['disent.dataset.sampling._groundtruth__dist.GroundTruthDistSampler']          = _import_info()
    GT_Pair:        _T['disent.dataset.sampling._groundtruth__pair.GroundTruthPairSampler']          = _import_info()
    GT_PairOrig:    _T['disent.dataset.sampling._groundtruth__pair_orig.GroundTruthPairOrigSampler'] = _import_info()
    GT_Single:      _T['disent.dataset.sampling._groundtruth__single.GroundTruthSingleSampler']      = _import_info()
    GT_Triple:      _T['disent.dataset.sampling._groundtruth__triplet.GroundTruthTripleSampler']     = _import_info()
    # [any dataset samplers]
    Single:         _T['disent.dataset.sampling._single.SingleSampler']                              = _import_info()
    Random:         _T['disent.dataset.sampling._random__any.RandomSampler']                         = _import_info()
    # [episode samplers]
    RandomEpisode:  _T['disent.dataset.sampling._random__episodes.RandomEpisodeSampler']             = _import_info()


if False:
    import disent.frameworks.ae
    import disent.frameworks.vae
    import disent.frameworks.ae.experimental
    import disent.frameworks.vae.experimental


# changes here should also update `disent/frameworks/ae/__init__.py` & `disent/frameworks/vae/__init__.py`
class FRAMEWORK(metaclass=_ImportRegistryMeta):
    # [AE]
    TripletAe:               _T['disent.frameworks.ae._supervised__tae.TripletAe']          = _import_info(aliases=['tae'])
    Ae:                      _T['disent.frameworks.ae._unsupervised__ae.Ae']                = _import_info(aliases=['ae'])
    # [VAE]
    TripletVae:              _T['disent.frameworks.vae._supervised__tvae.TripletVae']       = _import_info(aliases=['tvae'])
    BetaTcVae:               _T['disent.frameworks.vae._unsupervised__betatcvae.BetaTcVae'] = _import_info(aliases=['betatc_vae'])
    BetaVae:                 _T['disent.frameworks.vae._unsupervised__betavae.BetaVae']     = _import_info(aliases=['beta_vae'])
    DfcVae:                  _T['disent.frameworks.vae._unsupervised__dfcvae.DfcVae']       = _import_info(aliases=['dfc_vae'])
    DipVae:                  _T['disent.frameworks.vae._unsupervised__dipvae.DipVae']       = _import_info(aliases=['dip_vae'])
    InfoVae:                 _T['disent.frameworks.vae._unsupervised__infovae.InfoVae']     = _import_info(aliases=['info_vae'])
    Vae:                     _T['disent.frameworks.vae._unsupervised__vae.Vae']             = _import_info(aliases=['vae'])
    AdaVae:                  _T['disent.frameworks.vae._weaklysupervised__adavae.AdaVae']   = _import_info(aliases=['ada_vae'])
    # [AE - EXPERIMENTAL]                                                                                                                                           # pragma: delete-on-release
    E_AdaNegTripletAe:       _T['disent.frameworks.ae.experimental._supervised__adaneg_tae.AdaNegTripletAe']             = _import_info(aliases=['X_adaneg_tae'])   # pragma: delete-on-release
    E_DataOverlapTripletAe:  _T['disent.frameworks.ae.experimental._unsupervised__dotae.DataOverlapTripletAe']           = _import_info(aliases=['X_dot_ae'])       # pragma: delete-on-release
    E_AdaAe:                 _T['disent.frameworks.ae.experimental._weaklysupervised__adaae.AdaAe']                      = _import_info(aliases=['X_ada_ae'])       # pragma: delete-on-release
    # [VAE - EXPERIMENTAL]                                                                                                                                          # pragma: delete-on-release
    E_AdaAveTripletVae:      _T['disent.frameworks.vae.experimental._supervised__adaave_tvae.AdaAveTripletVae']          = _import_info(aliases=['X_adaave_tvae'])  # pragma: delete-on-release
    E_AdaNegTripletVae:      _T['disent.frameworks.vae.experimental._supervised__adaneg_tvae.AdaNegTripletVae']          = _import_info(aliases=['X_adaneg_tvae'])  # pragma: delete-on-release
    E_AdaTripletVae:         _T['disent.frameworks.vae.experimental._supervised__adatvae.AdaTripletVae']                 = _import_info(aliases=['X_ada_tvae'])     # pragma: delete-on-release
    E_BoundedAdaVae:         _T['disent.frameworks.vae.experimental._supervised__badavae.BoundedAdaVae']                 = _import_info(aliases=['X_bada_vae'])     # pragma: delete-on-release
    E_GuidedAdaVae:          _T['disent.frameworks.vae.experimental._supervised__gadavae.GuidedAdaVae']                  = _import_info(aliases=['X_gada_vae'])     # pragma: delete-on-release
    E_TripletBoundedAdaVae:  _T['disent.frameworks.vae.experimental._supervised__tbadavae.TripletBoundedAdaVae']         = _import_info(aliases=['X_tbada_vae'])    # pragma: delete-on-release
    E_TripletGuidedAdaVae:   _T['disent.frameworks.vae.experimental._supervised__tgadavae.TripletGuidedAdaVae']          = _import_info(aliases=['X_tgada_vae'])    # pragma: delete-on-release
    E_DataOverlapRankVae:    _T['disent.frameworks.vae.experimental._unsupervised__dorvae.DataOverlapRankVae']           = _import_info(aliases=['X_dor_vae'])      # pragma: delete-on-release
    E_DataOverlapTripletVae: _T['disent.frameworks.vae.experimental._unsupervised__dotvae.DataOverlapTripletVae']        = _import_info(aliases=['X_dot_vae'])      # pragma: delete-on-release
    E_AugPosTripletVae:      _T['disent.frameworks.vae.experimental._weaklysupervised__augpostriplet.AugPosTripletVae']  = _import_info(aliases=['X_augpos_tvae'])  # pragma: delete-on-release
    E_SwappedTargetAdaVae:   _T['disent.frameworks.vae.experimental._weaklysupervised__st_adavae.SwappedTargetAdaVae']   = _import_info(aliases=['X_st_ada_vae'])   # pragma: delete-on-release
    E_SwappedTargetBetaVae:  _T['disent.frameworks.vae.experimental._weaklysupervised__st_betavae.SwappedTargetBetaVae'] = _import_info(aliases=['X_st_beta_vae'])  # pragma: delete-on-release


if False:
    import disent.frameworks.helper.reconstructions
    import disent.frameworks.helper.latent_distributions


# changes here should also update `disent/frameworks/helper/reconstructions.py`
class RECON_LOSS(metaclass=_ImportRegistryMeta):
    # [STANDARD LOSSES]
    Mse:                 _T['disent.frameworks.helper.reconstructions.ReconLossHandlerMse']                 = _import_info(aliases=['mse'])  # from the normal distribution - real values in the range [0, 1]
    Mae:                 _T['disent.frameworks.helper.reconstructions.ReconLossHandlerMae']                 = _import_info(aliases=['mae'])  # mean absolute error
    # [STANDARD DISTRIBUTIONS]
    Bce:                 _T['disent.frameworks.helper.reconstructions.ReconLossHandlerBce']                 = _import_info(aliases=['bce'])        # from the bernoulli distribution - binary values in the set {0, 1}
    Bernoulli:           _T['disent.frameworks.helper.reconstructions.ReconLossHandlerBernoulli']           = _import_info(aliases=['bernoulli'])  # reduces to bce - binary values in the set {0, 1}
    ContinuousBernoulli: _T['disent.frameworks.helper.reconstructions.ReconLossHandlerContinuousBernoulli'] = _import_info(aliases=['continuous_bernoulli', 'c_bernoulli'])  # bernoulli with a computed offset to handle values in the range [0, 1]
    Normal:              _T['disent.frameworks.helper.reconstructions.ReconLossHandlerNormal']              = _import_info(aliases=['normal'])     # handle all real values
    # [EXPERIMENTAL LOSSES]                                                                                                                                                                                                # pragma: delete-on-release
    Mse4:                _T['disent.frameworks.helper.reconstructions.ReconLossHandlerMse4']                = _import_info(aliases=['mse4'])  # scaled as if computed over outputs of the range [-1, 1] instead of [0, 1]  # pragma: delete-on-release
    Mae2:                _T['disent.frameworks.helper.reconstructions.ReconLossHandlerMae2']                = _import_info(aliases=['mae2'])  # scaled as if computed over outputs of the range [-1, 1] instead of [0, 1]  # pragma: delete-on-release


# changes here should also update `disent/frameworks/helper/latent_distributions.py`
class LATENT_DIST(metaclass=_ImportRegistryMeta):
    Normal:  _T['disent.frameworks.helper.latent_distributions.LatentDistsHandlerNormal']  = _import_info()
    Laplace: _T['disent.frameworks.helper.latent_distributions.LatentDistsHandlerLaplace'] = _import_info()


if False:
    import torch.optim
    import torch_optimizer


# non-disent classes
class OPTIMIZER(metaclass=_ImportRegistryMeta):
    # [torch]
    Adadelta:   _T['torch.optim.adadelta.Adadelta']      = _import_info(aliases=['adadelta'])
    Adagrad:    _T['torch.optim.adagrad.Adagrad']        = _import_info(aliases=['adagrad'])
    Adam:       _T['torch.optim.adam.Adam']              = _import_info(aliases=['adam'])
    Adamax:     _T['torch.optim.adamax.Adamax']          = _import_info(aliases=['adamax'])
    AdamW:      _T['torch.optim.adamw.AdamW']            = _import_info(aliases=['adam_w'])
    ASGD:       _T['torch.optim.asgd.ASGD']              = _import_info(aliases=['asgd'])
    LBFGS:      _T['torch.optim.lbfgs.LBFGS']            = _import_info(aliases=['lbfgs'])
    RMSprop:    _T['torch.optim.rmsprop.RMSprop']        = _import_info(aliases=['rmsprop'])
    Rprop:      _T['torch.optim.rprop.Rprop']            = _import_info(aliases=['rprop'])
    SGD:        _T['torch.optim.sgd.SGD']                = _import_info(aliases=['sgd'])
    SparseAdam: _T['torch.optim.sparse_adam.SparseAdam'] = _import_info(aliases=['sparse_adam'])
    # [torch_optimizer]
    AccSGD:    _T['torch_optimizer.AccSGD']   = _import_info(aliases=['acc_sgd'])
    AdaBound:  _T['torch_optimizer.AdaBound'] = _import_info(aliases=['ada_bound'])
    AdaMod:    _T['torch_optimizer.AdaMod']   = _import_info(aliases=['ada_mod'])
    AdamP:     _T['torch_optimizer.AdamP']    = _import_info(aliases=['adam_p'])
    AggMo:     _T['torch_optimizer.AggMo']    = _import_info(aliases=['agg_mo'])
    DiffGrad:  _T['torch_optimizer.DiffGrad'] = _import_info(aliases=['diff_grad'])
    Lamb:      _T['torch_optimizer.Lamb']     = _import_info(aliases=['lamb'])
    # 'torch_optimizer.Lookahead' is skipped because it is wrapped
    NovoGrad:  _T['torch_optimizer.NovoGrad'] = _import_info(aliases=['novograd'])
    PID:       _T['torch_optimizer.PID']      = _import_info(aliases=['pid'])
    QHAdam:    _T['torch_optimizer.QHAdam']   = _import_info(aliases=['qh_adam'])
    QHM:       _T['torch_optimizer.QHM']      = _import_info(aliases=['qhm'])
    RAdam:     _T['torch_optimizer.RAdam']    = _import_info(aliases=['radam'])
    Ranger:    _T['torch_optimizer.Ranger']   = _import_info(aliases=['ranger'])
    RangerQH:  _T['torch_optimizer.RangerQH'] = _import_info(aliases=['ranger_qh'])
    RangerVA:  _T['torch_optimizer.RangerVA'] = _import_info(aliases=['ranger_va'])
    SGDW:      _T['torch_optimizer.SGDW']     = _import_info(aliases=['sgd_w'])
    SGDP:      _T['torch_optimizer.SGDP']     = _import_info(aliases=['sgd_p'])
    Shampoo:   _T['torch_optimizer.Shampoo']  = _import_info(aliases=['shampoo'])
    Yogi:      _T['torch_optimizer.Yogi']     = _import_info(aliases=['yogi'])


if False:
    import disent.metrics


# changes here should also update `disent/metrics/__init__.py`
class METRIC(metaclass=_ImportRegistryMeta):
    dci:                 _T['disent.metrics._dci.metric_dci']                                 = _import_info()
    factor_vae:          _T['disent.metrics._factor_vae.metric_factor_vae']                   = _import_info()
    flatness:            _T['disent.metrics._flatness.metric_flatness']                       = _import_info()  # pragma: delete-on-release
    flatness_components: _T['disent.metrics._flatness_components.metric_flatness_components'] = _import_info()  # pragma: delete-on-release
    mig:                 _T['disent.metrics._mig.metric_mig']                                 = _import_info()
    sap:                 _T['disent.metrics._sap.metric_sap']                                 = _import_info()
    unsupervised:        _T['disent.metrics._unsupervised.metric_unsupervised']               = _import_info()


if False:
    import disent.schedule


# changes here should also update `disent/schedule/__init__.py`
class SCHEDULE(metaclass=_ImportRegistryMeta):
    Clip:       _T['disent.schedule._schedule.ClipSchedule']       = _import_info()
    CosineWave: _T['disent.schedule._schedule.CosineWaveSchedule'] = _import_info()
    Cyclic:     _T['disent.schedule._schedule.CyclicSchedule']     = _import_info()
    Linear:     _T['disent.schedule._schedule.LinearSchedule']     = _import_info()
    Noop:       _T['disent.schedule._schedule.NoopSchedule']       = _import_info()


if False:
    import disent.model.ae


# changes here should also update `disent/model/ae/__init__.py`
class MODEL(metaclass=_ImportRegistryMeta):
    # [DECODER]
    EncoderConv64:     _T['disent.model.ae._vae_conv64.EncoderConv64']      = _import_info()
    EncoderConv64Norm: _T['disent.model.ae._norm_conv64.EncoderConv64Norm'] = _import_info()
    EncoderFC:         _T['disent.model.ae._vae_fc.EncoderFC']              = _import_info()
    EncoderTest:       _T['disent.model.ae._test.EncoderTest']              = _import_info()
    # [ENCODER]
    DecoderConv64:     _T['disent.model.ae._vae_conv64.DecoderConv64']      = _import_info()
    DecoderConv64Norm: _T['disent.model.ae._norm_conv64.DecoderConv64Norm'] = _import_info()
    DecoderFC:         _T['disent.model.ae._vae_fc.DecoderFC']              = _import_info()
    DecoderTest:       _T['disent.model.ae._test.DecoderTest']              = _import_info()


# ========================================================================= #
# Registry of all Registries                                                #
# ========================================================================= #


# self-reference -- for testing purposes
class REGISTRY(metaclass=_ImportRegistryMeta):
    DATASET:     _T['disent.registry.DATASET']     = _import_info()
    SAMPLER:     _T['disent.registry.SAMPLER']     = _import_info()
    FRAMEWORK:   _T['disent.registry.FRAMEWORK']   = _import_info()
    RECON_LOSS:  _T['disent.registry.RECON_LOSS']  = _import_info()
    LATENT_DIST: _T['disent.registry.LATENT_DIST'] = _import_info()
    OPTIMIZER:   _T['disent.registry.OPTIMIZER']   = _import_info()
    METRIC:      _T['disent.registry.METRIC']      = _import_info()
    SCHEDULE:    _T['disent.registry.SCHEDULE']    = _import_info()
    MODEL:       _T['disent.registry.MODEL']       = _import_info()


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
